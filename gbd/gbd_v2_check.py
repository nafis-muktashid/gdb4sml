"""
GBD v2 risk scale: reference prototype and numerical check (Nov 12, 2026).

This file checks the v2 equations in PROJECT_MEMORY.md (Entry 005) and in the pitch's Method
section. It is NOT the production builder: gbd.py is still v1 and must be ported.

Scope: one document, no role markers (S = empty). The boundary rule (Eq. 1) and the
turn-state / content projection (Eq. 2) are therefore not exercised here.

  python gbd_v2_check.py test    brute-force check of every v2 formula on a small corpus
  python gbd_v2_check.py bench   v1 (gbd.py) vs v2 on the synthetic benchmark, 3 corpus seeds

Needs numpy, scipy, faiss-cpu, gensim (bench only), and gbd.py + sanity_synthetic.py next to it.
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from gbd import GBDConfig, GBDTargets, OrderIndex, build_targets, context_features, nearest_contexts, _topr


@dataclass
class V2Config:
    K: int = 6
    Ks: int = 2
    M: int = 8
    M_search: int = 32
    R_pre: int = 64
    r: int = 16
    r_min: float = 1e-3
    gamma: float = 1.0
    alpha_max: float = 0.5
    gamma_alpha: float = 1.0
    use_semantic: bool = True
    n_fit: int = 20
    n_fit_pairs: int = 2000
    n0: float = 5.0
    beta0: float = 3.0
    chunk: int = 50_000
    seed: int = 0


# ---------------------------------------------------------------- Eq. 3: support prior, variance
def support_prior(orders, n_fit, V_act):
    """d0(k) = (D_k + 1) / (C_k + 1): D_k contexts seen exactly twice, C_k of them with equal next tokens."""
    d0 = np.zeros(len(orders))
    d0[0] = V_act
    for k in range(1, len(orders)):
        dbl = orders[k].n == 2
        D, C = int(dbl.sum()), int((dbl & (orders[k].d_obs == 1)).sum())
        d0[k] = d0[k - 1] if D < n_fit else float(np.clip((D + 1) / (C + 1), 1.0, V_act))
    return d0


def d_tilde(n, d_obs, f1, d0k, n0):
    return (n * (d_obs + f1 + 1.0) + n0 * d0k) / (n + n0)


def var_term(n, d_obs, f1, d0k, n0):
    """v = ln(1 + (d_tilde - 1) / n); +inf when n < 1."""
    n = np.asarray(n, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.log1p((d_tilde(n, d_obs, f1, d0k, n0) - 1.0) / n)
    return np.where(n >= 1, v, np.inf)


# ---------------------------------------------------------------- Eq. 3: bias fitting
def plugin_kl(Cp, Cq, beta0, prior):
    """KL(p_hat(c1) || p_tilde(c2)), p_tilde(w) = (N(c2,w) + beta0 * prior(w)) / (N(c2) + beta0)."""
    P, Q = Cp.toarray(), Cq.toarray()
    p = P / P.sum(1, keepdims=True)
    q = (Q + beta0 * prior[None, :]) / (Q.sum(1, keepdims=True) + beta0)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(p > 0, p * np.log(p / q), 0.0)
    return t.sum(1)


def fit_delta(orders, cfg, rng, prior):
    delta = np.zeros(cfg.K + 1)
    for j in range(cfg.K - 1, -1, -1):
        hi, lo = orders[j + 1], orders[j]
        cand = np.nonzero(hi.n >= cfg.n_fit)[0]
        step = 0.0
        if len(cand) >= 10:
            cand = rng.choice(cand, size=min(len(cand), cfg.n_fit_pairs), replace=False)
            lo_ids = lo.ctx[hi.rep_pos[cand]]
            vals = []
            for s in range(0, len(cand), 200):
                a, b = cand[s:s + 200], lo_ids[s:s + 200]
                kl = plugin_kl(hi.C[a], lo.C[b], cfg.beta0, prior)
                noise = (hi.d_obs[a] + hi.f1[a]) / 2.0 * (1.0 / hi.n[a] - 1.0 / lo.n[b])   # nested counts: minus
                vals.append(kl - noise)
            step = max(float(np.concatenate(vals).mean()), 0.0)                              # rectify after the mean
        delta[j] = delta[j + 1] + step
    return delta


def fit_semantic_bias(sidx, nD, nI, cfg, rng, prior):
    rows, cols = np.nonzero(np.ones_like(nI, dtype=bool))
    c, c2, dist = rows, nI[rows, cols], nD[rows, cols]
    sel = np.nonzero((sidx.n[c] >= cfg.n_fit) & (sidx.n[c2] >= cfg.n_fit))[0]
    if len(sel) < 10:
        return None                                   # too few reliable pairs: semantic branch is switched off
    sel = rng.choice(sel, size=min(len(sel), cfg.n_fit_pairs), replace=False)
    ys = []
    for s in range(0, len(sel), 200):
        i = sel[s:s + 200]
        kl = plugin_kl(sidx.C[c[i]], sidx.C[c2[i]], cfg.beta0, prior)
        noise = ((sidx.d_obs[c[i]] + sidx.f1[c[i]]) / (2.0 * sidx.n[c[i]])
                 + (sidx.d_obs[c2[i]] + sidx.f1[c2[i]]) / (2.0 * sidx.n[c2[i]]))           # disjoint counts: plus
        ys.append(kl - noise)                                                               # not rectified
    y, x = np.concatenate(ys), dist[sel]
    a, b = np.linalg.lstsq(np.stack([np.ones_like(x), x], 1), y, rcond=None)[0]
    if b < 0:
        a, b = float(y.mean()), 0.0
    return float(a), float(b)


def _exp_neg(r):
    return np.where(np.isfinite(r), np.exp(-np.where(np.isfinite(r), r, 0.0)), 0.0)


# ---------------------------------------------------------------- main builder
def build_targets_v2(tokens, V, E, cfg: V2Config):
    tokens = np.asarray(tokens, dtype=np.int64)
    T = len(tokens)
    rng = np.random.default_rng(cfg.seed)
    orders, h = [], np.zeros(T, dtype=np.uint64)
    for k in range(cfg.K + 1):
        idx = OrderIndex(tokens, k, V, h)
        h = idx.h
        idx.truncate(cfg.R_pre)
        orders.append(idx)
    uni = np.bincount(tokens, minlength=V).astype(np.float64)
    V_act = int((uni > 0).sum())
    prior = uni / uni.sum()
    d0 = support_prior(orders, cfg.n_fit, V_act)
    delta = fit_delta(orders, cfg, rng, prior)
    phi = lambda r: np.where(np.isfinite(r), 1.0 / np.maximum(r, cfg.r_min), 0.0)

    use_sem = cfg.use_semantic and E is not None and cfg.M > 0
    a = b = None
    if use_sem:
        sidx = orders[cfg.Ks]
        F = context_features(tokens, sidx, E, cfg.gamma)
        nD, nI = nearest_contexts(F, cfg.M_search)
        fit = fit_semantic_bias(sidx, nD, nI, cfg, rng, prior)
        use_sem = fit is not None
    if use_sem:
        a, b = fit
        bias_nb = np.maximum(a + b * nD, 0.0)
        r_nb = delta[cfg.Ks] + bias_nb + var_term(sidx.n[nI], sidx.d_obs[nI], sidx.f1[nI], d0[cfg.Ks], cfg.n0)
        keep = np.argsort(r_nb, axis=1, kind="stable")[:, : cfg.M]
        nI = np.take_along_axis(nI, keep, 1)
        r_nb = np.take_along_axis(r_nb, keep, 1)
        bias_nb = np.take_along_axis(bias_nb, keep, 1)
        wt = phi(r_nb)
        wt = wt / wt.sum(1, keepdims=True)
        N_sem = 1.0 / (wt ** 2 / sidx.n[nI]).sum(1)
        d_bar = (wt * d_tilde(sidx.n[nI], sidx.d_obs[nI], sidx.f1[nI], d0[cfg.Ks], cfg.n0)).sum(1)
        r_sem_ctx = delta[cfg.Ks] + (wt * bias_nb).sum(1) + np.log1p((d_bar - 1.0) / N_sem)
        Cs_norm = sp.diags(1.0 / np.maximum(sidx.tr_sum, 1e-12)) @ sidx.Ctr

    out_ids = np.zeros((T, cfg.r), np.int32)
    out_p = np.zeros((T, cfg.r), np.float32)
    o_R, o_r0, o_ro, o_rs, o_lam = (np.zeros(T) for _ in range(5))
    k_star = np.zeros(T, np.int16)
    for s in range(0, T, cfg.chunk):
        pos = np.arange(s, min(s + cfg.chunk, T))
        P, x = len(pos), tokens[pos]
        risks, mats, corr, n_all = [], [], [], []
        for k in range(cfg.K + 1):
            idx = orders[k]
            ctx = idx.ctx[pos]
            valid = ctx >= 0
            cs = np.where(valid, ctx, 0)
            cnt = np.asarray(idx.C[cs, x]).ravel() * valid
            n = (idx.n[cs] - 1.0) * valid
            d = idx.d_obs[cs] - (cnt == 1)
            f1 = idx.f1[cs] - (cnt == 1) + (cnt == 2)
            in_tr = (np.asarray(idx.Ctr[cs, x]).ravel() > 0) * valid
            s_tr = idx.tr_sum[cs] - in_tr
            r = delta[k] + var_term(n, d, f1, d0[k], cfg.n0)
            r[(~valid) | (s_tr <= 0)] = np.inf
            risks.append(r)
            mats.append((idx.Ctr, cs, s_tr))
            corr.append(in_tr.astype(float))
            n_all.append(np.where(valid, n, 0))
        Rm, n_all = np.stack(risks, 1), np.stack(n_all, 1)
        has, kk = n_all >= 1, np.arange(cfg.K + 1)
        k_star[pos] = np.where(has.any(1), kk[np.argmax(has * (kk + 1), axis=1)], 0)
        ph = phi(Rm)
        u = ph / np.maximum(ph.sum(1, keepdims=True), 1e-300)
        r_ord = -np.log(np.maximum((u * _exp_neg(Rm)).sum(1), 1e-300))          # Eq. 4: mixture bound
        if use_sem:
            cs_s = sidx.ctx[pos]
            vs = cs_s >= 0
            cs_s0 = np.where(vs, cs_s, 0)
            r_sem = np.where(vs, r_sem_ctx[cs_s0], np.inf)
        else:
            r_sem = np.full(P, np.inf)
        po, ps = phi(r_ord), phi(r_sem)
        lam = po / np.maximum(po + ps, 1e-300)
        R = -np.log(np.maximum(lam * _exp_neg(r_ord) + (1 - lam) * _exp_neg(r_sem), 1e-300))
        o_R[pos], o_r0[pos], o_ro[pos], o_rs[pos], o_lam[pos] = R, Rm[:, 0], r_ord, r_sem, lam

        Q = sp.csr_matrix((P, V))
        rowsP = np.arange(P)
        for j, (Ctr, cs, s_tr) in enumerate(mats):
            wj = lam * u[:, j] / np.maximum(s_tr, 1e-12)
            nz = wj > 0
            if not nz.any():
                continue
            Q = Q + sp.csr_matrix((wj[nz], (rowsP[nz], cs[nz])), shape=(P, Ctr.shape[0])) @ Ctr
            c = wj * corr[j]
            m = c > 0
            if m.any():
                Q = Q - sp.csr_matrix((c[m], (rowsP[m], x[m])), shape=(P, V))
        if use_sem:
            Wn = (1 - lam)[:, None] * wt[cs_s0] * vs[:, None]
            m = Wn > 0
            Q = Q + sp.csr_matrix((Wn[m], (np.nonzero(m)[0], nI[cs_s0][m])), shape=(P, Cs_norm.shape[0])) @ Cs_norm
        Q = Q.tocsr()
        Q.data[Q.data < 1e-12] = 0.0
        Q.eliminate_zeros()
        ids, pr = _topr(Q, cfg.r)
        empty = pr.sum(1) <= 0
        ids[empty, 0] = x[empty]
        pr[empty, 0] = 1.0
        out_ids[pos] = ids
        out_p[pos] = pr / pr.sum(1, keepdims=True)
    # Eq. 5: alpha from the order-branch risk, anchored on the unigram risk
    alpha = cfg.alpha_max * np.clip(1.0 - o_ro / np.maximum(o_r0, cfg.r_min), 0.0, 1.0) ** cfg.gamma_alpha
    tg = GBDTargets(ids=out_ids, probs=out_p, alpha=alpha.astype(np.float32), R=o_R.astype(np.float32),
                    w_exact=o_lam.astype(np.float32), k_star=k_star,
                    fitted={"delta": delta, "a": a, "b": b, "d0": d0})
    tg.r_ord, tg.r_sem, tg.r0 = o_ro, o_rs, o_r0
    return tg


# ---------------------------------------------------------------- brute-force check of every formula
def brute_force_check():
    rng = np.random.default_rng(3)
    V, T, D = 12, 3000, 8
    P = rng.dirichlet(np.ones(V) * 0.3, size=V)
    tokens = np.zeros(T, np.int64)
    for t in range(1, T):
        tokens[t] = rng.choice(V, p=P[tokens[t - 1]])
    E = rng.normal(size=(V, D))
    cfg = V2Config(K=3, Ks=2, M=4, M_search=10, R_pre=V, r=V, n_fit=5, n_fit_pairs=10**9)
    tg = build_targets_v2(tokens, V, E, cfg)

    occ = [defaultdict(list) for _ in range(cfg.K + 1)]            # context tuple -> positions
    for k in range(cfg.K + 1):
        for t in range(k, T):
            occ[k][tuple(tokens[t - k:t])].append(t)
    cnt = [{c: np.bincount(tokens[p], minlength=V).astype(float) for c, p in o.items()} for o in occ]
    uni = np.bincount(tokens, minlength=V).astype(float)
    V_act, prior = int((uni > 0).sum()), uni / uni.sum()

    def dhat(v):
        return (v > 0).sum() + (v == 1).sum() + 1.0

    d0 = [float(V_act)]
    for k in range(1, cfg.K + 1):
        dbl = [v for v in cnt[k].values() if v.sum() == 2]
        Dk, Ck = len(dbl), sum((v > 0).sum() == 1 for v in dbl)
        d0.append(d0[-1] if Dk < cfg.n_fit else float(np.clip((Dk + 1) / (Ck + 1), 1, V_act)))
    assert np.allclose(d0, tg.fitted["d0"]), "d0"

    def var(v, k):
        n = v.sum()
        return np.log(1 + ((n * dhat(v) + cfg.n0 * d0[k]) / (n + cfg.n0) - 1) / n)

    def kl_s(v1, v2):
        p = v1 / v1.sum()
        q = (v2 + cfg.beta0 * prior) / (v2.sum() + cfg.beta0)
        return float(np.sum(p[p > 0] * np.log(p[p > 0] / q[p > 0])))

    delta = np.zeros(cfg.K + 1)
    for j in range(cfg.K - 1, -1, -1):
        vals = [kl_s(v, cnt[j][c[1:]]) - (dhat(v) - 1) / 2 * (1 / v.sum() - 1 / cnt[j][c[1:]].sum())
                for c, v in cnt[j + 1].items() if v.sum() >= cfg.n_fit]
        delta[j] = delta[j + 1] + (max(np.mean(vals), 0.0) if len(vals) >= 10 else 0.0)
    assert np.allclose(delta, tg.fitted["delta"], atol=1e-9), "delta"

    # candidate neighbours: same kNN call as the builder (v1 code, already tested); everything after it is brute force
    h, sidx = np.zeros(T, dtype=np.uint64), None
    for k in range(cfg.Ks + 1):
        sidx = OrderIndex(tokens, k, V, h)
        h = sidx.h
    ctxs = [tuple(tokens[p - cfg.Ks:p]) for p in sidx.rep_pos]
    Dm, nbr = nearest_contexts(context_features(tokens, sidx, E, cfg.gamma), cfg.M_search)
    xs, ys = [], []
    for i, c in enumerate(ctxs):
        for dist, j in zip(Dm[i], nbr[i]):
            v1, v2 = cnt[cfg.Ks][c], cnt[cfg.Ks][ctxs[j]]
            if v1.sum() >= cfg.n_fit and v2.sum() >= cfg.n_fit:
                xs.append(dist)
                ys.append(kl_s(v1, v2) - (dhat(v1) - 1) / (2 * v1.sum()) - (dhat(v2) - 1) / (2 * v2.sum()))
    a, b = np.linalg.lstsq(np.stack([np.ones(len(xs)), xs], 1), np.array(ys), rcond=None)[0]
    if b < 0:
        a, b = np.mean(ys), 0.0
    assert abs(a - tg.fitted["a"]) < 1e-4 and abs(b - tg.fitted["b"]) < 1e-4, ("a,b", a, b, tg.fitted)

    phi = lambda r: 1.0 / max(r, cfg.r_min)
    cid = {c: i for i, c in enumerate(ctxs)}
    worst = 0.0
    for t in rng.choice(np.arange(cfg.K, T), size=60, replace=False):
        x = tokens[t]
        comps = []
        for k in range(cfg.K + 1):
            v = cnt[k][tuple(tokens[t - k:t])].copy()
            v[x] -= 1
            if v.sum() >= 1:
                comps.append((delta[k] + var(v, k), v / v.sum()))
        r0 = comps[0][0]
        u = np.array([phi(r) for r, _ in comps])
        u /= u.sum()
        p_ord = sum(ui * p for ui, (_, p) in zip(u, comps))
        r_ord = -np.log(sum(ui * np.exp(-r) for ui, (r, _) in zip(u, comps)))
        i = cid[tuple(tokens[t - cfg.Ks:t])]
        cand = []
        for dist, j in zip(Dm[i], nbr[i]):
            v = cnt[cfg.Ks][ctxs[j]]
            bias = max(a + b * dist, 0.0)
            cand.append((delta[cfg.Ks] + bias + var(v, cfg.Ks), bias, v))
        cand = sorted(cand, key=lambda z: z[0])[: cfg.M]
        w = np.array([phi(z[0]) for z in cand])
        w /= w.sum()
        p_sem = sum(wi * z[2] / z[2].sum() for wi, z in zip(w, cand))
        N_sem = 1 / sum(wi ** 2 / z[2].sum() for wi, z in zip(w, cand))
        d_bar = sum(wi * (z[2].sum() * dhat(z[2]) + cfg.n0 * d0[cfg.Ks]) / (z[2].sum() + cfg.n0) for wi, z in zip(w, cand))
        r_sem = delta[cfg.Ks] + sum(wi * z[1] for wi, z in zip(w, cand)) + np.log(1 + (d_bar - 1) / N_sem)
        lam = phi(r_ord) / (phi(r_ord) + phi(r_sem))
        q = lam * p_ord + (1 - lam) * p_sem
        R = -np.log(lam * np.exp(-r_ord) + (1 - lam) * np.exp(-r_sem))
        alpha = cfg.alpha_max * max(1 - r_ord / max(r0, cfg.r_min), 0.0) ** cfg.gamma_alpha
        q_impl = np.zeros(V)
        np.add.at(q_impl, tg.ids[t], tg.probs[t])
        errs = [np.abs(q - q_impl).max(), abs(r_ord - tg.r_ord[t]), abs(r_sem - tg.r_sem[t]), abs(lam - tg.w_exact[t]),
                abs(R - tg.R[t]), abs(alpha - tg.alpha[t])]
        worst = max(worst, max(errs))
        assert max(errs) < 1e-4, (t, errs)
    # numeric check of the two inequalities used in the spec
    for _ in range(2000):
        J, d = rng.integers(2, 6), rng.integers(3, 9)
        p = rng.dirichlet(np.ones(d))
        qs = rng.dirichlet(np.ones(d) * 0.5, size=J) * 0.98 + 0.02 / d
        w = rng.dirichlet(np.ones(J))
        kls = np.array([np.sum(p * np.log(p / qj)) for qj in qs])
        assert np.sum(p * np.log(p / (w @ qs))) <= -np.log(np.sum(w * np.exp(-kls))) + 1e-12
        assert -np.log(np.sum(w * np.exp(-kls))) <= np.sum(w * kls) + 1e-12
    for d, n in [(4, 1), (4, 3), (10, 1), (10, 5), (30, 2), (30, 20)]:
        p = rng.dirichlet(np.ones(d))
        e = np.mean([(lambda ph: np.sum(ph[ph > 0] * np.log(ph[ph > 0] / p[ph > 0])))(rng.multinomial(n, p) / n)
                     for _ in range(20000)])
        assert e <= np.log(1 + (d - 1) / n) + 0.01, (d, n, e)
    print(f"brute-force check passed: d0, delta, a, b exact; q_t, r_ord, r_sem, lambda_t, R_t, alpha_t on 60 positions "
          f"(max abs error {worst:.1e}); mixture bound on 2000 random cases; variance bound on 6 (d, n) settings")


# ---------------------------------------------------------------- synthetic benchmark, v1 vs v2
def bench(seeds=(0, 1, 2), T=200_000, n_eval=8000):
    from gensim.models import Word2Vec
    from scipy.stats import spearmanr
    from sanity_synthetic import make_language, sample, true_dist

    def score(tg, tokens, idx, cls, emit, trans):
        V = len(cls)
        kl, tv, top = [], [], []
        for t in idx:
            p = true_dist(tokens, t, cls, emit, trans)
            q = np.zeros(V)
            np.add.at(q, tg.ids[t], tg.probs[t])
            qs = 0.99 * q + 0.01 / V
            kl.append(np.sum(p * np.log(p / qs + 1e-300) * (p > 0)))
            tv.append(0.5 * np.abs(p - q).sum())
            top.append(q.max())
        return np.array(kl), np.array(tv), np.array(top)

    rows, groups, extra = defaultdict(list), defaultdict(list), defaultdict(list)
    for seed in seeds:
        cls, emit, trans = make_language(seed=seed)
        tokens = sample(T, cls, emit, trans, seed=seed + 100)
        V = len(cls)
        sents = [[str(x) for x in tokens[i:i + 1000]] for i in range(0, T, 1000)]
        w2v = Word2Vec(sents, vector_size=32, window=2, min_count=1, sg=1, epochs=5, workers=1, seed=0)
        E = np.stack([w2v.wv[str(v)] if str(v) in w2v.wv else np.random.default_rng(v).normal(size=32) for v in range(V)])
        idx = np.random.default_rng(5).choice(np.arange(8, T), size=n_eval, replace=False)
        h, ns = np.zeros(T, dtype=np.uint64), []
        for k in range(5):
            o = OrderIndex(tokens, k, V, h)
            h = o.h
            ns.append(o.n[o.ctx[idx]] - 1)
        rare = ns[2] <= 3
        g = {"order-4 context seen once elsewhere": ns[4] == 1,
             "no order-4 match, order-3 seen once": (ns[4] == 0) & (ns[3] == 1),
             "order-3 context seen 5+ times": ns[3] >= 5,
             "order-2 context seen 20+ times": ns[2] >= 20,
             "order-2 context never seen elsewhere": ns[2] == 0,
             "all positions": np.ones(n_eval, bool)}
        runs = {
            "v1 order backoff only": build_targets(tokens, V, None, GBDConfig(K=4, use_semantic=False, r=32, risk_version=1)),
            "v2 order backoff only": build_targets_v2(tokens, V, None, V2Config(K=4, use_semantic=False, r=32)),
            "v1 full": build_targets(tokens, V, E, GBDConfig(K=4, Ks=2, M=8, M_search=32, r=32, risk_version=1)),
            "v2 full": build_targets_v2(tokens, V, E, V2Config(K=4, Ks=2, M=8, M_search=32, r=32)),
        }
        for name, tg in runs.items():
            kl, tv, top = score(tg, tokens, idx, cls, emit, trans)
            a = tg.alpha[idx]
            driver = tg.r_ord[idx] if name.startswith("v2") else tg.R[idx]     # the risk that sets alpha_t
            rows[name].append([kl.mean(), tv.mean(), kl[rare].mean(), tv[rare].mean(),
                               spearmanr(driver, kl).correlation, spearmanr(tg.R[idx], kl).correlation,
                               np.quantile(a, 0.05), np.quantile(a, 0.95), (a == 0).mean()])
            groups[name].append([[m.mean(), top[m].mean(), driver[m].mean(), a[m].mean(), kl[m].mean()] for m in g.values()])
            if name == "v2 full":
                extra["fit"].append(tg.fitted)
                extra["r_sem_vs_full"].append(spearmanr(tg.r_sem[idx], kl).correlation)
        print(f"seed {seed} done", flush=True)

    print("\nTarget quality and risk calibration, mean over corpus seeds", list(seeds), f"({n_eval} positions each)")
    print(f"{'target':24s} {'KL all':>7s} {'TV all':>7s} {'KL rare':>8s} {'TV rare':>8s} {'Sp(alpha-risk,KL)':>18s} {'Sp(R_t,KL)':>11s} {'a 5%':>6s} {'a 95%':>6s} {'a=0':>6s}")
    for name, r in rows.items():
        m, sd = np.mean(r, 0), np.std(r, 0)
        print(f"{name:24s} {m[0]:7.2f} {m[1]:7.2f} {m[2]:8.2f} {m[3]:8.2f} {m[4]:11.2f} ± {sd[4]:.2f} {m[5]:6.2f} ± {sd[5]:.2f} {m[6]:6.2f} {m[7]:6.2f} {m[8]:6.1%}")
    for name in rows:
        print(f"\n[{name}]  {'positions':38s} {'share':>6s} {'top p':>6s} {'risk':>6s} {'alpha':>6s} {'trueKL':>7s}")
        gm = np.mean(groups[name], 0)
        for lab, v in zip(g, gm):
            print(f"  {lab:46s} {v[0]:6.1%} {v[1]:6.2f} {v[2]:6.2f} {v[3]:6.3f} {v[4]:7.2f}")
    print("\nv2 full, fitted constants per seed:")
    for f in extra["fit"]:
        print("  delta", np.round(f["delta"], 2), " a %.2f b %.2f" % (f["a"], f["b"]), " d0", np.round(f["d0"], 1))
    print("Spearman(r_sem, true KL of q_t), v2 full, per seed:", np.round(extra["r_sem_vs_full"], 2))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"
    brute_force_check() if mode == "test" else bench()
