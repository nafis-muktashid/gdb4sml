"""
Sanity check of target quality on a synthetic language with a KNOWN next-token distribution.

Generator: V tokens in C classes ("synonym sets"). Classes follow a sparse 2nd-order
Markov chain; a token is emitted from its class with a fixed within-class distribution.
The true p(x_t | x_{t-2}, x_{t-1}) is therefore known exactly.

This is built so that synonym structure exists. It checks that the estimator does
what the math says; it is NOT evidence about real language.

Run: python sanity_synthetic.py
"""
import time

import numpy as np
from gensim.models import Word2Vec

from gbd import GBDConfig, build_targets


def make_language(V=300, C=60, conc=0.08, seed=0):
    rng = np.random.default_rng(seed)
    cls = np.repeat(np.arange(C), V // C)
    emit = np.zeros(V)
    for c in range(C):
        m = cls == c
        emit[m] = rng.dirichlet(np.ones(m.sum()))
    trans = rng.dirichlet(np.ones(C) * conc, size=(C, C))  # P(c_t | c_{t-2}, c_{t-1})
    return cls, emit, trans


def sample(T, cls, emit, trans, seed=1):
    rng = np.random.default_rng(seed)
    C = trans.shape[0]
    members = [np.nonzero(cls == c)[0] for c in range(C)]
    cs = np.zeros(T, dtype=np.int64)
    cs[:2] = rng.integers(0, C, 2)
    u = rng.random(T)
    cdf = np.cumsum(trans, axis=2)
    for t in range(2, T):
        cs[t] = min(np.searchsorted(cdf[cs[t - 2], cs[t - 1]], u[t]), C - 1)
    toks = np.array([rng.choice(members[c], p=emit[members[c]] / emit[members[c]].sum()) for c in cs])
    return toks


def true_dist(tokens, t, cls, emit, trans):
    pc = trans[cls[tokens[t - 2]], cls[tokens[t - 1]]]
    return pc[cls] * emit


def evaluate(name, tg, tokens, cls, emit, trans, idx, rare_mask):
    V = len(cls)
    kls, tvs = [], []
    for t in idx:
        p = true_dist(tokens, t, cls, emit, trans)
        q = np.zeros(V)
        np.add.at(q, tg.ids[t], tg.probs[t])
        qs = 0.99 * q + 0.01 / V                     # floor so KL is finite
        kls.append(np.sum(p * np.log(p / qs + 1e-300) * (p > 0)))
        tvs.append(0.5 * np.abs(p - q).sum())
    kls, tvs = np.array(kls), np.array(tvs)
    return {
        "name": name,
        "KL_all": kls.mean(), "TV_all": tvs.mean(),
        "KL_rare": kls[rare_mask].mean(), "TV_rare": tvs[rare_mask].mean(),
                "kl": kls,
    }


def main(T=200_000, n_eval=4000):
    cls, emit, trans = make_language()
    tokens = sample(T, cls, emit, trans)
    V = len(cls)
    t0 = time.time()
    sents = [[str(x) for x in tokens[i:i + 1000]] for i in range(0, T, 1000)]  # gensim caps sentences at 10k
    w2v = Word2Vec(sents, vector_size=32, window=2, min_count=1,
                   sg=1, epochs=5, workers=2, seed=0)
    E = np.stack([w2v.wv[str(v)] if str(v) in w2v.wv else np.random.default_rng(v).normal(size=32)
                  for v in range(V)])
    print(f"word2vec on the same corpus: {time.time() - t0:.1f}s")

    rng = np.random.default_rng(5)
    idx = rng.choice(np.arange(8, T), size=n_eval, replace=False)

    # order-2 LOO count at each eval position, to define "rare context"
    base = GBDConfig(K=2, exact_orders=(2,), use_semantic=False, delta=np.zeros(3), r=32)
    tg_ex2 = build_targets(tokens, V, None, base)
    from gbd import OrderIndex
    h = np.zeros(T, dtype=np.uint64)
    for k in range(3):
        o = OrderIndex(tokens, k, V, h)
        h = o.h
    n2 = o.n[o.ctx[idx]] - 1
    rare = n2 <= 3
    print(f"eval positions: {n_eval}; rare (order-2 LOO count <= 3): {rare.mean():.1%}")

    runs = {}
    t0 = time.time()
    from gbd import GBDTargets
    oh_ids = np.zeros((T, 32), np.int32); oh_ids[:, 0] = tokens
    oh_p = np.zeros((T, 32), np.float32); oh_p[:, 0] = 1
    z = np.zeros(T, np.float32)
    runs["one-hot label (plain CE)"] = GBDTargets(oh_ids, oh_p, z + 0.5, z, z, z.astype(np.int16))
    runs["exact order-2, label included (CoCoNTs-style)"] = build_targets(
        tokens, V, None, GBDConfig(K=2, exact_orders=(2,), use_semantic=False, leave_one_out=False,
                                   delta=np.zeros(3), r=32))
    runs["exact order-2, leave-one-out"] = tg_ex2
    runs["GBD exact backoff only (orders 0-4)"] = build_targets(
        tokens, V, None, GBDConfig(K=4, use_semantic=False, r=32))
    runs["GBD full (orders 0-4 + semantic)"] = build_targets(
        tokens, V, E, GBDConfig(K=4, Ks=2, M=8, M_search=32, r=32))
    print(f"targets built: {time.time() - t0:.1f}s; fitted (GBD full): "
          f"delta={np.round(runs['GBD full (orders 0-4 + semantic)'].fitted['delta'], 3)}, "
          f"a={runs['GBD full (orders 0-4 + semantic)'].fitted['a']:.3f}, "
          f"b={runs['GBD full (orders 0-4 + semantic)'].fitted['b']:.3f}")

    print(f"\n{'target':48s} {'KL all':>7s} {'TV all':>7s} {'KL rare':>8s} {'TV rare':>8s}")
    res = {}
    for name, tg in runs.items():
        r = evaluate(name, tg, tokens, cls, emit, trans, idx, rare)
        res[name] = r
        print(f"{name:48s} {r['KL_all']:7.3f} {r['TV_all']:7.3f} {r['KL_rare']:8.3f} {r['TV_rare']:8.3f}")

    # does the risk proxy predict actual target error? (contribution 2)
    from scipy.stats import spearmanr
    full = runs["GBD full (orders 0-4 + semantic)"]
    rho = spearmanr(full.R[idx], res["GBD full (orders 0-4 + semantic)"]["kl"]).correlation
    print(f"\nSpearman(R_t, true KL of q_t) for GBD full: {rho:.3f}")
    a = full.alpha[idx]
    print(f"alpha_t (alpha_max=0.5, rho=median R): mean {a.mean():.3f}; rare {a[rare].mean():.3f}; frequent {a[~rare].mean():.3f}")
    q1, q4 = np.quantile(full.R[idx], [0.25, 0.75])
    kl = res["GBD full (orders 0-4 + semantic)"]["kl"]
    print(f"true KL of q_t: lowest-risk quartile {kl[full.R[idx] <= q1].mean():.3f}, highest-risk quartile {kl[full.R[idx] >= q4].mean():.3f}")
    print(f"mean weight on exact components: all {full.w_exact[idx].mean():.3f}, "
          f"rare {full.w_exact[idx][rare].mean():.3f}, frequent {full.w_exact[idx][~rare].mean():.3f}")


if __name__ == "__main__":
    main()
