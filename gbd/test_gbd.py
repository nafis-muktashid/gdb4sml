"""Correctness tests for GBD v2 in gbd.py. Run: python -I gbd/test_gbd.py"""
from collections import defaultdict
import tempfile

import numpy as np

from gbd import (
    GBDConfig,
    build_targets,
    conversation_starts,
    gbd_loss,
    load_targets,
    save_targets,
)
from gbd_v2_check import V2Config, build_targets_v2


def test_v2_matches_reference_prototype():
    """Ported gbd.py (v2) matches gbd_v2_check.build_targets_v2 to machine precision."""
    rng = np.random.default_rng(7)
    V, T, D = 15, 2500, 8
    P = rng.dirichlet(np.ones(V) * 0.4, size=V)
    tokens = np.zeros(T, dtype=np.int64)
    for t in range(1, T):
        tokens[t] = rng.choice(V, p=P[tokens[t - 1]])
    E = rng.normal(size=(V, D))

    cfg_prod = GBDConfig(K=4, Ks=2, M=5, M_search=12, R_pre=10, r=8, n_fit=6, chunk=600, seed=11)
    cfg_ref = V2Config(K=4, Ks=2, M=5, M_search=12, R_pre=10, r=8, n_fit=6, chunk=600, seed=11)
    tg_prod = build_targets(tokens, V, E, cfg_prod)
    tg_ref = build_targets_v2(tokens, V, E, cfg_ref)

    assert np.allclose(tg_prod.fitted["d0"], tg_ref.fitted["d0"], atol=1e-12)
    assert np.allclose(tg_prod.fitted["delta"], tg_ref.fitted["delta"], atol=1e-12)
    assert abs(tg_prod.fitted["a"] - tg_ref.fitted["a"]) < 1e-10
    assert abs(tg_prod.fitted["b"] - tg_ref.fitted["b"]) < 1e-10
    assert np.array_equal(tg_prod.ids, tg_ref.ids)
    assert np.allclose(tg_prod.probs, tg_ref.probs, atol=1e-6)
    assert np.allclose(tg_prod.alpha, tg_ref.alpha, atol=1e-6)
    assert np.allclose(tg_prod.r_ord, tg_ref.r_ord, atol=1e-6)
    assert np.allclose(tg_prod.R, tg_ref.R, atol=1e-6)
    print("ok  gbd.py v2 matches gbd_v2_check reference prototype across all outputs")


def test_conversation_boundaries_and_role_projection_vs_brute_force():
    """Brute-force verification of Eq. 1 (eos_id boundary) and Eq. 2 (role_ids turn-state & skipping)."""
    rng = np.random.default_rng(19)
    EOS, USER, ASST = 0, 1, 2
    V, D = 14, 6
    # Build 80 short synthetic dialogues: USER u1 u2 ASST a1 a2 EOS
    convs = []
    for _ in range(80):
        u_len = int(rng.integers(2, 5))
        a_len = int(rng.integers(2, 5))
        u_toks = rng.integers(3, V, size=u_len).tolist()
        a_toks = rng.integers(3, V, size=a_len).tolist()
        convs.extend([USER] + u_toks + [ASST] + a_toks + [EOS])
    tokens = np.asarray(convs, dtype=np.int64)
    T = len(tokens)
    E = rng.normal(size=(V, D))
    En = E / np.linalg.norm(E, axis=1, keepdims=True)

    delta = np.array([1.8, 1.1, 0.4, 0.0])
    d0 = np.array([14.0, 10.0, 6.0, 6.0])
    d0_sem = 5.0
    a_val, b_val, g = -0.2, 1.5, 0.73
    cfg = GBDConfig(
        K=3, Ks=2, M=3, M_search=8, R_pre=V, r=V, gamma=g,
        eos_id=EOS, role_ids=(USER, ASST),
        delta=delta, d0=d0, d0_sem=d0_sem, a=a_val, b=b_val,
        chunk=150,
    )
    tg = build_targets(tokens, V, E, cfg)

    # Brute-force b(t)
    b_bf = np.zeros(T, dtype=np.int64)
    cur_b = 0
    for t in range(T):
        b_bf[t] = cur_b
        if tokens[t] == EOS:
            cur_b = t + 1
    assert np.array_equal(conversation_starts(tokens, EOS), b_bf)

    # Brute-force exact counts inside conversations
    role_set = {USER, ASST}
    ex_cnt = [defaultdict(lambda: np.zeros(V, dtype=float)) for _ in range(cfg.K + 1)]
    for k in range(cfg.K + 1):
        for t in range(T):
            if t - k >= b_bf[t]:
                ex_cnt[k][tuple(tokens[t - k:t])][tokens[t]] += 1.0

    # Brute-force semantic contexts (s_t, x_{tau_1}, x_{tau_2})
    def sem_key(t):
        st = int(tokens[t - 1]) if (t > b_bf[t] and tokens[t - 1] in role_set) else -1
        content = [int(tokens[p]) for p in range(t - 1, b_bf[t] - 1, -1) if tokens[p] not in role_set]
        if len(content) < cfg.Ks:
            return None
        return (st, *content[: cfg.Ks])

    sem_cnt = defaultdict(lambda: np.zeros(V, dtype=float))
    for t in range(T):
        sk = sem_key(t)
        if sk is not None:
            sem_cnt[sk][tokens[t]] += 1.0
    sem_keys = list(sem_cnt.keys())
    sem_feat = {
        sk: np.concatenate([(g ** m) * En[sk[1 + m]] for m in range(cfg.Ks)])
        for sk in sem_keys
    }

    phi = lambda r: 1.0 / max(r, cfg.r_min)

    def var_bf(v, d0k):
        n = v.sum()
        dh = (v > 0).sum() + (v == 1).sum() + 1.0
        dt = (n * dh + cfg.n0 * d0k) / (n + cfg.n0)
        return np.log(1.0 + (dt - 1.0) / n)

    # Verify at conversation starts (t == b_bf[t]): orders k >= 1 and semantic must be inactive -> alpha_t == 0!
    conv_starts = np.nonzero(np.arange(T) == b_bf)[0]
    assert np.all(tg.alpha[conv_starts] == 0.0), "conversation-initial positions must have alpha_t == 0"

    # Verify 50 random positions including reply starts (right after ASST)
    reply_starts = [t for t in range(1, T) if tokens[t - 1] == ASST and sem_key(t) is not None]
    assert len(reply_starts) > 10
    check_pos = list(reply_starts[:15]) + rng.choice(T, size=35, replace=False).tolist()

    for t in check_pos:
        x = tokens[t]
        comps = []
        for k in range(cfg.K + 1):
            if t - k < b_bf[t]:
                continue
            v = ex_cnt[k][tuple(tokens[t - k:t])].copy()
            v[x] -= 1.0
            if v.sum() >= 1:
                comps.append((delta[k] + var_bf(v, d0[k]), v / v.sum()))
        r0 = comps[0][0]
        u = np.array([phi(r) for r, _ in comps])
        u /= u.sum()
        p_ord = sum(ui * p for ui, (_, p) in zip(u, comps))
        r_ord = -np.log(sum(ui * np.exp(-r) for ui, (r, _) in zip(u, comps)))

        sk = sem_key(t)
        cands = []
        if sk is not None:
            same_bucket = [k2 for k2 in sem_keys if k2[0] == sk[0] and k2 != sk]
            same_bucket.sort(key=lambda k2: np.linalg.norm(sem_feat[sk] - sem_feat[k2]))
            for k2 in same_bucket[: cfg.M_search]:
                dist = float(np.linalg.norm(sem_feat[sk] - sem_feat[k2]))
                v = sem_cnt[k2]
                bias = max(a_val + b_val * dist, 0.0)
                cands.append((delta[cfg.Ks] + bias + var_bf(v, d0_sem), bias, v))
            cands.sort(key=lambda z: z[0])
            cands = cands[: cfg.M]

        if cands:
            w = np.array([phi(z[0]) for z in cands])
            w /= w.sum()
            p_sem = sum(wi * (z[2] / z[2].sum()) for wi, z in zip(w, cands))
            N_sem = 1.0 / sum((wi ** 2) / z[2].sum() for wi, z in zip(w, cands))
            d_bar = sum(
                wi * ((z[2].sum() * ((z[2] > 0).sum() + (z[2] == 1).sum() + 1.0) + cfg.n0 * d0_sem) / (z[2].sum() + cfg.n0))
                for wi, z in zip(w, cands)
            )
            r_sem = delta[cfg.Ks] + sum(wi * z[1] for wi, z in zip(w, cands)) + np.log(1.0 + (d_bar - 1.0) / N_sem)
            lam = phi(r_ord) / (phi(r_ord) + phi(r_sem))
            q_bf = lam * p_ord + (1.0 - lam) * p_sem
        else:
            r_sem = np.inf
            lam = 1.0
            q_bf = p_ord

        alpha_bf = cfg.alpha_max * max(1.0 - r_ord / max(r0, cfg.r_min), 0.0)
        q_got = np.zeros(V)
        np.add.at(q_got, tg.ids[t], tg.probs[t])
        assert np.abs(q_got - q_bf).max() < 1e-5, (t, np.abs(q_got - q_bf).max())
        assert abs(tg.alpha[t] - alpha_bf) < 1e-5, (t, tg.alpha[t], alpha_bf)
    print("ok  Eq. 1 conversation boundaries + Eq. 2 role-marker projection match brute force (50 positions)")


def test_leave_one_out_and_alpha_modes():
    rng = np.random.default_rng(0)
    tokens = rng.permutation(4000) % 4000
    tg = build_targets(tokens, 4000, None, GBDConfig(K=3, exact_orders=(3,), use_semantic=False, r=4))
    assert np.all(tg.alpha == 0.0) and np.all(tg.probs[:, 0] == 1.0)

    tg_inc = build_targets(
        tokens, 4000, None,
        GBDConfig(K=3, exact_orders=(3,), use_semantic=False, leave_one_out=False, alpha_mode="const", r=4),
    )
    assert np.all(tg_inc.ids[3:, 0] == tokens[3:])
    assert np.allclose(tg_inc.alpha[3:], 0.5)
    assert np.all(tg_inc.alpha[:3] == 0.0)
    print("ok  leave-one-out & constant-alpha sliding-window ablation behave as specified")


def test_memmap_save_load_roundtrip():
    rng = np.random.default_rng(2)
    V, T = 60, 1500
    tokens = rng.choice(V, size=T)
    E = rng.normal(size=(V, 8))
    tg = build_targets(tokens, V, E, GBDConfig(K=3, Ks=2, M=4, M_search=8, r=8))
    assert np.allclose(tg.probs.sum(1), 1.0, atol=1e-5)
    assert ((tg.alpha >= 0) & (tg.alpha <= 0.5 + 1e-6)).all()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        save_targets(tmp, tg, fp16_probs=True, max_vocab=V)
        loaded = load_targets(tmp, mmap_mode="r")
        assert loaded.ids.dtype == np.uint16
        assert loaded.probs.dtype == np.float16
        assert np.array_equal(loaded.ids, tg.ids)
        assert np.allclose(loaded.probs.astype(np.float32), tg.probs, atol=2e-3)
        assert np.allclose(loaded.alpha, tg.alpha, atol=1e-6)
        del loaded
    print("ok  valid distributions + uint16/fp16 memmap save/load roundtrip")


def test_loss_gradient_is_soft_label_ce():
    try:
        import torch
    except ImportError:
        print("skip  torch not installed yet; skipping autograd loss test")
        return
    torch.manual_seed(0)
    B, L, V, r = 2, 5, 11, 4
    z = torch.randn(B, L, V, requires_grad=True)
    y = torch.randint(0, V, (B, L))
    ids = torch.stack([torch.randperm(V)[:r] for _ in range(B * L)]).view(B, L, r)
    p = torch.softmax(torch.randn(B, L, r), -1)
    a = torch.rand(B, L) * 0.5
    loss, _ = gbd_loss(z, y, ids, p, a)
    loss.backward()
    soft = torch.zeros(B, L, V).scatter_add_(-1, ids, p)
    ytilde = (1 - a)[..., None] * torch.nn.functional.one_hot(y, V) + a[..., None] * soft
    g_expected = (torch.softmax(z, -1) - ytilde) / (B * L)
    assert torch.allclose(z.grad, g_expected.detach(), atol=1e-6)
    l0, _ = gbd_loss(z, y, ids, p, torch.zeros(B, L))
    assert torch.allclose(l0, torch.nn.functional.cross_entropy(z.view(-1, V), y.view(-1)))
    print("ok  gradient = softmax(z) - [(1-a) e_y + a q]; alpha=0 recovers cross-entropy")


def test_orthogonal_component_and_embedding_switches():
    """Verify switchable exact, order-backoff, semantic, fixed/adaptive lambda & alpha, PPMI, and student_hidden hook."""
    rng = np.random.default_rng(29)
    V, T, D = 12, 1600, 12
    P = rng.dirichlet(np.ones(V) * 0.25, size=V)
    tokens = np.zeros(T, dtype=np.int64)
    for t in range(1, T):
        tokens[t] = rng.choice(V, p=P[tokens[t - 1]])
    E = rng.normal(size=(V, D)).astype(np.float32)

    # 1. Pure semantic retrieval (use_exact=False): lambda_t == 0, adaptive alpha > 0 where semantic exists
    tg_sem_only = build_targets(
        tokens, V, E,
        GBDConfig(K=3, Ks=2, M=4, M_search=10, r=8, n_fit=4, use_exact=False, use_semantic=True, alpha_mode="adaptive"),
    )
    assert tg_sem_only.fitted["switches"]["use_semantic"] is True
    assert np.all(tg_sem_only.w_exact == 0.0)
    assert np.allclose(tg_sem_only.probs.sum(1), 1.0, atol=1e-5)
    assert (tg_sem_only.alpha[2:] >= 0.0).all() and (tg_sem_only.alpha[2:] > 0.0).any()

    # 2. Exact single-order (use_order_backoff=False) + semantic with fixed lambda = 0.35 and fixed alpha = 0.4
    tg_fix = build_targets(
        tokens, V, E,
        GBDConfig(
            K=3, Ks=2, M=4, M_search=10, r=8, n_fit=4,
            use_exact=True, use_order_backoff=False, use_semantic=True,
            lambda_mode="fixed", fixed_lambda=0.35,
            alpha_mode="fixed", alpha_max=0.40,
            pool_semantic=False, phi="uniform",
        ),
    )
    assert tg_fix.fitted["switches"]["use_semantic"] is True
    both_active = np.isfinite(tg_fix.r_ord) & np.isfinite(tg_fix.r_sem)
    only_sem = (~np.isfinite(tg_fix.r_ord)) & np.isfinite(tg_fix.r_sem)
    assert both_active.any() and only_sem.any()
    assert np.allclose(tg_fix.w_exact[both_active], 0.35, atol=1e-6)
    assert np.allclose(tg_fix.w_exact[only_sem], 0.0, atol=1e-6)
    assert np.allclose(tg_fix.alpha[np.isfinite(tg_fix.R)], 0.40, atol=1e-6)

    # 3. Both branches off -> clean plain-CE fallback (alpha == 0 everywhere)
    tg_none = build_targets(tokens, V, E, GBDConfig(use_exact=False, use_semantic=False, r=4))
    assert np.all(tg_none.alpha == 0.0) and np.array_equal(tg_none.ids[:, 0], tokens)

    # 4. Count-only PPMI+SVD embeddings and student_hidden sem_feature_fn hook
    from embeddings import build_ppmi_svd_embeddings, make_student_hidden_feature_fn
    E_ppmi = build_ppmi_svd_embeddings(tokens, vocab_size=V, dim=8, window=3, seed=0)
    assert E_ppmi.shape == (V, 8) and np.isfinite(E_ppmi).all()

    try:
        import torch
        from model import CausalLM, ModelConfig
        tiny_m = CausalLM(ModelConfig.from_preset("tiny", vocab_size=V, max_seq_len=64))
        sem_fn = make_student_hidden_feature_fn(tiny_m, layer=2, proj_dim=8, seq_len=64, batch_size=8)
        tg_hid = build_targets(tokens, V, None, GBDConfig(K=3, Ks=2, M=4, M_search=8, r=8, n_fit=4), sem_feature_fn=sem_fn)
        assert tg_hid.fitted["switches"]["use_semantic"] is True
        assert np.allclose(tg_hid.probs.sum(1), 1.0, atol=1e-5)
    except ImportError:
        pass
    print("ok  all orthogonal switches (exact/backoff/semantic/fixed-vs-adaptive lambda & alpha/PPMI/student_hidden) verified")


def test_coconts_eq4_consistency_and_cache_isolation():
    """
    Verify:
      1. `build_coconts_targets` matches Sathe & Sarawagi (2024) Eq. (4) brute force (including
         prefixes crossing EOS tokens) AND satisfies `E[y_CC] == y_all` over prefix occurrences.
      2. `build_random_entropy_matched_targets` draws distinct IDs per row (1:1 entropy preservation).
      3. `_target_cache_dir_and_spec` never collides across distinct hyperparameters or token slices.
    """
    from pathlib import Path
    from train import (
        _target_cache_dir_and_spec,
        build_coconts_targets,
        build_random_entropy_matched_targets,
        make_gbd_config_from_args,
        parse_args_with_config,
    )

    rng = np.random.default_rng(101)
    V, T, L, k_pre, r, gamma_cc = 10, 2049, 16, 4, 3, 1.5
    # Include EOS_ID=3 frequently so prefixes cross EOS inside blocks
    tokens = rng.choice(V, size=T, p=rng.dirichlet(np.ones(V) * 0.8))
    tg_cc = build_coconts_targets(tokens, V=V, L=L, k_prefix=k_pre, r=r, gamma_cc=gamma_cc)
    assert tg_cc.nll_weight is not None

    # Brute-force corpus-wide n-gram counts (raw sequences, no EOS reset)
    full_cnt = [defaultdict(lambda: np.zeros(V, dtype=np.float64)) for _ in range(k_pre + 1)]
    for ord_k in range(1, k_pre + 1):
        for t in range(ord_k, T):
            full_cnt[ord_k][tuple(tokens[t - ord_k : t])][tokens[t]] += 1.0

    n_chunks = (T - 1) // L
    for ci in range(n_chunks):
        s = ci * L
        for p in range(L):
            pos = s + 1 + p
            x_t = int(tokens[pos])
            # Reconstruct effective supervision vector y_eff = nll_weight * e_{x_t} + alpha * q_t
            y_eff = np.zeros(V, dtype=np.float64)
            y_eff[x_t] += float(tg_cc.nll_weight[pos])
            np.add.at(y_eff, tg_cc.ids[pos], float(tg_cc.alpha[pos]) * tg_cc.probs[pos].astype(np.float64))
            if p >= k_pre:
                expected = np.zeros(V, dtype=np.float64)
                expected[x_t] = 1.0
                assert np.allclose(y_eff, expected, atol=1e-6)
            else:
                ord_k = p + 1
                cnt_vec = full_cnt[ord_k][tuple(tokens[pos - ord_k : pos])]
                y_all = cnt_vec / cnt_vec.sum()
                # Exact top-r with tie-breaking matching _topr (lexsort by -cnt, token_id)
                top_idx = np.lexsort((np.arange(V), -cnt_vec))[:r]
                top_idx = top_idx[cnt_vec[top_idx] > 0]
                y_topr = np.zeros(V, dtype=np.float64)
                y_topr[top_idx] = y_all[top_idx]
                p_mass = float(y_topr.sum())
                u = 1.0 / (gamma_cc - p_mass)
                v = (1.0 - (1.0 - p_mass) * u) / p_mass
                if x_t in set(top_idx.tolist()):
                    y_cc_bf = v * y_topr
                else:
                    y_cc_bf = u * y_topr
                    y_cc_bf[x_t] += 1.0
                assert np.allclose(y_eff, y_cc_bf, atol=1e-5), (pos, y_eff, y_cc_bf)

    # Verify exact consistency identity E_{x_t ~ y_all}[y_CC(x_t)] == y_all for an arbitrary distribution
    y_all_test = rng.dirichlet(np.ones(8) * 0.6)
    top3 = np.argsort(-y_all_test)[:3]
    y_topr_t = np.zeros(8)
    y_topr_t[top3] = y_all_test[top3]
    pm = y_topr_t.sum()
    u_t = 1.0 / (1.5 - pm)
    v_t = (1.0 - (1.0 - pm) * u_t) / pm
    exp_y_cc = np.zeros(8)
    for xt in range(8):
        if xt in set(top3.tolist()):
            y_cc_xt = v_t * y_topr_t
        else:
            y_cc_xt = u_t * y_topr_t.copy()
            y_cc_xt[xt] += 1.0
        exp_y_cc += y_all_test[xt] * y_cc_xt
    assert np.allclose(exp_y_cc, y_all_test, atol=1e-12)

    # Verify random_target draws r distinct IDs per row -> identical Shannon entropy to reference GBD
    ref_tg = build_targets(tokens, V, None, GBDConfig(K=3, use_semantic=False, r=5))
    rand_tg = build_random_entropy_matched_targets(tokens, V, ref_tg, seed=7)
    sorted_ids = np.sort(rand_tg.ids, axis=1)
    assert not (sorted_ids[:, 1:] == sorted_ids[:, :-1]).any(), "duplicate IDs in random_target row would shrink entropy!"

    # Verify cache keys never collide across M_search, R_pre, tau, embed_dim, random_target ref, or token edits
    root = Path("dummy_cache")
    base_argv = ["--method", "gbd_full", "--K", "6", "--Ks", "2", "--M", "8"]
    variants = [
        base_argv,
        base_argv + ["--M_search", "64"],
        base_argv + ["--R_pre", "128"],
        base_argv + ["--phi", "exp", "--tau", "1.0"],
        base_argv + ["--phi", "exp", "--tau", "2.0"],
        base_argv + ["--embed_dim", "256"],
        base_argv + ["--no-leave_one_out"],
        base_argv + ["--embedding_source", "random"],
        ["--method", "random_target", "--K", "6", "--alpha_max", "0.5", "--seed", "0"],
        ["--method", "random_target", "--K", "4", "--alpha_max", "0.5", "--seed", "0"],
        ["--method", "random_target", "--K", "6", "--no-leave_one_out", "--alpha_max", "0.5", "--seed", "0"],
        ["--method", "coconts", "--coconts_k", "8", "--coconts_gamma", "1.5"],
        ["--method", "coconts", "--coconts_k", "8", "--coconts_gamma", "2.0"],
    ]
    seen_dirs = set()
    for av in variants:
        a_parsed = parse_args_with_config(av)
        cfg_v = None if a_parsed.method in ("coconts", "random_target") else make_gbd_config_from_args(a_parsed)
        cdir, _ = _target_cache_dir_and_spec(a_parsed.method, tokens, V, cfg_v, a_parsed, root)
        assert cdir not in seen_dirs, f"Cache collision for argv={av}: {cdir}"
        seen_dirs.add(cdir)

    # Mutating a single token at identical length T must also change the cache key!
    tokens_mut = tokens.copy()
    tokens_mut[10] = (tokens_mut[10] + 1) % V
    a0 = parse_args_with_config(base_argv)
    cdir1, _ = _target_cache_dir_and_spec(a0.method, tokens, V, make_gbd_config_from_args(a0), a0, root)
    cdir2, _ = _target_cache_dir_and_spec(a0.method, tokens_mut, V, make_gbd_config_from_args(a0), a0, root)
    assert cdir1 != cdir2, "Mutating token content at same length T must produce a distinct cache directory"
    print("ok  CoCoNTs Eq. 4 (u,v) consistency, without-replacement random_target, and collision-free cache keys verified")


def _naive_kn_counts(tokens, V, K, eos_id, remove_t=None):
    """Dict multiset of raw orders 1..K and continuation orders 0..K-1.

    `remove_t` drops every n-gram event ending at that position (KN-LOO).
    Continuation N1+(· h w) is the number of distinct in-bounds left tokens.
    """
    from collections import defaultdict
    from gbd import conversation_starts

    tokens = np.asarray(tokens, dtype=np.int64)
    T = len(tokens)
    b = conversation_starts(tokens, eos_id)
    raw = [None] + [defaultdict(lambda: defaultdict(int)) for _ in range(K)]
    for t in range(T):
        if remove_t is not None and t == int(remove_t):
            continue
        w = int(tokens[t])
        for k in range(1, K + 1):
            if t - k >= int(b[t]):
                h = tuple(int(tokens[t - k + j]) for j in range(k))
                raw[k][h][w] += 1
    cont = []
    for k in range(K):
        lefts = defaultdict(set)
        for h, words in raw[k + 1].items():
            suf, left = h[1:], h[0]
            for w, c in words.items():
                if c > 0:
                    lefts[(suf, w)].add(left)
        acc = defaultdict(lambda: defaultdict(int))
        for (suf, w), group in lefts.items():
            acc[suf][w] = len(group)
        cont.append(acc)
    return raw, cont, b


def _naive_kn_discounts(raw, cont, K):
    from kn import modified_kn_discounts

    discounts = []
    for k in range(K + 1):
        words = raw[K] if k == K else cont[k]
        vals = [c for hist in words.values() for c in hist.values() if c > 0]
        discounts.append(modified_kn_discounts(np.asarray(vals, dtype=np.int64)))
    return discounts


def _naive_kn_at(t, tokens, raw, cont, discounts, b, K, V):
    """Full P_K(· | history at t) with the given (possibly LOO-edited) counts."""

    def mix(words, D, lower):
        c = float(sum(words.values())) if words else 0.0
        if c <= 0.0:
            return lower
        D1, D2, D3 = D
        N1 = sum(v == 1 for v in words.values())
        N2 = sum(v == 2 for v in words.values())
        N3 = sum(v >= 3 for v in words.values())
        gamma = (D1 * N1 + D2 * N2 + D3 * N3) / c
        num = np.zeros(V, dtype=np.float64)
        for w, cw in words.items():
            Dw = D1 if cw == 1 else D2 if cw == 2 else (D3 if cw >= 3 else 0.0)
            num[int(w)] = max(float(cw) - Dw, 0.0)
        return num / c + gamma * lower

    p = mix(cont[0][()], discounts[0], np.full(V, 1.0 / V))
    for k in range(1, K + 1):
        if t - k < int(b[t]):
            continue
        h = tuple(int(tokens[t - k + j]) for j in range(k))
        words = raw[k][h] if k == K else cont[k][h]
        p = mix(words, discounts[k], p)
    return p


def _dense_target(tg, V):
    T = tg.ids.shape[0]
    out = np.zeros((T, V), dtype=np.float64)
    np.add.at(out, (np.arange(T)[:, None], tg.ids.astype(np.int64)), tg.probs.astype(np.float64))
    return out


def test_kn_and_d2gpo_targets():
    """KN matches a dict multiset; D2GPo rows follow Li et al. Eqs. 7-9; cache keys do not collide."""
    from pathlib import Path

    from d2gpo import build_d2gpo_prior, build_d2gpo_targets
    from kn import build_kn_targets
    from train import _target_cache_dir_and_spec, parse_args_with_config

    rng = np.random.default_rng(11)
    V, T, K = 30, 2000, 3
    # Markov tokens plus frequent eos (id 3) so boundaries are exercised.
    P = rng.dirichlet(np.ones(V) * 0.35, size=V)
    P[:, 3] += 0.15
    P /= P.sum(1, keepdims=True)
    tokens = np.zeros(T, dtype=np.int64)
    for t in range(1, T):
        tokens[t] = rng.choice(V, p=P[tokens[t - 1]])

    eos = 3
    tg = build_kn_targets(tokens, V, K=K, r=V, alpha_max=0.4, kn_loo=False, respect_eos=True, eos_id=eos)
    raw, cont, b = _naive_kn_counts(tokens, V, K, eos_id=eos, remove_t=None)
    discounts = _naive_kn_discounts(raw, cont, K)
    got = _dense_target(tg, V)
    max_err = 0.0
    for t in range(T):
        ref = _naive_kn_at(t, tokens, raw, cont, discounts, b, K, V)
        max_err = max(max_err, float(np.max(np.abs(got[t] - ref))))
    assert max_err < 1e-6, max_err
    assert np.allclose(got.sum(1), 1.0, atol=1e-6)
    assert np.allclose(tg.alpha, 0.4)
    assert tg.nll_weight is None

    # (b) LOO at sampled positions == naive KN with events ending at t removed, discounts fixed.
    tg_loo = build_kn_targets(tokens, V, K=K, r=V, alpha_max=0.4, kn_loo=True, respect_eos=True, eos_id=eos)
    got_loo = _dense_target(tg_loo, V)
    sample = [0, 1, 2, T // 2, T - 1]
    sample += [int(t) for t in rng.choice(T, size=25, replace=False)]
    eos_pos = np.nonzero(tokens == eos)[0]
    if len(eos_pos):
        sample.append(int(min(eos_pos[0] + 1, T - 1)))
    loo_err = 0.0
    for t in sample:
        raw_t, cont_t, b_t = _naive_kn_counts(tokens, V, K, eos_id=eos, remove_t=t)
        ref = _naive_kn_at(t, tokens, raw_t, cont_t, discounts, b_t, K, V)
        loo_err = max(loo_err, float(np.max(np.abs(got_loo[t] - ref))))
    assert loo_err < 1e-6, loo_err
    assert np.allclose(got_loo.sum(1), 1.0, atol=1e-6)

    # (c) A context seen once, without LOO, puts the most mass on its label.
    # The body is repeated twice so most n-grams have count 2 (D1 < 0.5). The tail is unique.
    body = np.arange(40, dtype=np.int64) % 17
    once = np.concatenate([body, body, np.array([17, 18, 19, 20], dtype=np.int64)])
    t_once = len(once) - 1  # history (17,18,19) occurs once, label 20
    tg_once = build_kn_targets(once, V=21, K=3, r=21, alpha_max=0.5, kn_loo=False, respect_eos=False, eos_id=None)
    row = _dense_target(tg_once, 21)[t_once]
    assert int(np.argmax(row)) == 20, (int(np.argmax(row)), float(row[20]), float(row.max()))

    # (d) D2GPo prior rows.
    E = rng.normal(size=(V, 8))
    E[1] = E[0]  # identical neighbour must not steal rank 0 from either gold token
    sigma, temp, lam, r_d = 1.0, 2.0, 0.1, 7
    ids, q = build_d2gpo_prior(E, sigma=sigma, temp=temp, r_d=r_d)
    assert ids.shape == (V, r_d)
    assert np.allclose(q.sum(), 1.0, atol=1e-8)
    assert np.all(q[:-1] >= q[1:] - 1e-12)
    assert np.all(ids[:, 0] == np.arange(V))
    tg_d = build_d2gpo_targets(tokens, E, sigma=sigma, temp=temp, lam=lam, r_d=r_d)
    assert np.allclose(tg_d.probs.sum(1), 1.0, atol=1e-6)
    assert np.allclose(tg_d.alpha, lam) and np.allclose(tg_d.nll_weight, 1.0)
    step = max(T // 20, 1)
    for t in range(0, T, step):
        assert np.array_equal(tg_d.ids[t], ids[int(tokens[t])])
        assert np.allclose(tg_d.probs[t], q.astype(np.float32), atol=1e-6)

    # (e) Cache keys move when any KN / D2GPo parameter moves.
    root = Path("dummy_cache_kn_d2gpo")
    variants = [
        ["--method", "kn", "--K", "3", "--r", "16", "--alpha_max", "0.5"],
        ["--method", "kn", "--K", "4", "--r", "16", "--alpha_max", "0.5"],
        ["--method", "kn", "--K", "3", "--r", "8", "--alpha_max", "0.5"],
        ["--method", "kn", "--K", "3", "--r", "16", "--alpha_max", "0.25"],
        ["--method", "kn", "--K", "3", "--r", "16", "--no-kn_loo"],
        ["--method", "kn", "--K", "3", "--r", "16", "--no-respect_eos"],
        ["--method", "d2gpo", "--d2gpo_sigma", "1.0", "--d2gpo_temp", "2.0", "--d2gpo_lambda", "0.1", "--d2gpo_r", "64", "--embed_dim", "128", "--seed", "0"],
        ["--method", "d2gpo", "--d2gpo_sigma", "2.0", "--d2gpo_temp", "2.0", "--d2gpo_lambda", "0.1", "--d2gpo_r", "64", "--embed_dim", "128", "--seed", "0"],
        ["--method", "d2gpo", "--d2gpo_sigma", "1.0", "--d2gpo_temp", "1.0", "--d2gpo_lambda", "0.1", "--d2gpo_r", "64", "--embed_dim", "128", "--seed", "0"],
        ["--method", "d2gpo", "--d2gpo_sigma", "1.0", "--d2gpo_temp", "2.0", "--d2gpo_lambda", "0.2", "--d2gpo_r", "64", "--embed_dim", "128", "--seed", "0"],
        ["--method", "d2gpo", "--d2gpo_sigma", "1.0", "--d2gpo_temp", "2.0", "--d2gpo_lambda", "0.1", "--d2gpo_r", "32", "--embed_dim", "128", "--seed", "0"],
        ["--method", "d2gpo", "--d2gpo_sigma", "1.0", "--d2gpo_temp", "2.0", "--d2gpo_lambda", "0.1", "--d2gpo_r", "64", "--embed_dim", "64", "--seed", "0"],
        ["--method", "d2gpo", "--d2gpo_sigma", "1.0", "--d2gpo_temp", "2.0", "--d2gpo_lambda", "0.1", "--d2gpo_r", "64", "--embed_dim", "128", "--seed", "1"],
        ["--method", "d2gpo", "--d2gpo_sigma", "1.0", "--embedding_source", "random", "--embed_dim", "128", "--seed", "0"],
    ]
    seen = set()
    for av in variants:
        a = parse_args_with_config(av)
        cdir, spec = _target_cache_dir_and_spec(a.method, tokens, V, None, a, root)
        assert cdir not in seen, av
        seen.add(cdir)
        if a.method == "kn":
            assert spec["discounts"] == "chen_goodman_1998_modified"
            assert "K" in spec and "kn_loo" in spec and "respect_eos" in spec
        else:
            for key in ("sigma", "temp", "lambda", "r_d", "emb_src", "embed_dim", "emb_seed"):
                assert key in spec, key
    print("ok  KN matches dict multiset (incl. LOO); D2GPo prior rows; KN/D2GPo cache keys isolated")


def test_self_kd_loss():
    """Born-Again loss: alpha=0 is CE; equal logits make the KD term vanish; T^2 scales the KL."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from train import mean_teacher_logits, self_kd_loss

    torch.manual_seed(0)
    B, V = 4, 7
    z = torch.randn(B, V)
    y = torch.randint(0, V, (B,))
    y[0] = -100
    loss0 = self_kd_loss(z, torch.randn(B, V), y, alpha=0.0, temperature=2.0)
    ce = F.cross_entropy(z.float(), y, ignore_index=-100)
    assert torch.allclose(loss0, ce, atol=1e-6)

    z2 = torch.randn(B, V, requires_grad=True)
    kd_only = self_kd_loss(z2, z2.detach(), y, alpha=1.0, temperature=1.0)
    assert abs(float(kd_only)) < 1e-6
    kd_only.backward()
    assert torch.allclose(z2.grad, torch.zeros_like(z2.grad), atol=1e-6)

    teacher = nn.Linear(5, V)
    student = nn.Linear(5, V)
    teacher.requires_grad_(False)
    x = torch.randn(B, 5)
    with torch.no_grad():
        z_teacher = teacher(x)
    z_student = student(x)
    self_kd_loss(z_student, z_teacher, torch.randint(0, V, (B,)), alpha=0.5, temperature=1.5).backward()
    assert all(p.grad is None for p in teacher.parameters())
    assert student.weight.grad is not None and float(student.weight.grad.abs().sum()) > 0.0

    z_s = torch.randn(B, V)
    z_t = torch.randn(B, V)
    y3 = torch.randint(0, V, (B,))
    alpha = 0.4
    nll = F.cross_entropy(z_s.float(), y3)

    def kl_at(temp: float) -> torch.Tensor:
        log_s = F.log_softmax(z_s.float() / temp, dim=-1)
        log_t = F.log_softmax(z_t.float() / temp, dim=-1)
        p_t = log_t.exp()
        return (p_t * (log_t - log_s)).sum(dim=-1).mean()

    for temp in (0.5, 1.0, 2.0):
        loss = self_kd_loss(z_s, z_t, y3, alpha=alpha, temperature=temp)
        predicted = (1.0 - alpha) * nll + alpha * (temp ** 2) * kl_at(temp)
        assert torch.allclose(loss, predicted, atol=1e-5), (temp, float(loss), float(predicted))

    # Two copies of one teacher: mean of logits equals that teacher, so the loss matches.
    z_one = torch.randn(B, V)
    y_ens = torch.randint(0, V, (B,))
    single = self_kd_loss(z_s, z_one, y_ens, alpha=0.5, temperature=2.0)
    doubled = self_kd_loss(z_s, mean_teacher_logits([z_one, z_one.clone()]), y_ens, alpha=0.5, temperature=2.0)
    assert torch.allclose(single, doubled, atol=1e-6)
    print("ok  self-KD: alpha=0 is CE; equal logits zero the KD term and its gradient; T^2*KL holds; teacher gets no gradient; duplicate teachers match")


if __name__ == "__main__":
    test_v2_matches_reference_prototype()
    test_conversation_boundaries_and_role_projection_vs_brute_force()
    test_leave_one_out_and_alpha_modes()
    test_memmap_save_load_roundtrip()
    test_loss_gradient_is_soft_label_ce()
    test_orthogonal_component_and_embedding_switches()
    test_coconts_eq4_consistency_and_cache_isolation()
    test_kn_and_d2gpo_targets()
    test_self_kd_loss()
