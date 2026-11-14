"""
Geometric Backoff Distillation (GBD): teacher-free soft next-token targets.

Implements Version 2 of the GBD estimator and risk scale (Nov 12, 2026;
Pitch Method sections 1-6 and PROJECT_MEMORY.md Entry 005), with optional
`risk_version=1` compatibility for historical v1 benchmarks.

Offline:  build_targets(tokens, V, E, cfg)  ->  per-position top-r soft targets q_t
          and per-position distillation weights alpha_t.
Training: gbd_loss(logits, labels, tgt_ids, tgt_probs, alpha)
          = (1 - alpha_t) * CE(x_t) + alpha_t * H(q_t, p_theta).

No pretrained model is used anywhere. E is a static token-embedding matrix
(e.g. word2vec trained on the same corpus).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from typing import Callable

import numpy as np
import scipy.sparse as sp

_P = np.uint64(0x9E3779B97F4A7C15)  # odd multiplier for context hashing


# --------------------------------------------------------------------------
# Config and outputs
# --------------------------------------------------------------------------
@dataclass
class GBDConfig:
    K: int = 6                        # max exact order
    Ks: int = 2                       # order of contexts used for semantic neighbours
    M: int = 8                        # neighbours kept per context (after risk ranking)
    M_search: int = 32                # candidates retrieved by embedding distance
    R_pre: int = 64                   # per-context row truncation before mixing
    r: int = 16                       # stored top-r entries of q_t
    # --- Component switches (orthogonal ablations) ---
    use_exact: bool = True            # master switch for exact-context branch (False -> semantic-only)
    use_order_backoff: bool = True    # True -> orders 0..K; False -> single order (K,) only (unless exact_orders set)
    exact_orders: tuple[int, ...] | None = None  # explicit order tuple override, e.g. (0,) or (2, 4, 6)
    leave_one_out: bool = True        # True -> subtract position t (Eq. 1); False -> include label
    respect_eos: bool = True          # True -> contexts do not cross eos_id (Eq. 1)
    use_semantic: bool = True         # master switch for semantic retrieval branch
    skip_role_markers: bool = True    # True -> semantic context skips role_ids (Eq. 2)
    partition_turn_state: bool = True # True -> semantic kNN restricted to matching turn state s_t (Eq. 2)
    pool_semantic: bool = True        # True -> pooled N_sem variance (Eq. 4); False -> unpooled mixture bound
    # --- Weighting & schedule switches (fixed vs adaptive lambda / alpha) ---
    phi: str = "inv"                  # "inv": 1/max(r, r_min); "exp": exp(-tau * r); "uniform": equal weights
    tau: float = 1.0
    r_min: float = 1e-3               # floor on risk for phi="inv" and unigram anchor
    r_floor: float | None = None      # legacy alias for r_min
    gamma: float = 1.0                # positional decay in context features (1.0 = Frobenius)
    lambda_mode: str = "adaptive"     # "adaptive" (Eq. 4 phi-ratio) | "fixed" (constant fixed_lambda when both active)
    fixed_lambda: float = 0.5         # exact-branch weight in [0, 1] used when lambda_mode == "fixed"
    alpha_max: float = 0.5
    gamma_alpha: float = 1.0          # exponent in Eq. 5
    alpha_mode: str = "r_ord"         # "adaptive"/"r_ord" (Eq. 5), "R", "r_sem", or "fixed"/"const"
    # --- Corpus & estimation settings ---
    eos_id: int | None = None         # conversation separator id for Eq. 1 boundary rule t - k >= b(t)
    role_ids: tuple[int, ...] = ()    # role-marker token ids S for Eq. 2 turn-state / content projection
    n_fit: int = 20                   # min count for contexts used to fit bias / support constants
    n_fit_pairs: int = 2000
    n0: float = 5.0                   # empirical-Bayes shrinkage weight toward support prior d0
    beta0: float = 3.0                # total unigram pseudo-count in plug-in KL
    hnsw_threshold: int = 100_000     # switch FAISS from exact FlatL2 to HNSW above this context count
    chunk: int = 50_000
    seed: int = 0
    # overrides for ablations (None -> fitted on training counts)
    delta: np.ndarray | None = None   # delta[k]: order-truncation bias of order k vs order K
    a: float | None = None            # semantic bias [a + b * ||f_c - f_c'||]_+
    b: float | None = None
    d0: np.ndarray | None = None      # support prior per order 0..K
    d0_sem: float | None = None       # support prior for semantic context family
    # legacy v1 fields (used only when risk_version == 1)
    risk_version: int = 2
    rho: float | None = None
    fit_eps: float = 0.01

    def __post_init__(self):
        if self.r_floor is not None:
            self.r_min = self.r_floor

    def resolve_exact_orders(self) -> list[int]:
        """Resolve active exact n-gram orders from `use_exact`, `use_order_backoff`, and `exact_orders`."""
        if not self.use_exact:
            return []
        if self.exact_orders is not None:
            return [int(k) for k in self.exact_orders]
        if self.use_order_backoff:
            return list(range(self.K + 1))
        return [int(self.K)]


@dataclass
class GBDTargets:
    ids: np.ndarray                   # (T, r) int32 / uint16
    probs: np.ndarray                 # (T, r) float32 / float16, rows sum to 1
    alpha: np.ndarray                 # (T,)   float32 soft-target weight
    R: np.ndarray                     # (T,)   float32 combined mixture risk diagnostic
    w_exact: np.ndarray               # (T,)   float32 lambda_t (exact branch share)
    k_star: np.ndarray                # (T,)   int16 longest active exact order
    fitted: dict = field(default_factory=dict)
    r_ord: np.ndarray | None = None   # (T,)   float32 order-branch mixture risk
    r_sem: np.ndarray | None = None   # (T,)   float32 pooled semantic-branch risk
    r0: np.ndarray | None = None      # (T,)   float32 unigram LOO risk
    nll_weight: np.ndarray | None = None  # (T,) optional explicit hard-label NLL weight (None -> 1 - alpha; used by CoCoNTs Eq. 4)


def save_targets(
    directory: str | Path,
    tg: GBDTargets,
    fp16_probs: bool = True,
    max_vocab: int | None = None,
    cache_spec: dict | None = None,
):
    """Save targets as separate .npy files in `directory` for zero-RAM OS memory-mapping."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    vmax = int(tg.ids.max()) if tg.ids.size else 0
    use_u16 = (max_vocab is not None and max_vocab <= 65535) or (max_vocab is None and vmax <= 65535 and tg.ids.min() >= 0)
    ids_arr = tg.ids.astype(np.uint16 if use_u16 else np.int32)
    probs_arr = tg.probs.astype(np.float16 if fp16_probs else np.float32)
    if fp16_probs:
        # re-normalise in float32 on load, but ensure non-zero sum in fp16
        s = probs_arr.astype(np.float32).sum(1, keepdims=True)
        probs_arr = (probs_arr.astype(np.float32) / np.maximum(s, 1e-12)).astype(np.float16)
    np.save(d / "ids.npy", ids_arr)
    np.save(d / "probs.npy", probs_arr)
    np.save(d / "alpha.npy", tg.alpha.astype(np.float32))
    np.save(d / "R.npy", tg.R.astype(np.float32))
    np.save(d / "w_exact.npy", tg.w_exact.astype(np.float32))
    np.save(d / "k_star.npy", tg.k_star.astype(np.int16))
    if tg.r_ord is not None:
        np.save(d / "r_ord.npy", tg.r_ord.astype(np.float32))
    if tg.r_sem is not None:
        np.save(d / "r_sem.npy", tg.r_sem.astype(np.float32))
    if tg.r0 is not None:
        np.save(d / "r0.npy", tg.r0.astype(np.float32))
    if tg.nll_weight is not None:
        np.save(d / "nll_weight.npy", tg.nll_weight.astype(np.float32))
    serial_fitted = {}
    for k, v in tg.fitted.items():
        serial_fitted[k] = v.tolist() if isinstance(v, np.ndarray) else v
    (d / "fitted.json").write_text(json.dumps(serial_fitted, indent=2), encoding="utf-8")
    if cache_spec is not None:
        (d / "cache_spec.json").write_text(json.dumps(cache_spec, indent=2, sort_keys=True), encoding="utf-8")


def load_targets(
    directory: str | Path,
    mmap_mode: str | None = "r",
    expected_spec: dict | None = None,
) -> GBDTargets:
    """Load targets saved by `save_targets` (verifies `cache_spec.json` if `expected_spec` is given)."""
    d = Path(directory)
    if expected_spec is not None:
        spec_file = d / "cache_spec.json"
        if not spec_file.exists():
            raise ValueError(f"Cached target directory {d} is missing cache_spec.json; pass --rebuild_targets")
        saved_spec = json.loads(spec_file.read_text(encoding="utf-8"))
        if saved_spec != expected_spec:
            raise ValueError(f"Cache spec mismatch in {d}: saved={saved_spec} vs expected={expected_spec}")
    fitted = json.loads((d / "fitted.json").read_text(encoding="utf-8")) if (d / "fitted.json").exists() else {}
    for k in ("delta", "d0"):
        if k in fitted and isinstance(fitted[k], list):
            fitted[k] = np.asarray(fitted[k], dtype=np.float64)
    opt = lambda name: np.load(d / f"{name}.npy", mmap_mode=mmap_mode) if (d / f"{name}.npy").exists() else None
    return GBDTargets(
        ids=np.load(d / "ids.npy", mmap_mode=mmap_mode),
        probs=np.load(d / "probs.npy", mmap_mode=mmap_mode),
        alpha=np.load(d / "alpha.npy", mmap_mode=mmap_mode),
        R=np.load(d / "R.npy", mmap_mode=mmap_mode),
        w_exact=np.load(d / "w_exact.npy", mmap_mode=mmap_mode),
        k_star=np.load(d / "k_star.npy", mmap_mode=mmap_mode),
        fitted=fitted,
        r_ord=opt("r_ord"),
        r_sem=opt("r_sem"),
        r0=opt("r0"),
        nll_weight=opt("nll_weight"),
    )


# --------------------------------------------------------------------------
# Conversation boundaries & Counting (Eqs. 1 & 2)
# --------------------------------------------------------------------------
def conversation_starts(tokens: np.ndarray, eos_id: int | None) -> np.ndarray:
    """b(t): index of the first token of the conversation containing position t."""
    T = len(tokens)
    if T == 0:
        return np.zeros(0, dtype=np.int64)
    if eos_id is None:
        return np.zeros(T, dtype=np.int64)
    is_start = np.zeros(T, dtype=bool)
    is_start[0] = True
    if T > 1:
        is_start[1:] = tokens[:-1] == eos_id
    return np.maximum.accumulate(np.where(is_start, np.arange(T, dtype=np.int64), 0))


def _build_count_stats(obj, valid: np.ndarray, h: np.ndarray, tokens: np.ndarray, V: int):
    T = len(tokens)
    ctx = np.full(T, -1, dtype=np.int64)
    if not valid.any():
        obj.ctx = ctx
        obj.rep_pos = np.zeros(0, dtype=np.int64)
        obj.C = sp.csr_matrix((0, V), dtype=np.float64)
        obj.n = np.zeros(0, dtype=np.float64)
        obj.d_obs = np.zeros(0, dtype=np.int64)
        obj.f1 = np.zeros(0, dtype=np.int64)
        obj.Ctr = obj.C
        obj.tr_sum = np.zeros(0, dtype=np.float64)
        return
    uniq, first, inv = np.unique(h[valid], return_index=True, return_inverse=True)
    ctx[valid] = inv
    obj.ctx = ctx
    pos_valid = np.nonzero(valid)[0]
    obj.rep_pos = pos_valid[first]
    n_ctx = len(uniq)
    C = sp.csr_matrix(
        (np.ones(len(pos_valid), dtype=np.float64), (inv, tokens[pos_valid])),
        shape=(n_ctx, V),
    )
    C.sum_duplicates()
    obj.C = C
    obj.n = np.asarray(C.sum(axis=1)).ravel()
    obj.d_obs = np.diff(C.indptr)
    ones = (C.data == 1).astype(np.int64)
    obj.f1 = np.add.reduceat(ones, C.indptr[:-1]) if C.nnz else np.zeros(n_ctx, np.int64)
    obj.f1[obj.d_obs == 0] = 0
    obj.Ctr = C
    obj.tr_sum = obj.n


class OrderIndex:
    """Exact counts for one context order k, respecting conversation starts b_pos (Eq. 1)."""

    def __init__(
        self,
        tokens: np.ndarray,
        k: int,
        V: int,
        h_prev: np.ndarray | None,
        b_pos: np.ndarray | None = None,
    ):
        T = len(tokens)
        self.k = k
        if b_pos is None:
            b_pos = np.zeros(T, dtype=np.int64)
        if k == 0:
            h = np.zeros(T, dtype=np.uint64)
            valid = np.ones(T, dtype=bool)
        else:
            shifted = np.zeros(T, dtype=np.uint64)
            if T > k:
                shifted[k:] = tokens[:-k].astype(np.uint64) + np.uint64(1)
            h = h_prev * _P + shifted
            valid = (np.arange(T, dtype=np.int64) - k) >= b_pos
        self.h = h
        _build_count_stats(self, valid, h, tokens, V)

    def truncate(self, R: int | None):
        self.Ctr = _truncate_rows(self.C, R)
        self.tr_sum = np.asarray(self.Ctr.sum(axis=1)).ravel()


class SemanticIndex:
    """Turn-state partitioned, content-projected semantic contexts of order Ks (Eq. 2)."""

    def __init__(
        self,
        tokens: np.ndarray,
        Ks: int,
        V: int,
        b_pos: np.ndarray,
        role_ids: tuple[int, ...],
        skip_roles: bool = True,
        use_turn_state: bool = True,
    ):
        T = len(tokens)
        self.k = Ks
        role_arr = np.asarray(role_ids, dtype=np.int64)
        has_roles = role_arr.size > 0
        is_role_marker = np.isin(tokens, role_arr) if has_roles else np.zeros(T, dtype=bool)

        # s_t = x_{t-1} if t > b(t) and x_{t-1} in S, else -1 ("0 / otherwise" bucket)
        state = np.full(T, -1, dtype=np.int64)
        if T > 1 and has_roles and use_turn_state:
            in_conv = np.arange(T, dtype=np.int64) > b_pos
            prev_role = np.zeros(T, dtype=bool)
            prev_role[1:] = is_role_marker[:-1]
            m_st = in_conv & prev_role
            state[m_st] = tokens[np.nonzero(m_st)[0] - 1]

        # tau_1(t) > ... > tau_{Ks}(t) >= b(t): Ks most recent non-role (or raw) positions before t
        skip_mask = is_role_marker if skip_roles else np.zeros(T, dtype=bool)
        non_role_pos = np.where(~skip_mask, np.arange(T, dtype=np.int64), -1)
        last_non_role = np.maximum.accumulate(non_role_pos) if T > 0 else np.zeros(0, dtype=np.int64)
        taus = np.full((Ks, T), -1, dtype=np.int64)
        if Ks >= 1 and T > 1:
            taus[0, 1:] = last_non_role[:-1]
            for m in range(1, Ks):
                prev_t = taus[m - 1]
                ok = prev_t >= 1
                taus[m, ok] = last_non_role[prev_t[ok] - 1]
        valid = (taus[-1] >= b_pos) if Ks >= 1 else np.ones(T, dtype=bool)

        h = (state + 2).astype(np.uint64)
        for m in range(Ks):
            tok_m = tokens[np.maximum(taus[m], 0)].astype(np.uint64) + np.uint64(1)
            h = h * _P + tok_m
        self.h = h
        _build_count_stats(self, valid, h, tokens, V)
        self.ctx_state = state[self.rep_pos] if len(self.rep_pos) else np.zeros(0, dtype=np.int64)
        self.ctx_toks = tokens[taus[:, self.rep_pos]] if len(self.rep_pos) else np.zeros((Ks, 0), dtype=np.int64)

    def truncate(self, R: int | None):
        self.Ctr = _truncate_rows(self.C, R)
        self.tr_sum = np.asarray(self.Ctr.sum(axis=1)).ravel()


def _truncate_rows(C: sp.csr_matrix, R: int | None) -> sp.csr_matrix:
    """Keep the R largest entries of each CSR row (vectorised)."""
    if R is None or C.nnz == 0:
        return C
    rows = np.repeat(np.arange(C.shape[0]), np.diff(C.indptr))
    order = np.lexsort((-C.data, rows))
    rank = np.arange(C.nnz) - C.indptr[rows[order]]
    keep = order[rank < R]
    return sp.csr_matrix((C.data[keep], (rows[keep], C.indices[keep])), shape=C.shape)


# --------------------------------------------------------------------------
# Context features & kNN search (Eq. 2)
# --------------------------------------------------------------------------
def context_features(tokens: np.ndarray, idx: OrderIndex | SemanticIndex, E: np.ndarray, gamma: float) -> np.ndarray:
    """f(c) = [e(x_{tau_1}); gamma e(x_{tau_2}); ...] with unit-norm token vectors."""
    if len(idx.rep_pos) == 0:
        return np.zeros((0, E.shape[1] * idx.k), dtype=np.float32)
    En = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    if isinstance(idx, SemanticIndex):
        parts = [(gamma ** m) * En[idx.ctx_toks[m]] for m in range(idx.k)]
    else:
        t = idx.rep_pos
        parts = [(gamma ** (j - 1)) * En[tokens[t - j]] for j in range(1, idx.k + 1)]
    return np.concatenate(parts, axis=1).astype(np.float32)


def nearest_contexts(F: np.ndarray, m: int, hnsw_threshold: int = 100_000):
    """L2 kNN (exact FlatL2 below hnsw_threshold, HNSW above; numpy fallback). Drops self."""
    n = len(F)
    m = min(m, n - 1)
    if m <= 0:
        return np.zeros((n, 0), dtype=np.float64), np.zeros((n, 0), dtype=np.int64)
    F = np.ascontiguousarray(F, dtype=np.float32)
    try:
        import faiss
        if n > hnsw_threshold:
            index = faiss.IndexHNSWFlat(F.shape[1], 32)
            index.hnsw.efConstruction = 64
            index.hnsw.efSearch = max(64, m * 2)
        else:
            index = faiss.IndexFlatL2(F.shape[1])
        index.add(F)
        D2, I = index.search(F, m + 1)
    except ImportError:  # pragma: no cover
        sq = (F ** 2).sum(1)
        D2 = sq[:, None] + sq[None, :] - 2 * F @ F.T
        I = np.argsort(D2, axis=1)[:, : m + 1]
        D2 = np.take_along_axis(D2, I, axis=1)
    self_mask = I == np.arange(n)[:, None]
    order = np.argsort(self_mask, axis=1, kind="stable")[:, :m]
    out_I = np.take_along_axis(I, order, 1).astype(np.int64)
    out_D = np.sqrt(np.maximum(np.take_along_axis(D2, order, 1), 0.0)).astype(np.float64)
    bad = out_I < 0
    if bad.any():
        out_D[bad] = np.inf
    return out_D, out_I


def nearest_contexts_by_state(F: np.ndarray, ctx_state: np.ndarray | None, m: int, hnsw_threshold: int = 100_000):
    """Retrieve up to m nearest neighbours c_i != c_t within the same turn state s_i == s_t."""
    n = len(F)
    if n <= 1 or m <= 0:
        return np.zeros((n, 0), dtype=np.float64), np.zeros((n, 0), dtype=np.int64)
    if ctx_state is None:
        return nearest_contexts(F, m, hnsw_threshold=hnsw_threshold)
    uniq_states = np.unique(ctx_state)
    if len(uniq_states) == 1:
        return nearest_contexts(F, m, hnsw_threshold=hnsw_threshold)
    m_eff = min(m, n - 1)
    out_D = np.full((n, m_eff), np.inf, dtype=np.float64)
    out_I = np.full((n, m_eff), -1, dtype=np.int64)
    for st in uniq_states:
        idx_st = np.nonzero(ctx_state == st)[0]
        if len(idx_st) <= 1:
            continue
        d_b, i_b = nearest_contexts(F[idx_st], m_eff, hnsw_threshold=hnsw_threshold)
        mb = i_b.shape[1]
        valid_b = i_b >= 0
        out_D[idx_st, :mb] = d_b
        out_I[idx_st, :mb] = np.where(valid_b, idx_st[np.maximum(i_b, 0)], -1)
    return out_D, out_I


# --------------------------------------------------------------------------
# Version 2 Risk Scale & Bias Fitting (Eqs. 3, 4, 5)
# --------------------------------------------------------------------------
def _family_d0(idx_obj, n_fit: int, V_act: int, fallback: float) -> float:
    """Doubleton-collision support prior d0 = clip((D + 1) / (C + 1), 1, V_act)."""
    dbl = idx_obj.n == 2
    D = int(dbl.sum())
    if D < n_fit:
        return float(fallback)
    C = int((dbl & (idx_obj.d_obs == 1)).sum())
    return float(np.clip((D + 1.0) / (C + 1.0), 1.0, float(V_act)))


def support_prior(orders: list[OrderIndex], n_fit: int, V_act: int) -> np.ndarray:
    d0 = np.zeros(len(orders), dtype=np.float64)
    d0[0] = float(max(V_act, 1))
    for k in range(1, len(orders)):
        d0[k] = _family_d0(orders[k], n_fit, V_act, d0[k - 1])
    return d0


def d_tilde(n, d_obs, f1, d0k: float, n0: float):
    return (n * (d_obs + f1 + 1.0) + n0 * d0k) / (n + n0)


def var_term_v2(n, d_obs, f1, d0k: float, n0: float):
    """Paninski/Jensen KL variance bound ln(1 + (d_tilde - 1) / n); +inf when n < 1."""
    n = np.asarray(n, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.log1p(np.maximum(d_tilde(n, d_obs, f1, d0k, n0) - 1.0, 0.0) / n)
    return np.where(n >= 1, v, np.inf)


def plugin_kl_v2(Cp: sp.csr_matrix, Cq: sp.csr_matrix, beta0: float, prior: np.ndarray) -> np.ndarray:
    """Sparse row-wise KL(p_hat(c1) || p_tilde(c2)) with total unigram pseudo-count beta0."""
    Cp = Cp.tocsr()
    Cq = Cq.tocsr()
    B = Cp.shape[0]
    if B == 0 or Cp.nnz == 0:
        return np.zeros(B, dtype=np.float64)
    rows = np.repeat(np.arange(B, dtype=np.int64), np.diff(Cp.indptr))
    cols = Cp.indices
    p_cnt = Cp.data
    np_row = np.asarray(Cp.sum(axis=1)).ravel()
    nq_row = np.asarray(Cq.sum(axis=1)).ravel()
    q_cnt = np.asarray(Cq[rows, cols]).ravel()
    p_val = p_cnt / np_row[rows]
    q_val = (q_cnt + beta0 * prior[cols]) / (nq_row[rows] + beta0)
    out = np.zeros(B, dtype=np.float64)
    np.add.at(out, rows, p_val * np.log(p_val / np.maximum(q_val, 1e-300)))
    return out


def fit_delta_v2(orders: list[OrderIndex], cfg: GBDConfig, rng, prior: np.ndarray) -> np.ndarray:
    """Order-truncation bias delta[k]: nested minus-sign noise, rectified once after mean."""
    delta = np.zeros(cfg.K + 1, dtype=np.float64)
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
                kl = plugin_kl_v2(hi.C[a], lo.C[b], cfg.beta0, prior)
                noise = (hi.d_obs[a] + hi.f1[a]) / 2.0 * (1.0 / hi.n[a] - 1.0 / lo.n[b])
                vals.append(kl - noise)
            step = max(float(np.concatenate(vals).mean()), 0.0)
        delta[j] = delta[j + 1] + step
    return delta


def fit_semantic_bias_v2(sidx, nD: np.ndarray, nI: np.ndarray, cfg: GBDConfig, rng, prior: np.ndarray):
    """OLS fit of unrectified y on ||df|| under b >= 0 (KKT); None if < 10 reliable pairs."""
    if nI.size == 0:
        return None
    rows, cols = np.nonzero(nI >= 0)
    if len(rows) == 0:
        return None
    c, c2, dist = rows, nI[rows, cols], nD[rows, cols]
    sel = np.nonzero((sidx.n[c] >= cfg.n_fit) & (sidx.n[c2] >= cfg.n_fit) & np.isfinite(dist))[0]
    if len(sel) < 10:
        return None
    sel = rng.choice(sel, size=min(len(sel), cfg.n_fit_pairs), replace=False)
    ys = []
    for s in range(0, len(sel), 200):
        i = sel[s:s + 200]
        kl = plugin_kl_v2(sidx.C[c[i]], sidx.C[c2[i]], cfg.beta0, prior)
        noise = ((sidx.d_obs[c[i]] + sidx.f1[c[i]]) / (2.0 * sidx.n[c[i]])
                 + (sidx.d_obs[c2[i]] + sidx.f1[c2[i]]) / (2.0 * sidx.n[c2[i]]))
        ys.append(kl - noise)
    y, x = np.concatenate(ys), dist[sel]
    a, b = np.linalg.lstsq(np.stack([np.ones_like(x), x], 1), y, rcond=None)[0]
    if b < 0:
        a, b = float(y.mean()), 0.0
    return float(a), float(b)


def _phi_weights(r: np.ndarray, cfg: GBDConfig) -> np.ndarray:
    """Compute unnormalised weights phi(r); 0 where r is +inf."""
    fin = np.isfinite(r)
    if cfg.phi == "uniform":
        return np.where(fin, 1.0, 0.0)
    if cfg.phi == "inv":
        return np.where(fin, 1.0 / np.maximum(r, cfg.r_min), 0.0)
    if cfg.phi == "exp":
        if r.ndim == 2:
            r_min_row = np.min(r, axis=1, keepdims=True)
            shift = np.where(np.isfinite(r_min_row), r_min_row, 0.0)
            return np.where(fin, np.exp(-cfg.tau * np.where(fin, r - shift, 0.0)), 0.0)
        return np.where(fin, np.exp(-cfg.tau * np.where(fin, r, 0.0)), 0.0)
    raise ValueError(f"Unknown phi weighting rule: {cfg.phi!r}")


def _mixture_risk(weights: np.ndarray, risks: np.ndarray) -> np.ndarray:
    """Stabilised log-sum-exp bound -ln sum_j w_j exp(-r_j); +inf if no component has positive weight."""
    active = (weights > 0) & np.isfinite(risks)
    has_any = active.any(axis=1)
    r_clean = np.where(active, risks, np.inf)
    m = np.min(r_clean, axis=1, keepdims=True)
    m_safe = np.where(np.isfinite(m), m, 0.0)
    terms = np.where(active, weights * np.exp(-(np.where(active, risks, m_safe) - m_safe)), 0.0)
    s = terms.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = m_safe[:, 0] - np.log(np.maximum(s, 1e-300))
    return np.where(has_any, out, np.inf)


# --------------------------------------------------------------------------
# Legacy Version 1 helpers (retained solely for risk_version=1 comparisons)
# --------------------------------------------------------------------------
def _var_term_v1(n, d_obs, f1):
    with np.errstate(divide="ignore", invalid="ignore"):
        v = (d_obs + f1) / (2.0 * n)
    v = np.asarray(v, dtype=np.float64)
    v[n < 1] = np.inf
    return v


def _plugin_kl_v1(Cp: sp.csr_matrix, Cq: sp.csr_matrix, eps: float) -> np.ndarray:
    V = Cp.shape[1]
    P, Q = Cp.toarray(), Cq.toarray()
    p = P / P.sum(1, keepdims=True)
    q = (Q + eps) / (Q.sum(1, keepdims=True) + eps * V)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(p > 0, p * np.log(p / q), 0.0)
    return t.sum(1)


def _build_targets_v1(tokens: np.ndarray, V: int, E: np.ndarray | None, cfg: GBDConfig) -> GBDTargets:
    T = len(tokens)
    rng = np.random.default_rng(cfg.seed)
    b_pos = conversation_starts(tokens, cfg.eos_id)
    orders, h = [], np.zeros(T, dtype=np.uint64)
    for k in range(cfg.K + 1):
        idx = OrderIndex(tokens, k, V, h, b_pos=b_pos)
        h = idx.h
        idx.truncate(cfg.R_pre)
        orders.append(idx)
    exact_orders = list(range(cfg.K + 1)) if cfg.exact_orders is None else list(cfg.exact_orders)

    if cfg.delta is not None:
        delta = np.asarray(cfg.delta, dtype=np.float64)
    else:
        step = np.zeros(cfg.K + 1)
        last = 0.0
        for j in range(cfg.K - 1, -1, -1):
            hi, lo = orders[j + 1], orders[j]
            cand = np.nonzero(hi.n >= cfg.n_fit)[0]
            if len(cand) == 0:
                step[j] = last
                continue
            cand = rng.choice(cand, size=min(len(cand), cfg.n_fit_pairs), replace=False)
            lo_ids = lo.ctx[hi.rep_pos[cand]]
            vals = []
            for s in range(0, len(cand), 200):
                ca, cb = cand[s:s + 200], lo_ids[s:s + 200]
                kl = _plugin_kl_v1(hi.C[ca], lo.C[cb], cfg.fit_eps)
                noise = _var_term_v1(hi.n[ca], hi.d_obs[ca], hi.f1[ca]) + _var_term_v1(lo.n[cb], lo.d_obs[cb], lo.f1[cb])
                vals.append(np.maximum(kl - noise, 0.0))
            step[j] = last = float(np.mean(np.concatenate(vals)))
        delta = np.zeros(cfg.K + 1)
        for k in range(cfg.K - 1, -1, -1):
            delta[k] = delta[k + 1] + step[k]

    use_sem = cfg.use_semantic and E is not None and cfg.M > 0
    a = b = None
    if use_sem:
        sidx = orders[cfg.Ks]
        F = context_features(tokens, sidx, E, cfg.gamma)
        nD, nI = nearest_contexts(F, cfg.M_search, hnsw_threshold=cfg.hnsw_threshold)
        if cfg.a is None or cfg.b is None:
            rows, cols = np.nonzero(nI >= 0)
            c, c2, dist = rows, nI[rows, cols], nD[rows, cols]
            ok = (sidx.n[c] >= cfg.n_fit) & (sidx.n[c2] >= cfg.n_fit)
            if ok.sum() < 10:
                a, b = 0.0, 1.0
            else:
                sel = rng.choice(np.nonzero(ok)[0], size=min(int(ok.sum()), cfg.n_fit_pairs), replace=False)
                ys = []
                for s in range(0, len(sel), 200):
                    i = sel[s:s + 200]
                    kl = _plugin_kl_v1(sidx.C[c[i]], sidx.C[c2[i]], cfg.fit_eps)
                    noise = _var_term_v1(sidx.n[c[i]], sidx.d_obs[c[i]], sidx.f1[c[i]]) + _var_term_v1(sidx.n[c2[i]], sidx.d_obs[c2[i]], sidx.f1[c2[i]])
                    ys.append(np.maximum(kl - noise, 0.0))
                aa, bb = np.linalg.lstsq(np.stack([np.ones(len(sel)), dist[sel]], 1), np.concatenate(ys), rcond=None)[0]
                a, b = float(max(aa, 0.0)), float(max(bb, 0.0))
        else:
            a, b = float(cfg.a), float(cfg.b)
        var_nb = _var_term_v1(sidx.n[nI], sidx.d_obs[nI], sidx.f1[nI])
        r_nb_ctx = delta[cfg.Ks] + a + b * nD + var_nb
        keep = np.argsort(r_nb_ctx, axis=1)[:, : cfg.M]
        nI = np.take_along_axis(nI, keep, 1)
        r_nb_ctx = np.take_along_axis(r_nb_ctx, keep, 1)
        Cs_norm = sp.diags(1.0 / np.maximum(sidx.tr_sum, 1e-12)) @ sidx.Ctr

    out_ids = np.zeros((T, cfg.r), dtype=np.int32)
    out_p = np.zeros((T, cfg.r), dtype=np.float32)
    out_R = np.zeros(T, dtype=np.float32)
    out_wex = np.zeros(T, dtype=np.float32)
    k_star = np.zeros(T, dtype=np.int16)
    loo = 1.0 if cfg.leave_one_out else 0.0

    for s in range(0, T, cfg.chunk):
        pos = np.arange(s, min(s + cfg.chunk, T))
        P, x = len(pos), tokens[pos]
        risks, mats, corr, n_k_all = [], [], [], []
        for k in exact_orders:
            idx = orders[k]
            ctx = idx.ctx[pos]
            valid = ctx >= 0
            cs = np.where(valid, ctx, 0)
            cnt = np.asarray(idx.C[cs, x]).ravel() * valid
            n = (idx.n[cs] - loo) * valid
            d = idx.d_obs[cs] - loo * (cnt == 1)
            f1 = idx.f1[cs] - loo * (cnt == 1) + loo * (cnt == 2)
            in_tr = (np.asarray(idx.Ctr[cs, x]).ravel() > 0) * valid
            s_tr = idx.tr_sum[cs] - loo * in_tr
            r = delta[k] + _var_term_v1(n, d, f1)
            r[(~valid) | (s_tr <= 0)] = np.inf
            risks.append(r)
            mats.append((idx.Ctr, cs, s_tr))
            corr.append((in_tr * loo, s_tr))
            n_k_all.append(np.where(valid, n, 0))
        n_k_all = np.stack(n_k_all, 1)
        has = n_k_all >= 1
        kk = np.array(exact_orders)
        k_star[pos] = np.where(has.any(1), kk[np.argmax(has * (kk + 1), axis=1)], 0)
        if use_sem:
            cs_s = sidx.ctx[pos]
            valid_s = cs_s >= 0
            r_nb = np.where(valid_s[:, None], r_nb_ctx[np.where(valid_s, cs_s, 0)], np.inf)
            risks.extend(list(r_nb.T))
        Rm = np.stack(risks, 1)
        phi = _phi_weights(Rm, cfg)
        W = phi / np.maximum(phi.sum(1, keepdims=True), 1e-300)
        out_R[pos] = (W * np.where(np.isfinite(Rm), Rm, 0.0)).sum(1)
        n_ex = len(exact_orders)
        out_wex[pos] = W[:, :n_ex].sum(1)

        Q = sp.csr_matrix((P, V))
        rowsP = np.arange(P)
        for j, (Ctr, cs, s_tr) in enumerate(mats):
            wj = W[:, j] / np.maximum(s_tr, 1e-12)
            nz = wj > 0
            if not nz.any():
                continue
            Q = Q + sp.csr_matrix((wj[nz], (rowsP[nz], cs[nz])), shape=(P, Ctr.shape[0])) @ Ctr
            in_tr, _ = corr[j]
            c = wj * in_tr
            if (c > 0).any():
                m = c > 0
                Q = Q - sp.csr_matrix((c[m], (rowsP[m], x[m])), shape=(P, V))
        if use_sem:
            nb_ids = nI[np.where(sidx.ctx[pos] >= 0, sidx.ctx[pos], 0)]
            Wn = W[:, n_ex:]
            m = Wn > 0
            Q = Q + sp.csr_matrix((Wn[m], (np.nonzero(m)[0], nb_ids[m])), shape=(P, Cs_norm.shape[0])) @ Cs_norm
        Q = Q.tocsr()
        Q.data[Q.data < 1e-12] = 0.0
        Q.eliminate_zeros()
        ids, pr = _topr(Q, cfg.r)
        empty = pr.sum(1) <= 0
        ids[empty, 0] = x[empty]
        pr[empty, 0] = 1.0
        out_ids[pos] = ids
        out_p[pos] = pr / pr.sum(1, keepdims=True)
        out_R[pos[empty]] = np.inf

    fin = np.isfinite(out_R)
    rho = cfg.rho if cfg.rho is not None else float(np.median(out_R[fin])) if fin.any() else 1.0
    rho = max(rho, 1e-6)
    alpha = cfg.alpha_max * np.exp(-np.where(fin, out_R, np.inf) / rho)
    return GBDTargets(
        ids=out_ids, probs=out_p, alpha=alpha.astype(np.float32), R=out_R,
        w_exact=out_wex, k_star=k_star,
        fitted={"delta": delta, "a": a, "b": b, "rho": rho},
    )


# --------------------------------------------------------------------------
# Main entry point (GBD v2 default)
# --------------------------------------------------------------------------
def build_targets(
    tokens: np.ndarray,
    V: int,
    E: np.ndarray | None,
    cfg: GBDConfig,
    sem_feature_fn: Callable | None = None,
) -> GBDTargets:
    tokens = np.asarray(tokens, dtype=np.int64)
    if cfg.risk_version == 1:
        return _build_targets_v1(tokens, V, E, cfg)

    T = len(tokens)
    rng = np.random.default_rng(cfg.seed)
    eff_eos = cfg.eos_id if cfg.respect_eos else None
    b_pos = conversation_starts(tokens, eff_eos)

    # 1. Exact counts (boundary-safe Eq. 1 when respect_eos=True)
    exact_orders = cfg.resolve_exact_orders()
    use_role_sem_idx = len(cfg.role_ids) > 0 and (cfg.skip_role_markers or cfg.partition_turn_state)
    max_k = max(
        [cfg.K]
        + exact_orders
        + ([cfg.Ks] if (cfg.use_semantic and not use_role_sem_idx) else [])
    )
    orders, h = [], np.zeros(T, dtype=np.uint64)
    for k in range(max_k + 1):
        idx = OrderIndex(tokens, k, V, h, b_pos=b_pos)
        h = idx.h
        idx.truncate(cfg.R_pre)
        orders.append(idx)

    # 2. Unigram prior, support priors d0, and order-truncation biases delta
    uni = np.bincount(tokens, minlength=V).astype(np.float64)
    V_act = int((uni > 0).sum())
    prior = uni / max(uni.sum(), 1.0)
    d0 = np.asarray(cfg.d0, dtype=np.float64) if cfg.d0 is not None else support_prior(orders[: max_k + 1], cfg.n_fit, V_act)
    delta = np.asarray(cfg.delta, dtype=np.float64) if cfg.delta is not None else fit_delta_v2(orders[: cfg.K + 1], cfg, rng, prior)
    if len(delta) <= max_k:
        delta = np.pad(delta, (0, max_k + 1 - len(delta)), mode="edge")
    if len(d0) <= max_k:
        d0 = np.pad(d0, (0, max_k + 1 - len(d0)), mode="edge")

    # 3. Semantic branch (Eqs. 2, 3, 4)
    has_geom = (E is not None) or (sem_feature_fn is not None)
    use_sem = cfg.use_semantic and has_geom and cfg.M > 0
    a = b = d0_sem = None
    if use_sem:
        if use_role_sem_idx:
            sidx = SemanticIndex(
                tokens,
                cfg.Ks,
                V,
                b_pos=b_pos,
                role_ids=cfg.role_ids,
                skip_roles=cfg.skip_role_markers,
                use_turn_state=cfg.partition_turn_state,
            )
            sidx.truncate(cfg.R_pre)
            ctx_state = sidx.ctx_state if cfg.partition_turn_state else None
        else:
            sidx = orders[cfg.Ks]
            ctx_state = None
        d0_fallback = d0[min(cfg.Ks, len(d0) - 1)]
        d0_sem = float(cfg.d0_sem) if cfg.d0_sem is not None else _family_d0(sidx, cfg.n_fit, V_act, d0_fallback)
        F = sem_feature_fn(tokens, sidx) if sem_feature_fn is not None else context_features(tokens, sidx, E, cfg.gamma)
        nD, nI = nearest_contexts_by_state(F, ctx_state, cfg.M_search, hnsw_threshold=cfg.hnsw_threshold)
        if nI.shape[1] == 0:
            use_sem = False
        elif cfg.a is not None and cfg.b is not None:
            a, b = float(cfg.a), float(cfg.b)
        else:
            fit = fit_semantic_bias_v2(sidx, nD, nI, cfg, rng, prior)
            if fit is None:
                use_sem = False
            else:
                a, b = fit

    if use_sem:
        valid_nb = (nI >= 0) & np.isfinite(nD)
        safe_nI = np.where(valid_nb, nI, 0)
        valid_nb &= sidx.tr_sum[safe_nI] > 0
        bias_nb = np.where(valid_nb, np.maximum(a + b * nD, 0.0), 0.0)
        v_nb = var_term_v2(sidx.n[safe_nI], sidx.d_obs[safe_nI], sidx.f1[safe_nI], d0_sem, cfg.n0)
        delta_ks = delta[min(cfg.Ks, len(delta) - 1)]
        r_nb = np.where(valid_nb, delta_ks + bias_nb + v_nb, np.inf)

        m_keep = min(cfg.M, nI.shape[1])
        keep = np.argsort(r_nb, axis=1, kind="stable")[:, :m_keep]
        nI = np.take_along_axis(nI, keep, 1)
        r_nb = np.take_along_axis(r_nb, keep, 1)
        bias_nb = np.take_along_axis(bias_nb, keep, 1)
        valid_nb = np.isfinite(r_nb) & (nI >= 0)
        safe_nI = np.where(valid_nb, nI, 0)

        ph_nb = _phi_weights(r_nb, cfg)
        ph_sum = ph_nb.sum(1, keepdims=True)
        has_sem_ctx = ph_sum[:, 0] > 0
        wt = np.where(ph_sum > 0, ph_nb / np.maximum(ph_sum, 1e-300), 0.0)
        if cfg.pool_semantic:
            n_nb = np.where(valid_nb, sidx.n[safe_nI], 1.0)
            denom_N = (wt ** 2 / n_nb).sum(1)
            N_sem = np.where(has_sem_ctx, 1.0 / np.maximum(denom_N, 1e-300), 1.0)
            d_bar = (wt * d_tilde(n_nb, sidx.d_obs[safe_nI], sidx.f1[safe_nI], d0_sem, cfg.n0)).sum(1)
            r_sem_ctx = np.where(
                has_sem_ctx,
                delta_ks + (wt * bias_nb).sum(1) + np.log1p(np.maximum(d_bar - 1.0, 0.0) / N_sem),
                np.inf,
            )
        else:
            r_sem_ctx = _mixture_risk(wt, r_nb)
        Cs_norm = sp.diags(1.0 / np.maximum(sidx.tr_sum, 1e-12)) @ sidx.Ctr

    out_ids = np.zeros((T, cfg.r), dtype=np.int32)
    out_p = np.zeros((T, cfg.r), dtype=np.float32)
    o_R = np.zeros(T, dtype=np.float64)
    o_r0 = np.zeros(T, dtype=np.float64)
    o_ro = np.zeros(T, dtype=np.float64)
    o_rs = np.zeros(T, dtype=np.float64)
    o_lam = np.zeros(T, dtype=np.float64)
    k_star = np.full(T, -1 if len(exact_orders) == 0 else 0, dtype=np.int16)
    loo = 1.0 if cfg.leave_one_out else 0.0

    # 4. Chunked target mixing
    idx0 = orders[0]
    for s in range(0, T, cfg.chunk):
        pos = np.arange(s, min(s + cfg.chunk, T))
        P, x = len(pos), tokens[pos]

        # unigram LOO risk r_{0,t} (always measured for Eq. 5 anchor)
        cs0 = np.maximum(idx0.ctx[pos], 0)
        cnt0 = np.asarray(idx0.C[cs0, x]).ravel()
        n0_pos = idx0.n[cs0] - loo
        d0_pos = idx0.d_obs[cs0] - loo * (cnt0 == 1)
        f1_0_pos = idx0.f1[cs0] - loo * (cnt0 == 1) + loo * (cnt0 == 2)
        in_tr0 = np.asarray(idx0.Ctr[cs0, x]).ravel() > 0
        s_tr0 = idx0.tr_sum[cs0] - loo * in_tr0
        r0_pos = delta[0] + var_term_v2(n0_pos, d0_pos, f1_0_pos, d0[0], cfg.n0)
        r0_pos[s_tr0 <= 0] = np.inf

        mats, corr = [], []
        if len(exact_orders) > 0:
            risks, n_all = [], []
            for k in exact_orders:
                idx = orders[k]
                ctx = idx.ctx[pos]
                valid = ctx >= 0
                cs = np.where(valid, ctx, 0)
                cnt = np.asarray(idx.C[cs, x]).ravel() * valid
                n = (idx.n[cs] - loo) * valid
                d = idx.d_obs[cs] - loo * (cnt == 1)
                f1 = idx.f1[cs] - loo * (cnt == 1) + loo * (cnt == 2)
                in_tr = (np.asarray(idx.Ctr[cs, x]).ravel() > 0) * valid
                s_tr = idx.tr_sum[cs] - loo * in_tr
                r = delta[k] + var_term_v2(n, d, f1, d0[k], cfg.n0)
                r[(~valid) | (s_tr <= 0)] = np.inf
                risks.append(r)
                mats.append((idx.Ctr, cs, s_tr))
                corr.append(in_tr * loo)
                n_all.append(np.where(valid, n, 0))

            Rm = np.stack(risks, 1)
            n_all = np.stack(n_all, 1)
            has = n_all >= 1
            kk = np.asarray(exact_orders, dtype=np.int16)
            k_star[pos] = np.where(has.any(1), kk[np.argmax(has * (kk + 1), axis=1)], 0)

            ph = _phi_weights(Rm, cfg)
            ph_sum = ph.sum(1, keepdims=True)
            u = np.where(ph_sum > 0, ph / np.maximum(ph_sum, 1e-300), 0.0)
            r_ord = _mixture_risk(u, Rm)
        else:
            u = np.zeros((P, 0), dtype=np.float64)
            r_ord = np.full(P, np.inf, dtype=np.float64)

        if use_sem:
            cs_s = sidx.ctx[pos]
            vs = cs_s >= 0
            cs_s0 = np.where(vs, cs_s, 0)
            r_sem = np.where(vs, r_sem_ctx[cs_s0], np.inf)
        else:
            r_sem = np.full(P, np.inf)

        has_ord = np.isfinite(r_ord)
        has_sem = np.isfinite(r_sem)
        branch_r = np.stack([r_ord, r_sem], axis=1)
        if cfg.lambda_mode == "fixed":
            fl = float(np.clip(cfg.fixed_lambda, 0.0, 1.0))
            lam = np.where(has_ord & has_sem, fl, np.where(has_ord, 1.0, 0.0))
        elif cfg.lambda_mode == "adaptive":
            branch_ph = _phi_weights(branch_r, cfg)
            po, ps = branch_ph[:, 0], branch_ph[:, 1]
            lam = np.where((po + ps) > 0, po / np.maximum(po + ps, 1e-300), 0.0)
        else:
            raise ValueError(f"Unknown lambda_mode: {cfg.lambda_mode!r}")

        w_sem_share = np.where(has_ord | has_sem, 1.0 - lam, 0.0)
        branch_w = np.stack([lam, w_sem_share], axis=1)
        R = _mixture_risk(branch_w, branch_r)

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
            Wn = w_sem_share[:, None] * wt[cs_s0] * vs[:, None]
            nb_mat = np.maximum(nI[cs_s0], 0)
            m = (Wn > 0) & (nI[cs_s0] >= 0)
            if m.any():
                Q = Q + sp.csr_matrix((Wn[m], (np.nonzero(m)[0], nb_mat[m])), shape=(P, Cs_norm.shape[0])) @ Cs_norm

        Q = Q.tocsr()
        Q.data[Q.data < 1e-12] = 0.0
        Q.eliminate_zeros()

        ids, pr = _topr(Q, cfg.r)
        empty = pr.sum(1) <= 0
        ids[empty, 0] = x[empty]
        pr[empty, 0] = 1.0
        out_ids[pos] = ids
        out_p[pos] = pr / pr.sum(1, keepdims=True)
        R[empty] = np.inf
        r_ord[empty & ~np.isfinite(r_ord)] = np.inf

        o_R[pos], o_r0[pos], o_ro[pos], o_rs[pos], o_lam[pos] = R, r0_pos, r_ord, r_sem, lam

    # 5. Per-token distillation strength alpha_t (Eq. 5: fixed vs adaptive)
    amode = cfg.alpha_mode.strip()
    if amode in ("fixed", "const"):
        alpha = np.where(np.isfinite(o_R), cfg.alpha_max, 0.0)
    else:
        if amode in ("adaptive", "r_ord"):
            driver = o_ro if len(exact_orders) > 0 else o_R
        elif amode == "r_ord_strict":
            driver = o_ro
        elif amode == "R":
            driver = o_R
        elif amode == "r_sem":
            driver = o_rs
        else:
            raise ValueError(f"Unknown alpha_mode: {cfg.alpha_mode!r}")
        denom = np.maximum(o_r0, cfg.r_min)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(np.isfinite(driver) & np.isfinite(o_r0), 1.0 - driver / denom, 0.0)
        alpha = cfg.alpha_max * (np.clip(ratio, 0.0, 1.0) ** cfg.gamma_alpha)

    switches = {
        "use_exact": cfg.use_exact,
        "use_order_backoff": cfg.use_order_backoff,
        "exact_orders": exact_orders,
        "leave_one_out": cfg.leave_one_out,
        "respect_eos": cfg.respect_eos,
        "use_semantic": bool(use_sem),
        "skip_role_markers": cfg.skip_role_markers,
        "partition_turn_state": cfg.partition_turn_state,
        "pool_semantic": cfg.pool_semantic,
        "phi": cfg.phi,
        "lambda_mode": cfg.lambda_mode,
        "fixed_lambda": cfg.fixed_lambda if cfg.lambda_mode == "fixed" else None,
        "alpha_mode": cfg.alpha_mode,
        "alpha_max": cfg.alpha_max,
    }
    return GBDTargets(
        ids=out_ids,
        probs=out_p,
        alpha=alpha.astype(np.float32),
        R=o_R.astype(np.float32),
        w_exact=o_lam.astype(np.float32),
        k_star=k_star,
        fitted={"delta": delta[: cfg.K + 1], "a": a, "b": b, "d0": d0[: cfg.K + 1], "d0_sem": d0_sem, "switches": switches},
        r_ord=o_ro.astype(np.float32),
        r_sem=o_rs.astype(np.float32),
        r0=o_r0.astype(np.float32),
    )


def _topr(Q: sp.csr_matrix, r: int):
    P = Q.shape[0]
    ids = np.zeros((P, r), dtype=np.int32)
    pr = np.zeros((P, r), dtype=np.float32)
    if Q.nnz == 0:
        return ids, pr
    rows = np.repeat(np.arange(P), np.diff(Q.indptr))
    order = np.lexsort((-Q.data, rows))
    rs = rows[order]
    rank = np.arange(Q.nnz) - Q.indptr[rs]
    keep = rank < r
    ids[rs[keep], rank[keep]] = Q.indices[order][keep]
    pr[rs[keep], rank[keep]] = Q.data[order][keep]
    return ids, pr


# --------------------------------------------------------------------------
# Training loss & DataLoader chunks (PyTorch, Eq. 6)
# --------------------------------------------------------------------------
def gbd_loss(logits, labels, tgt_ids, tgt_probs, alpha, ignore_index: int = -100, nll_weight=None):
    """
    logits (B,L,V); labels (B,L); tgt_ids (B,L,r) long; tgt_probs (B,L,r); alpha (B,L).
    Default (`nll_weight=None`): `(1 - alpha) * NLL(x_t) + alpha * H(q_t, p)`, whose gradient wrt
    logits is `softmax(z) - [(1 - a) e_{x_t} + a q_t]`.
    When `nll_weight` is provided (e.g. CoCoNTs Eq. 4 where `w_nll + alpha != 1` per sample,
    matching AllNTs in expectation), computes `nll_weight * NLL(x_t) + alpha * H(q_t, p)`.
    """
    import torch
    import torch.nn.functional as Fnn

    logp = Fnn.log_softmax(logits.float(), dim=-1)
    mask = labels != ignore_index
    lab = labels.clamp_min(0)
    tp = tgt_probs.float()
    nll = -logp.gather(-1, lab.unsqueeze(-1)).squeeze(-1)
    soft = -(tp * logp.gather(-1, tgt_ids.long())).sum(-1)
    a = alpha.to(logp.dtype)
    w_nll = (1.0 - a) if nll_weight is None else nll_weight.to(logp.dtype)
    per_tok = w_nll * nll + a * soft
    denom = mask.sum().clamp_min(1)
    loss = (per_tok * mask).sum() / denom
    with torch.no_grad():
        ent = -(tp * torch.log(tp.clamp_min(1e-12))).sum(-1)
        kl = ((soft - ent) * mask).sum() / denom
        ce = (nll * mask).sum() / denom
    return loss, {"ce": ce.item(), "kl": kl.item()}


class GBDChunks:
    """Non-overlapping training chunks aligned with precomputed targets (supports memmap arrays).
    Item i: input x[s:s+L], labels y[s+1:s+L+1] (masked with -100 where train_mask is False),
    and the targets of positions s+1..s+L (returning a 6-tuple if `tg.nll_weight` is present)."""

    def __init__(
        self,
        tokens: np.ndarray,
        tg: GBDTargets | None,
        L: int,
        train_mask: np.ndarray | None = None,
        prefix_only_k: int | None = None,
    ):
        self.x = tokens
        self.tg = tg
        self.L = L
        self.train_mask = train_mask
        self.prefix_only_k = prefix_only_k
        self.n = (len(tokens) - 1) // L

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        import torch

        s = i * self.L
        sl = slice(s + 1, s + self.L + 1)
        x = np.asarray(self.x[s:s + self.L], dtype=np.int64)
        y = np.asarray(self.x[sl], dtype=np.int64).copy()
        if self.train_mask is not None:
            m = np.asarray(self.train_mask[sl], dtype=bool)
            y[~m] = -100
        if self.tg is None:
            return torch.from_numpy(x), torch.from_numpy(y)
        ids = np.asarray(self.tg.ids[sl], dtype=np.int64)
        probs = np.asarray(self.tg.probs[sl], dtype=np.float32)
        ps = probs.sum(axis=-1, keepdims=True)
        probs = probs / np.maximum(ps, 1e-12)
        alpha = np.asarray(self.tg.alpha[sl], dtype=np.float32).copy()
        if self.tg.nll_weight is not None:
            w_nll = np.asarray(self.tg.nll_weight[sl], dtype=np.float32).copy()
            if self.prefix_only_k is not None:
                alpha[self.prefix_only_k :] = 0.0
                w_nll[self.prefix_only_k :] = 1.0
            return (
                torch.from_numpy(x),
                torch.from_numpy(y),
                torch.from_numpy(ids),
                torch.from_numpy(probs),
                torch.from_numpy(alpha),
                torch.from_numpy(w_nll),
            )
        if self.prefix_only_k is not None:
            alpha[self.prefix_only_k :] = 0.0
        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(ids),
            torch.from_numpy(probs),
            torch.from_numpy(alpha),
        )
