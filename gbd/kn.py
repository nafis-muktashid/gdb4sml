"""
Interpolated modified Kneser–Ney targets (Chen & Goodman 1998).

Orders k = 0..K. Order K uses raw counts. Orders k < K use continuation
counts N1+(· h w) = number of distinct in-bounds left tokens preceding (h, w).
Discounts are the modified-KN triple from full-corpus count-of-counts:

    Y  = n1 / (n1 + 2 n2)
    D1 = 1 - 2 Y n2 / n1
    D2 = 2 - 3 Y n3 / n2
    D3 = 3 - 4 Y n4 / n3

    P_k(w|h) = max(c_k(h,w) - D(c_k(h,w)), 0) / c_k(h) + gamma_k(h) P_{k-1}(w|h')
    gamma_k(h) = (D1 N1(h) + D2 N2(h) + D3 N3+(h)) / c_k(h)

Order 0 is the continuation unigram, interpolated with uniform 1/V.
If c_k(h) = 0 the order is skipped and P_k = P_{k-1}.
Contexts do not cross eos_id when respect_eos is set (GBD Eq. 1, via OrderIndex).

Leave-one-out removes every n-gram event ending at t:
  - raw c_K(h_K(t), x_t) loses 1
  - continuation N1+(· h_k(t), x_t) loses 1 iff the left-extended raw event had count 1
Discounts stay at the full-corpus values. The stored target is the exact P_K
on (union of nonzero count-rows at this position) ∪ (top-64 of order 0),
truncated to top-r and renormalised. alpha is the constant alpha_max
(nll_weight left unset, so the loss is (1-a) NLL + a H(q, p) and
--alpha_schedule / --anneal_alpha apply unchanged).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from gbd import GBDTargets, OrderIndex, conversation_starts

# Head of the order-0 distribution added to every position's candidate set.
KN_UNIGRAM_CANDIDATES = 64


def modified_kn_discounts(count_values: np.ndarray) -> tuple[float, float, float]:
    """Chen & Goodman 1998 D1, D2, D3+ from count-of-counts n1..n4.

    A zero denominator (no n-grams of that count) cannot estimate that discount;
    D2 falls back to D1, and D3+ falls back to D2. n1 = 0 yields (0, 0, 0).
    """
    c = np.asarray(count_values)
    n1 = float(np.sum(c == 1))
    n2 = float(np.sum(c == 2))
    n3 = float(np.sum(c == 3))
    n4 = float(np.sum(c == 4))
    if n1 <= 0.0:
        return 0.0, 0.0, 0.0
    Y = n1 / (n1 + 2.0 * n2)
    D1 = 1.0 - 2.0 * Y * n2 / n1
    D2 = (2.0 - 3.0 * Y * n3 / n2) if n2 > 0.0 else D1
    D3 = (3.0 - 4.0 * Y * n4 / n3) if n3 > 0.0 else D2
    return float(D1), float(D2), float(D3)


def _discount_lookup(counts: np.ndarray, D1: float, D2: float, D3: float) -> np.ndarray:
    out = np.zeros(np.shape(counts), dtype=np.float64)
    out = np.where(counts == 1, D1, out)
    out = np.where(counts == 2, D2, out)
    out = np.where(counts >= 3, D3, out)
    return out


def _numerators(counts: np.ndarray, D1: float, D2: float, D3: float) -> np.ndarray:
    return np.maximum(np.asarray(counts, dtype=np.float64) - _discount_lookup(counts, D1, D2, D3), 0.0)


def _as_count_csr(C: sp.csr_matrix) -> sp.csr_matrix:
    C = C.tocsr().copy()
    C.sum_duplicates()
    if C.nnz:
        C.data[:] = np.rint(C.data)
    C.sort_indices()
    C.eliminate_zeros()
    return C


def _continuation_csr(raw_hi: OrderIndex, suffix_of_ext: np.ndarray, n_suffix: int, V: int) -> sp.csr_matrix:
    """N1+(· h w): one per higher-order context row that contains w, mapped to suffix h."""
    C = raw_hi.C.tocsr()
    if C.nnz == 0 or n_suffix <= 0:
        return sp.csr_matrix((max(n_suffix, 0), V), dtype=np.float64)
    rows = np.repeat(np.arange(C.shape[0], dtype=np.int64), np.diff(C.indptr))
    suf = suffix_of_ext[rows]
    keep = suf >= 0
    mat = sp.csr_matrix(
        (np.ones(int(keep.sum()), dtype=np.float64), (suf[keep], C.indices[keep])),
        shape=(n_suffix, V),
    )
    return _as_count_csr(mat)


def _row_buckets(C: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = C.shape[0]
    counts = np.rint(C.data).astype(np.int64) if C.nnz else np.zeros(0, dtype=np.int64)
    N1 = np.zeros(n, dtype=np.int64)
    N2 = np.zeros(n, dtype=np.int64)
    N3p = np.zeros(n, dtype=np.int64)
    row_sum = np.zeros(n, dtype=np.float64)
    if C.nnz == 0:
        return counts, row_sum, N1, N2, N3p
    rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(C.indptr))
    np.add.at(N1, rows, counts == 1)
    np.add.at(N2, rows, counts == 2)
    np.add.at(N3p, rows, counts >= 3)
    np.add.at(row_sum, rows, counts.astype(np.float64))
    return counts, row_sum, N1, N2, N3p


def _lookup(C: sp.csr_matrix, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """C[rows[i], cols[i]] as int counts. rows < 0 contribute 0."""
    n = len(rows)
    out = np.zeros(n, dtype=np.int64)
    if C.nnz == 0 or n == 0:
        return out
    ridx = np.repeat(np.arange(C.shape[0], dtype=np.int64), np.diff(C.indptr))
    key_dt = np.dtype([("row", np.int64), ("col", np.int32)])
    keys = np.empty(C.nnz, dtype=key_dt)
    keys["row"] = ridx
    keys["col"] = C.indices.astype(np.int32, copy=False)
    q = np.empty(n, dtype=key_dt)
    valid = rows >= 0
    q["row"] = np.where(valid, rows, np.int64(-1))
    q["col"] = np.where(valid, cols, np.int32(-1)).astype(np.int32, copy=False)
    idx = np.searchsorted(keys, q)
    inb = idx < C.nnz
    safe = np.where(inb, idx, 0)
    hit = inb & valid & (keys["row"][safe] == q["row"]) & (keys["col"][safe] == q["col"])
    out[hit] = np.rint(C.data[safe[hit]]).astype(np.int64)
    return out


def _expand_rows(ctx: np.ndarray, indptr: np.ndarray, indices: np.ndarray, values: np.ndarray):
    """Repeat each position's CSR row. Returns local_row, token, value."""
    valid_local = np.nonzero(ctx >= 0)[0]
    if len(valid_local) == 0:
        return None
    cs = ctx[valid_local]
    nnz = np.diff(indptr)[cs]
    total = int(nnz.sum())
    if total == 0:
        return None
    starts = indptr[cs]
    within = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(nnz) - nnz, nnz)
    ptr = np.repeat(starts, nnz) + within
    local = np.repeat(valid_local, nnz)
    return local, indices[ptr], values[ptr]


def _topk_renorm(S: np.ndarray, cand: np.ndarray, r: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-r by score descending, ties broken by smaller token id, then renormalise."""
    n, M = S.shape
    ids = np.zeros((n, r), dtype=np.int32)
    pr = np.zeros((n, r), dtype=np.float64)
    if n == 0 or M == 0:
        return ids, pr
    order_id = np.argsort(cand, kind="mergesort")
    cand_s = cand[order_id]
    Ss = S[:, order_id]
    take = min(r, M)
    order_sc = np.argsort(-Ss, axis=1, kind="mergesort")[:, :take]
    top_s = np.take_along_axis(Ss, order_sc, axis=1)
    top_i = cand_s[order_sc]
    z = top_s.sum(axis=1, keepdims=True)
    good = z[:, 0] > 0
    top_p = np.zeros_like(top_s)
    top_p[good] = top_s[good] / z[good]
    ids[:, :take] = top_i
    pr[:, :take] = top_p
    return ids, pr


def _order0_top(d0_num: np.ndarray, c0: float, disc0: float, V: int, n_keep: int) -> np.ndarray:
    if c0 > 0.0:
        P0 = d0_num / c0 + (disc0 / c0) / float(V)
    else:
        P0 = np.full(V, 1.0 / float(V), dtype=np.float64)
    keep = min(n_keep, V)
    return np.lexsort((np.arange(V), -P0))[:keep].astype(np.int32)


def build_kn_targets(
    tokens: np.ndarray,
    V: int,
    K: int = 6,
    r: int = 16,
    alpha_max: float = 0.5,
    kn_loo: bool = True,
    respect_eos: bool = True,
    eos_id: int | None = 3,
    unigram_candidates: int = KN_UNIGRAM_CANDIDATES,
) -> GBDTargets:
    tokens = np.asarray(tokens, dtype=np.int64)
    T = int(len(tokens))
    K = int(K)
    r = int(r)
    V = int(V)
    if T == 0:
        z = np.zeros(0, dtype=np.float32)
        return GBDTargets(
            ids=np.zeros((0, r), np.int32),
            probs=np.zeros((0, r), np.float32),
            alpha=z,
            R=z,
            w_exact=z,
            k_star=np.zeros(0, np.int16),
            fitted={"discounts": [], "kn_loo": kn_loo},
        )

    b_pos = conversation_starts(tokens, eos_id if respect_eos else None)
    raw: list[OrderIndex] = []
    h = np.zeros(T, dtype=np.uint64)
    for k in range(K + 1):
        idx = OrderIndex(tokens, k, V, h, b_pos=b_pos)
        h = idx.h
        raw.append(idx)

    # Count matrix used at order k: raw at K, continuation below.
    matrices: list[sp.csr_matrix] = []
    for k in range(K + 1):
        if k == K and K > 0:
            matrices.append(_as_count_csr(raw[K].C))
        elif k < K:
            n_suf = int(raw[k].C.shape[0])
            if n_suf == 0 or raw[k + 1].C.shape[0] == 0:
                matrices.append(sp.csr_matrix((n_suf, V), dtype=np.float64))
            else:
                suffix = raw[k].ctx[raw[k + 1].rep_pos]
                matrices.append(_continuation_csr(raw[k + 1], suffix, n_suf, V))
        else:
            # K == 0: continuation unigram needs the order-1 raw index, which we did not build.
            # Build it just for the unigram continuation.
            idx1 = OrderIndex(tokens, 1, V, raw[0].h, b_pos=b_pos)
            suffix = raw[0].ctx[idx1.rep_pos] if idx1.C.shape[0] else np.zeros(0, dtype=np.int64)
            matrices.append(_continuation_csr(idx1, suffix, int(raw[0].C.shape[0]), V))

    discounts: list[tuple[float, float, float]] = []
    num_data: list[np.ndarray] = []
    row_sum: list[np.ndarray] = []
    N1s: list[np.ndarray] = []
    N2s: list[np.ndarray] = []
    N3s: list[np.ndarray] = []
    for k in range(K + 1):
        counts, rsum, N1, N2, N3p = _row_buckets(matrices[k])
        D = modified_kn_discounts(counts)
        discounts.append(D)
        num_data.append(_numerators(counts, *D))
        row_sum.append(rsum)
        N1s.append(N1)
        N2s.append(N2)
        N3s.append(N3p)

    # Per-position LOO adjustment of the observed cell only. Discounts stay fixed.
    s = np.zeros((T, K + 1), dtype=np.float64)
    delta = np.zeros((T, K + 1), dtype=np.float64)
    c_adj = np.zeros((T, K + 1), dtype=np.float64)
    disc_mass = np.zeros((T, K + 1), dtype=np.float64)
    in_ctx = np.zeros((T, K + 1), dtype=bool)
    for k in range(K + 1):
        ctx = raw[k].ctx
        m = _lookup(matrices[k], ctx, tokens)
        if not kn_loo:
            dec = np.zeros(T, dtype=bool)
        elif k == K:
            dec = (ctx >= 0) & (m >= 1)
        else:
            ext_ctx = raw[k + 1].ctx if k + 1 <= K else None
            if ext_ctx is None:
                # K == 0 unigram: left extension is the bigram, not stored on `raw`.
                dec = np.zeros(T, dtype=bool)
            else:
                ext_m = _lookup(raw[k + 1].C, ext_ctx, tokens)
                dec = (ext_ctx >= 0) & (ext_m == 1) & (m >= 1)
        in_k = ctx >= 0
        in_ctx[:, k] = in_k
        if len(row_sum[k]) == 0:
            c = np.zeros(T, dtype=np.float64)
            N1 = np.zeros(T, dtype=np.int64)
            N2 = np.zeros(T, dtype=np.int64)
            N3 = np.zeros(T, dtype=np.int64)
        else:
            safe = np.clip(ctx, 0, len(row_sum[k]) - 1)
            c = np.where(in_k, row_sum[k][safe], 0.0)
            N1 = np.where(in_k, N1s[k][safe], 0)
            N2 = np.where(in_k, N2s[k][safe], 0)
            N3 = np.where(in_k, N3s[k][safe], 0)
        N1a = N1 - (dec & (m == 1)).astype(np.int64) + (dec & (m == 2)).astype(np.int64)
        N2a = N2 - (dec & (m == 2)).astype(np.int64) + (dec & (m == 3)).astype(np.int64)
        N3a = N3 - (dec & (m == 3)).astype(np.int64)
        c_a = np.maximum(c - dec.astype(np.float64), 0.0)
        D1, D2, D3 = discounts[k]
        disc_mass[:, k] = D1 * N1a + D2 * N2a + D3 * N3a
        c_adj[:, k] = c_a
        num_o = _numerators(m, D1, D2, D3)
        m_loo = np.where(dec, m - 1, m)
        num_l = _numerators(np.maximum(m_loo, 0), D1, D2, D3)
        num_l = np.where(dec & (m - 1 <= 0), 0.0, num_l)
        delta[:, k] = num_l - num_o

    # K == 0 unigram LOO: left extension is the raw bigram, built above only inside matrices.
    # Recompute that decrement from a dedicated order-1 index when K == 0.
    if K == 0 and kn_loo:
        idx1 = OrderIndex(tokens, 1, V, raw[0].h, b_pos=b_pos)
        ext_m = _lookup(idx1.C, idx1.ctx, tokens)
        m0 = _lookup(matrices[0], raw[0].ctx, tokens)
        dec = (idx1.ctx >= 0) & (ext_m == 1) & (m0 >= 1)
        D1, D2, D3 = discounts[0]
        # Rebuild order-0 adjusted stats. There is a single context row.
        if len(row_sum[0]):
            N1 = N1s[0][0]
            N2 = N2s[0][0]
            N3 = N3s[0][0]
            c = row_sum[0][0]
            N1a = N1 - (dec & (m0 == 1)).astype(np.int64) + (dec & (m0 == 2)).astype(np.int64)
            N2a = N2 - (dec & (m0 == 2)).astype(np.int64) + (dec & (m0 == 3)).astype(np.int64)
            N3a = N3 - (dec & (m0 == 3)).astype(np.int64)
            disc_mass[:, 0] = D1 * N1a + D2 * N2a + D3 * N3a
            c_adj[:, 0] = np.maximum(c - dec.astype(np.float64), 0.0)
            num_o = _numerators(m0, D1, D2, D3)
            num_l = _numerators(np.maximum(m0 - dec.astype(np.int64), 0), D1, D2, D3)
            num_l = np.where(dec & (m0 - 1 <= 0), 0.0, num_l)
            delta[:, 0] = num_l - num_o

    beta = np.ones(T, dtype=np.float64)
    for k in range(K, -1, -1):
        active = in_ctx[:, k] & (c_adj[:, k] > 0)
        s[:, k] = np.where(active, beta / np.maximum(c_adj[:, k], 1.0), 0.0)
        gamma = np.ones(T, dtype=np.float64)
        if np.any(active):
            gamma[active] = disc_mass[active, k] / c_adj[active, k]
        beta = beta * gamma
    uniform = beta / float(V)

    d0 = np.zeros(V, dtype=np.float64)
    if matrices[0].nnz:
        d0[matrices[0].indices] = num_data[0]
    c0_full = float(row_sum[0][0]) if len(row_sum[0]) else 0.0
    # Full-corpus order-0 head (LOO moves one cell; the label is already in some row or this head).
    D1_0, D2_0, D3_0 = discounts[0]
    if len(row_sum[0]):
        disc0_full = D1_0 * float(N1s[0][0]) + D2_0 * float(N2s[0][0]) + D3_0 * float(N3s[0][0])
    else:
        disc0_full = 0.0
    top0 = _order0_top(d0, c0_full, disc0_full, V, unigram_candidates)

    ctx1 = raw[1].ctx if K >= 1 else np.full(T, -1, dtype=np.int64)
    out_ids = np.zeros((T, r), dtype=np.int32)
    out_p = np.zeros((T, r), dtype=np.float64)
    order = np.argsort(ctx1, kind="mergesort")
    sorted_ctx = ctx1[order]
    bounds = np.concatenate((
        np.array([0], dtype=np.int64),
        np.flatnonzero(np.diff(sorted_ctx)) + 1,
        np.array([T], dtype=np.int64),
    ))

    for g in range(len(bounds) - 1):
        pos = order[bounds[g]:bounds[g + 1]]
        c1 = int(sorted_ctx[bounds[g]])
        row_tok = np.zeros(0, dtype=np.int32)
        row_val = np.zeros(0, dtype=np.float64)
        if K >= 1 and c1 >= 0 and matrices[1].shape[0] > c1:
            a, b = int(matrices[1].indptr[c1]), int(matrices[1].indptr[c1 + 1])
            row_tok = matrices[1].indices[a:b].astype(np.int32, copy=False)
            row_val = num_data[1][a:b]
        extra = top0[~np.isin(top0, row_tok)] if len(row_tok) else top0
        cand = np.concatenate([row_tok, extra]).astype(np.int32, copy=False)
        if len(cand) == 0:
            continue
        col_of = np.full(V, -1, dtype=np.int32)
        col_of[cand] = np.arange(len(cand), dtype=np.int32)
        d0_c = d0[cand]
        d1 = np.zeros(len(cand), dtype=np.float64)
        if len(row_tok):
            d1[:len(row_tok)] = row_val

        step = 1024
        for s0 in range(0, len(pos), step):
            sub = pos[s0:s0 + step]
            n = len(sub)
            S = uniform[sub, None] + s[sub, 0:1] * d0_c[None, :]
            if K >= 1:
                S = S + s[sub, 1:2] * d1[None, :]
            for k in range(2, K + 1):
                if not np.any(s[sub, k] != 0.0):
                    continue
                expanded = _expand_rows(
                    raw[k].ctx[sub], matrices[k].indptr, matrices[k].indices, num_data[k],
                )
                if expanded is None:
                    continue
                local, toks, vals = expanded
                cols = col_of[toks]
                keep = cols >= 0
                if not np.any(keep):
                    continue
                local, cols, vals = local[keep], cols[keep], vals[keep]
                np.add.at(S, (local, cols), vals * s[sub[local], k])
            corr = (s[sub] * delta[sub]).sum(axis=1)
            lab_cols = col_of[tokens[sub]]
            has = lab_cols >= 0
            if np.any(has):
                ii = np.nonzero(has)[0]
                S[ii, lab_cols[ii]] += corr[ii]
            ids_sub, pr_sub = _topk_renorm(S, cand, r)
            out_ids[sub] = ids_sub
            out_p[sub] = pr_sub

    missing = out_p.sum(axis=1) <= 0
    if np.any(missing):
        out_ids[missing, 0] = tokens[missing]
        out_p[missing, 0] = 1.0

    k_star = np.full(T, -1, dtype=np.int16)
    for k in range(K, -1, -1):
        active = in_ctx[:, k] & (c_adj[:, k] > 0) & (k_star < 0)
        k_star[active] = np.int16(k)

    z = np.zeros(T, dtype=np.float32)
    return GBDTargets(
        ids=out_ids.astype(np.int32, copy=False),
        probs=out_p.astype(np.float64, copy=False),
        alpha=np.full(T, float(alpha_max), dtype=np.float32),
        R=z,
        w_exact=np.ones(T, dtype=np.float32),
        k_star=k_star,
        fitted={
            "discounts": [list(d) for d in discounts],
            "kn_loo": bool(kn_loo),
            "respect_eos": bool(respect_eos),
            "K": K,
            "r": r,
            "unigram_candidates": int(unigram_candidates),
            "discounts_version": "chen_goodman_1998_modified",
        },
    )
