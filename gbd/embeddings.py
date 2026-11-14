"""
Pluggable, teacher-free embedding & context-feature sources for GBD semantic retrieval.

Supported `embedding_source` modes:
  1. "w2v":             Static skip-gram Word2Vec trained on the training corpus (`w2v_embeddings.npy`).
  2. "ppmi_svd":        100% count-only Shifted Positive PMI co-occurrence matrix + truncated SVD
                        (Levy & Goldberg 2014 — zero neural network, zero SGD, pure count geometry).
  3. "random":          Random unit-norm Gaussian vectors (geometry control ablation).
  4. "student_tok_emb": Token embedding table `tok_emb.weight` extracted from the student itself
                        (either after a short CE warmup or from a saved student checkpoint).
  5. "student_hidden":  Contextual hidden states `h_{t-1}` from layer `hidden_layer` of the student
                        itself (defaults to mid-layer `n_layer // 2` to avoid pre-logit label memorisation),
                        pooled per semantic context key in `SemanticIndex` and projected via SVD.
  6. "file":            Arbitrary precomputed `(V, D)` `.npy` matrix via `embedding_path`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

from data import ASST_ID, EOS_ID, PAD_ID, USER_ID, train_same_corpus_word2vec


EMBEDDING_SOURCES = (
    "w2v",
    "ppmi_svd",
    "random",
    "student_tok_emb",
    "student_hidden",
    "file",
)


def build_ppmi_svd_embeddings(
    tokens: np.ndarray,
    vocab_size: int,
    dim: int = 128,
    window: int = 5,
    cds_alpha: float = 0.75,
    shift_k: float = 5.0,
    seed: int = 0,
) -> np.ndarray:
    """
    Pure count-based static token embeddings via Shifted PPMI + sparse truncated SVD.
    Zero neural training: builds directional skip-window co-occurrence counts strictly inside
    conversations (never crossing EOS_ID and ignoring structural role markers), applies
    context-distribution smoothing `cds_alpha=0.75` and log-shift `ln(k)` (matching SGNS k=5
    negative samples, Levy & Goldberg 2014), and takes top-`dim` singular vectors.
    """
    tokens = np.asarray(tokens, dtype=np.int64)
    role_arr = np.asarray([PAD_ID, EOS_ID, USER_ID, ASST_ID], dtype=np.int64)
    eos_locs = np.nonzero(tokens == EOS_ID)[0]

    rows_all = []
    cols_all = []
    vals_all = []

    start = 0
    boundaries = list(eos_locs) + ([len(tokens)] if (len(eos_locs) == 0 or eos_locs[-1] != len(tokens) - 1) else [])
    for end in boundaries:
        seg = tokens[start:end]
        start = int(end) + 1
        if seg.size < 2:
            continue
        keep = ~np.isin(seg, role_arr)
        seg = seg[keep]
        L = len(seg)
        if L < 2:
            continue
        for off in range(1, min(window + 1, L)):
            w_decay = 1.0 / off
            rows_all.append(seg[:-off])
            cols_all.append(seg[off:])
            vals_all.append(np.full(L - off, w_decay, dtype=np.float64))
            rows_all.append(seg[off:])
            cols_all.append(seg[:-off])
            vals_all.append(np.full(L - off, w_decay, dtype=np.float64))

    rng = np.random.default_rng(seed)
    if not rows_all:
        E = rng.normal(size=(vocab_size, dim)).astype(np.float32)
        return E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)

    r_cat = np.concatenate(rows_all)
    c_cat = np.concatenate(cols_all)
    v_cat = np.concatenate(vals_all)
    C = sp.csr_matrix((v_cat, (r_cat, c_cat)), shape=(vocab_size, vocab_size))
    C.sum_duplicates()

    row_sum = np.asarray(C.sum(axis=1)).ravel()
    col_pow = np.power(np.asarray(C.sum(axis=0)).ravel(), cds_alpha)
    Z = float(col_pow.sum())
    r_nz = np.repeat(np.arange(vocab_size, dtype=np.int64), np.diff(C.indptr))
    c_nz = C.indices
    p_wc = C.data / max(float(C.data.sum()), 1e-12)
    p_w = np.maximum(row_sum[r_nz] / max(float(row_sum.sum()), 1e-12), 1e-12)
    p_c = np.maximum(col_pow[c_nz] / max(Z, 1e-12), 1e-12)
    sppmi = np.maximum(np.log(np.maximum(p_wc / (p_w * p_c), 1e-30)) - np.log(max(shift_k, 1.0)), 0.0)

    M = sp.csr_matrix((sppmi, (r_nz, c_nz)), shape=(vocab_size, vocab_size))
    M.eliminate_zeros()
    k_svd = min(dim, vocab_size - 2)
    if M.nnz == 0 or k_svd < 1:
        E = rng.normal(size=(vocab_size, dim)).astype(np.float32)
        return E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)

    v0 = rng.normal(size=min(M.shape))
    U, S, _ = svds(M, k=k_svd, v0=v0)
    order = np.argsort(-S)
    E = (U[:, order] * np.sqrt(np.maximum(S[order], 0.0))).astype(np.float32)
    if k_svd < dim:
        pad = rng.normal(scale=0.01, size=(vocab_size, dim - k_svd)).astype(np.float32)
        E = np.concatenate([E, pad], axis=1)

    norms = np.linalg.norm(E, axis=1, keepdims=True)
    unseen = norms[:, 0] < 1e-10
    if unseen.any():
        r_vec = rng.normal(size=(int(unseen.sum()), dim)).astype(np.float32)
        E[unseen] = r_vec / np.maximum(np.linalg.norm(r_vec, axis=1, keepdims=True), 1e-12)
    return E


def extract_student_token_embeddings(model) -> np.ndarray:
    """Extract detached `(V, d_model)` token embedding table from a `CausalLM`."""
    import torch

    with torch.no_grad():
        return model.tok_emb.weight.detach().float().cpu().numpy().copy()


def resolve_token_embeddings(
    source: str,
    tokens: np.ndarray,
    vocab_size: int,
    corpus_dir: Path,
    embed_dim: int = 128,
    seed: int = 0,
    embedding_path: str | Path | None = None,
    student_model=None,
) -> np.ndarray | None:
    """
    Resolve static token-level embeddings `E: (V, D)` for token-table sources.
    Returns `None` when `source == "student_hidden"` (which uses `sem_feature_fn` instead).
    """
    src = source.lower().strip()
    if src == "student_hidden":
        return None
    if src == "w2v":
        default_w2v = corpus_dir / "w2v_embeddings.npy"
        if default_w2v.exists() and len(tokens) == len(np.load(corpus_dir / "train_tokens.npy", mmap_mode="r")):
            return np.load(default_w2v)
        cache_p = corpus_dir.parent / "embeddings" / f"w2v_T{len(tokens)}_D{embed_dim}_s{seed}.npy"
        if cache_p.exists():
            return np.load(cache_p)
        cache_p.parent.mkdir(parents=True, exist_ok=True)
        E = train_same_corpus_word2vec(tokens, vocab_size=vocab_size, dim=embed_dim, seed=seed)
        np.save(cache_p, E)
        return E
    if src == "ppmi_svd":
        cache_p = corpus_dir.parent / "embeddings" / f"ppmi_svd_T{len(tokens)}_D{embed_dim}_s{seed}.npy"
        if cache_p.exists():
            return np.load(cache_p)
        cache_p.parent.mkdir(parents=True, exist_ok=True)
        E = build_ppmi_svd_embeddings(tokens, vocab_size=vocab_size, dim=embed_dim, seed=seed)
        np.save(cache_p, E)
        return E
    if src == "random":
        rng = np.random.default_rng(seed + 999)
        E = rng.normal(size=(vocab_size, embed_dim)).astype(np.float32)
        return E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    if src == "student_tok_emb":
        if student_model is None:
            raise ValueError("embedding_source='student_tok_emb' requires a student model or --embedding_ckpt")
        return extract_student_token_embeddings(student_model)
    if src == "file":
        if not embedding_path:
            raise ValueError("embedding_source='file' requires --embedding_path")
        return np.load(Path(embedding_path)).astype(np.float32)
    raise ValueError(f"Unknown embedding_source: {source!r}; expected one of {EMBEDDING_SOURCES}")


def make_student_hidden_feature_fn(
    model,
    layer: int | None = None,
    proj_dim: int = 128,
    seq_len: int = 512,
    batch_size: int = 32,
    device=None,
) -> Callable:
    """
    Build a `sem_feature_fn(tokens, sidx) -> F (n_ctx, D)` that extracts contextual hidden
    states `h_{t-1}` from the student causal LM at `layer` (defaults to middle layer
    `n_layer // 2` to avoid pre-logit target memorisation), pools unit-normalised `h_{t-1}`
    across all corpus occurrences of each semantic context key in `sidx`, and projects to
    `proj_dim` via centered SVD.
    """
    import torch

    if device is None:
        device = next(model.parameters()).device
    use_layer = (model.cfg.n_layer // 2) if layer is None else int(layer)

    @torch.no_grad()
    def _feature_fn(tokens: np.ndarray, sidx) -> np.ndarray:
        n_ctx = len(sidx.rep_pos)
        D_m = model.cfg.d_model
        out_dim = min(proj_dim, D_m) if proj_dim > 0 else D_m
        if n_ctx == 0:
            return np.zeros((0, out_dim), dtype=np.float32)

        was_training = model.training
        model.eval()
        T = len(tokens)
        H_sum = np.zeros((n_ctx, D_m), dtype=np.float64)
        H_cnt = np.zeros(n_ctx, dtype=np.int64)

        # Evaluate causal windows of length `seq_len`: hidden state at offset `p` (token x[s+p])
        # is the causal representation h_{(s+p+1)-1} predicting position `t = s + p + 1`.
        starts = list(range(0, max(1, T - 1), seq_len))
        use_amp = device.type == "cuda"
        for b0 in range(0, len(starts), batch_size):
            bs = starts[b0 : b0 + batch_size]
            max_l = min(seq_len, max((T - 1) - s for s in bs))
            if max_l <= 0:
                continue
            x_np = np.full((len(bs), max_l), PAD_ID, dtype=np.int64)
            t_pos = np.full((len(bs), max_l), -1, dtype=np.int64)
            for r_i, s in enumerate(bs):
                ln = min(max_l, (T - 1) - s)
                x_np[r_i, :ln] = tokens[s : s + ln]
                t_pos[r_i, :ln] = np.arange(s + 1, s + 1 + ln, dtype=np.int64)

            x_t = torch.from_numpy(x_np).to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                h = model.forward_hidden(x_t, layer=use_layer)
            h_f = h.float()
            h_f = h_f / h_f.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            h_cpu = h_f.cpu().numpy()

            flat_t = t_pos.ravel()
            ok_t = flat_t >= 0
            valid_t = flat_t[ok_t]
            cids = sidx.ctx[valid_t]
            ok_c = cids >= 0
            if ok_c.any():
                np.add.at(H_sum, cids[ok_c], h_cpu.reshape(-1, D_m)[ok_t][ok_c])
                np.add.at(H_cnt, cids[ok_c], 1)

        if was_training:
            model.train()

        # Position t == 0 has no t-1 predecessor; fallback to token embedding if needed
        unseen = H_cnt == 0
        if unseen.any():
            H_cnt[unseen] = 1
        F_raw = H_sum / H_cnt[:, None]
        F_raw /= np.maximum(np.linalg.norm(F_raw, axis=1, keepdims=True), 1e-12)

        if out_dim < D_m and n_ctx > out_dim:
            F_centered = F_raw - F_raw.mean(axis=0, keepdims=True)
            _, _, Vt = np.linalg.svd(F_centered, full_matrices=False)
            F_proj = F_centered @ Vt[:out_dim].T
            F_proj /= np.maximum(np.linalg.norm(F_proj, axis=1, keepdims=True), 1e-12)
            return F_proj.astype(np.float32)
        return F_raw[:, :out_dim].astype(np.float32)

    return _feature_fn
