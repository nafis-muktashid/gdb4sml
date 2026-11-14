"""
Stage 1 Training Harness for GBD and Baselines (Config-Switchable Architecture).

Every GBD component is orthogonally switchable via CLI flags or `--config <path.json>`:
  - Exact-context branch:     `--use_exact` / `--no_exact`
  - Multi-order backoff:      `--use_order_backoff` / `--no_order_backoff`, `--exact_orders`
  - Leave-one-out:            `--leave_one_out` / `--no_loo`
  - Conversation boundary:    `--respect_eos` / `--no_respect_eos`
  - Semantic retrieval:       `--use_semantic` / `--no_semantic`
  - Dialogue role handling:   `--skip_role_markers` / `--no_skip_role_markers`,
                              `--partition_turn_state` / `--no_partition_turn_state`
  - Semantic variance pool:   `--pool_semantic` / `--no_pool_semantic`
  - Fixed vs adaptive lambda: `--lambda_mode {adaptive,fixed}`, `--fixed_lambda 0.5`
  - Fixed vs adaptive alpha:  `--alpha_mode {adaptive,r_ord,R,r_sem,fixed,const}`, `--alpha_max 0.5`
  - Curriculum annealing:     `--alpha_schedule {constant,linear,cosine}`
  - Embedding source:         `--embedding_source {w2v,ppmi_svd,random,student_tok_emb,student_hidden,file}`
  - Kneser–Ney baseline:      `--method kn` (`--kn_loo` / `--no-kn_loo`, `--alpha_max`)
  - D2GPo-NN64:               `--method d2gpo` is the neighbour-restricted variant of D2GPo
                              (`--d2gpo_sigma`, `--d2gpo_temp`, `--d2gpo_lambda`, `--d2gpo_r`)
  - Born-again / BabyLlama-2: `--method self_kd` (`--teacher_ckpt` comma-separated, `--kd_alpha`, `--kd_temp`)
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import ASST_ID, EOS_ID, PAD_ID, ROLE_IDS, USER_ID
from embeddings import (
    EMBEDDING_SOURCES,
    make_student_hidden_feature_fn,
    resolve_token_embeddings,
)
from eval import evaluate_blimp, evaluate_split, load_checkpoint_model
from gbd import (
    GBDChunks,
    GBDConfig,
    GBDTargets,
    OrderIndex,
    _topr,
    build_targets,
    conversation_starts,
    gbd_loss,
    load_targets,
    save_targets,
)
from d2gpo import build_d2gpo_targets
from kn import build_kn_targets
from model import CausalLM, ModelConfig


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_int_tuple(spec: str) -> tuple[int, ...] | None:
    s = spec.strip()
    if not s:
        return None
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def _parse_float_array(spec: str) -> np.ndarray | None:
    s = spec.strip()
    if not s:
        return None
    return np.asarray([float(x.strip()) for x in s.split(",") if x.strip()], dtype=np.float64)


def _hash_token_slice(tokens: np.ndarray) -> str:
    """Fast deterministic content digest of the exact training token slice."""
    arr = np.ascontiguousarray(tokens, dtype=np.int32)
    h = hashlib.blake2b(digest_size=8)
    h.update(str(arr.shape).encode("utf-8"))
    if arr.size <= 131_072:
        h.update(arr.tobytes())
    else:
        step = max(1, arr.size // 65_536)
        h.update(arr[:32_768].tobytes())
        h.update(arr[-32_768:].tobytes())
        h.update(arr[::step].tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------
# Target builders for CoCoNTs proper (7a) and Random entropy-matched (3)
# --------------------------------------------------------------------------
def build_coconts_targets(
    tokens: np.ndarray,
    V: int,
    L: int,
    k_prefix: int = 8,
    r: int = 8,
    gamma_cc: float = 1.5,
) -> GBDTargets:
    """
    Baseline 7a — CoCoNTs proper (Sathe & Sarawagi 2024, Sec. 3-4 Eqs. 3-4):
    Disjoint blocks of length L aligned with GBDChunks (`labels` are positions `s+1 .. s+L`).
    Counts corpus-wide exact n-grams over raw sequences (`eos_id=None`).
    At block offset `p in 0 .. k_prefix-1` (predicting `x_t = tokens[s+1+p]` from prefix
    `tokens[s : s+1+p]` of length `p+1`), let `y_topr` be the unnormalised MLE fractions of the
    top-r continuations and `p_mass = sum(y_topr)`. With `u = 1 / (gamma_cc - p_mass)` and
    `v = (1 - (1 - p_mass) * u) / p_mass`:
      - if `x_t in top-r`:   `y_CC = v * y_topr`             -> `nll_weight = 0`, `alpha = v * p_mass`
      - if `x_t not in top-r`: `y_CC = u * y_topr + 1_{x_t}` -> `nll_weight = 1`, `alpha = u * p_mass`
    where `probs = y_topr / p_mass` sums to 1. In expectation over occurrences of each prefix,
    `E[y_CC] = y_all` (exact statistical consistency with AllNTs). Positions `p >= k_prefix`
    use plain next-token CE (`nll_weight = 1`, `alpha = 0`).
    """
    assert gamma_cc > 1.0, f"CoCoNTs requires gamma_cc > 1.0, got {gamma_cc}"
    tokens = np.asarray(tokens, dtype=np.int64)
    T = len(tokens)
    orders, h = [], np.zeros(T, dtype=np.uint64)
    for ord_k in range(k_prefix + 1):
        idx = OrderIndex(tokens, ord_k, V, h, b_pos=None)
        h = idx.h
        idx.truncate(r)
        orders.append(idx)

    ids = np.zeros((T, r), dtype=np.int32)
    probs = np.zeros((T, r), dtype=np.float32)
    ids[:, 0] = tokens
    probs[:, 0] = 1.0
    alpha = np.zeros(T, dtype=np.float32)
    nll_weight = np.ones(T, dtype=np.float32)

    n_chunks = (T - 1) // L
    for p in range(min(k_prefix, L)):
        ord_k = p + 1
        oidx = orders[ord_k]
        pos = np.arange(n_chunks, dtype=np.int64) * L + 1 + p
        ctx = oidx.ctx[pos]
        valid = ctx >= 0
        if not valid.any():
            continue
        pos_v = pos[valid]
        ctx_v = ctx[valid]
        tot_n = oidx.n[ctx_v]
        top_i, top_cnt = _topr(oidx.Ctr[ctx_v], r)
        top_sum = top_cnt.sum(1).astype(np.float64)
        ok = (tot_n > 0) & (top_sum > 0)
        if not ok.any():
            continue

        pos_ok = pos_v[ok]
        ti_ok = top_i[ok]
        tc_ok = top_cnt[ok].astype(np.float64)
        n_ok = tot_n[ok]
        p_mass = np.clip(tc_ok.sum(1) / n_ok, 1e-12, 1.0)
        q_norm = (tc_ok / np.maximum(tc_ok.sum(1, keepdims=True), 1e-12)).astype(np.float32)

        x_true = tokens[pos_ok]
        in_topr = (ti_ok == x_true[:, None]).any(axis=1)
        u = 1.0 / (gamma_cc - p_mass)
        vp = 1.0 - (1.0 - p_mass) * u          # equals v * p_mass
        up = u * p_mass                        # equals u * p_mass

        ids[pos_ok] = ti_ok
        probs[pos_ok] = q_norm
        alpha[pos_ok] = np.where(in_topr, vp, up).astype(np.float32)
        nll_weight[pos_ok] = np.where(in_topr, 0.0, 1.0).astype(np.float32)

    z = np.zeros(T, dtype=np.float32)
    return GBDTargets(
        ids=ids,
        probs=probs,
        alpha=alpha,
        R=z,
        w_exact=np.ones(T, dtype=np.float32),
        k_star=np.zeros(T, dtype=np.int16),
        fitted={"gamma_cc": gamma_cc, "k_prefix": k_prefix, "r": r},
        nll_weight=nll_weight,
    )


def build_random_entropy_matched_targets(tokens: np.ndarray, V: int, ref_tg: GBDTargets, seed: int = 0) -> GBDTargets:
    """
    Baseline 3: random distinct support IDs (sampled without replacement per position) paired
    with the exact probability vector & alpha_t of GBD so per-token Shannon entropy matches 1:1.
    """
    rng = np.random.default_rng(seed)
    T, r = ref_tg.ids.shape
    assert r <= V, f"r={r} cannot exceed vocab size V={V}"
    rand_ids = np.empty((T, r), dtype=np.int32)
    chunk_sz = 25_000
    for s in range(0, T, chunk_sz):
        e = min(s + chunk_sz, T)
        B = e - s
        # Draw r distinct IDs per row: generate initial random IDs, then resolve rare collisions row-wise
        block = rng.integers(0, V, size=(B, r), dtype=np.int32)
        s_block = np.sort(block, axis=1)
        has_dup = (s_block[:, 1:] == s_block[:, :-1]).any(axis=1)
        for row_i in np.nonzero(has_dup)[0]:
            block[row_i] = rng.choice(V, size=r, replace=False)
        rand_ids[s:e] = block
    return GBDTargets(
        ids=rand_ids,
        probs=np.asarray(ref_tg.probs, dtype=np.float32).copy(),
        alpha=np.asarray(ref_tg.alpha, dtype=np.float32).copy(),
        R=np.asarray(ref_tg.R, dtype=np.float32).copy(),
        w_exact=np.zeros(T, dtype=np.float32),
        k_star=np.zeros(T, dtype=np.int16),
    )


def make_gbd_config_from_args(args, method_override: str | None = None) -> GBDConfig:
    m = method_override or args.method
    exact_ord_override = _parse_int_tuple(args.exact_orders)
    delta_override = _parse_float_array(args.fixed_delta)

    # Preset method conveniences map directly onto orthogonal GBDConfig switches
    if m == "exact_ngram":
        return GBDConfig(
            K=args.K,
            use_exact=True,
            use_order_backoff=False,
            exact_orders=(args.K,),
            leave_one_out=False,
            respect_eos=args.respect_eos,
            use_semantic=False,
            R_pre=args.R_pre,
            r=args.r,
            alpha_max=args.alpha_max,
            alpha_mode="fixed",
            eos_id=EOS_ID,
            role_ids=ROLE_IDS,
            seed=0,
        )
    if m == "unigram":
        return GBDConfig(
            K=0,
            use_exact=True,
            use_order_backoff=False,
            exact_orders=(0,),
            leave_one_out=False,
            respect_eos=args.respect_eos,
            use_semantic=False,
            R_pre=args.R_pre,
            r=args.r,
            alpha_max=args.alpha_max,
            alpha_mode="fixed",
            eos_id=EOS_ID,
            role_ids=ROLE_IDS,
            seed=0,
        )

    use_sem = False if m == "gbd_order" else bool(args.use_semantic)
    return GBDConfig(
        K=args.K,
        Ks=args.Ks,
        M=args.M,
        M_search=args.M_search,
        R_pre=args.R_pre,
        r=args.r,
        use_exact=bool(args.use_exact),
        use_order_backoff=bool(args.use_order_backoff),
        exact_orders=exact_ord_override,
        leave_one_out=bool(args.leave_one_out),
        respect_eos=bool(args.respect_eos),
        use_semantic=use_sem,
        skip_role_markers=bool(args.skip_role_markers),
        partition_turn_state=bool(args.partition_turn_state),
        pool_semantic=bool(args.pool_semantic),
        phi=args.phi,
        tau=args.tau,
        gamma=args.gamma,
        lambda_mode=args.lambda_mode,
        fixed_lambda=args.fixed_lambda,
        alpha_max=args.alpha_max,
        gamma_alpha=args.gamma_alpha,
        alpha_mode=args.alpha_mode,
        eos_id=EOS_ID,
        role_ids=ROLE_IDS,
        delta=delta_override,
        a=args.fixed_a,
        b=args.fixed_b,
        seed=0,
    )


def _canonical_gbd_spec(cfg: GBDConfig, tokens_hash: str, V: int, args) -> dict:
    emb_src = "random" if args.random_vectors else args.embedding_source
    norm_amode = "adaptive" if cfg.alpha_mode in ("adaptive", "r_ord") else ("fixed" if cfg.alpha_mode in ("fixed", "const") else cfg.alpha_mode)
    return {
        "schema_v": 2,
        "tokens_hash": tokens_hash,
        "T": int(args._resolved_T),
        "V": int(V),
        "risk_version": int(cfg.risk_version),
        "K": int(cfg.K),
        "Ks": int(cfg.Ks),
        "M": int(cfg.M),
        "M_search": int(cfg.M_search),
        "R_pre": int(cfg.R_pre),
        "r": int(cfg.r),
        "r_min": float(cfg.r_min),
        "exact_orders": cfg.resolve_exact_orders(),
        "leave_one_out": bool(cfg.leave_one_out),
        "respect_eos": bool(cfg.respect_eos),
        "eos_id": int(cfg.eos_id) if (cfg.respect_eos and cfg.eos_id is not None) else None,
        "role_ids": list(cfg.role_ids) if cfg.use_semantic else [],
        "use_semantic": bool(cfg.use_semantic),
        "skip_role_markers": bool(cfg.skip_role_markers) if cfg.use_semantic else False,
        "partition_turn_state": bool(cfg.partition_turn_state) if cfg.use_semantic else False,
        "pool_semantic": bool(cfg.pool_semantic) if cfg.use_semantic else False,
        "phi": str(cfg.phi),
        "tau": float(cfg.tau) if cfg.phi == "exp" else None,
        "gamma": float(cfg.gamma) if cfg.use_semantic else None,
        "lambda_mode": str(cfg.lambda_mode),
        "fixed_lambda": float(cfg.fixed_lambda) if cfg.lambda_mode == "fixed" else None,
        "alpha_mode": norm_amode,
        "alpha_max": float(cfg.alpha_max),
        "gamma_alpha": float(cfg.gamma_alpha),
        "n_fit": int(cfg.n_fit),
        "n_fit_pairs": int(cfg.n_fit_pairs),
        "n0": float(cfg.n0),
        "beta0": float(cfg.beta0),
        "hnsw_threshold": int(cfg.hnsw_threshold),
        "cfg_seed": int(cfg.seed),
        "delta": cfg.delta.tolist() if cfg.delta is not None else None,
        "a": float(cfg.a) if cfg.a is not None else None,
        "b": float(cfg.b) if cfg.b is not None else None,
        "d0": cfg.d0.tolist() if cfg.d0 is not None else None,
        "d0_sem": float(cfg.d0_sem) if cfg.d0_sem is not None else None,
        "emb_src": emb_src if cfg.use_semantic else "none",
        "embed_dim": int(args.embed_dim) if cfg.use_semantic else None,
        "embedding_path": str(Path(args.embedding_path).resolve()) if (cfg.use_semantic and emb_src == "file" and args.embedding_path) else "",
        "embedding_ckpt": str(Path(args.embedding_ckpt).resolve()) if (cfg.use_semantic and emb_src in ("student_tok_emb", "student_hidden") and args.embedding_ckpt) else "",
        "warmup_ce_epochs": int(args.warmup_ce_epochs) if (cfg.use_semantic and emb_src in ("student_tok_emb", "student_hidden") and not args.embedding_ckpt) else 0,
        "warmup_model": str(args.model) if (cfg.use_semantic and emb_src in ("student_tok_emb", "student_hidden") and not args.embedding_ckpt) else "",
        "hidden_layer": args.hidden_layer if (cfg.use_semantic and emb_src == "student_hidden") else None,
        "emb_seed": int(args.seed) if (cfg.use_semantic and emb_src in ("random", "student_tok_emb", "student_hidden")) else 0,
    }


def _target_cache_dir_and_spec(
    method: str,
    tokens: np.ndarray,
    V: int,
    cfg: GBDConfig | None,
    args,
    cache_root: Path,
) -> tuple[Path, dict]:
    """
    Build a 100% collision-free target cache path + full JSON spec dictionary.
    Every parameter that can change `ids`, `probs`, `alpha`, or `nll_weight` (including a
    content digest of `tokens` itself) enters the `blake2b` hash and is verified on load.
    """
    args._resolved_T = len(tokens)
    tok_hash = _hash_token_slice(tokens)
    if method == "coconts":
        spec = {
            "schema_v": 2,
            "method": "coconts",
            "tokens_hash": tok_hash,
            "T": int(len(tokens)),
            "V": int(V),
            "seq_len": int(args.seq_len),
            "coconts_k": int(args.coconts_k),
            "coconts_r": int(args.coconts_r),
            "coconts_gamma": float(args.coconts_gamma),
        }
        tag = f"coconts__T{len(tokens)}_L{args.seq_len}_k{args.coconts_k}_r{args.coconts_r}_g{args.coconts_gamma}"
    elif method == "random_target":
        ref_cfg = make_gbd_config_from_args(args, method_override="gbd_full")
        spec = {
            "schema_v": 2,
            "method": "random_target",
            "random_seed": int(args.seed),
            "ref_spec": _canonical_gbd_spec(ref_cfg, tok_hash, V, args),
        }
        tag = f"random_target__T{len(tokens)}_a{args.alpha_max}_s{args.seed}"
    elif method == "kn":
        spec = {
            "schema_v": 2,
            "method": "kn",
            "tokens_hash": tok_hash,
            "T": int(len(tokens)),
            "V": int(V),
            "K": int(args.K),
            "r": int(args.r),
            "kn_loo": bool(args.kn_loo),
            "respect_eos": bool(args.respect_eos),
            "eos_id": int(EOS_ID) if args.respect_eos else None,
            "alpha_max": float(args.alpha_max),
            "unigram_candidates": 64,
            "discounts": "chen_goodman_1998_modified",
        }
        tag = f"kn__T{len(tokens)}__K{args.K}"
    elif method == "d2gpo":
        emb_src = "random" if args.random_vectors else args.embedding_source
        spec = {
            "schema_v": 2,
            "method": "d2gpo",
            "tokens_hash": tok_hash,
            "T": int(len(tokens)),
            "V": int(V),
            "sigma": float(args.d2gpo_sigma),
            "temp": float(args.d2gpo_temp),
            "lambda": float(args.d2gpo_lambda),
            "r_d": int(args.d2gpo_r),
            "emb_src": emb_src,
            "embed_dim": int(args.embed_dim),
            "emb_seed": int(args.seed),
            "embedding_path": str(Path(args.embedding_path).resolve()) if (emb_src == "file" and args.embedding_path) else "",
        }
        tag = f"d2gpo__T{len(tokens)}__{emb_src}"
    else:
        assert cfg is not None
        spec = _canonical_gbd_spec(cfg, tok_hash, V, args)
        spec["method_alias"] = method
        emb_lbl = spec["emb_src"] if cfg.use_semantic else "no_sem"
        tag = f"{method}__T{len(tokens)}__{emb_lbl}"

    digest = hashlib.blake2b(json.dumps(spec, sort_keys=True).encode("utf-8"), digest_size=10).hexdigest()
    return cache_root / f"{tag}__{digest}", spec


def _train_warmup_student_for_geometry(
    train_tokens: np.ndarray,
    V: int,
    args,
    device: torch.device,
) -> CausalLM:
    """Train a short teacher-free CE warmup student on the same training corpus to bootstrap student geometry."""
    print(
        f"bootstrapping student geometry via {args.warmup_ce_epochs}-epoch same-corpus CE warmup "
        f"(model={args.model}, seed={args.seed})...",
        flush=True,
    )
    set_seed(args.seed)
    mcfg = ModelConfig.from_preset(args.model, vocab_size=V, max_seq_len=args.seq_len, dropout=args.dropout)
    w_model = CausalLM(mcfg).to(device)
    ds = GBDChunks(train_tokens, tg=None, L=args.seq_len, train_mask=None)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
    decay_p = [p for p in w_model.parameters() if p.requires_grad and p.ndim >= 2]
    nodecay_p = [p for p in w_model.parameters() if p.requires_grad and p.ndim < 2]
    opt = torch.optim.AdamW(
        [{"params": decay_p, "weight_decay": args.weight_decay}, {"params": nodecay_p, "weight_decay": 0.0}],
        lr=args.lr,
        betas=(0.9, 0.95),
        fused=(device.type == "cuda"),
    )
    use_amp = device.type == "cuda"
    w_model.train()
    for ep in range(1, args.warmup_ce_epochs + 1):
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for b_i, (x, y) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                logits = w_model(x)
                loss = F.cross_entropy(logits.float().view(-1, V), y.view(-1), ignore_index=-100)
            (loss / args.grad_accum).backward()
            loss_sum += float(loss.item())
            if ((b_i + 1) % args.grad_accum == 0) or ((b_i + 1) == len(loader)):
                torch.nn.utils.clip_grad_norm_(w_model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
        print(f"  warmup epoch {ep}/{args.warmup_ce_epochs} | ce_loss={loss_sum/max(len(loader),1):.4f} ({time.time()-t0:.1f}s)", flush=True)
    w_model.eval()
    return w_model


def get_or_build_targets(
    method: str,
    tokens: np.ndarray,
    V: int,
    corpus_dir: Path,
    args,
    cache_root: Path,
    device: torch.device,
) -> GBDTargets | None:
    if method in ("ce", "label_smoothing", "static_init", "mir", "self_kd"):
        return None

    cfg = None if method in ("coconts", "random_target", "kn", "d2gpo") else make_gbd_config_from_args(args, method_override=method)
    cache_dir, cache_spec = _target_cache_dir_and_spec(method, tokens, V, cfg, args, cache_root)

    if (cache_dir / "ids.npy").exists() and not args.rebuild_targets:
        t0 = time.time()
        tg = load_targets(cache_dir, mmap_mode="r", expected_spec=cache_spec)
        print(f"loaded verified cached targets from {cache_dir} ({time.time() - t0:.2f}s)", flush=True)
        return tg

    t0 = time.time()
    print(f"building offline targets for method={method!r} on {len(tokens):,} tokens...", flush=True)

    if method == "coconts":
        tg = build_coconts_targets(
            tokens,
            V,
            L=args.seq_len,
            k_prefix=args.coconts_k,
            r=args.coconts_r,
            gamma_cc=args.coconts_gamma,
        )
    elif method == "random_target":
        ref = get_or_build_targets("gbd_full", tokens, V, corpus_dir, args, cache_root, device=device)
        tg = build_random_entropy_matched_targets(tokens, V, ref, seed=args.seed)
    elif method == "kn":
        tg = build_kn_targets(
            tokens,
            V,
            K=args.K,
            r=args.r,
            alpha_max=args.alpha_max,
            kn_loo=bool(args.kn_loo),
            respect_eos=bool(args.respect_eos),
            eos_id=EOS_ID,
        )
    elif method == "d2gpo":
        emb_src = "random" if args.random_vectors else args.embedding_source
        if emb_src == "student_hidden":
            raise ValueError("method 'd2gpo' needs token embeddings; student_hidden has no token table")
        E_d = resolve_token_embeddings(
            source=emb_src,
            tokens=tokens,
            vocab_size=V,
            corpus_dir=corpus_dir,
            embed_dim=args.embed_dim,
            seed=args.seed,
            embedding_path=args.embedding_path or None,
        )
        if E_d is None:
            raise ValueError(f"method 'd2gpo' could not resolve embeddings for source={emb_src!r}")
        tg = build_d2gpo_targets(
            tokens,
            E_d,
            sigma=args.d2gpo_sigma,
            temp=args.d2gpo_temp,
            lam=args.d2gpo_lambda,
            r_d=args.d2gpo_r,
        )
    elif method in ("gbd", "gbd_full", "gbd_order", "exact_ngram", "unigram"):
        E_use = None
        sem_fn = None
        geom_model = None
        emb_src = "random" if args.random_vectors else args.embedding_source
        if cfg.use_semantic:
            if emb_src in ("student_tok_emb", "student_hidden"):
                if args.embedding_ckpt:
                    geom_model, _ = load_checkpoint_model(args.embedding_ckpt, device=device)
                elif args.warmup_ce_epochs > 0:
                    geom_model = _train_warmup_student_for_geometry(tokens, V, args, device=device)
                else:
                    raise ValueError(
                        f"embedding_source={emb_src!r} requires either --embedding_ckpt <path.pt> "
                        "or --warmup_ce_epochs >= 1"
                    )
            if emb_src == "student_hidden":
                sem_fn = make_student_hidden_feature_fn(
                    geom_model,
                    layer=args.hidden_layer,
                    proj_dim=args.embed_dim,
                    seq_len=args.seq_len,
                    batch_size=args.batch_size,
                    device=device,
                )
            else:
                E_use = resolve_token_embeddings(
                    source=emb_src,
                    tokens=tokens,
                    vocab_size=V,
                    corpus_dir=corpus_dir,
                    embed_dim=args.embed_dim,
                    seed=args.seed,
                    embedding_path=args.embedding_path or None,
                    student_model=geom_model,
                )
        tg = build_targets(tokens, V, E_use, cfg, sem_feature_fn=sem_fn)
        if geom_model is not None:
            del geom_model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    else:
        raise ValueError(f"Unknown method: {method!r}")

    build_sec = time.time() - t0
    save_targets(cache_dir, tg, fp16_probs=True, max_vocab=V, cache_spec=cache_spec)
    print(f"built & saved targets to {cache_dir} in {build_sec:.1f}s (mean alpha={float(tg.alpha.mean()):.4f})", flush=True)
    del tg
    return load_targets(cache_dir, mmap_mode="r", expected_spec=cache_spec)


def mean_teacher_logits(teacher_logits: list) -> torch.Tensor:
    """BabyLlama-2 ensemble (Tastet & Timiryasov 2024): mean of logits, not of probabilities."""
    if len(teacher_logits) == 1:
        return teacher_logits[0]
    return torch.stack([z.float() for z in teacher_logits], dim=0).mean(dim=0)


def _parse_teacher_ckpts(raw: str) -> list[str]:
    paths = [part.strip() for part in str(raw).split(",") if part.strip()]
    if not paths:
        raise ValueError("method 'self_kd' requires --teacher_ckpt")
    return paths


def self_kd_loss(student_logits, teacher_logits, labels, alpha: float, temperature: float, ignore_index: int = -100):
    """Born-Again loss (Furlanello et al. 2018), Hinton T^2 on the KL (also BabyLlama-2).

    (1 - a) * NLL(x_t) + a * T^2 * KL(softmax(z_teacher / T) || softmax(z_student / T)).
    Softmaxes are float32. Ignored labels (ignore_index) are dropped, matching CE.
    `z_teacher` may already be the mean of several teachers' logits.
    """
    if temperature <= 0.0:
        raise ValueError(f"kd_temp must be positive, got {temperature}")
    T = float(temperature)
    z_s = student_logits.float()
    z_t = teacher_logits.float()
    log_p_s = F.log_softmax(z_s / T, dim=-1)
    log_p_t = F.log_softmax(z_t / T, dim=-1)
    p_t = log_p_t.exp()
    kl = (p_t * (log_p_t - log_p_s)).sum(dim=-1)
    nll = F.cross_entropy(
        z_s.reshape(-1, z_s.size(-1)),
        labels.reshape(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).reshape(labels.shape)
    mask = labels != ignore_index
    per = (1.0 - float(alpha)) * nll + float(alpha) * (T * T) * kl
    return (per * mask).sum() / mask.sum().clamp_min(1)


def _alpha_curriculum_scale(schedule: str, step: int, total_steps: int) -> float:
    if schedule == "constant":
        return 1.0
    prog = min(max(step / max(total_steps, 1), 0.0), 1.0)
    if schedule == "linear":
        return 1.0 - prog
    if schedule == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * prog))
    raise ValueError(f"Unknown alpha_schedule: {schedule!r}")


def _default_run_name(args, n_tokens: int) -> str:
    """Collision-free default run filename encoding full config digest when `--run_name` is omitted."""
    cfg_digest = hashlib.blake2b(
        json.dumps(vars(args), sort_keys=True, default=str).encode("utf-8"),
        digest_size=4,
    ).hexdigest()
    return f"{args.model}_{args.method}_T{n_tokens}_a{args.alpha_max}_s{args.seed}_{cfg_digest}"
def train_run(args) -> dict:
    if args.anneal_alpha and args.alpha_schedule == "constant":
        args.alpha_schedule = "linear"

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    corpus_dir = Path(args.corpus_dir)
    meta = json.loads((corpus_dir / "corpus_meta.json").read_text(encoding="utf-8"))
    V = int(meta["vocab_size"])

    full_train_tokens = np.load(corpus_dir / "train_tokens.npy", mmap_mode="r")
    full_train_reply = np.load(corpus_dir / "train_reply_mask.npy", mmap_mode="r")
    T_use = min(len(full_train_tokens), args.train_tokens) if args.train_tokens > 0 else len(full_train_tokens)
    train_tokens = np.asarray(full_train_tokens[:T_use], dtype=np.int32)
    train_reply = np.asarray(full_train_reply[:T_use], dtype=bool)
    train_unigram_counts = np.bincount(train_tokens, minlength=V)

    val_tokens = np.load(corpus_dir / "val_tokens.npy")
    val_reply = np.load(corpus_dir / "val_reply_mask.npy")
    test_tokens = np.load(corpus_dir / "test_tokens.npy")
    test_reply = np.load(corpus_dir / "test_reply_mask.npy")

    cache_root = corpus_dir.parent / "targets"
    cache_root.mkdir(parents=True, exist_ok=True)
    tg = get_or_build_targets(args.method, train_tokens, V, corpus_dir, args, cache_root=cache_root, device=device)

    # Reset RNG after target building so student initialisation & batch order match across methods at fixed `seed`
    set_seed(args.seed)

    train_mask = train_reply if args.reply_loss_only else None
    dataset = GBDChunks(
        train_tokens,
        tg=tg,
        L=args.seq_len,
        train_mask=train_mask,
        prefix_only_k=args.prefix_only_k if args.prefix_only_k > 0 else None,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    mcfg = ModelConfig.from_preset(args.model, vocab_size=V, max_seq_len=args.seq_len, dropout=args.dropout)
    model = CausalLM(mcfg).to(device)
    if args.method == "static_init":
        E_init = resolve_token_embeddings(
            source="random" if args.random_vectors else args.embedding_source,
            tokens=train_tokens,
            vocab_size=V,
            corpus_dir=corpus_dir,
            embed_dim=args.embed_dim,
            seed=args.seed,
            embedding_path=args.embedding_path or None,
        )
        model.init_from_static_embeddings(E_init, seed=args.seed)

    teachers: list[CausalLM] = []
    if args.method == "self_kd":
        teacher_paths = _parse_teacher_ckpts(args.teacher_ckpt)
        args.teacher_ckpts = teacher_paths
        for path in teacher_paths:
            teacher, _ = load_checkpoint_model(path, device=device)
            teacher.eval()
            teacher.requires_grad_(False)
            if teacher.cfg.vocab_size != V or teacher.cfg.max_seq_len != args.seq_len:
                raise ValueError(
                    f"teacher checkpoint {path} has vocab_size={teacher.cfg.vocab_size}, "
                    f"max_seq_len={teacher.cfg.max_seq_len}; student has vocab_size={V}, max_seq_len={args.seq_len}"
                )
            teachers.append(teacher)

    decay_params = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    nodecay_params = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
        fused=(device.type == "cuda"),
    )

    steps_per_epoch = max(1, math.ceil(len(loader) / args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_frac))
    min_lr = args.lr * 0.1

    def get_lr(step: int) -> float:
        if step < warmup_steps:
            return args.lr * (step + 1) / warmup_steps
        if total_steps <= warmup_steps:
            return args.lr
        prog = (step - warmup_steps) / (total_steps - warmup_steps)
        return min_lr + 0.5 * (args.lr - min_lr) * (1.0 + math.cos(math.pi * min(prog, 1.0)))

    print(
        f"\n=== Run: method={args.method} | model={args.model} ({model.num_parameters()/1e6:.2f}M params) "
        f"| train_tokens={len(train_tokens):,} | epochs={args.epochs} ({total_steps} steps) "
        f"| schedule={args.alpha_schedule} | device={device} ===",
        flush=True,
    )

    use_amp = device.type == "cuda"
    global_step = 0
    best_val_bpb = float("inf")
    best_state = None
    history = []
    t_train_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_t0 = time.time()
        ep_loss_sum = 0.0
        ep_batches = 0
        optimizer.zero_grad(set_to_none=True)

        for b_idx, batch in enumerate(loader):
            t_nll_w = None
            if tg is None:
                x, y = batch
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
            elif len(batch) == 6:
                x, y, t_ids, t_probs, t_alpha, t_nll_w = batch
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                t_ids = t_ids.to(device, non_blocking=True)
                t_probs = t_probs.to(device, non_blocking=True)
                t_alpha = t_alpha.to(device, non_blocking=True)
                t_nll_w = t_nll_w.to(device, non_blocking=True)
            else:
                x, y, t_ids, t_probs, t_alpha = batch
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                t_ids = t_ids.to(device, non_blocking=True)
                t_probs = t_probs.to(device, non_blocking=True)
                t_alpha = t_alpha.to(device, non_blocking=True)

            if args.method == "mir" and args.mir_prob > 0.0:
                m_in = (torch.rand_like(x, dtype=torch.float32) < args.mir_prob) & (x >= 6)
                x = torch.where(m_in, torch.full_like(x, PAD_ID), x)

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                logits = model(x)
                if tg is not None:
                    scale_a = _alpha_curriculum_scale(args.alpha_schedule, global_step, total_steps)
                    if scale_a != 1.0:
                        if t_nll_w is not None:
                            t_nll_w = (1.0 - scale_a) + scale_a * t_nll_w
                        t_alpha = t_alpha * scale_a
                    loss, _ = gbd_loss(logits, y, t_ids, t_probs, t_alpha, ignore_index=-100, nll_weight=t_nll_w)
                elif args.method == "label_smoothing":
                    scale_a = _alpha_curriculum_scale(args.alpha_schedule, global_step, total_steps)
                    loss = F.cross_entropy(
                        logits.float().view(-1, V),
                        y.view(-1),
                        ignore_index=-100,
                        label_smoothing=args.alpha_max * scale_a,
                    )
                elif args.method == "self_kd":
                    scale_a = _alpha_curriculum_scale(args.alpha_schedule, global_step, total_steps)
                    a_kd = float(args.kd_alpha) * scale_a
                    with torch.no_grad():
                        teacher_logits = mean_teacher_logits([teacher(x) for teacher in teachers])
                    loss = self_kd_loss(logits, teacher_logits, y, a_kd, args.kd_temp)
                else:
                    loss = F.cross_entropy(logits.float().view(-1, V), y.view(-1), ignore_index=-100)

            (loss / args.grad_accum).backward()
            ep_loss_sum += float(loss.item())
            ep_batches += 1

            is_step = ((b_idx + 1) % args.grad_accum == 0) or ((b_idx + 1) == len(loader))
            if is_step:
                lr_now = get_lr(global_step)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_now
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

        val_metrics = evaluate_split(
            model,
            val_tokens,
            val_reply,
            reply_bytes=meta["splits"]["val"]["reply_bytes"],
            total_bytes=meta["splits"]["val"]["total_bytes"],
            train_unigram_counts=train_unigram_counts,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            device=device,
        )
        ep_sec = time.time() - ep_t0
        improved = val_metrics["reply_bpb"] < best_val_bpb
        if improved:
            best_val_bpb = val_metrics["reply_bpb"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        rec = {
            "epoch": epoch,
            "train_loss": round(ep_loss_sum / max(ep_batches, 1), 4),
            "val_reply_bpb": val_metrics["reply_bpb"],
            "val_all_bpb": val_metrics["all_bpb"],
            "val_reply_ppl": val_metrics["reply_ppl"],
            "epoch_sec": round(ep_sec, 1),
        }
        history.append(rec)
        print(
            f"  epoch {epoch:2d}/{args.epochs} | train_loss={rec['train_loss']:.4f} "
            f"| val_reply_bpb={rec['val_reply_bpb']:.4f} (all_bpb={rec['val_all_bpb']:.4f}, ppl={rec['val_reply_ppl']:.2f}) "
            f"| {ep_sec:.1f}s {'*' if improved else ''}",
            flush=True,
        )

    train_wall_sec = time.time() - t_train_start
    if best_state is not None:
        model.load_state_dict(best_state)

    run_name = args.run_name or _default_run_name(args, len(train_tokens))
    ckpt_path = None
    if args.save_ckpt:
        ckpt_dir = Path(args.ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"{run_name}.pt"
        bf16_state = {
            k: (v.detach().to(torch.bfloat16).cpu() if v.is_floating_point() else v.detach().cpu())
            for k, v in model.state_dict().items()
        }
        torch.save(
            {
                "model_config": asdict(mcfg),
                "train_config": vars(args),
                "state_dict": bf16_state,
                "best_val_reply_bpb": best_val_bpb,
            },
            ckpt_path,
        )

    val_best = evaluate_split(
        model, val_tokens, val_reply,
        reply_bytes=meta["splits"]["val"]["reply_bytes"],
        total_bytes=meta["splits"]["val"]["total_bytes"],
        train_unigram_counts=train_unigram_counts,
        seq_len=args.seq_len, batch_size=args.batch_size, device=device,
    )
    test_best = evaluate_split(
        model, test_tokens, test_reply,
        reply_bytes=meta["splits"]["test"]["reply_bytes"],
        total_bytes=meta["splits"]["test"]["total_bytes"],
        train_unigram_counts=train_unigram_counts,
        seq_len=args.seq_len, batch_size=args.batch_size, device=device,
    )

    blimp_res = None
    if args.eval_blimp:
        import sentencepiece as spm
        sp_proc = spm.SentencePieceProcessor(model_file=str(corpus_dir / "tokenizer.model"))
        blimp_res = evaluate_blimp(model, sp_proc, corpus_dir.parent / ".hf_cache", device=device)

    out_summary = {
        "config": vars(args),
        "target_fitted_switches": tg.fitted.get("switches") if (tg is not None and tg.fitted) else None,
        "checkpoint_path": str(ckpt_path) if ckpt_path is not None else None,
        "model_params": model.num_parameters(),
        "train_tokens_used": int(len(train_tokens)),
        "train_wall_seconds": round(train_wall_sec, 2),
        "best_val": val_best,
        "test": test_best,
        "blimp": blimp_res,
        "history": history,
    }
    print(
        f"DONE {args.method} (seed={args.seed}) in {train_wall_sec:.1f}s -> "
        f"TEST reply_bpb={test_best['reply_bpb']:.4f} | all_bpb={test_best['all_bpb']:.4f} "
        f"| ECE={test_best['ece']:.4f} | rare_<10 NLL={test_best['rare_token_nll']['rare_<10']}",
        flush=True,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{run_name}.json").write_text(json.dumps(out_summary, indent=2), encoding="utf-8")
    del tg
    return out_summary


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Config-Switchable GBD & Baseline Trainer")
    ap.add_argument("--config", type=str, default="", help="Optional JSON config file; CLI flags override JSON values")
    ap.add_argument("--corpus_dir", type=str, default="artifacts/corpus")
    ap.add_argument("--out_dir", type=str, default="artifacts/runs")
    ap.add_argument("--ckpt_dir", type=str, default="artifacts/checkpoints")
    ap.add_argument("--run_name", type=str, default="")
    ap.add_argument(
        "--method",
        type=str,
        default="ce",
        choices=[
            "ce", "label_smoothing", "unigram", "random_target",
            "exact_ngram", "coconts", "static_init", "mir",
            "gbd", "gbd_order", "gbd_full",
            "kn", "d2gpo", "self_kd",
        ],
        help=(
            "Training objective. d2gpo is D2GPo-NN64: neighbour-restricted variant of D2GPo, "
            "not the full-vocabulary Eq. 7 prior. self_kd is Born-Again self-distillation "
            "(Furlanello et al. 2018) and requires --teacher_ckpt."
        ),
    )
    ap.add_argument("--model", type=str, default="30m", choices=["tiny", "30m", "39m", "gpt2-small", "124m"])
    ap.add_argument("--train_tokens", type=int, default=1_000_000)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--warmup_frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)

    # --- Orthogonal GBD Component Switches ---
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--Ks", type=int, default=2)
    ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--M_search", type=int, default=32)
    ap.add_argument("--R_pre", type=int, default=64)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--gamma", type=float, default=1.0)

    ap.add_argument("--use_exact", action=argparse.BooleanOptionalAction, default=True, help="Enable/disable exact n-gram branch (--no-use_exact -> semantic-only)")
    ap.add_argument("--use_order_backoff", action=argparse.BooleanOptionalAction, default=True, help="Enable/disable multi-order 0..K backoff (--no-use_order_backoff -> single order K)")
    ap.add_argument("--exact_orders", type=str, default="", help="Comma-separated exact orders override, e.g. '0,2,4,6'")
    ap.add_argument("--leave_one_out", action=argparse.BooleanOptionalAction, default=True, help="Enable/disable leave-one-out counts")
    ap.add_argument("--no_loo", action="store_true", help="Legacy alias for --no-leave_one_out")
    ap.add_argument("--respect_eos", action=argparse.BooleanOptionalAction, default=True, help="Enable/disable conversation boundary reset")

    ap.add_argument("--use_semantic", action=argparse.BooleanOptionalAction, default=True, help="Enable/disable semantic retrieval branch")
    ap.add_argument("--skip_role_markers", action=argparse.BooleanOptionalAction, default=True, help="Enable/disable skipping role markers in semantic context")
    ap.add_argument("--partition_turn_state", action=argparse.BooleanOptionalAction, default=True, help="Enable/disable turn-state bucket restriction in semantic kNN")
    ap.add_argument("--pool_semantic", action=argparse.BooleanOptionalAction, default=True, help="Enable/disable N_sem pooled variance in semantic branch")

    # --- Fixed vs Adaptive Lambda / Alpha Switches ---
    ap.add_argument("--lambda_mode", type=str, default="adaptive", choices=["adaptive", "fixed"])
    ap.add_argument("--fixed_lambda", type=float, default=0.5)
    ap.add_argument(
        "--alpha_mode",
        type=str,
        default="adaptive",
        choices=["adaptive", "r_ord", "r_ord_strict", "R", "r_sem", "fixed", "const"],
    )
    ap.add_argument("--alpha_max", type=float, default=0.5)
    ap.add_argument("--gamma_alpha", type=float, default=1.0)
    ap.add_argument("--alpha_schedule", type=str, default="constant", choices=["constant", "linear", "cosine"])
    ap.add_argument("--anneal_alpha", action="store_true", help="Shorthand alias for --alpha_schedule linear")
    ap.add_argument("--phi", type=str, default="inv", choices=["inv", "exp", "uniform"])
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--fixed_delta", type=str, default="", help="Optional comma-separated delta[0..K] override")
    ap.add_argument("--fixed_a", type=float, default=None)
    ap.add_argument("--fixed_b", type=float, default=None)

    # --- Pluggable Embedding / Representation Source Switches ---
    ap.add_argument("--embedding_source", type=str, default="w2v", choices=list(EMBEDDING_SOURCES))
    ap.add_argument("--embed_dim", type=int, default=128)
    ap.add_argument("--embedding_path", type=str, default="", help="Path to custom .npy matrix when --embedding_source file")
    ap.add_argument("--embedding_ckpt", type=str, default="", help="Student .pt checkpoint for student_tok_emb / student_hidden")
    ap.add_argument("--warmup_ce_epochs", type=int, default=0, help="If >0 and no --embedding_ckpt, run N epochs of same-corpus CE to bootstrap student geometry")
    ap.add_argument("--hidden_layer", type=int, default=None, help="Transformer block index for student_hidden (None -> mid-layer n_layer // 2; -1 -> final ln_f)")
    ap.add_argument("--random_vectors", action="store_true", help="Legacy alias for --embedding_source random")

    # --- Other Baselines & Runtime Flags ---
    ap.add_argument("--teacher_ckpt", type=str, default="", help="Comma-separated teacher .pt checkpoints for --method self_kd. Each is loaded from its own config; logits are averaged.")
    ap.add_argument("--kd_alpha", type=float, default=0.5, help="Born-Again mixing weight on the tempered KL (scaled by --alpha_schedule)")
    ap.add_argument("--kd_temp", type=float, default=1.0, help="Born-Again softmax temperature T; the KL term is multiplied by T^2")
    ap.add_argument("--kn_loo", action=argparse.BooleanOptionalAction, default=True, help="KN leave-one-out (--no-kn_loo keeps the current event in the counts)")
    ap.add_argument("--d2gpo_sigma", type=float, default=1.0, help="D2GPo Gaussian std on rank (Li et al. 2020; released generator default 1)")
    ap.add_argument("--d2gpo_temp", type=float, default=2.0, help="D2GPo prior softmax temperature T in Eq. 7 (released criterion default 2.0)")
    ap.add_argument("--d2gpo_lambda", type=float, default=0.1, help="D2GPo KL weight lambda in Eq. 6 (released --d2gpo-alpha default 0.1)")
    ap.add_argument("--d2gpo_r", type=int, default=64, help="D2GPo prior row width after truncate-and-renormalise")
    ap.add_argument("--coconts_k", type=int, default=8)
    ap.add_argument("--coconts_r", type=int, default=8)
    ap.add_argument("--coconts_gamma", type=float, default=1.5, help="CoCoNTs Eq. 4 consistency parameter gamma > 1 (paper default: 1.5)")
    ap.add_argument("--mir_prob", type=float, default=0.10)
    ap.add_argument("--prefix_only_k", type=int, default=0)
    ap.add_argument("--reply_loss_only", action="store_true")
    ap.add_argument("--rebuild_targets", action="store_true")
    ap.add_argument("--save_ckpt", action=argparse.BooleanOptionalAction, default=True, help="Save bf16 checkpoint for standalone eval.py")
    ap.add_argument("--eval_blimp", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    return ap


def parse_args_with_config(argv: list[str] | None = None) -> argparse.Namespace:
    ap = build_argparser()
    prelim, _ = ap.parse_known_args(argv)
    if prelim.config:
        cfg_data = json.loads(Path(prelim.config).read_text(encoding="utf-8"))
        ap.set_defaults(**cfg_data)
    args = ap.parse_args(argv)
    if args.no_loo:
        args.leave_one_out = False
    if args.random_vectors:
        args.embedding_source = "random"
    return args


if __name__ == "__main__":
    train_run(parse_args_with_config(sys.argv[1:]))
