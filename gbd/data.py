"""
Human-only English conversation pipeline, SentencePiece tokeniser, same-corpus word2vec,
and Phase 0 diagnostics for GBD.

Strictly excludes LLM-generated synthetic chat datasets. Supported human-written sources:
  - "oasst1":      OpenAssistant/oasst1 (English crowd-sourced trees, zero-node-overlap decomposition)
  - "oasst2":      OpenAssistant/oasst2 (English crowd-sourced trees, zero-node-overlap decomposition)
  - "no_robots":   HuggingFaceH4/no_robots (human-written multi-turn instructions/dialogues)
  - "dailydialog": li2017dailydialog/daily_dialog (Parquet branch, multi-turn everyday dialogues)
  - "prosocial":   allenai/prosocial-dialog (Parquet branch, human-written multi-turn dialogues)

Special token IDs (fixed across all runs):
  0: <|pad|>, 1: <|unk|>, 2: <|bos|>, 3: <|eos|>, 4: <|user|>, 5: <|assistant|>
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

import numpy as np

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
USER_ID = 4
ASST_ID = 5
ROLE_IDS = (USER_ID, ASST_ID)

_WS_RE = re.compile(r"\s+")


def _norm_text(s: str) -> str:
    return _WS_RE.sub(" ", s.strip().lower())


@dataclass
class Conversation:
    conv_id: str
    group_id: str                    # tree/group id so related branches stay in the same split
    source: str
    turns: list[tuple[str, str]]     # list of ("user" | "assistant", text)


# --------------------------------------------------------------------------
# Loading strictly human-written English conversation datasets
# --------------------------------------------------------------------------
def _extract_disjoint_oasst_convs(rows, source_name: str) -> list[Conversation]:
    """
    Decompose OpenAssistant message trees into strictly node-disjoint English conversations.
    Every `message_id` is used at most once across the entire corpus so shared tree prefixes
    never duplicate text across conversations (preserving Leave-One-Out integrity).
    """
    by_id = {}
    children = defaultdict(list)
    roots = []
    for r in rows:
        if r.get("deleted", False):
            continue
        if r.get("lang") != "en":
            continue
        if r.get("review_result") is False:
            continue
        txt = (r.get("text") or "").strip()
        if not txt:
            continue
        mid = r["message_id"]
        pid = r.get("parent_id")
        role = "user" if r.get("role") == "prompter" else "assistant"
        rank = r.get("rank")
        rank_key = rank if (rank is not None and rank >= 0) else 999
        node = {
            "id": mid,
            "pid": pid,
            "tree": r.get("message_tree_id", mid),
            "role": role,
            "text": txt,
            "rank": rank_key,
        }
        by_id[mid] = node
        if pid is None:
            roots.append(mid)
        else:
            children[pid].append(mid)

    # Keep only nodes reachable from valid English roots with alternating roles
    memo_depth: dict[str, int] = {}

    def best_Order_key(mid: str):
        node = by_id[mid]
        nxt_role = "assistant" if node["role"] == "user" else "user"
        valid_ch = [c for c in children[mid] if c in by_id and by_id[c]["role"] == nxt_role]
        children[mid] = valid_ch
        if not valid_ch:
            memo_depth[mid] = 1
            return
        for c in valid_ch:
            best_Order_key(c)
        # Prefer top human rank first, then deeper multi-turn thread, then longer text
        valid_ch.sort(key=lambda c: (by_id[c]["rank"], -memo_depth[c], -len(by_id[c]["text"])))
        memo_depth[mid] = 1 + memo_depth[valid_ch[0]]

    for r_id in roots:
        if r_id in by_id and by_id[r_id]["role"] == "user":
            best_Order_key(r_id)

    convs: list[Conversation] = []
    used = set()

    def decompose_from_user(start_uid: str, tree_id: str):
        path = []
        curr = start_uid
        while curr is not None and curr not in used and curr in by_id:
            used.add(curr)
            path.append(curr)
            ch = [c for c in children[curr] if c not in used]
            if not ch:
                break
            best_c = ch[0]
            # Secondary branches starting at a user turn become disjoint conversations;
            # secondary assistant replies on the same user prompt are skipped because their
            # prompt was already consumed, except their downstream user follow-ups!
            for extra_c in ch[1:]:
                if by_id[extra_c]["role"] == "user":
                    decompose_from_user(extra_c, tree_id)
                else:
                    for grand_u in children[extra_c]:
                        if grand_u not in used and by_id[grand_u]["role"] == "user":
                            decompose_from_user(grand_u, tree_id)
            curr = best_c

        # Trim trailing user turn so every conversation ends on an assistant reply
        while path and by_id[path[-1]]["role"] != "assistant":
            path.pop()
        if len(path) >= 2 and by_id[path[0]]["role"] == "user":
            turns = [(by_id[m]["role"], by_id[m]["text"]) for m in path]
            convs.append(Conversation(conv_id=f"{source_name}:{path[0]}", group_id=f"{source_name}:{tree_id}", source=source_name, turns=turns))

    for r_id in roots:
        if r_id in by_id and by_id[r_id]["role"] == "user" and r_id not in used:
            decompose_from_user(r_id, by_id[r_id]["tree"])
    return convs


def load_human_conversations(sources: list[str], hf_cache_dir: str | Path) -> list[Conversation]:
    from datasets import load_dataset

    os.environ["HF_HOME"] = str(Path(hf_cache_dir).resolve())
    all_convs: list[Conversation] = []

    for src in sources:
        s = src.strip().lower()
        t0 = time.time()
        before = len(all_convs)
        if s in ("oasst1", "oasst2"):
            repo = f"OpenAssistant/{s}"
            ds_tr = load_dataset(repo, split="train", cache_dir=str(hf_cache_dir))
            ds_va = load_dataset(repo, split="validation", cache_dir=str(hf_cache_dir))
            rows = list(ds_tr) + list(ds_va)
            all_convs.extend(_extract_disjoint_oasst_convs(rows, s))
        elif s == "no_robots":
            for sp_name in ("train", "test"):
                ds = load_dataset("HuggingFaceH4/no_robots", split=sp_name, cache_dir=str(hf_cache_dir))
                for idx, row in enumerate(ds):
                    msgs = row.get("messages") or []
                    sys_prefix = ""
                    turns: list[tuple[str, str]] = []
                    for m in msgs:
                        role = m.get("role", "")
                        content = (m.get("content") or "").strip()
                        if not content:
                            continue
                        if role == "system":
                            sys_prefix = content
                        elif role == "user":
                            if sys_prefix and not turns:
                                content = f"{sys_prefix}\n{content}"
                                sys_prefix = ""
                            if turns and turns[-1][0] == "user":
                                turns[-1] = ("user", turns[-1][1] + "\n" + content)
                            else:
                                turns.append(("user", content))
                        elif role == "assistant":
                            if not turns:
                                continue
                            if turns[-1][0] == "assistant":
                                turns[-1] = ("assistant", turns[-1][1] + "\n" + content)
                            else:
                                turns.append(("assistant", content))
                    while turns and turns[-1][0] != "assistant":
                        turns.pop()
                    if len(turns) >= 2:
                        cid = f"no_robots:{sp_name}:{idx}"
                        all_convs.append(Conversation(conv_id=cid, group_id=cid, source="no_robots", turns=turns))
        elif s == "dailydialog":
            for sp_name in ("train", "validation", "test"):
                ds = load_dataset(
                    "li2017dailydialog/daily_dialog",
                    revision="refs/convert/parquet",
                    split=sp_name,
                    cache_dir=str(hf_cache_dir),
                )
                for idx, row in enumerate(ds):
                    utts = [u.strip() for u in (row.get("dialog") or row.get("utterances") or []) if u and u.strip()]
                    if len(utts) < 2:
                        continue
                    if len(utts) % 2 == 1:
                        utts = utts[:-1]
                    turns = [("user" if j % 2 == 0 else "assistant", utts[j]) for j in range(len(utts))]
                    cid = f"dailydialog:{sp_name}:{idx}"
                    all_convs.append(Conversation(conv_id=cid, group_id=cid, source="dailydialog", turns=turns))
        elif s == "prosocial":
            for sp_name in ("train", "validation", "test"):
                ds = load_dataset(
                    "allenai/prosocial-dialog",
                    revision="refs/convert/parquet",
                    split=sp_name,
                    cache_dir=str(hf_cache_dir),
                )
                grouped = defaultdict(list)
                for row in ds:
                    did = row.get("dialogue_id", 0)
                    eid = row.get("response_id", 0)
                    ctx = (row.get("context") or "").strip()
                    resp = (row.get("response") or "").strip()
                    if ctx and resp:
                        grouped[(sp_name, did)].append((eid, ctx, resp))
                for (sp_n, did), items in grouped.items():
                    items.sort(key=lambda z: z[0])
                    turns = []
                    for _, ctx, resp in items:
                        turns.append(("user", ctx))
                        turns.append(("assistant", resp))
                    cid = f"prosocial:{sp_n}:{did}"
                    all_convs.append(Conversation(conv_id=cid, group_id=cid, source="prosocial", turns=turns))
        else:
            raise ValueError(f"Unknown human dialogue source: {src!r}")
        print(f"loaded {s:12s}: {len(all_convs) - before:6d} conversations ({time.time() - t0:.1f}s)", flush=True)
    return all_convs


def deduplicate_conversations(convs: list[Conversation]) -> list[Conversation]:
    """Deduplicate identical conversations and repeated prompt-reply pairs."""
    seen_conv = set()
    seen_first_pair = set()
    clean: list[Conversation] = []
    for c in convs:
        full_sig = hashlib.blake2b(
            "\n".join(f"{r}:{_norm_text(t)}" for r, t in c.turns).encode("utf-8"),
            digest_size=16,
        ).digest()
        if full_sig in seen_conv:
            continue
        pair_sig = hashlib.blake2b(
            f"{_norm_text(c.turns[0][1])}||{_norm_text(c.turns[1][1])}".encode("utf-8"),
            digest_size=16,
        ).digest()
        if pair_sig in seen_first_pair:
            continue
        seen_conv.add(full_sig)
        seen_first_pair.add(pair_sig)
        clean.append(c)
    return clean


def split_conversations(
    convs: list[Conversation],
    val_frac: float = 0.05,
    test_frac: float = 0.05,
    seed: int = 42,
) -> tuple[list[Conversation], list[Conversation], list[Conversation]]:
    """Split at the `group_id` (message-tree / dialogue) level so no shared tree crosses splits."""
    groups = defaultdict(list)
    for c in convs:
        groups[c.group_id].append(c)
    g_keys = sorted(groups.keys())
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(g_keys))
    n_test = max(1, int(len(g_keys) * test_frac))
    n_val = max(1, int(len(g_keys) * val_frac))
    test_g = {g_keys[i] for i in perm[:n_test]}
    val_g = {g_keys[i] for i in perm[n_test : n_test + n_val]}
    train, val, test = [], [], []
    for k in g_keys:
        if k in test_g:
            test.extend(groups[k])
        elif k in val_g:
            val.extend(groups[k])
        else:
            train.extend(groups[k])
    rng.shuffle(train)
    return train, val, test


# --------------------------------------------------------------------------
# SentencePiece Tokeniser & Encoding
# --------------------------------------------------------------------------
def train_sentencepiece(train_convs: list[Conversation], out_prefix: Path, vocab_size: int = 16_000):
    import sentencepiece as spm

    txt_path = out_prefix.parent / "spm_train_corpus.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        for c in train_convs:
            for _, text in c.turns:
                for line in text.splitlines():
                    line = line.strip()
                    if line:
                        f.write(line + "\n")
    spm.SentencePieceTrainer.train(
        input=str(txt_path),
        model_prefix=str(out_prefix),
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        byte_fallback=True,
        split_digits=True,
        normalization_rule_name="identity",
        pad_id=PAD_ID,
        unk_id=UNK_ID,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        pad_piece="<|pad|>",
        unk_piece="<|unk|>",
        bos_piece="<|bos|>",
        eos_piece="<|eos|>",
        user_defined_symbols=["<|user|>", "<|assistant|>"],
        num_threads=max(1, (os.cpu_count() or 4) // 2),
    )
    txt_path.unlink(missing_ok=True)
    sp_proc = spm.SentencePieceProcessor(model_file=str(out_prefix) + ".model")
    assert sp_proc.piece_to_id("<|user|>") == USER_ID, sp_proc.piece_to_id("<|user|>")
    assert sp_proc.piece_to_id("<|assistant|>") == ASST_ID, sp_proc.piece_to_id("<|assistant|>")
    return sp_proc


def encode_split(convs: list[Conversation], sp_proc, max_tokens: int | None = None) -> dict:
    """
    Encode conversations into flat token stream separated by EOS_ID:
      <|user|> u_tokens <|assistant|> a_tokens ... <|eos|>
    Tracks `reply_mask` (True on assistant reply tokens and their terminating turn/EOS token)
    and exact UTF-8 byte counts for held-out Bits-Per-Byte (BPB) computation.
    """
    toks: list[int] = []
    reply_mask: list[bool] = []
    reply_bytes = 0
    total_bytes = 0
    n_convs_used = 0

    for c in convs:
        c_toks: list[int] = []
        c_rep: list[bool] = []
        c_rbytes = 0
        c_tbytes = 0
        for idx_t, (role, text) in enumerate(c.turns):
            ids = sp_proc.encode(text, out_type=int)
            if not ids:
                continue
            b_len = len(text.encode("utf-8")) + 1  # +1 byte delimiter per turn
            c_tbytes += b_len
            if role == "user":
                c_toks.append(USER_ID)
                c_rep.append(False)
                c_toks.extend(ids)
                c_rep.extend([False] * len(ids))
            else:
                c_toks.append(ASST_ID)
                c_rep.append(False)
                c_toks.extend(ids)
                c_rep.extend([True] * len(ids))
                c_rbytes += b_len
                # Next structural boundary token (<|user|> or final <|eos|>) terminates the reply
                # and must be scored in reply NLL so probability over variable-length replies sums to 1
        if not c_toks:
            continue
        c_toks.append(EOS_ID)
        c_rep.append(True)  # conversation ends after assistant reply

        # Also mark <|user|> immediately following an assistant turn as reply-stopping token
        for i in range(1, len(c_toks)):
            if c_toks[i] == USER_ID and c_rep[i - 1]:
                c_rep[i] = True

        if max_tokens is not None and len(toks) + len(c_toks) > max_tokens and len(toks) > 0:
            break
        toks.extend(c_toks)
        reply_mask.extend(c_rep)
        reply_bytes += c_rbytes
        total_bytes += c_tbytes
        n_convs_used += 1

    return {
        "tokens": np.asarray(toks, dtype=np.int32),
        "reply_mask": np.asarray(reply_mask, dtype=bool),
        "reply_bytes": int(reply_bytes),
        "total_bytes": int(total_bytes),
        "n_conversations": int(n_convs_used),
    }


# --------------------------------------------------------------------------
# Teacher-free same-corpus Word2Vec embeddings
# --------------------------------------------------------------------------
def train_same_corpus_word2vec(
    tokens: np.ndarray,
    vocab_size: int,
    dim: int = 128,
    window: int = 5,
    epochs: int = 8,
    seed: int = 0,
) -> np.ndarray:
    """Train static skip-gram vectors strictly on the training token stream (no external data)."""
    from gensim.models import Word2Vec

    # Split into conversation sentences at EOS_ID, stripping structural role markers
    eos_locs = np.nonzero(tokens == EOS_ID)[0]
    sents: list[list[str]] = []
    start = 0
    role_set = {USER_ID, ASST_ID, EOS_ID, PAD_ID}
    for end in eos_locs:
        seg = [str(int(x)) for x in tokens[start:end] if int(x) not in role_set]
        for s in range(0, len(seg), 2000):
            chunk = seg[s : s + 2000]
            if len(chunk) >= 2:
                sents.append(chunk)
        start = int(end) + 1

    w2v = Word2Vec(
        sentences=sents,
        vector_size=dim,
        window=window,
        min_count=1,
        sg=1,
        negative=5,
        epochs=epochs,
        workers=max(1, min(6, (os.cpu_count() or 4) // 2)),
        seed=seed,
    )
    rng = np.random.default_rng(seed)
    E = np.zeros((vocab_size, dim), dtype=np.float32)
    for v in range(vocab_size):
        key = str(v)
        if key in w2v.wv:
            E[v] = w2v.wv[key]
        else:
            vec = rng.normal(size=dim).astype(np.float32)
            E[v] = vec / max(float(np.linalg.norm(vec)), 1e-12)
    return E


# --------------------------------------------------------------------------
# Phase 0 Diagnostics on Real Corpus
# --------------------------------------------------------------------------
def run_phase0_diagnostics(
    train_tokens: np.ndarray,
    val_tokens: np.ndarray,
    V: int,
    E: np.ndarray,
    sp_proc,
    out_dir: Path,
    K: int = 6,
    Ks: int = 2,
    M: int = 8,
) -> dict:
    from scipy.stats import spearmanr
    from gbd import (
        GBDConfig,
        OrderIndex,
        SemanticIndex,
        build_targets,
        context_features,
        conversation_starts,
        nearest_contexts_by_state,
    )

    diag: dict = {}
    T_full = len(train_tokens)

    # 1. Unique-context rate for k = 1..8 across corpus subsamples
    scales = sorted({min(s, T_full) for s in (1_000_000, 5_000_000, T_full)})
    uniq_rates = {}
    for sc in scales:
        sub = train_tokens[:sc]
        b_sub = conversation_starts(sub, EOS_ID)
        h = np.zeros(len(sub), dtype=np.uint64)
        rates_k = {}
        for k in range(1, 9):
            oidx = OrderIndex(sub, k, V, h, b_pos=b_sub)
            h = oidx.h
            valid = oidx.ctx >= 0
            n_loo = (oidx.n[np.maximum(oidx.ctx, 0)] - 1.0) * valid
            # Fraction of valid order-k positions whose k-token context occurs ONLY once (LOO count == 0)
            rates_k[f"k={k}"] = round(float((n_loo[valid] == 0).mean()), 4) if valid.any() else 1.0
        uniq_rates[f"{sc:,}_tokens"] = rates_k
    diag["unique_context_rate_by_scale_and_order"] = uniq_rates

    # 2. Build GBD v2 targets on a diagnostic slice (up to 1M tokens for fast Phase 0 turnaround)
    T_diag = min(T_full, 1_000_000)
    sub_toks = train_tokens[:T_diag]
    t0 = time.time()
    cfg = GBDConfig(
        K=K, Ks=Ks, M=M, M_search=32, R_pre=64, r=16,
        eos_id=EOS_ID, role_ids=ROLE_IDS, seed=0,
    )
    tg = build_targets(sub_toks, V, E, cfg)
    build_sec = time.time() - t0
    diag["diagnostic_slice_tokens"] = int(T_diag)
    diag["target_build_seconds"] = round(build_sec, 2)
    diag["fitted"] = {
        "delta": [round(float(x), 4) for x in tg.fitted["delta"]],
        "d0": [round(float(x), 2) for x in tg.fitted["d0"]],
        "d0_sem": round(float(tg.fitted["d0_sem"]), 2) if tg.fitted["d0_sem"] is not None else None,
        "a": round(float(tg.fitted["a"]), 4) if tg.fitted["a"] is not None else None,
        "b": round(float(tg.fitted["b"]), 4) if tg.fitted["b"] is not None else None,
    }
    diag["alpha_summary"] = {
        "zero_fraction": round(float((tg.alpha == 0).mean()), 4),
        "mean": round(float(tg.alpha.mean()), 4),
        "q05": round(float(np.quantile(tg.alpha, 0.05)), 4),
        "q50": round(float(np.quantile(tg.alpha, 0.50)), 4),
        "q95": round(float(np.quantile(tg.alpha, 0.95)), 4),
        "mean_lambda_exact": round(float(tg.w_exact.mean()), 4),
    }

    # 3. Empirical calibration check on real training LOO positions:
    # Since q_t is strictly Leave-One-Out (never sees x_t at position t), -log(0.99 * q_t(x_t) + 0.01 / V)
    # is an unbiased sample of E_{x_t ~ P(.|c_t)}[-log q_t(x_t)] = H(P) + KL(P || q_t)!
    rng = np.random.default_rng(0)
    eval_n = min(20_000, T_diag - 20)
    idx = rng.choice(np.arange(10, T_diag), size=eval_n, replace=False)
    x_true = sub_toks[idx]
    match = tg.ids[idx] == x_true[:, None]
    q_on_true = np.where(match, tg.probs[idx], 0.0).sum(1)
    nll_qt = -np.log(0.99 * q_on_true + 0.01 / V)
    top_p = tg.probs[idx, 0]

    # Compute LOO counts per order on sub_toks for grouping
    b_sub = conversation_starts(sub_toks, EOS_ID)
    h = np.zeros(T_diag, dtype=np.uint64)
    ns_idx = []
    for k in range(K + 1):
        o = OrderIndex(sub_toks, k, V, h, b_pos=b_sub)
        h = o.h
        vld = o.ctx[idx] >= 0
        ns_idx.append(np.where(vld, o.n[np.maximum(o.ctx[idx], 0)] - 1.0, 0.0))

    r_ord_i = tg.r_ord[idx]
    r_sem_i = tg.r_sem[idx]
    R_i = tg.R[idx]
    fin_sem = np.isfinite(r_sem_i)
    diag["empirical_calibration"] = {
        "spearman_r_ord_vs_loo_nll": round(float(spearmanr(r_ord_i, nll_qt).correlation), 4),
        "spearman_r_sem_vs_loo_nll": round(float(spearmanr(r_sem_i[fin_sem], nll_qt[fin_sem]).correlation), 4) if fin_sem.sum() > 20 else None,
        "spearman_R_vs_loo_nll": round(float(spearmanr(R_i, nll_qt).correlation), 4),
    }

    groups = {
        f"order-{K} seen once elsewhere": ns_idx[K] == 1,
        "order-3 seen once, no longer match": (ns_idx[min(4, K)] == 0) & (ns_idx[min(3, K)] == 1),
        "order-3 seen 5+ times": ns_idx[min(3, K)] >= 5,
        "order-2 seen 20+ times": ns_idx[2] >= 20,
        "order-2 never seen elsewhere": ns_idx[2] == 0,
        "all sampled positions": np.ones(eval_n, dtype=bool),
    }
    grp_table = {}
    for gname, mask in groups.items():
        if mask.sum() == 0:
            continue
        grp_table[gname] = {
            "share": round(float(mask.mean()), 4),
            "top_prob": round(float(top_p[mask].mean()), 3),
            "mean_r_ord": round(float(np.mean(r_ord_i[mask][np.isfinite(r_ord_i[mask])])), 3) if np.isfinite(r_ord_i[mask]).any() else None,
            "mean_alpha": round(float(tg.alpha[idx][mask].mean()), 4),
            "mean_loo_nll": round(float(nll_qt[mask].mean()), 3),
        }
    diag["calibration_groups"] = grp_table

    # 4. Manual nearest-neighbour inspection (both reply-start s_t = ASST_ID and mid-turn s_t = -1)
    sidx = SemanticIndex(sub_toks, Ks, V, b_pos=b_sub, role_ids=ROLE_IDS)
    F = context_features(sub_toks, sidx, E, gamma=1.0)
    nD, nI = nearest_contexts_by_state(F, sidx.ctx_state, 5)
    samples_nn = []
    for bucket_label, target_state in [("reply_start_after_assistant_marker", ASST_ID), ("mid_turn_content", -1)]:
        cand_c = np.nonzero((sidx.ctx_state == target_state) & (sidx.n >= 5))[0]
        if len(cand_c) == 0:
            continue
        chosen = rng.choice(cand_c, size=min(4, len(cand_c)), replace=False)
        for ci in chosen:
            # ctx_toks[:, ci] is [tau_1, tau_2] (most recent first) -> reverse to chronological order
            chron = [int(t) for t in sidx.ctx_toks[::-1, ci]]
            q_str = sp_proc.decode(chron)
            nbrs = []
            for m_idx in range(min(4, nI.shape[1])):
                nj = int(nI[ci, m_idx])
                if nj < 0:
                    continue
                n_chron = [int(t) for t in sidx.ctx_toks[::-1, nj]]
                nbrs.append({
                    "context": sp_proc.decode(n_chron),
                    "dist": round(float(nD[ci, m_idx]), 3),
                    "count": int(sidx.n[nj]),
                })
            samples_nn.append({"bucket": bucket_label, "query": q_str, "count": int(sidx.n[ci]), "neighbours": nbrs})
    diag["nearest_neighbour_samples"] = samples_nn

    out_path = out_dir / "phase0_diagnostics.json"
    out_path.write_text(json.dumps(diag, indent=2, ensure_ascii=False), encoding="utf-8")
    return diag


def prepare_dataset_pipeline(
    sources: list[str],
    out_dir: str | Path,
    vocab_size: int = 16_000,
    embed_dim: int = 128,
    max_train_tokens: int = 20_000_000,
    seed: int = 42,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    hf_cache = out.parent / ".hf_cache"
    hf_cache.mkdir(parents=True, exist_ok=True)

    raw_convs = load_human_conversations(sources, hf_cache_dir=hf_cache)
    clean_convs = deduplicate_conversations(raw_convs)
    print(f"deduplicated: {len(raw_convs):,} -> {len(clean_convs):,} conversations", flush=True)

    train_c, val_c, test_c = split_conversations(clean_convs, val_frac=0.05, test_frac=0.05, seed=seed)
    print(f"splits: train={len(train_c):,}, val={len(val_c):,}, test={len(test_c):,} conversations", flush=True)

    spm_prefix = out / "tokenizer"
    sp_proc = train_sentencepiece(train_c, spm_prefix, vocab_size=vocab_size)
    print(f"trained SentencePiece BPE vocab={sp_proc.vocab_size()} (byte_fallback=True)", flush=True)

    tr = encode_split(train_c, sp_proc, max_tokens=max_train_tokens)
    va = encode_split(val_c, sp_proc, max_tokens=None)
    te = encode_split(test_c, sp_proc, max_tokens=None)

    for sp_name, d in [("train", tr), ("val", va), ("test", te)]:
        np.save(out / f"{sp_name}_tokens.npy", d["tokens"])
        np.save(out / f"{sp_name}_reply_mask.npy", d["reply_mask"])
        print(
            f"  {sp_name:5s}: {len(d['tokens']):10,d} tokens | reply tokens: {int(d['reply_mask'].sum()):9,d} "
            f"| reply UTF-8 bytes: {d['reply_bytes']:10,d}",
            flush=True,
        )

    t0 = time.time()
    E = train_same_corpus_word2vec(tr["tokens"], vocab_size=sp_proc.vocab_size(), dim=embed_dim, seed=seed)
    np.save(out / "w2v_embeddings.npy", E)
    print(f"trained same-corpus word2vec ({E.shape[0]}x{E.shape[1]}) in {time.time() - t0:.1f}s", flush=True)

    meta = {
        "sources": sources,
        "vocab_size": sp_proc.vocab_size(),
        "embed_dim": embed_dim,
        "special_ids": {"pad": PAD_ID, "unk": UNK_ID, "bos": BOS_ID, "eos": EOS_ID, "user": USER_ID, "assistant": ASST_ID},
        "splits": {
            name: {
                "tokens": int(len(d["tokens"])),
                "reply_tokens": int(d["reply_mask"].sum()),
                "reply_bytes": d["reply_bytes"],
                "total_bytes": d["total_bytes"],
                "conversations": d["n_conversations"],
            }
            for name, d in [("train", tr), ("val", va), ("test", te)]
        },
    }
    (out / "corpus_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("running Phase 0 diagnostics...", flush=True)
    diag = run_phase0_diagnostics(tr["tokens"], va["tokens"], sp_proc.vocab_size(), E, sp_proc, out_dir=out)
    return {"meta": meta, "diagnostics": diag}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", type=str, default="oasst2,no_robots,dailydialog")
    ap.add_argument("--out_dir", type=str, default="artifacts/corpus")
    ap.add_argument("--vocab_size", type=int, default=16_000)
    ap.add_argument("--embed_dim", type=int, default=128)
    ap.add_argument("--max_train_tokens", type=int, default=20_000_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    prepare_dataset_pipeline(
        sources=[s for s in args.sources.split(",") if s.strip()],
        out_dir=args.out_dir,
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        max_train_tokens=args.max_train_tokens,
        seed=args.seed,
    )
