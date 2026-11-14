r"""
Standalone Evaluation Harness for Stage 1 / Stage 2 GBD Student Checkpoints.

Computes:
  - Held-out Reply Bits-Per-Byte (BPB, primary metric) and All-token BPB
  - Reply NLL & Perplexity
  - Rare-token NLL stratified by training-corpus token frequency (<10, 10-99, 100-999, 1000+)
  - 15-bin Expected Calibration Error (ECE) on reply tokens
  - Optional BLiMP minimal-pair grammaticality accuracy (`--eval_blimp`)
  - Optional qualitative reply generation on held-out test conversations (`--num_samples`)

Usage:
  .\.venv\Scripts\python.exe gbd\eval.py --checkpoint artifacts/checkpoints/<run_name>.pt --eval_blimp
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import ASST_ID, EOS_ID, PAD_ID, USER_ID
from model import CausalLM, ModelConfig


@torch.no_grad()
def evaluate_split(
    model: CausalLM,
    tokens: np.ndarray,
    reply_mask: np.ndarray,
    reply_bytes: int,
    total_bytes: int,
    train_unigram_counts: np.ndarray,
    seq_len: int,
    batch_size: int,
    device: torch.device,
) -> dict:
    model.eval()
    T = len(tokens)
    starts = list(range(0, T - 1, seq_len))

    all_nll_sum = 0.0
    all_tok_cnt = 0
    rep_nll_sum = 0.0
    rep_tok_cnt = 0

    bin_edges = [
        (0, 9, "rare_<10"),
        (10, 99, "low_10_99"),
        (100, 999, "mid_100_999"),
        (1000, 10**12, "freq_1000+"),
    ]
    bin_nll = {name: 0.0 for _, _, name in bin_edges}
    bin_cnt = {name: 0 for _, _, name in bin_edges}

    n_ece_bins = 15
    ece_conf_sum = np.zeros(n_ece_bins, dtype=np.float64)
    ece_acc_sum = np.zeros(n_ece_bins, dtype=np.float64)
    ece_tot = np.zeros(n_ece_bins, dtype=np.int64)

    use_amp = device.type == "cuda"
    for b_start in range(0, len(starts), batch_size):
        b_s = starts[b_start : b_start + batch_size]
        max_l = min(seq_len, max((T - 1) - s for s in b_s))
        x_np = np.full((len(b_s), max_l), PAD_ID, dtype=np.int64)
        y_np = np.full((len(b_s), max_l), -100, dtype=np.int64)
        rm_np = np.zeros((len(b_s), max_l), dtype=bool)
        for row_i, s in enumerate(b_s):
            ln = min(max_l, (T - 1) - s)
            x_np[row_i, :ln] = tokens[s : s + ln]
            y_np[row_i, :ln] = tokens[s + 1 : s + 1 + ln]
            rm_np[row_i, :ln] = reply_mask[s + 1 : s + 1 + ln]

        x_t = torch.from_numpy(x_np).to(device, non_blocking=True)
        y_t = torch.from_numpy(y_np).to(device, non_blocking=True)
        rm_t = torch.from_numpy(rm_np).to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            logits = model(x_t)
        logp = F.log_softmax(logits.float(), dim=-1)
        valid_t = y_t != -100
        safe_y = y_t.clamp_min(0)
        tok_nll = -logp.gather(-1, safe_y.unsqueeze(-1)).squeeze(-1)

        all_nll_sum += float(tok_nll[valid_t].sum().item())
        all_tok_cnt += int(valid_t.sum().item())

        rep_valid = valid_t & rm_t
        if rep_valid.any():
            r_nll = tok_nll[rep_valid].detach().cpu().numpy()
            r_y = safe_y[rep_valid].detach().cpu().numpy()
            rep_nll_sum += float(r_nll.sum())
            rep_tok_cnt += int(r_nll.size)

            freqs = train_unigram_counts[r_y]
            for lo_f, hi_f, bname in bin_edges:
                m_f = (freqs >= lo_f) & (freqs <= hi_f)
                if m_f.any():
                    bin_nll[bname] += float(r_nll[m_f].sum())
                    bin_cnt[bname] += int(m_f.sum())

            max_lp, pred_id = logp[rep_valid].max(dim=-1)
            conf = max_lp.exp().detach().cpu().numpy()
            acc = (pred_id == safe_y[rep_valid]).float().detach().cpu().numpy()
            b_idx = np.clip((conf * n_ece_bins).astype(np.int64), 0, n_ece_bins - 1)
            np.add.at(ece_conf_sum, b_idx, conf)
            np.add.at(ece_acc_sum, b_idx, acc)
            np.add.at(ece_tot, b_idx, 1)

    ln2 = math.log(2.0)
    reply_bpb = rep_nll_sum / max(reply_bytes * ln2, 1e-12)
    all_bpb = all_nll_sum / max(total_bytes * ln2, 1e-12)
    reply_nll = rep_nll_sum / max(rep_tok_cnt, 1)
    all_nll = all_nll_sum / max(all_tok_cnt, 1)

    ece = 0.0
    if ece_tot.sum() > 0:
        nz = ece_tot > 0
        ece = float(np.sum(np.abs(ece_acc_sum[nz] - ece_conf_sum[nz])) / ece_tot.sum())

    rare_out = {
        bname: (round(bin_nll[bname] / bin_cnt[bname], 4) if bin_cnt[bname] > 0 else None)
        for _, _, bname in bin_edges
    }
    return {
        "reply_bpb": round(reply_bpb, 4),
        "all_bpb": round(all_bpb, 4),
        "reply_nll": round(reply_nll, 4),
        "all_nll": round(all_nll, 4),
        "reply_ppl": round(math.exp(min(reply_nll, 20.0)), 3),
        "ece": round(ece, 4),
        "rare_token_nll": rare_out,
        "rare_token_counts": bin_cnt,
    }


@torch.no_grad()
def evaluate_blimp(
    model: CausalLM,
    sp_proc,
    hf_cache_dir: Path,
    device: torch.device,
    max_paradigms: int = 12,
    pairs_per_paradigm: int = 200,
) -> dict:
    """Evaluate minimal-pair grammaticality accuracy on BLiMP subsets."""
    from datasets import get_dataset_config_names, load_dataset

    model.eval()
    all_cfgs = sorted(get_dataset_config_names("nyu-mll/blimp"))
    configs = all_cfgs if max_paradigms <= 0 else all_cfgs[:max_paradigms]
    use_amp = device.type == "cuda"
    accs = {}
    for cfg_name in configs:
        ds = load_dataset("nyu-mll/blimp", cfg_name, split="train", cache_dir=str(hf_cache_dir))
        n_use = len(ds) if pairs_per_paradigm <= 0 else min(len(ds), pairs_per_paradigm)
        wins = 0
        for i in range(n_use):
            good_ids = [USER_ID] + sp_proc.encode(ds[i]["sentence_good"], out_type=int)
            bad_ids = [USER_ID] + sp_proc.encode(ds[i]["sentence_bad"], out_type=int)
            good_ids = good_ids[: model.cfg.max_seq_len + 1]
            bad_ids = bad_ids[: model.cfg.max_seq_len + 1]
            scores = []
            for seq in (good_ids, bad_ids):
                inp = torch.tensor([seq[:-1]], dtype=torch.long, device=device)
                tgt = torch.tensor([seq[1:]], dtype=torch.long, device=device)
                with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                    logits = model(inp)
                lp = F.log_softmax(logits.float(), dim=-1)
                ll = lp.gather(-1, tgt.unsqueeze(-1)).sum().item()
                scores.append(ll)
            if scores[0] > scores[1]:
                wins += 1
        accs[cfg_name] = round(wins / max(n_use, 1), 4)
    return {"mean_acc": round(float(np.mean(list(accs.values()))), 4), "paradigms": accs}


@torch.no_grad()
def generate_sample_replies(
    model: CausalLM,
    test_tokens: np.ndarray,
    sp_proc,
    device: torch.device,
    num_samples: int = 5,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    top_p: float = 0.9,
    seed: int = 0,
) -> list[dict]:
    """Generate qualitative reply samples on held-out test prompts."""
    model.eval()
    rng = torch.Generator(device=device).manual_seed(seed)
    eos_locs = [-1] + np.nonzero(test_tokens == EOS_ID)[0].tolist()
    samples = []
    use_amp = device.type == "cuda"

    for c_idx in range(min(num_samples, len(eos_locs) - 1)):
        seg = test_tokens[eos_locs[c_idx] + 1 : eos_locs[c_idx + 1]]
        asst_pos = np.nonzero(seg == ASST_ID)[0]
        if len(asst_pos) == 0:
            continue
        p_end = int(asst_pos[0])
        prompt_ids = seg[: p_end + 1].tolist()  # includes <|assistant|>
        # reference first reply
        ref_end = len(seg)
        for j in range(p_end + 1, len(seg)):
            if seg[j] in (USER_ID, EOS_ID):
                ref_end = j
                break
        ref_ids = [int(t) for t in seg[p_end + 1 : ref_end] if int(t) >= 6]
        user_ids = [int(t) for t in seg[1:p_end] if int(t) >= 6]

        cur = prompt_ids[-(model.cfg.max_seq_len - max_new_tokens) :]
        gen_ids = []
        for _ in range(max_new_tokens):
            inp = torch.tensor([cur[-model.cfg.max_seq_len :]], dtype=torch.long, device=device)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                logits = model(inp)[:, -1, :]
            if temperature <= 0:
                nxt = int(logits.argmax(dim=-1).item())
            else:
                probs = F.softmax(logits.float() / temperature, dim=-1)
                sorted_p, sorted_i = torch.sort(probs, descending=True, dim=-1)
                cum_p = torch.cumsum(sorted_p, dim=-1)
                cut = cum_p > top_p
                cut[..., 1:] = cut[..., :-1].clone()
                cut[..., 0] = False
                sorted_p[cut] = 0.0
                sorted_p /= sorted_p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                pick = torch.multinomial(sorted_p, num_samples=1, generator=rng)
                nxt = int(sorted_i.gather(-1, pick).item())
            if nxt in (EOS_ID, USER_ID, ASST_ID):
                break
            cur.append(nxt)
            if nxt >= 6:
                gen_ids.append(nxt)

        samples.append({
            "user_prompt": sp_proc.decode(user_ids),
            "reference_reply": sp_proc.decode(ref_ids),
            "model_reply": sp_proc.decode(gen_ids),
        })
    return samples


def load_checkpoint_model(ckpt_path: str | Path, device: torch.device) -> tuple[CausalLM, dict]:
    payload = torch.load(Path(ckpt_path), map_location="cpu", weights_only=False)
    mcfg = ModelConfig(**payload["model_config"])
    model = CausalLM(mcfg)
    state = {k: v.float() if v.is_floating_point() else v for k, v in payload["state_dict"].items()}
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, payload


def main():
    ap = argparse.ArgumentParser(description="Standalone GBD Student Checkpoint Evaluator")
    ap.add_argument("--checkpoint", type=str, required=True, help="Path to saved .pt checkpoint")
    ap.add_argument("--corpus_dir", type=str, default="artifacts/corpus")
    ap.add_argument("--train_tokens", type=int, default=0, help="Override train_tokens slice for rare-token bin counts (0 -> read from checkpoint)")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--splits", type=str, default="val,test", help="Comma-separated splits to evaluate: val,test")
    ap.add_argument("--eval_blimp", action="store_true")
    ap.add_argument("--blimp_paradigms", type=int, default=12, help="Number of BLiMP paradigms (0 -> all 67)")
    ap.add_argument("--blimp_pairs", type=int, default=200, help="Pairs per paradigm (0 -> all 1000)")
    ap.add_argument("--num_samples", type=int, default=0, help="Generate N sample conversation replies on test set")
    ap.add_argument("--out_json", type=str, default="", help="Optional path to save evaluation JSON")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, payload = load_checkpoint_model(args.checkpoint, device=device)
    run_cfg = payload.get("train_config", {})

    corpus_dir = Path(args.corpus_dir)
    meta = json.loads((corpus_dir / "corpus_meta.json").read_text(encoding="utf-8"))
    V = int(meta["vocab_size"])

    full_train = np.load(corpus_dir / "train_tokens.npy", mmap_mode="r")
    t_used = args.train_tokens if args.train_tokens > 0 else int(run_cfg.get("train_tokens", len(full_train)))
    t_used = min(len(full_train), t_used) if t_used > 0 else len(full_train)
    train_unigram_counts = np.bincount(np.asarray(full_train[:t_used], dtype=np.int32), minlength=V)

    results = {
        "checkpoint": str(args.checkpoint),
        "model_config": payload["model_config"],
        "train_tokens_reference": t_used,
        "splits": {},
    }

    for sp_name in [s.strip() for s in args.splits.split(",") if s.strip()]:
        toks = np.load(corpus_dir / f"{sp_name}_tokens.npy")
        rmask = np.load(corpus_dir / f"{sp_name}_reply_mask.npy")
        res = evaluate_split(
            model,
            toks,
            rmask,
            reply_bytes=meta["splits"][sp_name]["reply_bytes"],
            total_bytes=meta["splits"][sp_name]["total_bytes"],
            train_unigram_counts=train_unigram_counts,
            seq_len=model.cfg.max_seq_len,
            batch_size=args.batch_size,
            device=device,
        )
        results["splits"][sp_name] = res
        print(
            f"[{sp_name:4s}] reply_bpb={res['reply_bpb']:.4f} | all_bpb={res['all_bpb']:.4f} "
            f"| ppl={res['reply_ppl']:.2f} | ECE={res['ece']:.4f} | rare_<10={res['rare_token_nll']['rare_<10']}",
            flush=True,
        )

    if args.eval_blimp or args.num_samples > 0:
        import sentencepiece as spm
        sp_proc = spm.SentencePieceProcessor(model_file=str(corpus_dir / "tokenizer.model"))
        if args.eval_blimp:
            blimp = evaluate_blimp(
                model,
                sp_proc,
                corpus_dir.parent / ".hf_cache",
                device=device,
                max_paradigms=args.blimp_paradigms,
                pairs_per_paradigm=args.blimp_pairs,
            )
            results["blimp"] = blimp
            print(f"[BLiMP] mean_acc={blimp['mean_acc']:.4f} ({len(blimp['paradigms'])} paradigms)", flush=True)
        if args.num_samples > 0:
            test_toks = np.load(corpus_dir / "test_tokens.npy")
            samples = generate_sample_replies(model, test_toks, sp_proc, device=device, num_samples=args.num_samples)
            results["samples"] = samples
            for idx_s, s in enumerate(samples, 1):
                print(f"\n--- Sample {idx_s} ---\nUser:  {s['user_prompt']}\nModel: {s['model_reply']}\nRef:   {s['reference_reply']}")

    if args.out_json:
        out_p = Path(args.out_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
