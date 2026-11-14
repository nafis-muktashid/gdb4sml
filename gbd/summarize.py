"""
Deterministic markdown tables from saved run JSON. Does not train or evaluate.

Reads artifacts/results_groups.json, artifacts/runs/<name>.json, and
artifacts/eval/<name>_blimp.json. Missing files are reported as missing.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "artifacts" / "runs"
EVALS = ROOT / "artifacts" / "eval"
GROUPS_PATH = ROOT / "artifacts" / "results_groups.json"
OUT_PATH = ROOT / "artifacts" / "results_summary.md"

BINS = ("rare_<10", "low_10_99", "mid_100_999", "freq_1000+")


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "missing"
    return f"{value:.{digits}f}"


def _mean(xs: list[float]) -> float | None:
    if not xs:
        return None
    return float(statistics.mean(xs))


def _sample_sd(xs: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    return float(statistics.stdev(xs))


def _same_sign(xs: list[float]) -> bool:
    return all(x > 0.0 for x in xs) or all(x < 0.0 for x in xs)


def _decisive(gaps: list[float]) -> str:
    """True when mean > max(0.01, 3 * sample sd) and every gap has the same strict sign."""
    if len(gaps) < 2:
        return "n/a"
    sd = _sample_sd(gaps)
    mean = _mean(gaps)
    if sd is None or mean is None:
        return "n/a"
    ok = mean > max(0.01, 3.0 * sd) and _same_sign(gaps)
    return "yes" if ok else "no"


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _reply_numbers(payload: dict | None) -> tuple[float | None, float | None]:
    if not payload:
        return None, None
    test = payload.get("test") if isinstance(payload.get("test"), dict) else {}
    best = payload.get("best_val") if isinstance(payload.get("best_val"), dict) else {}
    test_bpb = test.get("reply_bpb")
    val_bpb = best.get("reply_bpb")
    return (
        float(test_bpb) if isinstance(test_bpb, (int, float)) else None,
        float(val_bpb) if isinstance(val_bpb, (int, float)) else None,
    )


def _blimp_acc(name: str) -> float | None:
    payload = _load_json(EVALS / f"{name}_blimp.json")
    if not payload:
        return None
    direct = payload.get("mean_acc")
    if isinstance(direct, (int, float)):
        return float(direct)
    nested = payload.get("blimp")
    if isinstance(nested, dict) and isinstance(nested.get("mean_acc"), (int, float)):
        return float(nested["mean_acc"])
    return None


def _bin_fields(payload: dict | None) -> tuple[dict | None, dict | None]:
    if not payload or not isinstance(payload.get("test"), dict):
        return None, None
    test = payload["test"]
    nll = test.get("rare_token_nll")
    counts = test.get("rare_token_counts")
    if not isinstance(nll, dict) or not isinstance(counts, dict):
        return None, None
    return nll, counts


def _seed_label(index: int, seed_order: list[int], group_name: str, n_runs: int) -> str:
    if "baseline" in group_name or n_runs > len(seed_order):
        return "—"
    if index < len(seed_order):
        return str(seed_order[index])
    return "—"


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([head, sep, *body])


def build_summary(spec: dict) -> str:
    seed_order = [int(s) for s in spec.get("seed_order", [0, 1, 2])]
    groups: dict[str, list[str]] = spec["groups"]
    pairs: list[list[str]] = spec["pairs"]
    loaded: dict[str, dict | None] = {}
    lines = [
        "# Results summary",
        "",
        "Sample standard deviation uses the n−1 denominator. A gap is decisive when "
        "its mean is greater than max(0.01, 3 × sd) and every per-seed gap has the same strict sign. "
        "Fewer than two paired seeds is n/a. Missing files are not filled in.",
        "",
        "## Groups",
        "",
    ]
    group_test: dict[str, list[tuple[int, float]]] = {}
    group_blimp: dict[str, list[tuple[int, float]]] = {}

    for name, runs in groups.items():
        lines.append(f"### {name}")
        lines.append("")
        rows = []
        tests = []
        vals = []
        blimps = []
        for i, run in enumerate(runs):
            payload = loaded.setdefault(run, _load_json(RUNS / f"{run}.json"))
            test_bpb, val_bpb = _reply_numbers(payload)
            blimp = _blimp_acc(run)
            if payload is None:
                status = "missing"
            else:
                status = "ok"
                if test_bpb is not None:
                    tests.append(test_bpb)
                    group_test.setdefault(name, []).append((i, test_bpb))
                if val_bpb is not None:
                    vals.append(val_bpb)
                if blimp is not None:
                    blimps.append(blimp)
                    group_blimp.setdefault(name, []).append((i, blimp))
            rows.append([
                _seed_label(i, seed_order, name, len(runs)),
                run if payload is not None else f"{run} (missing)",
                status if payload is not None else "missing",
                _fmt(test_bpb),
                _fmt(val_bpb),
                _fmt(blimp),
            ])
        rows.append([
            "",
            "mean ± sample sd",
            str(len(tests)),
            f"{_fmt(_mean(tests))} ± {_fmt(_sample_sd(tests))}",
            f"{_fmt(_mean(vals))} ± {_fmt(_sample_sd(vals))}",
            f"{_fmt(_mean(blimps))} ± {_fmt(_sample_sd(blimps))}",
        ])
        lines.append(_md_table(
            ["seed", "run", "status", "test reply BPB", "best-val reply BPB", "BLiMP mean_acc"],
            rows,
        ))
        lines.append("")

    lines.append("## Paired test reply-BPB gaps")
    lines.append("")
    lines.append("Gap = test(A) − test(B) on seeds present in both groups, matched by list position.")
    lines.append("")
    gap_rows = []
    for pair in pairs:
        a_name, b_name = pair[0], pair[1]
        a_runs = groups.get(a_name, [])
        b_runs = groups.get(b_name, [])
        gaps = []
        per = []
        n = max(len(a_runs), len(b_runs))
        for i in range(n):
            a_run = a_runs[i] if i < len(a_runs) else None
            b_run = b_runs[i] if i < len(b_runs) else None
            if a_run is None or b_run is None:
                per.append(f"{i}: missing")
                continue
            a_payload = loaded.setdefault(a_run, _load_json(RUNS / f"{a_run}.json"))
            b_payload = loaded.setdefault(b_run, _load_json(RUNS / f"{b_run}.json"))
            a_test, _ = _reply_numbers(a_payload)
            b_test, _ = _reply_numbers(b_payload)
            if a_test is None or b_test is None:
                per.append(f"{i}: missing")
                continue
            gap = a_test - b_test
            gaps.append(gap)
            per.append(f"{i}: {_fmt(gap)}")
        gap_rows.append([
            a_name,
            b_name,
            ", ".join(per) if per else "missing",
            _fmt(_mean(gaps)),
            _fmt(_sample_sd(gaps)),
            _decisive(gaps),
        ])
    lines.append(_md_table(
        ["A", "B", "per-seed gap", "mean", "sample sd", "decisive"],
        gap_rows,
    ))
    lines.append("")

    lines.append("## Frequency-bin share of test-NLL change")
    lines.append("")
    lines.append(
        "For each matched seed, bin contribution = count_A × (NLL_A − NLL_B). "
        "Share = contribution / sum of contributions, pooled over seeds where both runs exist."
    )
    lines.append("")
    for pair in pairs:
        a_name, b_name = pair[0], pair[1]
        a_runs = groups.get(a_name, [])
        b_runs = groups.get(b_name, [])
        lines.append(f"### {a_name} vs {b_name}")
        lines.append("")
        pooled = {b: 0.0 for b in BINS}
        used = {b: 0 for b in BINS}
        missing_slots = []
        n_pair = max(len(a_runs), len(b_runs))
        for i in range(n_pair):
            if i >= len(a_runs) or i >= len(b_runs):
                missing_slots.append(str(i))
                continue
            a_payload = loaded.setdefault(a_runs[i], _load_json(RUNS / f"{a_runs[i]}.json"))
            b_payload = loaded.setdefault(b_runs[i], _load_json(RUNS / f"{b_runs[i]}.json"))
            a_nll, a_cnt = _bin_fields(a_payload)
            b_nll, _b_cnt = _bin_fields(b_payload)
            if a_nll is None or b_nll is None or a_cnt is None:
                missing_slots.append(str(i))
                continue
            for b in BINS:
                nll_a = a_nll.get(b)
                nll_b = b_nll.get(b)
                cnt = a_cnt.get(b)
                if not isinstance(nll_a, (int, float)) or not isinstance(nll_b, (int, float)) or not isinstance(cnt, (int, float)):
                    continue
                pooled[b] += float(cnt) * (float(nll_a) - float(nll_b))
                used[b] += 1
        total = sum(pooled.values())
        rows = []
        for b in BINS:
            if used[b] == 0 or total == 0.0:
                share = None
            else:
                share = pooled[b] / total
            rows.append([b, str(used[b]), _fmt(pooled[b]), _fmt(share)])
        if missing_slots:
            lines.append("Missing slots: " + ", ".join(missing_slots))
            lines.append("")
        lines.append(_md_table(["bin", "seeds used", "pooled count×ΔNLL", "share"], rows))
        lines.append("")

    lines.append("## Paired BLiMP gaps")
    lines.append("")
    lines.append("Gap = mean_acc(A) − mean_acc(B). The decisive column uses the same rule as reply BPB.")
    lines.append("")
    blimp_rows = []
    for pair in pairs:
        a_name, b_name = pair[0], pair[1]
        a_map = {i: v for i, v in group_blimp.get(a_name, [])}
        b_map = {i: v for i, v in group_blimp.get(b_name, [])}
        keys = sorted(set(a_map) & set(b_map))
        gaps = [a_map[i] - b_map[i] for i in keys]
        per = [f"{i}: {_fmt(a_map[i] - b_map[i])}" for i in keys] or ["missing"]
        blimp_rows.append([
            a_name,
            b_name,
            ", ".join(per),
            _fmt(_mean(gaps)),
            _fmt(_sample_sd(gaps)),
            _decisive(gaps),
        ])
    lines.append(_md_table(
        ["A", "B", "per-seed gap", "mean", "sample sd", "decisive"],
        blimp_rows,
    ))
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    spec = json.loads(GROUPS_PATH.read_text(encoding="utf-8"))
    text = build_summary(spec)
    OUT_PATH.write_text(text, encoding="utf-8")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
