"""CLI: evaluate qlib recorders + compare them.

Usage:
  python -m production.eval_recorder list
  python -m production.eval_recorder eval <recorder_id> [--top-k 30] [--bps 10] [--out reports/]
  python -m production.eval_recorder compare <recorder_a_id> <recorder_b_id> [--out reports/]

Wraps backend/app/evaluation/service.py — same in-process cache, same metrics.
Output:
  list   → prints a table to stdout
  eval   → prints scorecard summary + writes JSON + Markdown to --out (default: production/reports/)
  compare → prints side-by-side + writes JSON + Markdown to --out
"""
from __future__ import annotations

# Ensure conda-env qlib resolves before any in-repo qlib/ source (mirrors
# production/rolling_train.py).
import sys as _sys
import sysconfig as _sysconfig
_PURELIB = _sysconfig.get_paths().get("purelib")
if _PURELIB and _PURELIB not in _sys.path[:1]:
    _sys.path.insert(0, _PURELIB)

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Add the backend src dir so we can import the service in-process
_BACKEND = REPO_ROOT / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


def cmd_list(args: argparse.Namespace) -> int:
    from app.evaluation.service import list_recorders_with_summary

    summaries = list_recorders_with_summary()
    if not summaries:
        print("(no recorders found)", file=sys.stderr)
        return 1

    cols = ["recorder_id", "experiment", "run_name", "pred_start", "pred_end", "rows", "evaluated"]
    print("\t".join(cols))
    for s in summaries:
        row = [
            s.recorder_id[:12],
            s.experiment,
            s.run_name[:30],
            s.pred_start or "—",
            s.pred_end or "—",
            str(s.pred_rows or "—"),
            "yes" if s.has_eval else "no",
        ]
        print("\t".join(row))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from app.evaluation.service import evaluate_recorder

    print(f"Evaluating recorder {args.recorder_id[:12]}... (first call ~30-90s)", file=sys.stderr)
    result = evaluate_recorder(
        args.recorder_id, top_k=args.top_k, cost_bps=args.bps,
        force_refresh=args.force_refresh,
    )

    # Print summary to stdout
    sc = result.scorecard
    print()
    print(f"=== {result.experiment} / {result.recorder_id[:12]} ({result.run_name}) ===")
    print(f"window: {result.window_start} .. {result.window_end}  ({result.sample_size} rows)")
    print(f"TopK={result.top_k}  cost_bps={result.cost_bps}")
    print()
    print(f"  IC mean              {sc.ic_mean:+.4f}  {_pf(sc.ic_mean >= 0.030)}")
    print(f"  RIC mean             {sc.ric_mean:+.4f}")
    print(f"  ICIR                 {sc.icir:+.4f}  {_pf(sc.icir >= 0.40)}")
    print(f"  Top-Bottom Spread    {sc.top_bottom_spread_monthly:+.2f}%/mo  {_pf(sc.top_bottom_spread_monthly >= 1.5)}")
    print(f"  Annual Excess Ret    {sc.annual_excess_return:+.2%}  {_pf(sc.annual_excess_return >= 0.15)}")
    print(f"  IR (cost-adj)        {sc.ir:+.4f}  {_pf(sc.ir >= 2.5)}")
    print(f"  Max Drawdown         {sc.max_drawdown:+.2%}  {_pf(sc.max_drawdown >= -0.15)}")
    print(f"  Daily Turnover       {sc.daily_turnover:.2%}  {_pf(sc.daily_turnover <= 0.20)}")
    print()
    if result.regimes:
        print("Regime breakdown:")
        for r in result.regimes:
            print(f"  {r.label:<30}  IR={r.scorecard.ir:+.4f}  IC={r.scorecard.ic_mean:+.4f}  (n={r.sample_size})")
        print()
    print(f"Acceptance: {'PASS' if result.acceptance.passed else 'FAIL'}")
    for k, ok in result.acceptance.details.items():
        print(f"  [{'x' if ok else ' '}] {k}")

    # Write reports
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"{result.recorder_id[:12]}_{stamp}.json"
    md_path = out_dir / f"{result.recorder_id[:12]}_{stamp}.md"
    json_path.write_text(json.dumps(result.model_dump(), indent=2, default=str), encoding="utf-8")
    md_path.write_text(_render_eval_md(result), encoding="utf-8")
    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")
    return 0 if result.acceptance.passed else 1


def cmd_compare(args: argparse.Namespace) -> int:
    from app.evaluation.service import compare_recorders

    print(f"Comparing {args.a[:12]} vs {args.b[:12]}... (each first eval ~30-90s)", file=sys.stderr)
    cmp = compare_recorders(args.a, args.b, top_k=args.top_k, cost_bps=args.bps)

    print()
    print(f"=== Compare ===")
    print(f"A: {cmp.a.experiment} / {cmp.a.recorder_id[:12]} ({cmp.a.run_name})")
    print(f"B: {cmp.b.experiment} / {cmp.b.recorder_id[:12]} ({cmp.b.run_name})")
    print()
    print(f"  Metric             A           B           Δ (B-A)")
    print(f"  IC mean            {cmp.a.scorecard.ic_mean:+.4f}    {cmp.b.scorecard.ic_mean:+.4f}    {cmp.ic_delta:+.4f}")
    print(f"  IR                 {cmp.a.scorecard.ir:+.4f}    {cmp.b.scorecard.ir:+.4f}    {cmp.ir_delta:+.4f}")
    print(f"  Max DD             {cmp.a.scorecard.max_drawdown:+.2%}    {cmp.b.scorecard.max_drawdown:+.2%}")
    print()
    print(f"Paired t-test on daily IC:")
    print(f"  t = {cmp.paired_t_stat:.3f}")
    print(f"  p = {cmp.paired_p_value:.4f}")
    print(f"  Significant @ 5%: {cmp.significant_at_05}")
    print(f"  Verdict: {cmp.verdict}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"compare_{cmp.a.recorder_id[:8]}_vs_{cmp.b.recorder_id[:8]}_{stamp}.json"
    md_path = out_dir / f"compare_{cmp.a.recorder_id[:8]}_vs_{cmp.b.recorder_id[:8]}_{stamp}.md"
    json_path.write_text(json.dumps(cmp.model_dump(), indent=2, default=str), encoding="utf-8")
    md_path.write_text(_render_compare_md(cmp), encoding="utf-8")
    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


def _pf(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def _render_eval_md(result) -> str:
    """Render a single evaluation result as Markdown."""
    sc = result.scorecard
    lines = [
        f"# Evaluation: {result.experiment} / `{result.recorder_id[:12]}`",
        "",
        f"- **Run name**: {result.run_name}",
        f"- **Window**: {result.window_start} → {result.window_end}",
        f"- **Sample size**: {result.sample_size} (date, symbol) pairs",
        f"- **TopK**: {result.top_k}  |  **Cost**: {result.cost_bps} bps",
        f"- **Computed at**: {result.computed_at}",
        "",
        "## Scorecard",
        "",
        "| Metric | Value | Threshold | Pass |",
        "|---|---|---|---|",
        f"| IC mean | {sc.ic_mean:+.4f} | ≥ 0.030 | {_pf(sc.ic_mean >= 0.030)} |",
        f"| RIC mean | {sc.ric_mean:+.4f} | — | — |",
        f"| ICIR | {sc.icir:+.4f} | ≥ 0.40 | {_pf(sc.icir >= 0.40)} |",
        f"| Top-Bottom Spread (monthly %) | {sc.top_bottom_spread_monthly:+.2f}% | ≥ 1.5% | {_pf(sc.top_bottom_spread_monthly >= 1.5)} |",
        f"| Annual Excess Return | {sc.annual_excess_return:+.2%} | ≥ +15% | {_pf(sc.annual_excess_return >= 0.15)} |",
        f"| IR (cost-adjusted) | {sc.ir:+.4f} | ≥ 2.5 | {_pf(sc.ir >= 2.5)} |",
        f"| Max Drawdown | {sc.max_drawdown:+.2%} | ≥ -15% | {_pf(sc.max_drawdown >= -0.15)} |",
        f"| Daily Turnover | {sc.daily_turnover:.2%} | ≤ 20% | {_pf(sc.daily_turnover <= 0.20)} |",
        "",
        "## Regime Breakdown",
        "",
        "| Segment | Start | End | IR | IC | Samples |",
        "|---|---|---|---|---|---|",
    ]
    for r in result.regimes:
        lines.append(
            f"| {r.label} | {r.start} | {r.end} | {r.scorecard.ir:+.4f} | {r.scorecard.ic_mean:+.4f} | {r.sample_size} |"
        )
    lines.append("")
    lines.append("## Acceptance")
    lines.append("")
    lines.append(f"**Overall: {'PASS ✓' if result.acceptance.passed else 'FAIL ✗'}**")
    lines.append("")
    for k, ok in result.acceptance.details.items():
        lines.append(f"- [{'x' if ok else ' '}] {k}")
    return "\n".join(lines) + "\n"


def _render_compare_md(cmp) -> str:
    a, b = cmp.a, cmp.b
    return "\n".join([
        f"# Compare: `{a.recorder_id[:12]}` (A) vs `{b.recorder_id[:12]}` (B)",
        "",
        f"- A: {a.experiment} / {a.run_name} ({a.window_start}..{a.window_end})",
        f"- B: {b.experiment} / {b.run_name} ({b.window_start}..{b.window_end})",
        "",
        "## Side-by-side",
        "",
        "| Metric | A | B | Δ (B-A) |",
        "|---|---|---|---|",
        f"| IC mean | {a.scorecard.ic_mean:+.4f} | {b.scorecard.ic_mean:+.4f} | {cmp.ic_delta:+.4f} |",
        f"| RIC mean | {a.scorecard.ric_mean:+.4f} | {b.scorecard.ric_mean:+.4f} | {b.scorecard.ric_mean - a.scorecard.ric_mean:+.4f} |",
        f"| ICIR | {a.scorecard.icir:+.4f} | {b.scorecard.icir:+.4f} | {b.scorecard.icir - a.scorecard.icir:+.4f} |",
        f"| IR | {a.scorecard.ir:+.4f} | {b.scorecard.ir:+.4f} | {cmp.ir_delta:+.4f} |",
        f"| Max Drawdown | {a.scorecard.max_drawdown:+.2%} | {b.scorecard.max_drawdown:+.2%} | — |",
        f"| Daily Turnover | {a.scorecard.daily_turnover:.2%} | {b.scorecard.daily_turnover:.2%} | — |",
        "",
        "## Paired t-test (daily IC)",
        "",
        f"- **t** = {cmp.paired_t_stat:.4f}",
        f"- **p** = {cmp.paired_p_value:.4f}",
        f"- Significant @ 5% : **{cmp.significant_at_05}**",
        f"- **Verdict**: {cmp.verdict}",
        "",
    ]) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate qlib recorders.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List all recorders.")
    p_list.set_defaults(func=cmd_list)

    p_eval = sub.add_parser("eval", help="Evaluate a single recorder.")
    p_eval.add_argument("recorder_id")
    p_eval.add_argument("--top-k", type=int, default=30)
    p_eval.add_argument("--bps", type=float, default=10.0)
    p_eval.add_argument("--out", type=str, default=str(REPO_ROOT / "production" / "reports"))
    p_eval.add_argument("--force-refresh", action="store_true")
    p_eval.set_defaults(func=cmd_eval)

    p_cmp = sub.add_parser("compare", help="Compare two recorders.")
    p_cmp.add_argument("a", help="recorder_id of baseline A")
    p_cmp.add_argument("b", help="recorder_id of challenger B")
    p_cmp.add_argument("--top-k", type=int, default=30)
    p_cmp.add_argument("--bps", type=float, default=10.0)
    p_cmp.add_argument("--out", type=str, default=str(REPO_ROOT / "production" / "reports"))
    p_cmp.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
