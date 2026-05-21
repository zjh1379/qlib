"""Acceptance criteria validator.

Per spec §11, returns {passed: bool, details: {criterion: bool}}.

Performance thresholds (cost-adjusted):
  IC mean >= 0.030
  IR >= 2.5
  max drawdown <= 15% (i.e. >= -0.15)
  daily turnover <= 20%
  all 5 regime IRs > 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


THRESHOLDS = {
    "ic_mean": 0.030,
    "ir": 2.5,
    "max_drawdown": -0.15,
    "daily_turnover": 0.20,
}


def check_acceptance(scorecard: dict, regime_irs: dict[str, float]) -> dict:
    details = {
        "ic_mean": scorecard.get("ic_mean", 0) >= THRESHOLDS["ic_mean"],
        "ir": scorecard.get("ir", 0) >= THRESHOLDS["ir"],
        "max_drawdown": scorecard.get("max_drawdown", -1) >= THRESHOLDS["max_drawdown"],
        "daily_turnover": scorecard.get("daily_turnover", 1) <= THRESHOLDS["daily_turnover"],
        "regimes_all_positive": all(ir > 0 for ir in regime_irs.values()) if regime_irs else False,
    }
    return {
        "passed": all(details.values()),
        "details": details,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scorecard", required=True, help="path to scorecard JSON")
    parser.add_argument("--regimes", required=True, help="path to regime IRs JSON {seg_name: ir}")
    args = parser.parse_args()

    sc = json.loads(Path(args.scorecard).read_text())
    rg = json.loads(Path(args.regimes).read_text())
    result = check_acceptance(sc, rg)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
