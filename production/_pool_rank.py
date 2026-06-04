"""Pool the LambdaRank experiment (rolling_v2_ensemble_rank) LGBM recorders over
2021-2026 into one OOF pickle for net-of-cost comparison vs the factor/mse model.
Run: python -m production._pool_rank"""
import sys as _sys
import sysconfig as _sysconfig
_P = _sysconfig.get_paths().get("purelib")
if _P and _P not in _sys.path[:1]:
    _sys.path.insert(0, _P)
try:
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from datetime import date
from production.backfill_pool import pool_range


def main() -> None:
    out = pool_range(
        date(2021, 1, 1), date(2026, 1, 1),
        models=("lgbm",),
        config_path="production/configs/rolling_ensemble_rank.yaml",
        out_path="production/reports/oof_lgbmrank_2021_2026.pkl",
    )
    print(f"POOLED -> {out}")


if __name__ == "__main__":
    main()
