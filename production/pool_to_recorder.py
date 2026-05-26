"""Pool the 12-week backfill into a single mlflow recorder.

Combines every examples/mlruns/pred_2026-0[1-4]-*.pkl file (each is one week
of LGBM × 3-horizon ensemble) into one big DataFrame, then writes it as
pred.pkl inside a new recorder under the `rolling_v2_ensemble` experiment.
The backend's `get_latest_recorder_id("rolling_v2_ensemble")` will then point
at this recorder.
"""
from __future__ import annotations

import sys as _sys
import sysconfig as _sysconfig
_PURELIB = _sysconfig.get_paths().get("purelib")
if _PURELIB and _PURELIB not in _sys.path[:1]:
    _sys.path.insert(0, _PURELIB)

import glob
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))


def main() -> int:
    import qlib
    from qlib.workflow import R

    qlib.init(
        provider_uri="~/.qlib/qlib_data/cn_data_bs",
        region="cn",
        exp_manager={
            "class": "MLflowExpManager",
            "module_path": "qlib.workflow.expm",
            "kwargs": {
                "uri": f"file:{REPO_ROOT / 'examples' / 'mlruns'}",
                "default_exp_name": "rolling_v2_ensemble",
            },
        },
    )

    # Pool all non-empty backfill weeks
    pattern = str(REPO_ROOT / "examples" / "mlruns" / "pred_2026-*.pkl")
    paths = sorted(glob.glob(pattern))
    print(f"Pooling {len(paths)} weeks:", file=sys.stderr)
    frames = []
    for p in paths:
        df = pd.read_pickle(p)
        if df.shape[0] == 0:
            print(f"  SKIP {Path(p).name} (0 rows)", file=sys.stderr)
            continue
        print(f"  + {Path(p).name} {df.shape}", file=sys.stderr)
        frames.append(df)

    pooled = pd.concat(frames, axis=0).sort_index()
    # Dedup any duplicate (date, instrument) — keep latest week's value
    pooled = pooled[~pooled.index.duplicated(keep="last")]
    pooled.index.names = ["datetime", "instrument"]
    print(f"\nPooled DataFrame: {pooled.shape}", file=sys.stderr)
    print(f"  dates: {pooled.index.get_level_values('datetime').nunique()}", file=sys.stderr)
    print(f"  syms: {pooled.index.get_level_values('instrument').nunique()}", file=sys.stderr)
    print(f"  cols: {list(pooled.columns)}", file=sys.stderr)
    print(f"  range: {pooled.index.get_level_values('datetime').min().date()} → "
          f"{pooled.index.get_level_values('datetime').max().date()}", file=sys.stderr)

    # Write as a new recorder under rolling_v2_ensemble
    with R.start(experiment_name="rolling_v2_ensemble",
                 recorder_name="ensemble_pooled_12wk_2026-01-30_to_2026-04-17") as run:
        R.save_objects(**{"pred.pkl": pooled})
        recorder_id = run.id
    print(f"\nWrote recorder: {recorder_id}", file=sys.stderr)
    print(recorder_id)  # to stdout for shell capture
    return 0


if __name__ == "__main__":
    sys.exit(main())
