# Prediction UX Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make multi-horizon (1d/5d/20d) predictions visible in the UI as calibrated expected returns + percentile ranks, with predictions automatically refreshed after data updates (event-driven daily inference subprocess).

**Architecture:** New `production/daily_inference.py` subprocess loads models from weekly mlflow recorders, runs inference on missing dates, applies per-horizon isotonic calibration, appends to pooled recorder. Triggered by data refresh job success callback. Backend schema gains per-horizon prediction data; frontend renders single-table 3-column mini-bars + K-line future markers.

**Tech Stack:** Python 3.10 + qlib + mlflow + scikit-learn (IsotonicRegression) + FastAPI + APScheduler · React 18 + TanStack Query + lightweight-charts + Vitest

**Reference Spec:** `docs/superpowers/specs/2026-05-28-prediction-ux-redesign-design.md`

---

## Phase A — Backend Core (calibration + inference)

### Task 1: Calibration module (`production/calibration.py`)

**Files:**
- Create: `production/calibration.py`
- Test: `tests/test_calibration.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_calibration.py`:
```python
import numpy as np
import pandas as pd
import pytest
from production.calibration import fit_calibration, apply_calibration, load_calibration, save_calibration


def _toy_pred_df(seed=0):
    rng = np.random.default_rng(seed)
    n_dates, n_inst = 20, 50
    dates = pd.date_range("2025-01-01", periods=n_dates, freq="B")
    idx = pd.MultiIndex.from_product([dates, [f"S{i:03d}" for i in range(n_inst)]],
                                      names=["datetime", "instrument"])
    data = {
        "lgbm_5d": rng.normal(size=len(idx)),
        "alstm_5d": rng.normal(size=len(idx)),
        "tra_5d": rng.normal(size=len(idx)),
        "lgbm_1d": rng.normal(size=len(idx)),
        "alstm_1d": rng.normal(size=len(idx)),
        "tra_1d": rng.normal(size=len(idx)),
    }
    return pd.DataFrame(data, index=idx)


def _toy_label_df(pred_df, signal=0.5):
    """Label correlated with rank-avg of pred cols (monotonic relationship)."""
    rng = np.random.default_rng(1)
    out = {}
    for h in ("1d", "5d"):
        cols = [c for c in pred_df.columns if c.endswith(f"_{h}")]
        ranks = pred_df[cols].groupby(level="datetime").rank(ascending=False, method="min")
        comp = -ranks.mean(axis=1)
        # label = comp * signal + noise
        out[f"label_{h}"] = comp * signal + rng.normal(scale=0.5, size=len(comp))
    return pd.DataFrame(out, index=pred_df.index)


def test_fit_calibration_returns_dict_keyed_by_horizon():
    pred = _toy_pred_df()
    label = _toy_label_df(pred)
    cal = fit_calibration(pred, label, horizons=("1d", "5d"))
    assert set(cal.keys()) == {"1d", "5d"}


def test_isotonic_is_monotonic_increasing():
    pred = _toy_pred_df()
    label = _toy_label_df(pred, signal=1.0)
    cal = fit_calibration(pred, label, horizons=("5d",))
    iso = cal["5d"]
    xs = np.linspace(-100, 100, 50)
    ys = iso.predict(xs)
    assert (np.diff(ys) >= -1e-9).all(), "isotonic must be non-decreasing"


def test_apply_calibration_returns_series_with_same_index():
    pred = _toy_pred_df()
    label = _toy_label_df(pred)
    cal = fit_calibration(pred, label, horizons=("5d",))
    cols = [c for c in pred.columns if c.endswith("_5d")]
    ranks = pred[cols].groupby(level="datetime").rank(ascending=False, method="min")
    comp = -ranks.mean(axis=1)
    out = apply_calibration(comp, cal["5d"])
    assert isinstance(out, pd.Series)
    assert out.index.equals(comp.index)
    assert out.notna().all()


def test_clip_out_of_bounds():
    pred = _toy_pred_df()
    label = _toy_label_df(pred)
    cal = fit_calibration(pred, label, horizons=("5d",))
    # x = 9999 should clip, not raise
    out = apply_calibration(pd.Series([9999.0, -9999.0]), cal["5d"])
    assert out.notna().all()
    assert np.isfinite(out).all()


def test_small_sample_skipped(caplog):
    pred = _toy_pred_df().head(50)  # < 100 rows after dropna
    label = _toy_label_df(pred)
    cal = fit_calibration(pred, label, horizons=("5d",))
    assert "5d" not in cal


def test_save_load_round_trip(tmp_path):
    pred = _toy_pred_df()
    label = _toy_label_df(pred)
    cal = fit_calibration(pred, label, horizons=("1d", "5d"))
    p = tmp_path / "cal.pkl"
    save_calibration(cal, p, meta={"trained_at": "2026-05-22"})
    loaded = load_calibration(p)
    assert set(loaded["maps"].keys()) == {"1d", "5d"}
    assert loaded["meta"]["trained_at"] == "2026-05-22"


def test_apply_calibration_handles_all_nan_input():
    cal = fit_calibration(_toy_pred_df(), _toy_label_df(_toy_pred_df()), horizons=("5d",))
    out = apply_calibration(pd.Series([np.nan, np.nan, np.nan]), cal["5d"])
    assert out.isna().all()
```

- [ ] **Step 2: Run tests to confirm they fail**

Run from `E:/Projects/qlib`:
```
F:/Tools/Anaconda/envs/qlib/python.exe -m pytest tests/test_calibration.py -v
```
Expected: 7 FAIL with `ModuleNotFoundError: No module named 'production.calibration'`

- [ ] **Step 3: Implement `production/calibration.py`**

```python
"""Isotonic regression calibration for model scores -> expected returns.

Per the design spec, each horizon (1d/5d/20d) has a separate isotonic map
fitted on the validation slice's (composite_score, realized_return) pairs.

composite_score := -rank_avg over that horizon's model columns.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

log = logging.getLogger(__name__)

MIN_SAMPLES = 100


def _composite_score(pred_df: pd.DataFrame, horizon: str) -> pd.Series:
    cols = [c for c in pred_df.columns if c.endswith(f"_{horizon}")]
    if not cols:
        return pd.Series(dtype=float, index=pred_df.index)
    ranks = pred_df[cols].groupby(level="datetime").rank(ascending=False, method="min")
    return -ranks.mean(axis=1, skipna=True)


def fit_calibration(
    pred_df: pd.DataFrame,
    label_df: pd.DataFrame,
    horizons: Iterable[str] = ("1d", "5d", "20d"),
) -> dict[str, IsotonicRegression]:
    """Fit one IsotonicRegression per horizon.

    pred_df: MultiIndex (datetime, instrument), columns include <model>_<horizon>
    label_df: same MultiIndex, columns include label_<horizon>
    Returns dict horizon -> fitted IsotonicRegression
    """
    out: dict[str, IsotonicRegression] = {}
    for h in horizons:
        label_col = f"label_{h}"
        if label_col not in label_df.columns:
            log.warning("calibration_skip horizon=%s reason=label_missing", h)
            continue
        x = _composite_score(pred_df, h)
        y = label_df[label_col]
        df = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
        if len(df) < MIN_SAMPLES:
            log.warning(
                "calibration_skip horizon=%s samples=%d threshold=%d",
                h, len(df), MIN_SAMPLES,
            )
            continue
        iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
        iso.fit(df["x"].values, df["y"].values)
        out[h] = iso
        log.info("calibration_fit horizon=%s samples=%d", h, len(df))
    return out


def apply_calibration(
    composite_scores: pd.Series,
    iso: IsotonicRegression,
) -> pd.Series:
    """Apply fitted isotonic to composite scores. NaN inputs -> NaN output."""
    if composite_scores.empty:
        return composite_scores
    mask = composite_scores.notna()
    out = pd.Series(np.nan, index=composite_scores.index, dtype=float)
    if mask.any():
        out.loc[mask] = iso.predict(composite_scores.loc[mask].values)
    return out


def save_calibration(
    cal: dict[str, IsotonicRegression],
    path: Path | str,
    meta: dict | None = None,
) -> None:
    payload = {"maps": cal, "meta": meta or {}}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        pickle.dump(payload, f)


def load_calibration(path: Path | str) -> dict:
    """Returns {'maps': {'1d': iso, ...}, 'meta': {...}} or empty dict if missing."""
    p = Path(path)
    if not p.exists():
        return {"maps": {}, "meta": {}}
    with p.open("rb") as f:
        return pickle.load(f)
```

- [ ] **Step 4: Run tests to verify pass**

```
F:/Tools/Anaconda/envs/qlib/python.exe -m pytest tests/test_calibration.py -v
```
Expected: 7 PASS

- [ ] **Step 5: Commit**

```
git add production/calibration.py tests/test_calibration.py
git commit -m "feat(calibration): isotonic regression module per horizon

Fits one IsotonicRegression per horizon on (composite_score, realized_return)
pairs from validation data. composite_score = -rank_avg of horizon's model
cols. Handles clip out-of-bounds, NaN inputs, small-sample fail-soft.
save/load round-trips via pickle."
```

---

### Task 2: Backfill calibration script (`production/backfill_calibration.py`)

**Files:**
- Create: `production/backfill_calibration.py`
- Test: `tests/test_backfill_calibration.py`

- [ ] **Step 1: Write smoke test**

`tests/test_backfill_calibration.py`:
```python
"""Smoke test for backfill_calibration script using fixture data.

We don't try to reload real mlflow recorders here; the dataframe-level
logic is tested in test_calibration.py. This test verifies the orchestration
function _fit_and_save runs end-to-end on a fake pred/label pair.
"""
import pickle
import numpy as np
import pandas as pd
import pytest
from production.backfill_calibration import _fit_and_save


def _fixture_pred_label(seed=0):
    rng = np.random.default_rng(seed)
    n_dates, n_inst = 60, 30  # bigger than MIN_SAMPLES
    dates = pd.date_range("2025-01-01", periods=n_dates, freq="B")
    instruments = [f"S{i:03d}" for i in range(n_inst)]
    idx = pd.MultiIndex.from_product([dates, instruments],
                                      names=["datetime", "instrument"])
    pred = pd.DataFrame({
        "lgbm_1d": rng.normal(size=len(idx)),
        "alstm_1d": rng.normal(size=len(idx)),
        "tra_1d": rng.normal(size=len(idx)),
        "lgbm_5d": rng.normal(size=len(idx)),
        "alstm_5d": rng.normal(size=len(idx)),
        "tra_5d": rng.normal(size=len(idx)),
    }, index=idx)
    label = pd.DataFrame({
        "label_1d": rng.normal(size=len(idx)),
        "label_5d": rng.normal(size=len(idx)),
    }, index=idx)
    return pred, label


def test_fit_and_save_writes_pickle(tmp_path):
    pred, label = _fixture_pred_label()
    out_path = tmp_path / "latest_calibration.pkl"
    _fit_and_save(pred, label, out_path, trained_at="2026-05-22")
    assert out_path.exists()
    with out_path.open("rb") as f:
        payload = pickle.load(f)
    assert set(payload["maps"].keys()) == {"1d", "5d"}
    assert payload["meta"]["trained_at"] == "2026-05-22"
```

- [ ] **Step 2: Run test, expect FAIL**

```
F:/Tools/Anaconda/envs/qlib/python.exe -m pytest tests/test_backfill_calibration.py -v
```
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: Implement `production/backfill_calibration.py`**

```python
"""One-off backfill: reload validation predictions + labels from latest
weekly recorders, fit isotonic calibration, save to
production/cache/latest_calibration.pkl.

Run once after deploying the calibration module so daily_inference and the
backend have a populated calibration file. Idempotent — overwrites
latest_calibration.pkl + writes a dated backup.

Usage:
  python -m production.backfill_calibration
  python -m production.backfill_calibration --end-date 2026-05-22
"""
from __future__ import annotations

import argparse
import logging
import sys as _sys
import sysconfig as _sysconfig
from datetime import date, datetime
from pathlib import Path

_PURELIB = _sysconfig.get_paths().get("purelib")
if _PURELIB and _PURELIB not in _sys.path[:1]:
    _sys.path.insert(0, _PURELIB)

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.append(str(REPO_ROOT))

from production.calibration import fit_calibration, save_calibration

log = logging.getLogger("backfill_calibration")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

CACHE_DIR = REPO_ROOT / "production" / "cache"
LATEST_PATH = CACHE_DIR / "latest_calibration.pkl"


def _fit_and_save(pred_df: pd.DataFrame, label_df: pd.DataFrame,
                  out_path: Path, trained_at: str) -> None:
    cal = fit_calibration(pred_df, label_df)
    save_calibration(cal, out_path, meta={"trained_at": trained_at,
                                          "saved_at": datetime.utcnow().isoformat()})
    log.info("saved calibration to %s horizons=%s", out_path, list(cal.keys()))


def _load_valid_slice_from_recorders(end_date: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """For each (model, horizon) load the matching <model>_<horizon>_<end_date>
    recorder, pull its 'pred.pkl' artifact for the validation slice (the rows
    BEFORE the test slice). Build pred_df (9 cols) + label_df (3 cols).

    Labels come from the handler's 'label' field — we reload one handler per
    horizon and extract the realized label series.
    """
    import qlib
    from qlib.workflow import R
    from production.rolling_train import load_config, init_qlib

    cfg = load_config(REPO_ROOT / "production/configs/rolling_ensemble.yaml")
    init_qlib(cfg)

    end_str = end_date.isoformat()
    exp = R.get_exp(experiment_name=cfg.experiment_name)
    recs = exp.list_recorders()
    if isinstance(recs, dict):
        recs = list(recs.values())

    pred_cols: dict[str, pd.Series] = {}
    label_cols: dict[str, pd.Series] = {}

    for model_id in ("lgbm", "alstm", "tra"):
        for h in ("1d", "5d", "20d"):
            target = f"{model_id}_{h}_{end_str}"
            matched = [
                r for r in recs
                if _recorder_name(r) == target
            ]
            if not matched:
                log.warning("recorder_missing %s", target)
                continue
            rec = matched[0]
            try:
                # Each weekly recorder saves pred.pkl for the test slice only.
                # For calibration we want the VALIDATION slice instead.
                # Convention: rolling_train stores valid predictions as
                # 'valid_pred.pkl' (added in Task 3). For backfill on recorders
                # that predate Task 3, we fall back to 'pred.pkl' (last week's
                # test set) — imperfect but acceptable for initial backfill.
                try:
                    pred = rec.load_object("valid_pred.pkl")
                except Exception:
                    pred = rec.load_object("pred.pkl")
                    log.warning("recorder_lacks_valid_pred falling back to pred.pkl %s", target)
            except Exception as exc:
                log.warning("load_failed %s: %s", target, exc)
                continue

            if isinstance(pred, pd.DataFrame):
                pred = pred["score"] if "score" in pred.columns else pred.iloc[:, 0]
            col = f"{model_id}_{h}"
            pred_cols[col] = pred.rename(col)

            if f"label_{h}" not in label_cols:
                # Load label series once per horizon
                try:
                    lab = rec.load_object("valid_label.pkl")
                except Exception:
                    try:
                        lab = rec.load_object("label.pkl")
                    except Exception:
                        log.warning("label_missing for %s", target)
                        continue
                if isinstance(lab, pd.DataFrame):
                    lab = lab.iloc[:, 0]
                label_cols[f"label_{h}"] = lab.rename(f"label_{h}")

    pred_df = pd.concat(list(pred_cols.values()), axis=1).sort_index() if pred_cols else pd.DataFrame()
    label_df = pd.concat(list(label_cols.values()), axis=1).sort_index() if label_cols else pd.DataFrame()
    return pred_df, label_df


def _recorder_name(rec) -> str:
    info = rec.info if hasattr(rec, "info") else {}
    name = info.get("name") if isinstance(info, dict) else getattr(rec, "name", "")
    if name:
        return name
    try:
        return rec.client.get_run(rec.id).data.tags.get("mlflow.runName", "")
    except Exception:
        return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--end-date", default=None, help="YYYY-MM-DD; default = latest")
    args = parser.parse_args()

    if args.end_date:
        end = date.fromisoformat(args.end_date)
    else:
        # Find latest weekly retrain end_date
        path = REPO_ROOT / "production" / "reports" / "latest_v10_recorder.txt"
        if path.exists():
            # The recorder name encodes end_date; for backfill use the most
            # recent weekly retrain we can identify from filesystem.
            end = date.fromisoformat("2026-05-22")  # current latest
        else:
            end = date.today()

    pred_df, label_df = _load_valid_slice_from_recorders(end)
    if pred_df.empty or label_df.empty:
        log.error("no_data_loaded — aborting")
        return 1

    out = LATEST_PATH
    _fit_and_save(pred_df, label_df, out, trained_at=end.isoformat())

    # Also write a dated backup
    backup = CACHE_DIR / f"calibration_{end.isoformat()}.pkl"
    _fit_and_save(pred_df, label_df, backup, trained_at=end.isoformat())

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run smoke test, verify pass**

```
F:/Tools/Anaconda/envs/qlib/python.exe -m pytest tests/test_backfill_calibration.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```
git add production/backfill_calibration.py tests/test_backfill_calibration.py
git commit -m "feat(calibration): backfill script for existing recorders

Reads latest weekly <model>_<horizon>_<date> recorders, pulls validation
slice pred + label, fits per-horizon isotonic, writes
production/cache/latest_calibration.pkl + dated backup. Falls back to
pred.pkl if valid_pred.pkl missing (pre-Task 3 recorders)."
```

---

### Task 3: Save validation predictions + labels + handler_config during training

**Goal:** Future training runs save the extra artifacts daily_inference + calibration need.

**Files:**
- Modify: `production/train_lgbm.py` (or wherever LightGBM training lives — check `production/rolling_train.py:run_once`)
- Modify: `production/train_alstm.py`
- Modify: `production/train_tra.py`

- [ ] **Step 1: Locate the per-model training entry points**

Run:
```
grep -n "save_objects\|R.save\|recorder.save" production/train_*.py production/rolling_train.py
```
Note line numbers where `pred.pkl` is currently saved per (model, horizon). Each save site is the place to add three new artifacts:
- `valid_pred.pkl` — model's predictions on validation slice
- `valid_label.pkl` — realized labels on validation slice
- `handler_config.pkl` — the handler init dict so daily_inference can rebuild features

- [ ] **Step 2: Add helper `_save_extra_artifacts` in each train_* module**

In `production/train_lgbm.py`, just before the existing `recorder.save_objects(**{"pred.pkl": ...})` call, add:
```python
def _save_extra_artifacts(recorder, model, dataset, handler_cfg):
    """Save valid_pred, valid_label, handler_config for daily inference + calibration."""
    # Validation slice predictions
    try:
        valid_pred = model.predict(dataset, segment="valid")
        if isinstance(valid_pred, pd.DataFrame):
            valid_pred = valid_pred["score"] if "score" in valid_pred.columns else valid_pred.iloc[:, 0]
        recorder.save_objects(**{"valid_pred.pkl": valid_pred})
    except Exception as exc:
        log.warning("valid_pred_save_failed: %s", exc)

    # Validation slice realized labels (from handler)
    try:
        # qlib's DatasetH.prepare("valid", col_set="label") returns the label DataFrame
        valid_label = dataset.prepare("valid", col_set="label", data_key="raw")
        if isinstance(valid_label, pd.DataFrame):
            valid_label = valid_label.iloc[:, 0]
        recorder.save_objects(**{"valid_label.pkl": valid_label})
    except Exception as exc:
        log.warning("valid_label_save_failed: %s", exc)

    # Handler init config — daily_inference needs this to rebuild features for new dates
    try:
        recorder.save_objects(**{"handler_config.pkl": handler_cfg})
    except Exception as exc:
        log.warning("handler_config_save_failed: %s", exc)
```

Add to `train_lgbm.py`'s main flow, right before `recorder.save_objects(**{"pred.pkl": pred})`:
```python
_save_extra_artifacts(recorder, model, dataset, handler_cfg)
```

Repeat in `train_alstm.py` and `train_tra.py`.

- [ ] **Step 3: Manual test — re-run smoke training once + verify artifacts**

```
F:/Tools/Anaconda/envs/qlib/python.exe -m production.rolling_train run-once --end-date 2026-05-22 --config production/configs/smoke_rolling.yaml --only-models lgbm --skip-pool
```
Then check the new recorder has the artifacts:
```
F:/Tools/Anaconda/envs/qlib/python.exe -c "
from qlib.workflow import R
import qlib
qlib.init(provider_uri='~/.qlib/qlib_data/cn_data_bs', region='cn',
          exp_manager={'class':'MLflowExpManager','module_path':'qlib.workflow.expm',
                       'kwargs':{'uri':'file:E:/Projects/qlib/examples/mlruns',
                                 'default_exp_name':'smoke'}})
exp = R.get_exp(experiment_name='smoke')
recs = exp.list_recorders()
rec = list(recs.values())[0] if isinstance(recs, dict) else recs[0]
print('artifacts:', rec.list_artifacts() if hasattr(rec,'list_artifacts') else 'n/a')
"
```
Expected output includes `valid_pred.pkl`, `valid_label.pkl`, `handler_config.pkl`.

- [ ] **Step 4: Commit**

```
git add production/train_lgbm.py production/train_alstm.py production/train_tra.py
git commit -m "feat(training): save valid_pred + valid_label + handler_config artifacts

daily_inference needs handler_config to rebuild features for new dates;
calibration needs valid_pred + valid_label to fit isotonic. All three
saved alongside existing pred.pkl at end of each per-(model,horizon)
training run. Fail-soft on save errors (log + continue)."
```

---

### Task 4: Run backfill_calibration once to populate latest_calibration.pkl

**Files:**
- Run: `production/backfill_calibration.py`
- Verify: `production/cache/latest_calibration.pkl`

- [ ] **Step 1: Pre-flight check**

```
ls E:/Projects/qlib/production/cache/ 2>/dev/null || mkdir -p E:/Projects/qlib/production/cache
```

- [ ] **Step 2: Run backfill**

```
F:/Tools/Anaconda/envs/qlib/python.exe -m production.backfill_calibration --end-date 2026-05-22
```
Expected log:
```
INFO calibration_fit horizon=1d samples=...
INFO calibration_fit horizon=5d samples=...
INFO calibration_fit horizon=20d samples=...
INFO saved calibration to E:/Projects/qlib/production/cache/latest_calibration.pkl horizons=['1d','5d','20d']
```

If the run reports `calibration_skip horizon=X samples=Y threshold=100`: the recorders predate Task 3's `valid_pred.pkl` and fell back to `pred.pkl` (only test slice ~5 days × 800 stocks = 4000 samples ≥ 100, OK). If still skipped, the recorder may have very few rows — note and continue.

- [ ] **Step 3: Verify the artifact**

```
F:/Tools/Anaconda/envs/qlib/python.exe -c "
import pickle
with open('E:/Projects/qlib/production/cache/latest_calibration.pkl','rb') as f:
    p = pickle.load(f)
print('horizons:', list(p['maps'].keys()))
print('meta:', p['meta'])
"
```

- [ ] **Step 4: Commit**

```
git add production/cache/latest_calibration.pkl
git commit -m "data: backfill latest_calibration.pkl from 2026-05-22 recorders

Initial population so backend + daily_inference have something to apply.
Will be overwritten by every weekly retrain going forward."
```

---

### Task 5: Daily inference module (`production/daily_inference.py`)

**Files:**
- Create: `production/daily_inference.py`
- Test: `tests/test_daily_inference.py`

- [ ] **Step 1: Write unit tests for helper logic**

`tests/test_daily_inference.py`:
```python
"""Unit tests for daily_inference helper functions (no mlflow / qlib needed)."""
from datetime import date
import numpy as np
import pandas as pd
import pytest
from production.daily_inference import (
    _missing_dates,
    _group_by_handler_signature,
    _composite_and_calibrate,
    _handler_signature,
    _default_handler_cfg,
)


def test_missing_dates_returns_only_new():
    qlib_dates = pd.DatetimeIndex(pd.date_range("2026-05-20", periods=8, freq="B"))
    pred_dates = pd.DatetimeIndex(pd.date_range("2026-05-20", periods=5, freq="B"))
    out = _missing_dates(qlib_dates, pred_dates)
    assert out == list(pd.date_range("2026-05-27", periods=3, freq="B").date)


def test_missing_dates_empty_when_pred_caught_up():
    dates = pd.DatetimeIndex(pd.date_range("2026-05-20", periods=5, freq="B"))
    assert _missing_dates(dates, dates) == []


def test_handler_signature_distinguishes_alpha158_vs_alpha360():
    cfg_158 = {"class": "Alpha158", "kwargs": {"start_time": "2020-01-01"}}
    cfg_360 = {"class": "Alpha360_OpenH", "kwargs": {"start_time": "2020-01-01"}}
    assert _handler_signature(cfg_158) != _handler_signature(cfg_360)


def test_group_by_handler_signature_collapses_alpha360_models():
    loaded = {
        "lgbm":  {"1d": ("model_l1", {"class": "Alpha158", "kwargs": {}})},
        "alstm": {"1d": ("model_a1", {"class": "Alpha360", "kwargs": {}}),
                  "5d": ("model_a5", {"class": "Alpha360", "kwargs": {}})},
        "tra":   {"1d": ("model_t1", {"class": "Alpha360", "kwargs": {}})},
    }
    groups = _group_by_handler_signature(loaded)
    # Two groups: Alpha158 (1 entry) and Alpha360 (3 entries)
    assert len(groups) == 2
    sizes = sorted(len(g) for g in groups.values())
    assert sizes == [1, 3]


def test_composite_and_calibrate_returns_three_horizons():
    rng = np.random.default_rng(0)
    n = 100
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2026-05-25", periods=2, freq="B"), [f"S{i:03d}" for i in range(50)]],
        names=["datetime", "instrument"],
    )
    raw = pd.DataFrame({
        "lgbm_1d": rng.normal(size=len(idx)),
        "alstm_1d": rng.normal(size=len(idx)),
        "tra_1d": rng.normal(size=len(idx)),
        "lgbm_5d": rng.normal(size=len(idx)),
        "alstm_5d": rng.normal(size=len(idx)),
        "tra_5d": rng.normal(size=len(idx)),
        "lgbm_20d": rng.normal(size=len(idx)),
        "alstm_20d": rng.normal(size=len(idx)),
        "tra_20d": rng.normal(size=len(idx)),
    }, index=idx)
    cal = {}  # no calibration -> still computes score+consensus
    out = _composite_and_calibrate(raw, cal)
    assert "score" in out.columns
    assert "consensus" in out.columns
    # score = -rank_avg of 1d+5d cols (v9 convention, exclude 20d)
    assert out["score"].notna().all()


def test_default_handler_cfg_returns_alpha360_for_alstm_tra():
    assert _default_handler_cfg("alstm")["class"] in ("Alpha360_OpenH", "Alpha360")
    assert _default_handler_cfg("tra")["class"] in ("Alpha360_OpenH", "Alpha360")
    assert _default_handler_cfg("lgbm")["class"] in ("Alpha158", "Alpha158_OpenH")
```

- [ ] **Step 2: Run tests, expect FAIL with ModuleNotFoundError**

```
F:/Tools/Anaconda/envs/qlib/python.exe -m pytest tests/test_daily_inference.py -v
```

- [ ] **Step 3: Implement `production/daily_inference.py`**

```python
"""Event-driven daily inference: load latest trained models, run on missing
dates, apply calibration, append to pooled ensemble recorder.

Triggered automatically by data refresh job's on_success callback. Can also
be invoked manually:

  python -m production.daily_inference
  python -m production.daily_inference --end-date 2026-05-27
  python -m production.daily_inference --force

Architecture:
  1. Locate the latest <model>_<horizon>_<date> weekly recorders (9 total)
  2. Load each model + its handler_config (or default)
  3. Find the pooled ensemble recorder (latest in experiment)
  4. missing_dates = qlib_calendar - pred.pkl's dates
  5. Group by handler signature (so Alpha360 models share feature build)
  6. For each group: build features once, run each model.predict()
  7. Apply latest_calibration.pkl per horizon
  8. Compute composite score + consensus, append rows to pred.pkl
  9. POST to backend cache invalidate endpoint
"""
from __future__ import annotations

import sys as _sys
import sysconfig as _sysconfig
_PURELIB = _sysconfig.get_paths().get("purelib")
if _PURELIB and _PURELIB not in _sys.path[:1]:
    _sys.path.insert(0, _PURELIB)

import argparse
import json
import logging
import time
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.append(str(REPO_ROOT))

from production.calibration import apply_calibration, load_calibration

log = logging.getLogger("daily_inference")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")

HORIZONS = ("1d", "5d", "20d")
MODELS = ("lgbm", "alstm", "tra")
CACHE_PATH = REPO_ROOT / "production" / "cache" / "latest_calibration.pkl"


# ---- helpers ---------------------------------------------------------------

def _recorder_name(rec) -> str:
    info = rec.info if hasattr(rec, "info") else {}
    name = info.get("name") if isinstance(info, dict) else getattr(rec, "name", "")
    if name:
        return name
    try:
        return rec.client.get_run(rec.id).data.tags.get("mlflow.runName", "")
    except Exception:
        return ""


def _handler_signature(cfg: dict) -> str:
    """Two configs are 'same handler' if class + kwargs (minus segment dates +
    instruments) match."""
    cls = cfg.get("class", "?")
    kw = {k: v for k, v in cfg.get("kwargs", {}).items()
          if k not in ("start_time", "end_time", "fit_start_time", "fit_end_time",
                       "instruments")}
    # processors as JSON-stable key
    return cls + ":" + json.dumps(kw, sort_keys=True, default=str)


def _default_handler_cfg(model_id: str) -> dict:
    """Fallback config when handler_config.pkl artifact is absent."""
    if model_id == "lgbm":
        return {
            "class": "Alpha158",
            "module_path": "qlib.contrib.data.handler",
            "kwargs": {},
        }
    # ALSTM + TRA both use Alpha360
    return {
        "class": "Alpha360_OpenH",
        "module_path": "custom_handler",
        "kwargs": {},
    }


def _missing_dates(qlib_dates, pred_dates) -> list[date]:
    """Set difference, returns sorted list of date objects."""
    q = {d.date() if hasattr(d, "date") else d for d in qlib_dates}
    p = {d.date() if hasattr(d, "date") else d for d in pred_dates}
    return sorted(q - p)


def _group_by_handler_signature(loaded: dict) -> dict[str, list]:
    """{model_id: {horizon: (model, cfg)}}  ->  {sig: [(mid, h, model, cfg), ...]}"""
    groups: dict[str, list] = {}
    for mid, hmap in loaded.items():
        for h, (model, cfg) in hmap.items():
            sig = _handler_signature(cfg)
            groups.setdefault(sig, []).append((mid, h, model, cfg))
    return groups


def _composite_and_calibrate(raw: pd.DataFrame, cal: dict) -> pd.DataFrame:
    """Given 9-col raw scores DataFrame, return enriched DataFrame with:
      - original 9 cols
      - composite_<h> (rank-avg per horizon)
      - expected_return_<h> (if cal[h] exists)
      - score (= -rank_avg of 1d+5d cols, v9 convention)
      - consensus (3-model directional agreement on the primary 5d)
    """
    df = raw.copy()
    for h in HORIZONS:
        cols = [c for c in df.columns if c.endswith(f"_{h}") and not c.startswith("expected_")]
        if not cols:
            continue
        ranks = df[cols].groupby(level="datetime").rank(ascending=False, method="min")
        comp = -ranks.mean(axis=1, skipna=True)
        df[f"composite_{h}"] = comp
        if h in cal.get("maps", {}):
            df[f"expected_return_{h}"] = apply_calibration(comp, cal["maps"][h])

    # Unified `score` = -rank_avg over 1d+5d cols (exclude 20d per v9 convention)
    score_cols = [c for c in df.columns
                  if (c.endswith("_1d") or c.endswith("_5d"))
                  and not (c.startswith("expected_") or c.startswith("composite_"))]
    if score_cols:
        ranks = df[score_cols].groupby(level="datetime").rank(ascending=False, method="min")
        df["score"] = -ranks.mean(axis=1, skipna=True)

    # Consensus on 5d direction (raw scores)
    five_cols = [c for c in df.columns
                 if c.endswith("_5d")
                 and not (c.startswith("expected_") or c.startswith("composite_"))]
    if five_cols:
        signs = df[five_cols].apply(lambda r: [1 if x > 0 else (-1 if x < 0 else 0)
                                                for x in r], axis=1, result_type="expand")
        df["consensus"] = signs.abs().sum(axis=1) / signs.shape[1]

    return df


# ---- main pipeline ---------------------------------------------------------

def _load_models(exp_name: str) -> dict:
    from qlib.workflow import R
    exp = R.get_exp(experiment_name=exp_name)
    recs = exp.list_recorders()
    if isinstance(recs, dict):
        recs = list(recs.values())

    out: dict = {}
    for mid in MODELS:
        out[mid] = {}
        for h in HORIZONS:
            prefix = f"{mid}_{h}_"
            matched = [r for r in recs if _recorder_name(r).startswith(prefix)]
            if not matched:
                log.warning("no_recorder model=%s horizon=%s", mid, h)
                continue
            latest = max(matched, key=lambda r: r.info.get("start_time", 0))
            try:
                model = latest.load_object("trained_model")
            except Exception as exc:
                log.warning("trained_model_load_failed model=%s horizon=%s: %s", mid, h, exc)
                continue
            try:
                cfg = latest.load_object("handler_config.pkl")
            except Exception:
                cfg = _default_handler_cfg(mid)
                log.warning("handler_config_missing model=%s horizon=%s fallback=default", mid, h)
            out[mid][h] = (model, cfg)
            log.info("loaded model=%s horizon=%s recorder=%s",
                     mid, h, _recorder_name(latest)[:50])
    return out


def _find_pooled_recorder(exp_name: str):
    """Latest ensemble pool recorder = the one whose name starts with 'ensemble_'."""
    from qlib.workflow import R
    exp = R.get_exp(experiment_name=exp_name)
    recs = exp.list_recorders()
    if isinstance(recs, dict):
        recs = list(recs.values())
    pooled = [r for r in recs if _recorder_name(r).startswith("ensemble_")]
    if not pooled:
        raise RuntimeError(f"no pooled recorder found in {exp_name}")
    return max(pooled, key=lambda r: r.info.get("start_time", 0))


def _infer_group(group, dates, instruments) -> dict[str, pd.Series]:
    """Build dataset once per handler group, predict each model in group."""
    from qlib.utils import init_instance_by_config
    from qlib.data.dataset import DatasetH

    cfg = dict(group[0][3])
    cfg["kwargs"] = dict(cfg.get("kwargs", {}))
    cfg["kwargs"].update(
        start_time=str(dates[0]), end_time=str(dates[-1]),
        instruments=instruments,
    )
    handler = init_instance_by_config(cfg)
    dataset = DatasetH(handler=handler,
                        segments={"test": (str(dates[0]), str(dates[-1]))})
    out: dict[str, pd.Series] = {}
    for mid, h, model, _ in group:
        try:
            pred = model.predict(dataset)
            if isinstance(pred, pd.DataFrame):
                pred = pred["score"] if "score" in pred.columns else pred.iloc[:, 0]
            out[f"{mid}_{h}"] = pred
        except Exception as exc:
            log.warning("predict_failed model=%s horizon=%s: %s", mid, h, exc)
    return out


def _post_invalidate_cache():
    """Best-effort: notify backend the candidates cache is stale."""
    import urllib.request
    url = "http://127.0.0.1:8000/api/internal/cache/invalidate"
    try:
        req = urllib.request.Request(url, method="POST")
        with urllib.request.urlopen(req, timeout=3) as r:
            log.info("cache_invalidate status=%d", r.status)
    except Exception as exc:
        log.warning("cache_invalidate_failed: %s — backend may be down", exc)


def run(end_date: date | None = None, force: bool = False,
        experiment: str = "rolling_v2_ensemble") -> int:
    import qlib
    from qlib.data import D
    from production.rolling_train import load_config, init_qlib

    cfg = load_config(REPO_ROOT / "production/configs/rolling_ensemble.yaml")
    init_qlib(cfg)

    pooled = _find_pooled_recorder(experiment)
    existing = pooled.load_object("pred.pkl")
    if not isinstance(existing, pd.DataFrame):
        existing = existing.to_frame(name="score") if hasattr(existing, "to_frame") else None
    pred_dates = set(existing.index.get_level_values("datetime").unique()) if existing is not None else set()

    # Determine target end_date from qlib calendar
    calendar = D.calendar(end_time=str(end_date) if end_date else None)
    if not len(calendar):
        log.error("empty_calendar")
        return 1
    qlib_latest = pd.Timestamp(calendar[-1])

    missing = _missing_dates(calendar[-30:], pred_dates) if not force else \
              [d.date() for d in calendar[-10:]]

    if not missing:
        log.info("no_missing_dates pred_latest=%s qlib_latest=%s",
                 max(pred_dates) if pred_dates else None, qlib_latest)
        return 0

    log.info("missing_dates count=%d range=%s..%s",
             len(missing), missing[0], missing[-1])

    loaded = _load_models(experiment)
    if not any(loaded.values()):
        log.error("no_models_loaded")
        return 1

    groups = _group_by_handler_signature(loaded)
    log.info("handler_groups count=%d sizes=%s",
             len(groups), [len(g) for g in groups.values()])

    instruments = sorted({inst for _, inst in existing.index})
    raw_scores: dict[str, pd.Series] = {}
    for sig, group in groups.items():
        log.info("group_start sig=%s models=%s",
                 sig[:60], [(m, h) for m, h, _, _ in group])
        t0 = time.time()
        out = _infer_group(group, missing, instruments)
        log.info("group_end elapsed=%.1fs cols=%s", time.time() - t0, list(out.keys()))
        raw_scores.update(out)

    if not raw_scores:
        log.error("no_predictions_produced")
        return 1

    raw_df = pd.concat(list(raw_scores.values()), axis=1).sort_index()
    raw_df = raw_df[~raw_df.index.duplicated(keep="last")]
    raw_df.index.names = ["datetime", "instrument"]

    cal = load_calibration(CACHE_PATH)
    enriched = _composite_and_calibrate(raw_df, cal)

    # Reindex to match existing columns (drop our extras that pool didn't have,
    # add NaN-filled cols for any pool col we don't produce).
    final_cols = list(existing.columns) if existing is not None else list(enriched.columns)
    for c in enriched.columns:
        if c not in final_cols:
            final_cols.append(c)
    enriched = enriched.reindex(columns=final_cols)

    combined = pd.concat([existing, enriched], axis=0).sort_index() \
               if existing is not None else enriched
    combined = combined[~combined.index.duplicated(keep="last")]

    pooled.save_objects(**{"pred.pkl": combined})
    log.info("appended new_rows=%d total_rows=%d new_dates=%d",
             len(enriched), len(combined), len(missing))

    _post_invalidate_cache()
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--end-date", default=None, help="YYYY-MM-DD; default = latest qlib date")
    parser.add_argument("--force", action="store_true",
                        help="Re-infer even if pred.pkl has the dates")
    parser.add_argument("--experiment", default="rolling_v2_ensemble")
    args = parser.parse_args()

    end = date.fromisoformat(args.end_date) if args.end_date else None
    rc = run(end_date=end, force=args.force, experiment=args.experiment)
    return rc


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run unit tests**

```
F:/Tools/Anaconda/envs/qlib/python.exe -m pytest tests/test_daily_inference.py -v
```
Expected: 6 PASS

- [ ] **Step 5: Manual smoke run**

```
F:/Tools/Anaconda/envs/qlib/python.exe -m production.daily_inference --end-date 2026-05-27
```
Expected: detects missing dates 5-25/5-26/5-27, loads 9 models, runs inference, writes back to recorder, logs `appended new_rows=...`.

If this fails on `valid_pred.pkl` missing or handler_config errors, that's expected for old recorders — note the fallback logs but continue. The fundamental pipeline should still produce a non-empty output.

- [ ] **Step 6: Commit**

```
git add production/daily_inference.py tests/test_daily_inference.py
git commit -m "feat(inference): production/daily_inference.py event-driven subprocess

Loads latest <model>_<horizon>_<date> weekly recorders, identifies missing
dates between qlib calendar and pooled pred.pkl, groups models by handler
signature (Alpha158 vs Alpha360) to share feature build, runs inference,
applies isotonic calibration, computes composite score + consensus,
appends to pooled recorder pred.pkl, POSTs cache invalidate."
```

---

### Task 6: Hook calibration fitting into `run_split._pool_from_recorders`

**Files:**
- Modify: `production/run_split.py` (the `_pool_from_recorders` function)

- [ ] **Step 1: Read current `_pool_from_recorders` end**

It currently builds `base` DataFrame, computes ranks/score/consensus/EWMA, writes pred.pkl. Add after writing pred.pkl:

```python
# === Refit calibration on the valid slices of the contributing recorders ===
try:
    from production.calibration import fit_calibration, save_calibration
    valid_pred_cols: dict[str, pd.Series] = {}
    valid_label_cols: dict[str, pd.Series] = {}
    for rec in recs:
        run_name = _recorder_name(rec)
        for mid in ("lgbm", "alstm", "tra"):
            for h in ("1d", "5d", "20d"):
                target = f"{mid}_{h}_{end_str}"
                if run_name != target:
                    continue
                col = f"{mid}_{h}"
                try:
                    vp = rec.load_object("valid_pred.pkl")
                    if isinstance(vp, pd.DataFrame):
                        vp = vp["score"] if "score" in vp.columns else vp.iloc[:, 0]
                    valid_pred_cols[col] = vp.rename(col)
                except Exception:
                    pass
                if f"label_{h}" not in valid_label_cols:
                    try:
                        vl = rec.load_object("valid_label.pkl")
                        if isinstance(vl, pd.DataFrame):
                            vl = vl.iloc[:, 0]
                        valid_label_cols[f"label_{h}"] = vl.rename(f"label_{h}")
                    except Exception:
                        pass

    if valid_pred_cols and valid_label_cols:
        vp_df = pd.concat(list(valid_pred_cols.values()), axis=1).sort_index()
        vl_df = pd.concat(list(valid_label_cols.values()), axis=1).sort_index()
        cal = fit_calibration(vp_df, vl_df)
        cache_path = REPO_ROOT / "production" / "cache" / "latest_calibration.pkl"
        save_calibration(cal, cache_path,
                         meta={"trained_at": end_str,
                               "saved_at": datetime.utcnow().isoformat()})
        _log.info("calibration_refit horizons=%s -> %s", list(cal.keys()), cache_path)
    else:
        _log.warning("calibration_skip no valid_pred/label artifacts found")
except Exception as exc:
    _log.warning("calibration_refit_failed: %s", exc)
```

Add a small helper to `run_split.py` at module level:

```python
def _recorder_name(rec) -> str:
    info = rec.info if hasattr(rec, "info") else {}
    name = info.get("name") if isinstance(info, dict) else getattr(rec, "name", "")
    if name:
        return name
    try:
        return rec.client.get_run(rec.id).data.tags.get("mlflow.runName", "")
    except Exception:
        return ""
```

Also add at top of file: `from datetime import datetime`.

- [ ] **Step 2: Verify by reading the modified file end-to-end**

Run:
```
F:/Tools/Anaconda/envs/qlib/python.exe -c "from production.run_split import _pool_from_recorders; print('ok')"
```
Expected: no import error.

- [ ] **Step 3: Commit**

```
git add production/run_split.py
git commit -m "feat(run_split): refit calibration at end of pooling

Every weekly retrain now refreshes production/cache/latest_calibration.pkl
using the valid_pred/valid_label artifacts from each weekly recorder.
Fail-soft if artifacts missing (logs warning, keeps existing cal file)."
```

---

## Phase B — Backend API

### Task 7: Inference router/service/schemas

**Files:**
- Create: `backend/app/inference/__init__.py`
- Create: `backend/app/inference/schemas.py`
- Create: `backend/app/inference/service.py`
- Create: `backend/app/inference/router.py`
- Modify: `backend/app/main.py` (register router)
- Test: `backend/tests/test_inference_router.py`

- [ ] **Step 1: Write failing router tests**

`backend/tests/test_inference_router.py`:
```python
from fastapi.testclient import TestClient
from app.main import create_app


def test_active_peek_returns_null_when_idle():
    client = TestClient(create_app())
    resp = client.get("/api/inference/active/peek")
    assert resp.status_code == 200
    assert resp.json() is None


def test_inference_status_returns_keys():
    client = TestClient(create_app())
    resp = client.get("/api/inference/status")
    assert resp.status_code == 200
    body = resp.json()
    assert "last_run_at" in body
    assert "last_success_at" in body
    assert "last_error" in body


def test_internal_cache_invalidate_localhost_only():
    client = TestClient(create_app())
    # TestClient uses localhost so this should succeed
    resp = client.post("/api/internal/cache/invalidate")
    assert resp.status_code == 200
    assert "cleared" in resp.json()


def test_run_now_returns_started_or_already_running():
    client = TestClient(create_app())
    resp = client.post("/api/inference/run-now")
    assert resp.status_code in (200, 202, 409)
    body = resp.json()
    assert body.get("status") in ("started", "already_running", "queued")
```

- [ ] **Step 2: Run, expect 404 NOT FOUND**

```
cd backend && F:/Tools/Anaconda/envs/qlib/python.exe -m pytest tests/test_inference_router.py -v
```

- [ ] **Step 3: Create schemas**

`backend/app/inference/__init__.py`:
```python
```
(empty)

`backend/app/inference/schemas.py`:
```python
from pydantic import BaseModel


class InferenceJob(BaseModel):
    job_id: str
    status: str  # "running" | "done" | "failed"
    started_at: str
    finished_at: str | None = None
    end_date: str | None = None
    error: str | None = None
    new_rows: int | None = None  # populated on success


class InferenceStatus(BaseModel):
    last_run_at: str | None = None
    last_success_at: str | None = None
    last_error: str | None = None
    is_running: bool = False


class TriggerResponse(BaseModel):
    status: str  # "started" | "already_running"
    job_id: str | None = None
```

- [ ] **Step 4: Create service**

`backend/app/inference/service.py`:
```python
"""In-memory inference job tracking + subprocess launch.

Mirrors backend/app/scheduling/service.py pattern: a module-level dict
holds job states, a lock prevents concurrent runs of the same kind.

Spawns production/daily_inference.py as a subprocess so model loading
doesn't bloat the API process.
"""
import logging
import subprocess
import sys
import threading
import uuid
from datetime import datetime, date
from pathlib import Path

from app.inference.schemas import InferenceJob, InferenceStatus, TriggerResponse

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]

_JOBS: dict[str, InferenceJob] = {}
_ACTIVE_JOB_ID: str | None = None
_LOCK = threading.Lock()
_LAST_RUN_AT: str | None = None
_LAST_SUCCESS_AT: str | None = None
_LAST_ERROR: str | None = None


def get_active_job() -> InferenceJob | None:
    if _ACTIVE_JOB_ID and _ACTIVE_JOB_ID in _JOBS:
        return _JOBS[_ACTIVE_JOB_ID]
    return None


def get_status() -> InferenceStatus:
    return InferenceStatus(
        last_run_at=_LAST_RUN_AT,
        last_success_at=_LAST_SUCCESS_AT,
        last_error=_LAST_ERROR,
        is_running=_ACTIVE_JOB_ID is not None,
    )


def get_job(job_id: str) -> InferenceJob | None:
    return _JOBS.get(job_id)


def trigger_inference(force: bool = False, end_date: date | None = None,
                     reason: str = "manual") -> TriggerResponse:
    """Start daily_inference subprocess if not already running."""
    global _ACTIVE_JOB_ID, _LAST_RUN_AT

    with _LOCK:
        if _ACTIVE_JOB_ID and _ACTIVE_JOB_ID in _JOBS \
           and _JOBS[_ACTIVE_JOB_ID].status == "running":
            return TriggerResponse(status="already_running",
                                    job_id=_ACTIVE_JOB_ID)

        job_id = uuid.uuid4().hex[:12]
        now = datetime.utcnow().isoformat()
        _JOBS[job_id] = InferenceJob(
            job_id=job_id, status="running", started_at=now,
            end_date=end_date.isoformat() if end_date else None,
        )
        _ACTIVE_JOB_ID = job_id
        _LAST_RUN_AT = now

    # Spawn outside the lock
    thread = threading.Thread(
        target=_run_subprocess, args=(job_id, end_date, force, reason), daemon=True,
    )
    thread.start()
    return TriggerResponse(status="started", job_id=job_id)


def _run_subprocess(job_id: str, end_date: date | None, force: bool, reason: str):
    global _ACTIVE_JOB_ID, _LAST_SUCCESS_AT, _LAST_ERROR

    cmd = [sys.executable, "-m", "production.daily_inference"]
    if end_date:
        cmd += ["--end-date", end_date.isoformat()]
    if force:
        cmd += ["--force"]

    log.info("inference_subprocess_start job_id=%s reason=%s cmd=%s",
             job_id, reason, " ".join(cmd))
    try:
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT),
                              capture_output=True, text=True, timeout=600)
        rc = proc.returncode
        log.info("inference_subprocess_end job_id=%s rc=%d", job_id, rc)
        # Best-effort parse "appended new_rows=N" from stdout/stderr
        new_rows = None
        for line in (proc.stderr or "").splitlines():
            if "appended new_rows=" in line:
                try:
                    new_rows = int(line.split("new_rows=")[1].split()[0])
                    break
                except Exception:
                    pass
        with _LOCK:
            job = _JOBS.get(job_id)
            if job:
                job.status = "done" if rc == 0 else "failed"
                job.finished_at = datetime.utcnow().isoformat()
                job.new_rows = new_rows
                if rc != 0:
                    job.error = (proc.stderr or "")[-2000:]
                    _LAST_ERROR = job.error
                else:
                    _LAST_SUCCESS_AT = job.finished_at
                    _LAST_ERROR = None
            _ACTIVE_JOB_ID = None
    except subprocess.TimeoutExpired:
        log.error("inference_subprocess_timeout job_id=%s", job_id)
        with _LOCK:
            job = _JOBS.get(job_id)
            if job:
                job.status = "failed"
                job.finished_at = datetime.utcnow().isoformat()
                job.error = "subprocess timed out after 600s"
            _LAST_ERROR = "timeout"
            _ACTIVE_JOB_ID = None
    except Exception as exc:
        log.exception("inference_subprocess_error job_id=%s: %s", job_id, exc)
        with _LOCK:
            job = _JOBS.get(job_id)
            if job:
                job.status = "failed"
                job.finished_at = datetime.utcnow().isoformat()
                job.error = str(exc)[-2000:]
            _LAST_ERROR = str(exc)
            _ACTIVE_JOB_ID = None
```

- [ ] **Step 5: Create router**

`backend/app/inference/router.py`:
```python
from fastapi import APIRouter, HTTPException, Request

from app.inference import service
from app.inference.schemas import InferenceJob, InferenceStatus, TriggerResponse

router = APIRouter(prefix="/api/inference", tags=["inference"])


@router.get("/active/peek", response_model=InferenceJob | None)
def active_peek():
    return service.get_active_job()


@router.get("/status", response_model=InferenceStatus)
def inference_status():
    return service.get_status()


@router.get("/jobs/{job_id}", response_model=InferenceJob)
def get_job(job_id: str):
    job = service.get_job(job_id)
    if not job:
        raise HTTPException(404, detail="job not found")
    return job


@router.post("/run-now", response_model=TriggerResponse)
def trigger(force: bool = False):
    return service.trigger_inference(force=force, reason="manual_ui")


# ---- internal: cache invalidate -------------------------------------------
internal_router = APIRouter(prefix="/api/internal", tags=["internal"])


@internal_router.post("/cache/invalidate")
def invalidate(request: Request):
    """Called by daily_inference subprocess after writing back to recorder.
    Localhost-only to avoid external cache flush."""
    client_host = request.client.host if request.client else None
    if client_host not in ("127.0.0.1", "localhost", "::1", "testclient"):
        raise HTTPException(403, detail="localhost only")
    from app.models.service import invalidate_candidates_cache
    cleared = invalidate_candidates_cache()
    return {"cleared": cleared}
```

- [ ] **Step 6: Register in `backend/app/main.py`**

Find existing `app.include_router(...)` calls and add:
```python
from app.inference.router import router as inference_router, internal_router
app.include_router(inference_router)
app.include_router(internal_router)
```

- [ ] **Step 7: Run tests, expect PASS**

```
cd backend && F:/Tools/Anaconda/envs/qlib/python.exe -m pytest tests/test_inference_router.py -v
```
Expected: 4 PASS.

- [ ] **Step 8: Commit**

```
git add backend/app/inference/ backend/app/main.py backend/tests/test_inference_router.py
git commit -m "feat(backend): inference router + service + schemas

POST /api/inference/run-now spawns daily_inference subprocess; tracks
job state in-memory like scheduling/evaluation modules. GET active/peek
+ status + jobs/{id} for UI polling. POST /api/internal/cache/invalidate
is localhost-only so the inference subprocess can notify the API to
drop the candidates lru_cache."
```

---

### Task 8: Refresh job triggers inference

**Files:**
- Modify: `backend/app/data/service.py`
- Test: `backend/tests/test_refresh_callback.py`

- [ ] **Step 1: Write failing test**

`backend/tests/test_refresh_callback.py`:
```python
"""Verify data refresh success triggers inference job."""
from unittest import mock

from app.data import service as data_service


def test_refresh_success_triggers_inference():
    with mock.patch("app.inference.service.trigger_inference") as mock_trigger:
        mock_trigger.return_value = mock.Mock(status="started", job_id="abc")
        data_service._on_refresh_success("job-xyz")
        mock_trigger.assert_called_once()
        kwargs = mock_trigger.call_args.kwargs
        assert kwargs.get("reason") == "data_refresh"
```

- [ ] **Step 2: Run, expect FAIL**

```
cd backend && F:/Tools/Anaconda/envs/qlib/python.exe -m pytest tests/test_refresh_callback.py -v
```

- [ ] **Step 3: Locate where refresh job marks itself "done"**

```
grep -n "status.*done\|RUNNING.*COMPLETED\|on_success" backend/app/data/service.py
```
Find the function that updates job state to terminal-success. Add a call to `_on_refresh_success` there.

- [ ] **Step 4: Add `_on_refresh_success` to `backend/app/data/service.py`**

Near the top of the file (after imports), add:
```python
def _on_refresh_success(job_id: str) -> None:
    """Triggered when a data refresh job completes successfully.

    Spawns daily_inference subprocess so predictions update to use the
    latest qlib bin data without waiting for the next weekly retrain.
    Best-effort: failures here are logged but don't affect the refresh
    job's result.
    """
    try:
        from app.inference.service import trigger_inference
        resp = trigger_inference(reason="data_refresh")
        logger.info(
            "refresh_success_triggered_inference refresh_job=%s inference_job=%s status=%s",
            job_id, getattr(resp, "job_id", None), getattr(resp, "status", None),
        )
    except Exception as exc:
        logger.exception("refresh_callback_failed job=%s: %s", job_id, exc)
```

Then in the place where refresh job transitions to "done":
```python
job["status"] = "done"
job["finished_at"] = datetime.utcnow().isoformat()
_on_refresh_success(job_id)   # <-- ADD THIS
```

- [ ] **Step 5: Run tests, verify PASS**

```
cd backend && F:/Tools/Anaconda/envs/qlib/python.exe -m pytest tests/test_refresh_callback.py -v
```

- [ ] **Step 6: Commit**

```
git add backend/app/data/service.py backend/tests/test_refresh_callback.py
git commit -m "feat(data): refresh success spawns daily_inference

After a data refresh job finishes successfully, automatically trigger
production/daily_inference.py via the inference service. Failures in
the callback don't affect the refresh job result."
```

---

### Task 9: ScreenItem schema + HorizonPrediction

**Files:**
- Modify: `backend/app/models/schemas.py`
- Test: `backend/tests/test_models_schemas.py`

- [ ] **Step 1: Read current `ScreenItem` schema**

```
grep -n "class ScreenItem" backend/app/models/schemas.py
```

- [ ] **Step 2: Add new schemas**

In `backend/app/models/schemas.py`, after existing classes:
```python
class HorizonPrediction(BaseModel):
    target_date: str  # ISO date, the trading day this horizon predicts
    pred_return: float | None = None  # calibrated expected return (decimal, e.g. 0.032 for 3.2%)
    percentile: float  # 0..100, higher is better (top 1% has percentile 99.0)
    model_agreement: float | None = None  # 0..1 fraction of models agreeing in direction
    raw_scores: dict[str, float] = Field(default_factory=dict)
```

Extend `ScreenItem`:
```python
class ScreenItem(BaseModel):
    # ... existing fields ...
    horizons: dict[str, HorizonPrediction] = Field(
        default_factory=dict,
        description="Keyed by horizon id: '1d', '5d', '20d'",
    )
```

Extend `CandidatesResponse` (or whichever response schema wraps the items):
```python
class CandidatesResponse(BaseModel):
    # ... existing fields ...
    as_of_date: str | None = None
    data_latest_date: str | None = None
    data_stale_days: int = 0
```

- [ ] **Step 3: Add tests**

`backend/tests/test_models_schemas.py`:
```python
from app.models.schemas import HorizonPrediction, ScreenItem


def test_horizon_prediction_defaults():
    h = HorizonPrediction(target_date="2026-06-03", percentile=95.0)
    assert h.pred_return is None
    assert h.model_agreement is None
    assert h.raw_scores == {}


def test_screen_item_horizons_default_empty():
    it = ScreenItem(rank=1, symbol="SH600519", name="贵州茅台",
                    score_today=0.1, score_avg=0.1, rank_avg=1.0, days_in_top=5)
    assert it.horizons == {}


def test_screen_item_with_horizons():
    it = ScreenItem(rank=1, symbol="SH600519", name="贵州茅台",
                    score_today=0.1, score_avg=0.1, rank_avg=1.0, days_in_top=5,
                    horizons={"5d": HorizonPrediction(target_date="2026-06-03",
                                                        pred_return=0.032, percentile=98.6)})
    assert it.horizons["5d"].pred_return == 0.032
```

- [ ] **Step 4: Run tests**

```
cd backend && F:/Tools/Anaconda/envs/qlib/python.exe -m pytest tests/test_models_schemas.py -v
```
Expected: 3 PASS

- [ ] **Step 5: Commit**

```
git add backend/app/models/schemas.py backend/tests/test_models_schemas.py
git commit -m "feat(models): HorizonPrediction + ScreenItem.horizons + staleness fields

Per-horizon target_date / pred_return / percentile / model_agreement / raw_scores.
CandidatesResponse gains as_of_date / data_latest_date / data_stale_days.
All fields optional with safe defaults for backward compat."
```

---

### Task 10: `candidates()` per-horizon computation + cache invalidate

**Files:**
- Modify: `backend/app/models/service.py`
- Test: `backend/tests/test_candidates_horizons.py`

- [ ] **Step 1: Write failing test**

`backend/tests/test_candidates_horizons.py`:
```python
"""Verify candidates() response includes per-horizon HorizonPrediction objects."""
import pytest
from unittest import mock

import numpy as np
import pandas as pd


def _toy_pred_df():
    dates = pd.date_range("2026-05-20", periods=5, freq="B")
    inst = [f"SH{600000+i}" for i in range(30)]
    idx = pd.MultiIndex.from_product([dates, inst], names=["datetime", "instrument"])
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "lgbm_1d": rng.normal(size=len(idx)),
        "alstm_1d": rng.normal(size=len(idx)),
        "tra_1d": rng.normal(size=len(idx)),
        "lgbm_5d": rng.normal(size=len(idx)),
        "alstm_5d": rng.normal(size=len(idx)),
        "tra_5d": rng.normal(size=len(idx)),
        "lgbm_20d": rng.normal(size=len(idx)),
        "alstm_20d": rng.normal(size=len(idx)),
        "tra_20d": rng.normal(size=len(idx)),
        "score": rng.normal(size=len(idx)),
        "consensus": np.abs(rng.normal(size=len(idx))),
    }, index=idx)


def test_candidates_returns_horizons(monkeypatch):
    from app.models import service as svc

    fake = _toy_pred_df()
    monkeypatch.setattr(svc, "load_pred", lambda *a, **kw: fake)
    monkeypatch.setattr(svc, "get_latest_recorder_id", lambda *a, **kw: "fake-rec")
    monkeypatch.setattr(svc, "init_qlib_once", lambda: None)
    monkeypatch.setattr(svc, "get_latest_close_prices", lambda syms: {})
    monkeypatch.setattr("app.core.qlib_adapter.get_filter_metrics", lambda syms: {})
    svc.invalidate_candidates_cache()

    resp = svc.candidates(top=5, days=3)
    assert "items" in resp
    for item in resp["items"]:
        assert "horizons" in item
        assert set(item["horizons"].keys()) >= {"1d", "5d", "20d"}
        for h in ("1d", "5d", "20d"):
            hp = item["horizons"][h]
            assert "target_date" in hp
            assert "percentile" in hp
            assert 0 <= hp["percentile"] <= 100


def test_candidates_includes_as_of_and_stale(monkeypatch):
    from app.models import service as svc

    fake = _toy_pred_df()
    monkeypatch.setattr(svc, "load_pred", lambda *a, **kw: fake)
    monkeypatch.setattr(svc, "get_latest_recorder_id", lambda *a, **kw: "fake-rec")
    monkeypatch.setattr(svc, "init_qlib_once", lambda: None)
    monkeypatch.setattr(svc, "get_latest_close_prices", lambda syms: {})
    monkeypatch.setattr("app.core.qlib_adapter.get_filter_metrics", lambda syms: {})
    svc.invalidate_candidates_cache()

    resp = svc.candidates(top=5, days=3)
    assert "as_of_date" in resp
    assert "data_latest_date" in resp
    assert "data_stale_days" in resp
    assert resp["data_stale_days"] >= 0
```

- [ ] **Step 2: Run, expect FAIL**

```
cd backend && F:/Tools/Anaconda/envs/qlib/python.exe -m pytest tests/test_candidates_horizons.py -v
```

- [ ] **Step 3: Modify `backend/app/models/service.py`**

Add helper imports at top:
```python
from datetime import date as _date
from pathlib import Path
```

Add module-level loaders:
```python
_CALIBRATION_CACHE: dict | None = None
_QLIB_LATEST_DATE_CACHE: tuple[str, _date] | None = None  # (asof_str, latest_date)


def _load_calibration() -> dict:
    global _CALIBRATION_CACHE
    if _CALIBRATION_CACHE is not None:
        return _CALIBRATION_CACHE
    try:
        from production.calibration import load_calibration
        repo_root = Path(__file__).resolve().parents[3]
        _CALIBRATION_CACHE = load_calibration(
            repo_root / "production" / "cache" / "latest_calibration.pkl"
        )
    except Exception:
        _CALIBRATION_CACHE = {"maps": {}, "meta": {}}
    return _CALIBRATION_CACHE


def _get_qlib_latest_date() -> _date | None:
    try:
        from qlib.data import D
        cal = D.calendar()
        if len(cal) == 0:
            return None
        return pd.Timestamp(cal[-1]).date()
    except Exception:
        return None


def _next_n_trading_days(start: _date, n: int) -> _date:
    """Returns the date n trading days after `start`."""
    try:
        from qlib.data import D
        cal = D.calendar(start_time=str(start))
        # cal[0] is start (or first >= start). We want cal[n].
        if len(cal) > n:
            return pd.Timestamp(cal[n]).date()
        return start
    except Exception:
        # Fall back to pandas BusinessDay
        return (pd.Timestamp(start) + pd.tseries.offsets.BDay(n)).date()


_H_TO_N = {"1d": 1, "5d": 5, "20d": 20}
```

Modify `_candidates_cached` end (after computing `items` and metrics):
```python
# === Per-horizon enrichment ===
cal_maps = _load_calibration().get("maps", {})
latest_ts = today  # pd.Timestamp
latest_date = latest_ts.date() if hasattr(latest_ts, "date") else _date.fromisoformat(str(latest_ts)[:10])
qlib_latest = _get_qlib_latest_date()

# Universe size at latest_date for percentile denominators
for it in items:
    horizons_data: dict[str, dict] = {}
    for h in ("1d", "5d", "20d"):
        cols = [c for c in df.columns
                if c.endswith(f"_{h}") and not c.startswith("expected_")
                and not c.startswith("composite_")]
        if not cols:
            continue
        # Composite score for this horizon @ latest_date
        sub = df[cols].xs(latest_ts, level="datetime")
        ranks = sub.rank(ascending=False, method="min")
        comp = -ranks.mean(axis=1, skipna=True)
        n = comp.notna().sum()
        sym_score = comp.get(it.symbol)
        if pd.isna(sym_score):
            continue
        sym_rank = comp.rank(ascending=False, method="min").get(it.symbol, n)
        percentile = float(100.0 * (1.0 - (sym_rank - 1) / n)) if n > 0 else 0.0

        pred_return = None
        if h in cal_maps:
            try:
                from production.calibration import apply_calibration
                pr = apply_calibration(pd.Series([sym_score]), cal_maps[h]).iloc[0]
                pred_return = None if pd.isna(pr) else float(pr)
            except Exception:
                pred_return = None

        raw = {}
        agreement = None
        try:
            row = df.loc[(latest_ts, it.symbol)]
            for m in ("lgbm", "alstm", "tra"):
                v = row.get(f"{m}_{h}")
                if pd.notna(v):
                    raw[m] = float(v)
            signs = [1 if v > 0 else (-1 if v < 0 else 0) for v in raw.values()]
            if signs:
                agreement = float(abs(sum(signs)) / len(signs))
        except KeyError:
            pass

        target = _next_n_trading_days(latest_date, _H_TO_N[h])
        horizons_data[h] = {
            "target_date": target.isoformat(),
            "pred_return": pred_return,
            "percentile": percentile,
            "model_agreement": agreement,
            "raw_scores": raw,
        }
    it.horizons = horizons_data  # type: ignore[attr-defined]

# Staleness
data_stale_days = 0
data_latest_str = latest_date.isoformat()
if qlib_latest and qlib_latest > latest_date:
    # Count trading days between latest_date and qlib_latest
    try:
        from qlib.data import D
        cal = D.calendar(start_time=str(latest_date), end_time=str(qlib_latest))
        data_stale_days = max(0, len(cal) - 1)
    except Exception:
        data_stale_days = (qlib_latest - latest_date).days
    data_latest_str = qlib_latest.isoformat()

return {
    "experiment": exp,
    "recorder_id": recorder_id,
    "latest_date": data_latest_str,           # historical compat: was asof
    "as_of_date": latest_date.isoformat(),    # the date the predictions are for
    "data_latest_date": data_latest_str,
    "data_stale_days": data_stale_days,
    "window_days": days,
    "universe_size": universe_size,
    "available_models": available_models,
    "active_models": list(score_cols) if score_cols else None,
    "items": [it.model_dump() for it in items],
}
```

Add to `invalidate_candidates_cache`:
```python
def invalidate_candidates_cache() -> int:
    global _CALIBRATION_CACHE
    _CALIBRATION_CACHE = None  # force reload on next call
    info = _candidates_cached.cache_info()
    _candidates_cached.cache_clear()
    return info.currsize
```

- [ ] **Step 4: Verify ScreenItem allows `horizons` attribute**

Confirm `app/models/schemas.py` ScreenItem has `horizons` field added in Task 9.

- [ ] **Step 5: Run tests, verify PASS**

```
cd backend && F:/Tools/Anaconda/envs/qlib/python.exe -m pytest tests/test_candidates_horizons.py -v
```

- [ ] **Step 6: Manual end-to-end probe**

Restart backend, then:
```
curl http://localhost:8000/api/models/candidates?top=3 | python -m json.tool
```
Expected: response includes `as_of_date`, `data_latest_date`, `data_stale_days`, and each item has `horizons` with 3 keys.

- [ ] **Step 7: Commit**

```
git add backend/app/models/service.py backend/tests/test_candidates_horizons.py
git commit -m "feat(models/service): per-horizon enrichment + staleness in candidates()

For each item at latest_date, compute composite_score = -rank_avg of
that horizon's model cols, derive percentile, apply isotonic calibration
if available, compute 3-model directional agreement, and look up the
target trading date via qlib calendar. Response gains as_of_date /
data_latest_date / data_stale_days. invalidate_candidates_cache also
drops the in-process calibration cache."
```

---

## Phase C — Frontend Core

### Task 11: API client + inference hooks + useActiveJobs extension

**Files:**
- Run: `npm run gen:api` (regenerate types)
- Modify: `frontend/src/api/client.ts`
- Create: `frontend/src/inference/hooks.ts`
- Modify: `frontend/src/jobs/useActiveJobs.ts`

- [ ] **Step 1: Regenerate types**

With backend running:
```
cd E:/Projects/qlib/frontend && npm run gen:api
```
Expected: `src/api/types.gen.ts` updates to include new schemas.

- [ ] **Step 2: Add inference methods to `src/api/client.ts`**

Find the existing object exposing api methods (look for `refreshActive`, `retrainActive`, `evalActive`) and add alongside them:
```typescript
inferenceActive: () =>
  fetchJson<InferenceJob | null>('/api/inference/active/peek'),

inferenceStatus: () =>
  fetchJson<InferenceStatus>('/api/inference/status'),

inferenceRunNow: (force = false) =>
  postJson<TriggerResponse>('/api/inference/run-now', { force }),

inferenceJob: (jobId: string) =>
  fetchJson<InferenceJob>(`/api/inference/jobs/${jobId}`),
```

Where `InferenceJob`, `InferenceStatus`, `TriggerResponse` are imported from the generated types (or defined inline if the gen step didn't pick them up — fallback:
```typescript
export interface InferenceJob {
  job_id: string;
  status: 'running' | 'done' | 'failed';
  started_at: string;
  finished_at?: string | null;
  end_date?: string | null;
  error?: string | null;
  new_rows?: number | null;
}
```
)

- [ ] **Step 3: Create `src/inference/hooks.ts`**

```typescript
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api } from '@/api/client';

export function useActiveInferenceJob() {
  return useQuery({
    queryKey: ['inference-active'],
    queryFn: () => api.inferenceActive(),
    refetchInterval: 3000,
    refetchIntervalInBackground: true,
  });
}

export function useInferenceStatus() {
  return useQuery({
    queryKey: ['inference-status'],
    queryFn: () => api.inferenceStatus(),
    staleTime: 10_000,
  });
}

export function useTriggerInference() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (force = false) => api.inferenceRunNow(force),
    onSuccess: async () => {
      const { toast } = await import('@/jobs/toast');
      toast.info('已触发推理任务，等待完成…');
      qc.invalidateQueries({ queryKey: ['inference-active'] });
    },
  });
}
```

- [ ] **Step 4: Extend `src/jobs/useActiveJobs.ts`**

Open the file, find the `JobKind` union and the array of polled queries. Add `'inference'`:

```typescript
export type JobKind = 'refresh' | 'retrain' | 'evaluation' | 'inference';
```

In the hook body, alongside the existing 3 polls, add:
```typescript
const inference = useQuery({
  queryKey: ['active-inference'],
  queryFn: () => api.inferenceActive(),
  refetchInterval: 3000,
  refetchIntervalInBackground: true,
});

// In the returned aggregated list:
if (inference.data) {
  jobs.push({
    id: inference.data.job_id,
    kind: 'inference',
    label: '模型推理',
    status: inference.data.status,
    href: '/picks',  // landing page where new predictions appear
    detail: inference.data.new_rows
      ? `+${inference.data.new_rows} 行`
      : inference.data.end_date ?? '',
  });
}
```

- [ ] **Step 5: Manual smoke test**

Restart frontend (`npm run dev`). Then trigger inference manually:
```
curl -X POST http://localhost:8000/api/inference/run-now
```
The header `<ActiveJobsBadge />` should show a "模型推理 · running" chip within 3 seconds.

- [ ] **Step 6: Commit**

```
git add frontend/src/api/client.ts frontend/src/api/types.gen.ts frontend/src/inference/hooks.ts frontend/src/jobs/useActiveJobs.ts
git commit -m "feat(frontend): inference hooks + active jobs chip

api.inferenceActive/Status/RunNow/Job + useActiveInferenceJob hook +
useTriggerInference mutation (with sticky info toast). useActiveJobs
adds 'inference' as 4th polled kind so the header badge shows running
inference like refresh/retrain/eval."
```

---

### Task 12: HorizonMiniBar component

**Files:**
- Create: `frontend/src/pages/picks/HorizonMiniBar.tsx`
- Test: `frontend/src/pages/picks/__tests__/HorizonMiniBar.test.tsx`

- [ ] **Step 1: Write failing test**

`frontend/src/pages/picks/__tests__/HorizonMiniBar.test.tsx`:
```typescript
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import HorizonMiniBar from '../HorizonMiniBar';

describe('HorizonMiniBar', () => {
  it('shows pred_return and percentile when both present', () => {
    render(
      <HorizonMiniBar horizon="5d" predReturn={0.032} percentile={98.6}
                       modelAgreement={1} />
    );
    expect(screen.getByText(/\+3\.2%/)).toBeTruthy();
    expect(screen.getByText(/top 1\.4%/)).toBeTruthy();
  });

  it('shows only percentile when pred_return is null', () => {
    render(
      <HorizonMiniBar horizon="5d" predReturn={null} percentile={98.6}
                       modelAgreement={null} />
    );
    expect(screen.queryByText(/\d+\.\d+%/)).toBeNull();
    expect(screen.getByText(/top 1\.4%/)).toBeTruthy();
  });

  it('uses A-share red for positive return', () => {
    const { container } = render(
      <HorizonMiniBar horizon="5d" predReturn={0.05} percentile={99}
                       modelAgreement={1} />
    );
    const bar = container.querySelector('[data-testid="mini-bar"]');
    expect(bar?.className).toMatch(/red|ef4444/);
  });

  it('uses A-share green for negative return', () => {
    const { container } = render(
      <HorizonMiniBar horizon="5d" predReturn={-0.05} percentile={5}
                       modelAgreement={null} />
    );
    const bar = container.querySelector('[data-testid="mini-bar"]');
    expect(bar?.className).toMatch(/green|22c55e/);
  });

  it('shows ★ when model_agreement >= 1', () => {
    render(
      <HorizonMiniBar horizon="5d" predReturn={0.03} percentile={95}
                       modelAgreement={1} />
    );
    expect(screen.getByText(/★/)).toBeTruthy();
  });
});
```

- [ ] **Step 2: Run, expect FAIL**

```
cd E:/Projects/qlib/frontend && npx vitest run src/pages/picks/__tests__/HorizonMiniBar.test.tsx
```

- [ ] **Step 3: Implement `HorizonMiniBar.tsx`**

```typescript
import { cn } from '@/lib/utils';

interface Props {
  horizon: '1d' | '5d' | '20d';
  predReturn: number | null;
  percentile: number;
  modelAgreement: number | null;
  // Optional: max abs return for the same column across the visible table —
  // lets us normalize bar widths across rows. If absent, single-row scale.
  maxAbsReturn?: number;
  isPrimary?: boolean;
}

const HORIZON_LABEL: Record<string, string> = {
  '1d': '次日',
  '5d': '5 日',
  '20d': '20 日',
};

export default function HorizonMiniBar({
  horizon, predReturn, percentile, modelAgreement, maxAbsReturn, isPrimary,
}: Props) {
  const isPositive = predReturn != null && predReturn > 0;
  const isNegative = predReturn != null && predReturn < 0;
  // A-share convention: red = up, green = down
  const baseColor = isPositive
    ? 'bg-red-500/40 border-red-500'
    : isNegative
    ? 'bg-green-500/40 border-green-500'
    : 'bg-gray-700/40 border-gray-700';

  const widthPct = predReturn != null
    ? Math.min(100, Math.abs(predReturn) / (maxAbsReturn ?? 0.05) * 100)
    : 0;

  const topPct = 100 - percentile; // "top X.X%"
  const topLabel = topPct < 0.1
    ? 'top 0.1%'
    : `top ${topPct.toFixed(1)}%`;

  const showStar = modelAgreement != null && modelAgreement >= 0.99;

  return (
    <div className={cn('flex flex-col gap-0.5 min-w-[78px]', isPrimary && 'ring-1 ring-blue-500/30 px-1 rounded')}>
      <div
        data-testid="mini-bar"
        className={cn(
          'h-4 rounded-sm border flex items-center justify-end pr-1',
          baseColor,
        )}
        style={{ width: `${Math.max(20, widthPct)}%`, minWidth: '40px' }}
        title={`${HORIZON_LABEL[horizon]} 预期收益 ${predReturn != null ? `${(predReturn * 100).toFixed(1)}%` : 'N/A'}`}
      >
        {predReturn != null && (
          <span className="text-[10px] font-medium text-white">
            {predReturn > 0 ? '+' : ''}{(predReturn * 100).toFixed(1)}%
          </span>
        )}
      </div>
      <div className="text-[10px] text-[#8b949e] flex items-center gap-1">
        <span>{topLabel}</span>
        {showStar && <span className="text-yellow-400" title="3 模型同向">★</span>}
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Run tests, expect PASS**

```
cd E:/Projects/qlib/frontend && npx vitest run src/pages/picks/__tests__/HorizonMiniBar.test.tsx
```

- [ ] **Step 5: Commit**

```
git add frontend/src/pages/picks/HorizonMiniBar.tsx frontend/src/pages/picks/__tests__/HorizonMiniBar.test.tsx
git commit -m "feat(picks): HorizonMiniBar component

Single-cell visualization for one (stock, horizon) prediction. Shows
+X.X% return (A-share red=up, green=down), 'top X.X%' percentile
underneath, ★ star when 3 models agree on direction. Bar width
normalizable across rows via maxAbsReturn prop."
```

---

### Task 13: StalenessBanner + TopInfoRow

**Files:**
- Create: `frontend/src/pages/picks/StalenessBanner.tsx`
- Create: `frontend/src/pages/picks/TopInfoRow.tsx`
- Test: `frontend/src/pages/picks/__tests__/StalenessBanner.test.tsx`

- [ ] **Step 1: Write failing test**

```typescript
// frontend/src/pages/picks/__tests__/StalenessBanner.test.tsx
import { render, screen, fireEvent } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { describe, it, expect, vi } from 'vitest';
import StalenessBanner from '../StalenessBanner';

const qc = new QueryClient();
const wrap = (ui: React.ReactNode) =>
  <QueryClientProvider client={qc}>{ui}</QueryClientProvider>;

describe('StalenessBanner', () => {
  it('returns null when not stale', () => {
    const { container } = render(wrap(
      <StalenessBanner staleDays={0} asOfDate="2026-05-27"
                       dataLatestDate="2026-05-27" />,
    ));
    expect(container.firstChild).toBeNull();
  });

  it('renders when stale', () => {
    render(wrap(
      <StalenessBanner staleDays={3} asOfDate="2026-05-22"
                       dataLatestDate="2026-05-27" />,
    ));
    expect(screen.getByText(/2026-05-27/)).toBeTruthy();
    expect(screen.getByText(/2026-05-22/)).toBeTruthy();
    expect(screen.getByText(/3/)).toBeTruthy();
  });
});
```

- [ ] **Step 2: Run, expect FAIL**

```
cd E:/Projects/qlib/frontend && npx vitest run src/pages/picks/__tests__/StalenessBanner.test.tsx
```

- [ ] **Step 3: Implement `StalenessBanner.tsx`**

```typescript
import { useTriggerInference } from '@/inference/hooks';

interface Props {
  staleDays: number;
  asOfDate: string;
  dataLatestDate: string;
}

export default function StalenessBanner({ staleDays, asOfDate, dataLatestDate }: Props) {
  const trigger = useTriggerInference();
  if (staleDays <= 0) return null;
  return (
    <div className="rounded-md border border-orange-800 bg-orange-950/40 px-4 py-2 text-sm text-orange-300 flex items-center justify-between gap-3 flex-wrap">
      <div>
        ⚠️ 数据已更新到 <span className="font-mono font-semibold">{dataLatestDate}</span>，
        但预测停留在 <span className="font-mono">{asOfDate}</span>
        （<span className="font-semibold">{staleDays}</span> 个交易日前）。
      </div>
      <button
        type="button"
        disabled={trigger.isPending}
        onClick={() => trigger.mutate(false)}
        className="px-3 py-1 rounded bg-orange-600 hover:bg-orange-500 text-white text-xs font-medium disabled:opacity-50"
      >
        {trigger.isPending ? '推理中…' : '立即重新推理 →'}
      </button>
    </div>
  );
}
```

- [ ] **Step 4: Implement `TopInfoRow.tsx`**

```typescript
interface Props {
  asOfDate: string;
  dataLatestDate: string;
  targetDates: Record<string, string>;  // {"1d":"2026-05-28","5d":"2026-06-03","20d":"2026-06-24"}
}

export default function TopInfoRow({ asOfDate, dataLatestDate, targetDates }: Props) {
  return (
    <div className="text-xs text-[#8b949e] flex flex-wrap gap-x-3 gap-y-1">
      <span>截至 <span className="font-mono text-[#e6edf3]">{dataLatestDate}</span>（最新数据）</span>
      <span>·</span>
      <span>预测目标日:</span>
      {(['1d','5d','20d'] as const).map((h) => (
        <span key={h} className="font-mono text-[#e6edf3]">
          {targetDates[h] ?? '?'} <span className="text-[#6e7681]">({h})</span>
        </span>
      ))}
      {asOfDate !== dataLatestDate && (
        <span className="text-orange-400">
          · 预测 as-of: <span className="font-mono">{asOfDate}</span>
        </span>
      )}
    </div>
  );
}
```

- [ ] **Step 5: Run tests, verify PASS**

```
cd E:/Projects/qlib/frontend && npx vitest run src/pages/picks/__tests__/StalenessBanner.test.tsx
```

- [ ] **Step 6: Commit**

```
git add frontend/src/pages/picks/StalenessBanner.tsx frontend/src/pages/picks/TopInfoRow.tsx frontend/src/pages/picks/__tests__/StalenessBanner.test.tsx
git commit -m "feat(picks): StalenessBanner + TopInfoRow

StalenessBanner shows orange warning + 'inference now' button when
data is newer than predictions. TopInfoRow shows the as-of/target
dates so users immediately see what each prediction is for."
```

---

### Task 14: PicksTable refactor (3-col horizon layout)

**Files:**
- Modify: `frontend/src/pages/picks/PicksPage.tsx` (or wherever the table is rendered today — locate via grep)
- Create: `frontend/src/pages/picks/PicksTable.tsx`

- [ ] **Step 1: Locate existing table render**

```
grep -rn "score_today\|days_in_top\|consensus" frontend/src/pages/picks/ frontend/src/pages/ --include="*.tsx" | head
```

- [ ] **Step 2: Create `PicksTable.tsx`** wrapping the existing render logic but adding 3 horizon columns

```typescript
import { useState, useMemo } from 'react';
import { Link } from 'react-router-dom';
import HorizonMiniBar from './HorizonMiniBar';
import type { ScreenItem } from '@/api/types';  // or appropriate import

type HorizonId = '1d' | '5d' | '20d';
type SortKey = `pred_${HorizonId}` | `pct_${HorizonId}` | 'rank';

interface Props {
  items: ScreenItem[];
}

export default function PicksTable({ items }: Props) {
  const [sortKey, setSortKey] = useState<SortKey>('pred_5d');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc');

  // Compute max abs return per horizon for bar normalization
  const maxAbsByHorizon = useMemo(() => {
    const out: Record<HorizonId, number> = { '1d': 0, '5d': 0, '20d': 0 };
    for (const it of items) {
      for (const h of ['1d', '5d', '20d'] as const) {
        const r = it.horizons?.[h]?.pred_return;
        if (r != null) out[h] = Math.max(out[h], Math.abs(r));
      }
    }
    // Avoid zero-division; minimum 1% scale
    for (const h of ['1d', '5d', '20d'] as const) {
      out[h] = Math.max(out[h], 0.01);
    }
    return out;
  }, [items]);

  const sorted = useMemo(() => {
    const arr = [...items];
    arr.sort((a, b) => {
      let av: number, bv: number;
      if (sortKey === 'rank') {
        av = a.rank; bv = b.rank;
      } else if (sortKey.startsWith('pred_')) {
        const h = sortKey.replace('pred_', '') as HorizonId;
        av = a.horizons?.[h]?.pred_return ?? -1e9;
        bv = b.horizons?.[h]?.pred_return ?? -1e9;
      } else { // pct_
        const h = sortKey.replace('pct_', '') as HorizonId;
        av = a.horizons?.[h]?.percentile ?? 0;
        bv = b.horizons?.[h]?.percentile ?? 0;
      }
      return sortDir === 'desc' ? bv - av : av - bv;
    });
    return arr;
  }, [items, sortKey, sortDir]);

  const setSort = (k: SortKey) => {
    if (k === sortKey) {
      setSortDir(d => d === 'desc' ? 'asc' : 'desc');
    } else {
      setSortKey(k);
      setSortDir('desc');
    }
  };

  return (
    <div className="overflow-x-auto rounded-lg border border-[#30363d]">
      <table className="w-full text-sm">
        <thead className="bg-[#161b22] text-[#8b949e] text-xs">
          <tr>
            <Th onClick={() => setSort('rank')} sortKey="rank" current={sortKey} dir={sortDir}>排名</Th>
            <th className="text-left p-2">股票</th>
            <Th onClick={() => setSort('pred_1d')} sortKey="pred_1d" current={sortKey} dir={sortDir}>1 日</Th>
            <Th onClick={() => setSort('pred_5d')} sortKey="pred_5d" current={sortKey} dir={sortDir}>5 日 (主)</Th>
            <Th onClick={() => setSort('pred_20d')} sortKey="pred_20d" current={sortKey} dir={sortDir}>20 日</Th>
            <th className="text-right p-2">最新价</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((it) => (
            <tr key={it.symbol} className="border-t border-[#21262d] hover:bg-[#161b22]">
              <td className="p-2 text-center font-mono text-[#8b949e]">{it.rank}</td>
              <td className="p-2">
                <Link to={`/charts/${it.symbol}`} className="text-[#58a6ff] hover:underline">
                  <div className="font-mono text-xs">{it.symbol}</div>
                  <div className="text-xs text-[#8b949e]">{it.name}</div>
                </Link>
              </td>
              {(['1d', '5d', '20d'] as const).map((h) => (
                <td key={h} className="p-2">
                  {it.horizons?.[h] ? (
                    <HorizonMiniBar
                      horizon={h}
                      predReturn={it.horizons[h]!.pred_return ?? null}
                      percentile={it.horizons[h]!.percentile}
                      modelAgreement={it.horizons[h]!.model_agreement ?? null}
                      maxAbsReturn={maxAbsByHorizon[h]}
                      isPrimary={h === '5d'}
                    />
                  ) : (
                    <span className="text-[#6e7681] text-xs">—</span>
                  )}
                </td>
              ))}
              <td className="p-2 text-right font-mono">
                {it.last_price != null ? `¥${it.last_price.toFixed(2)}` : '—'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Th({ onClick, sortKey, current, dir, children }: {
  onClick: () => void;
  sortKey: SortKey | 'rank';
  current: SortKey;
  dir: 'asc' | 'desc';
  children: React.ReactNode;
}) {
  const active = sortKey === current;
  return (
    <th
      onClick={onClick}
      className="p-2 cursor-pointer select-none hover:text-[#e6edf3]"
    >
      <span>{children}</span>
      {active && <span className="ml-1 text-[#58a6ff]">{dir === 'desc' ? '↓' : '↑'}</span>}
    </th>
  );
}
```

- [ ] **Step 3: Replace existing list render in PicksPage with `<PicksTable items={...} />`**

Open `frontend/src/pages/picks/PicksPage.tsx` (or wherever the candidates are currently rendered). Replace the old table/list JSX with:
```tsx
import PicksTable from './PicksTable';
import StalenessBanner from './StalenessBanner';
import TopInfoRow from './TopInfoRow';

// inside the component, where you have the candidates query data:
const targetDates: Record<string, string> = {};
for (const it of data.items.slice(0, 1)) {
  if (it.horizons) {
    for (const h of ['1d','5d','20d'] as const) {
      if (it.horizons[h]) targetDates[h] = it.horizons[h].target_date;
    }
  }
}

return (
  <div className="space-y-3">
    <StalenessBanner
      staleDays={data.data_stale_days ?? 0}
      asOfDate={data.as_of_date ?? ''}
      dataLatestDate={data.data_latest_date ?? ''}
    />
    <TopInfoRow
      asOfDate={data.as_of_date ?? ''}
      dataLatestDate={data.data_latest_date ?? ''}
      targetDates={targetDates}
    />
    {/* keep existing FilterBar */}
    <PicksTable items={data.items} />
  </div>
);
```

- [ ] **Step 4: Run frontend tests, manual smoke**

```
cd E:/Projects/qlib/frontend && npx vitest run
```
Then start dev server (`npm run dev`) and verify `/picks` renders the new 3-column layout, sorting by column headers works.

- [ ] **Step 5: Commit**

```
git add frontend/src/pages/picks/PicksTable.tsx frontend/src/pages/picks/PicksPage.tsx
git commit -m "feat(picks): single-table layout with 1d/5d/20d horizon mini-bars

Replaces the legacy candidate list with PicksTable. Each row has three
HorizonMiniBar cells side-by-side (5d marked as primary with ring),
default sort by 5d expected return desc, click any column header to
sort. StalenessBanner + TopInfoRow added at top."
```

---

## Phase D — Frontend Chart

### Task 15: Chart backend payload extension (`horizon_markers`)

**Files:**
- Modify: `backend/app/charts/schemas.py`
- Modify: `backend/app/charts/service.py`
- Test: `backend/tests/test_charts_horizon_markers.py`

- [ ] **Step 1: Write failing test**

`backend/tests/test_charts_horizon_markers.py`:
```python
def test_chart_payload_includes_horizon_markers():
    from app.charts.schemas import ChartPayload, HorizonMarker
    p = ChartPayload(
        symbol="SH600519", actual=[], predicted=[], forecast=[],
        horizon_markers=[
            HorizonMarker(horizon="5d", target_date="2026-06-03",
                          target_price=125.4, pred_return=0.032,
                          percentile=98.6, model_agreement=0.67,
                          raw_scores={"lgbm": 0.04, "alstm": 0.02}),
        ],
    )
    assert p.horizon_markers[0].horizon == "5d"
```

- [ ] **Step 2: Add `HorizonMarker` to schemas**

```python
# backend/app/charts/schemas.py
class HorizonMarker(BaseModel):
    horizon: str  # "1d" | "5d" | "20d"
    target_date: str
    target_price: float
    pred_return: float | None = None
    percentile: float
    model_agreement: float | None = None
    raw_scores: dict[str, float] = Field(default_factory=dict)


class ChartPayload(BaseModel):
    # ... existing fields ...
    horizon_markers: list[HorizonMarker] = Field(default_factory=list)
```

- [ ] **Step 3: Populate `horizon_markers` in `service.py`**

In `backend/app/charts/service.py`, at the end of the payload build, before `return ChartPayload(...)`:

```python
# Build horizon_markers from latest_candidates lookup
horizon_markers: list[HorizonMarker] = []
try:
    # Find this symbol in the latest candidates response (cached upstream)
    from app.models.service import candidates as _cands
    cands = _cands(top=2000, days=1)  # over-fetch to ensure we catch this symbol
    last_close = actual[-1].close if actual else 0.0
    found = next((it for it in cands["items"] if it["symbol"] == symbol), None)
    if found and found.get("horizons"):
        for h, hp in found["horizons"].items():
            if last_close <= 0:
                continue
            pred_ret = hp.get("pred_return")
            target_price = (
                last_close * (1 + pred_ret) if pred_ret is not None
                else last_close
            )
            horizon_markers.append(HorizonMarker(
                horizon=h,
                target_date=hp["target_date"],
                target_price=target_price,
                pred_return=pred_ret,
                percentile=hp["percentile"],
                model_agreement=hp.get("model_agreement"),
                raw_scores=hp.get("raw_scores", {}),
            ))
except Exception as exc:
    log.warning("horizon_markers_build_failed for %s: %s", symbol, exc)
```

Then update the return statement:
```python
return ChartPayload(
    symbol=symbol, actual=actual, predicted=predicted, forecast=forecast,
    horizon_markers=horizon_markers, meta=meta,
)
```

- [ ] **Step 4: Run tests, verify PASS**

```
cd backend && F:/Tools/Anaconda/envs/qlib/python.exe -m pytest tests/test_charts_horizon_markers.py -v
```

- [ ] **Step 5: Commit**

```
git add backend/app/charts/schemas.py backend/app/charts/service.py backend/tests/test_charts_horizon_markers.py
git commit -m "feat(charts): horizon_markers in payload

ChartPayload.horizon_markers carries 3 per-horizon predictions for the
symbol (target_date, target_price, pred_return, percentile,
model_agreement, raw_scores). Computed from the candidates response so
the chart and picks list show consistent numbers."
```

---

### Task 16: Chart frontend K-line extension + markers + tooltip

**Files:**
- Modify: `frontend/src/charts/PredictionChart.tsx`

- [ ] **Step 1: Regen API types**

```
cd E:/Projects/qlib/frontend && npm run gen:api
```

- [ ] **Step 2: Add `horizon_markers` to Props + render**

In `PredictionChart.tsx`:

```typescript
interface HorizonMarker {
  horizon: string;
  target_date: string;
  target_price: number;
  pred_return: number | null;
  percentile: number;
  model_agreement: number | null;
  raw_scores: Record<string, number>;
}

interface Props {
  // ... existing ...
  horizonMarkers?: HorizonMarker[];
}
```

Add a ref for the future-line series at the top of the component:
```typescript
const futureLineRef = useRef<ISeriesApi<'Line'> | null>(null);
const [showFuture, setShowFuture] = useState(true);
```

In the mount effect, after creating other series:
```typescript
futureLineRef.current = chart.addLineSeries({
  color: '#888',
  lineWidth: 1,
  lineStyle: 2,  // dashed
  priceLineVisible: false,
  lastValueVisible: false,
  title: '未来预测',
});
```

In cleanup:
```typescript
futureLineRef.current = null;
```

Add a sync effect for future markers + dashed line:
```typescript
useEffect(() => {
  if (!horizonMarkers || !actual.length || !futureLineRef.current || !actualSeriesRef.current) return;

  const sorted = [...horizonMarkers].sort((a, b) =>
    a.target_date.localeCompare(b.target_date));

  // Build the dashed line: from last actual close to each marker's target_price
  const lastClose = actual[actual.length - 1].close;
  const lineData: { time: Time; value: number }[] = [
    { time: lastActualDate as Time, value: lastClose },
    ...sorted.map(m => ({ time: m.target_date as Time, value: m.target_price })),
  ];

  futureLineRef.current.setData(showFuture ? lineData : []);
  futureLineRef.current.applyOptions({ visible: showFuture });

  // Place markers on the LAST ACTUAL series so they share the price scale.
  // Each marker gets a tooltip via the 'text' field.
  const markers = sorted.map(m => {
    const isUp = (m.pred_return ?? 0) >= 0;
    return {
      time: m.target_date as Time,
      position: 'inBar' as const,
      color: isUp ? '#ef4444' : '#22c55e',
      shape: 'diamond' as const,
      size: m.horizon === '20d' ? 2 : m.horizon === '5d' ? 1.5 : 1,
      text: `${m.horizon}: ${m.pred_return != null
        ? (m.pred_return >= 0 ? '+' : '') + (m.pred_return * 100).toFixed(1) + '%'
        : 'N/A'}`,
    };
  });

  // Keep prior "→ 未来" arrow on last actual + new diamond markers on future dates.
  const allMarkers = [
    { time: lastActualDate, position: 'aboveBar' as const, color: '#ff9800',
      shape: 'arrowDown' as const, text: '→ 未来' },
    ...(showFuture ? markers : []),
  ];
  actualSeriesRef.current.setMarkers?.(allMarkers);
}, [horizonMarkers, actual, lastActualDate, showFuture]);
```

Add a checkbox to the controls row (next to "实际 K 线", "预测 K 线" etc):
```jsx
<label className="flex items-center gap-2 cursor-pointer"
       title="未来 1/5/20 个交易日的预测目标价 (虚线) — A 股惯例:红=涨,绿=跌">
  <input type="checkbox" checked={showFuture}
         onChange={e => setShowFuture(e.target.checked)} />
  🎯 未来预测
  <span className="text-[10px] text-[#6e7681]">(虚线)</span>
</label>
```

- [ ] **Step 3: Pipe `horizonMarkers` from the page that renders the chart**

In `frontend/src/charts/ChartPage.tsx` (or wherever PredictionChart is mounted), pass:
```jsx
<PredictionChart
  symbol={symbol}
  actual={chartData.actual}
  predicted={chartData.predicted}
  forecast={chartData.forecast}
  horizonMarkers={chartData.horizon_markers}
  lastActualDate={chartData.actual[chartData.actual.length - 1]?.time ?? ''}
/>
```

- [ ] **Step 4: Manual smoke test**

Restart frontend, visit `/charts/SH600519`. The K-line should have a dashed gray line extending from the latest close to 3 diamond markers at future trading days. The 5d marker is mid-size, 20d is largest. Hover should show a tooltip with the horizon and return %. Toggle the "🎯 未来预测" checkbox to hide/show.

- [ ] **Step 5: Commit**

```
git add frontend/src/charts/PredictionChart.tsx frontend/src/charts/ChartPage.tsx
git commit -m "feat(charts): K-line extension + 3 future markers

Dashed gray line from the last actual close to 3 future diamond markers
at +1d/+5d/+20d (sizes scale with horizon). Marker color follows A-share
red=up/green=down. Hover marker shows horizon label + return %. New
'🎯 未来预测' chip in controls row toggles visibility (default on)."
```

---

## Phase E — Integration & Docs

### Task 17: End-to-end integration check

- [ ] **Step 1: Restart backend + frontend** (assumes both already running)

```
# In project root
F:/Tools/Anaconda/envs/qlib/python.exe -c "import requests; print(requests.get('http://localhost:8000/api/models/candidates?top=3').json())" > nul
```
Confirm the response includes `as_of_date`, `horizons`, `data_stale_days`.

- [ ] **Step 2: Drive the full happy path**

1. Browser: open `http://localhost:5173/picks`
   - Verify: TopInfoRow shows target dates; if data is fresh (stale_days=0) no banner; table shows 3 horizon columns
2. Click any column header to sort
3. Click a stock row → `/charts/SH600519`
   - Verify: dashed line + 3 future markers; tooltip works
4. Back to `/picks`. Click "立即重新推理" (if staleness banner present) or call directly:
   ```
   curl -X POST http://localhost:8000/api/inference/run-now
   ```
   - Verify: header `ActiveJobsBadge` shows "模型推理 · running" chip
5. Wait for the inference to complete (~30-60s).
   - Verify: chip turns green ("done"), `/api/models/candidates` returns updated `as_of_date`.

- [ ] **Step 3: Simulate data refresh callback**

```
curl -X POST http://localhost:8000/api/data/refresh
```
Wait for refresh to complete → confirm inference triggered automatically (visible in backend logs as `refresh_success_triggered_inference`).

- [ ] **Step 4: Commit any glue fixes from manual testing**

```
git add ...
git commit -m "fix(integration): <whatever was broken>"
```

(No commit if nothing needed fixing.)

---

### Task 18: Documentation update

**Files:**
- Modify: `production/CLAUDE.md` (or top-level `CLAUDE.md` if there isn't a production-specific one)
- Modify: `C:/Users/Administered/.claude/projects/E--Projects-qlib/memory/project_state.md`

- [ ] **Step 1: Update project state**

Append a section to `project_state.md`:
```markdown
## 2026-05-28: Multi-horizon prediction UX + daily inference

- New `production/daily_inference.py` subprocess triggered by data refresh
- Per-horizon isotonic calibration in `production/calibration.py`
- `production/cache/latest_calibration.pkl` refreshed every weekly retrain
- Picks page: single-table 3-col mini-bars (1d/5d/20d expected return %)
- Chart page: K-line dashed extension + 3 future markers (1d/5d/20d)
- Backend `ScreenItem.horizons` + `as_of_date`/`data_latest_date`/`data_stale_days`

Reference: `docs/superpowers/specs/2026-05-28-prediction-ux-redesign-design.md`
+ `docs/superpowers/plans/2026-05-28-prediction-ux-redesign.md`
```

- [ ] **Step 2: Update CLAUDE.md operational notes**

If a top-level `CLAUDE.md` exists, add to "Operations" section:
```markdown
### Daily inference

After a data refresh job succeeds, the backend automatically spawns
`production/daily_inference.py` as a subprocess. It loads the latest weekly
mlflow recorders (`<model>_<horizon>_<end_date>`), runs inference on
missing dates, applies calibration from
`production/cache/latest_calibration.pkl`, and appends to the pooled
ensemble recorder's `pred.pkl`. Manual trigger:

```
curl -X POST http://localhost:8000/api/inference/run-now
```

When the calibration cache is missing or stale (e.g. weekly retrain failed),
the candidates response sets `pred_return=null` and the UI falls back to
percentile-only display.
```

- [ ] **Step 3: Commit**

```
git add production/CLAUDE.md
git commit -m "docs: project state + ops notes for multi-horizon UX

Capture the daily_inference pipeline + calibration cache + UI changes
so future agents can find them."
```

---

## Self-Review Checklist (run after writing the plan)

### Spec Coverage

| Spec section | Tasks covering it |
|---|---|
| §2 Daily inference + Calibration | T1, T2, T4, T5, T6 |
| §3 Backend API | T7, T8, T9, T10 |
| §4 Picks Page UI | T12, T13, T14 |
| §5 Chart Page UI | T15, T16 |
| §6 Edge cases | T2 (small samples), T5 (fail-soft), T7 (timeout), T13 (banner), T15 (fail-soft markers) |
| §7 Schemas | T9 (ScreenItem), T15 (HorizonMarker) |
| §11 Migration | T3 (save artifacts), T4 (backfill) |
| §12 Testing | T1, T2, T5, T7-T10, T12, T13, T15, T17 |

### Placeholder Scan

- ✅ No "TBD" / "TODO" left in step bodies
- ✅ Every code step has actual code, not "implement X"
- ✅ Every test step has assert statements

### Type Consistency

- ✅ `HorizonPrediction` defined T9, consumed T10, T11 (FE), T12
- ✅ `InferenceJob` defined T7, consumed T11 (FE)
- ✅ `HorizonMarker` defined T15, consumed T16 (FE)
- ✅ `_h_to_n` / `_H_TO_N` mapping defined T10, used T16 (FE infers from target_date)
- ✅ `latest_calibration.pkl` path consistent T1, T2, T5, T6, T10

### Cross-Task Order

- T3 must precede T4 (need valid_pred.pkl before backfill)
- T1 must precede T2, T5, T6 (calibration module)
- T5 must precede T8 (inference subprocess used by callback)
- T7 must precede T8 (inference router)
- T9 must precede T10 (ScreenItem schema)
- T10 must precede T15 (candidates feeds chart markers)
- T11 must precede T12-T14 (frontend hooks)
- T15 must precede T16 (backend payload before frontend renderer)

---

## Execution
