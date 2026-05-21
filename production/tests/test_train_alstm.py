import pandas as pd
import pytest

from production import rolling_train
from production.train_alstm import _build_multihead_dataset


def _make_cfg():
    return rolling_train.load_config(rolling_train.REPO_ROOT / "production/configs/rolling_ensemble.yaml")


def test_multihead_dataset_has_three_label_columns():
    """The multi-head ALSTM dataset stacks 1d / 5d / 20d labels as 3 columns."""
    cfg = _make_cfg()
    ds = _build_multihead_dataset(
        cfg,
        universe_name="csi300",  # any string; not used when build_features=False
        end_date=pd.Timestamp("2026-05-10").date(),
        build_features=False,
    )
    assert ds.label_cols == ["LABEL_1d", "LABEL_5d", "LABEL_20d"]
