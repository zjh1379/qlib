# qlib 特征加载 seam — 实现计划 (arch #1)

> REQUIRED SUB-SKILL: subagent-driven-development. TDD,纯单测,在 worktree 当前目录工作(勿 cd)。Python `F:/Tools/Anaconda/envs/qlib/python.exe`。

**Goal:** 一个 deep `production/qlib_features.py` 收编 6+ loader 的 QlibDataLoader→干净帧归一;5 个领域 loader 改薄委托,行为/签名不变。

---

## Task 1: 创建 `production/qlib_features.py` + 纯单测

- [ ] Step 1: 写 `production/tests/test_qlib_features.py`:

```python
import pandas as pd, pytest
from production.qlib_features import _normalize


def _raw(idx_tuples, idx_names, cols, vals):
    idx = pd.MultiIndex.from_tuples(idx_tuples, names=idx_names)
    return pd.DataFrame(vals, index=idx, columns=cols)


def test_normalize_swaps_instrument_first_and_renames():
    raw = _raw([("SH600000", pd.Timestamp("2024-01-02")),
                ("SH600000", pd.Timestamp("2024-01-03"))],
               ["instrument", "datetime"], ["Ref($open,-1)"], [[1.0], [2.0]])
    out = _normalize(raw, ["entry_open"])
    assert list(out.columns) == ["entry_open"]
    assert out.index.names == ["datetime", "instrument"]
    assert out.loc[(pd.Timestamp("2024-01-02"), "SH600000"), "entry_open"] == 1.0


def test_normalize_multi_column_by_position():
    raw = _raw([("SH600000", pd.Timestamp("2024-01-02"))],
               ["instrument", "datetime"], ["$open", "$close"], [[10.0, 11.0]])
    out = _normalize(raw, ["open", "close"])
    assert list(out.columns) == ["open", "close"]
    assert out.iloc[0]["close"] == 11.0


def test_normalize_already_datetime_first_no_swap():
    raw = _raw([(pd.Timestamp("2024-01-02"), "SH600000")],
               ["datetime", "instrument"], ["x"], [[5.0]])
    out = _normalize(raw, ["v"])
    assert out.index.names == ["datetime", "instrument"]
    assert out.iloc[0]["v"] == 5.0


def test_normalize_column_count_mismatch_raises():
    raw = _raw([("SH600000", pd.Timestamp("2024-01-02"))],
               ["instrument", "datetime"], ["a", "b"], [[1.0, 2.0]])
    with pytest.raises(ValueError):
        _normalize(raw, ["only_one"])
```

- [ ] Step 2: run → FAIL (ModuleNotFound). `F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_qlib_features.py -q`
- [ ] Step 3: create `production/qlib_features.py`:

```python
# production/qlib_features.py
"""One deep seam for loading adjusted features from qlib as a clean
(datetime, instrument) frame. Collapses the swaplevel + set_names + column
normalization ritual that 6+ call sites each re-implemented — inconsistently
(some df.iloc[:,0], some df.columns=names; the latter divergence shipped as a bug).
Domain-named loaders (load_fwd_returns, load_entry_ohlc, daily_open_adj, ...) stay as
thin wrappers over this; the qlib-quirk handling lives here once."""
from __future__ import annotations

# Force installed (compiled) qlib ahead of the uncompiled ./qlib source tree.
import sys as _sys
import sysconfig as _sysconfig
_PURELIB = _sysconfig.get_paths().get("purelib")
if _PURELIB and _PURELIB not in _sys.path[:1]:
    _sys.path.insert(0, _PURELIB)

import pandas as pd

DEFAULT_CONFIG = "production/configs/rolling_ensemble.yaml"


def _normalize(df: pd.DataFrame, names: list) -> pd.DataFrame:
    """Pure: raw QlibDataLoader frame -> clean (datetime,instrument) DataFrame whose
    columns are exactly `names`. QlibDataLoader labels columns by the feature
    EXPRESSION (or a MultiIndex level) and indexes by (instrument, datetime); we force
    `names` by position and put datetime first. No qlib import."""
    if df is None:
        raise ValueError("qlib loader returned None")
    df = df.copy()
    if df.shape[1] != len(names):
        raise ValueError(f"expected {len(names)} columns, got {df.shape[1]}: {list(df.columns)}")
    df.columns = list(names)
    if df.index.names and df.index.names[0] == "instrument":
        df = df.swaplevel()
    df.index = df.index.set_names(["datetime", "instrument"])
    return df.sort_index()


def load_features(instruments, start: str, end: str, fields: list, names: list,
                  *, config_path: str = DEFAULT_CONFIG, dropna_subset=None) -> pd.DataFrame:
    """Load `fields` (qlib expressions) for `instruments` over [start,end] as a clean
    (datetime,instrument) DataFrame with columns `names`; optional dropna on a subset."""
    from qlib.data.dataset.loader import QlibDataLoader
    from production.backtest.data import init_qlib_from_config
    init_qlib_from_config(config_path)
    raw = QlibDataLoader(config={"feature": (list(fields), list(names))}).load(
        instruments=instruments, start_time=start, end_time=end)
    out = _normalize(raw, names)
    if dropna_subset is not None:
        out = out.dropna(subset=list(dropna_subset))
    return out


def load_series(instruments, start: str, end: str, field: str, name: str,
                *, config_path: str = DEFAULT_CONFIG, dropna: bool = False) -> pd.Series:
    """Single-field convenience -> Series named `name`, (datetime,instrument) index."""
    s = load_features(instruments, start, end, [field], [name], config_path=config_path)[name]
    return s.dropna() if dropna else s
```

- [ ] Step 4: run → PASS (4). Step 5: commit `git add production/qlib_features.py production/tests/test_qlib_features.py && git commit -m "feat(qlib): qlib_features loader seam (_normalize + load_features/load_series)"`

---

## Task 2: 5 个 loader 改薄委托(行为/签名不变)

对每个文件:READ 它,只替换目标函数的 **body**,保留签名、模块 docstring、其它函数、各自的 sys.path fixup 与 `init_qlib_from_config`(`data.py` 的 init 仍被 qlib_features 延迟 import)。委托用**延迟 import**(函数内 `from production.qlib_features import ...`)避免循环。

- [ ] `backtest/data.py::load_fwd_returns` body →
```python
    from production.qlib_features import load_series
    return load_series(instruments, start, end, FWD_RET_EXPR, "fwd_ret_1d",
                       config_path=config_path, dropna=True)
```
- [ ] `backtest/market.py::load_market_proxy` body →
```python
    from production.qlib_features import load_series
    s = load_series(instruments, start, end, MKT_RET_EXPR, "mkt_ret", config_path=config_path)
    return returns_to_close(mean_market_return(s.dropna()))
```
- [ ] `backtest/executability.py::load_entry_ohlc` body →
```python
    from production.qlib_features import load_features
    fields = ["Ref($open,-1)", "Ref($high,-1)", "Ref($low,-1)", "$close"]
    names = ["entry_open", "entry_high", "entry_low", "prev_close"]
    return load_features(instruments, start, end, fields, names,
                         config_path=config_path, dropna_subset=["entry_open", "prev_close"])
```
- [ ] `intraday/exec_backtest.py::daily_open_adj` body →
```python
    from production.qlib_features import load_series
    return load_series(instruments, start, end, "$open", "open", config_path=config)
```
- [ ] `research/_eval_user_exec.py::_daily_oc` body →
```python
    from production.qlib_features import load_features
    return load_features(insts, start, end, ["$open", "$close"], ["open", "close"], config_path=CONFIG)
```

- [ ] **验证(纯)**:`F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_qlib_features.py -q` 过;`python -c "import production.backtest.data, production.backtest.market, production.backtest.executability, production.intraday.exec_backtest, production.research._eval_user_exec; print('import OK')"` 无误(确认委托不破导入/不循环)。
- [ ] **数据回测测试**(test_backtest_data/test_market/test_executability/test_exec_backtest)若需 qlib 数据则在 worktree 可能报数据缺失 —— **跳过并记录**,合并到 main 后由控制者跑(行为锚)。
- [ ] commit `git add -A && git commit -m "refactor(qlib): 5 loaders delegate to qlib_features (behavior-preserving)"`

## 自评判据(控制者)
- `_normalize` 4 测过;5 处委托后 import 干净;领域 loader 签名/返回类型未变;无新接口泄漏给调用方。
