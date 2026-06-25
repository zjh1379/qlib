# qlib 特征加载 seam · 设计 (2026-06-25,arch #1)

## 背景与动机
6+ 个函数各自重写"`QlibDataLoader.load()` → 干净 `(datetime,instrument)` 帧"的同一套仪式(purelib fixup + load + swaplevel + set_names + 列归一),而且**列归一不一致**:`backtest/data.py`/`market.py`/`intraday/exec_backtest.py` 用 `df.iloc[:,0]`,`backtest/executability.py`/`research/_eval_user_exec.py` 用 `df.columns = names`。这个不一致本会话已以崩溃形式付过账(`load_entry_ohlc` 的 `dropna(subset=...)` KeyError,因 QlibDataLoader 返回 MultiIndex/表达式名列)。**没有稳定的"给我一个干净特征帧"接口。**

## 目标(本轮)
一个 deep module **`production/qlib_features.py`**:
- `_normalize(df, names) -> DataFrame` —— **纯函数**(无 qlib):按位置 `df.columns = names`、必要时 `swaplevel`、`set_names(["datetime","instrument"])`、排序。**唯一的列/索引归一处**。
- `load_features(instruments, start, end, fields, names, *, config_path=CONFIG, dropna_subset=None) -> DataFrame[(datetime,instrument), names]`。
- `load_series(instruments, start, end, field, name, *, config_path=CONFIG, dropna=False) -> Series`(单字段便捷)。

5 个现有 loader 改为**薄委托**,**名字/签名/行为不变**(其领域接口有意义,保留):
- `backtest/data.py::load_fwd_returns` → `load_series(..., FWD_RET_EXPR, "fwd_ret_1d", dropna=True)`
- `backtest/market.py::load_market_proxy` 内部 load → `load_series(..., MKT_RET_EXPR, "mkt_ret")`(后续 mean/cumprod 不变)
- `backtest/executability.py::load_entry_ohlc` → `load_features(..., 4 exprs, 4 names, dropna_subset=["entry_open","prev_close"])`
- `intraday/exec_backtest.py::daily_open_adj` → `load_series(..., "$open", "open")`
- `research/_eval_user_exec.py::_daily_oc` → `load_features(..., ["$open","$close"], ["open","close"])`

## 非目标
- **不改任何调用方接口**(`load_fwd_returns` 等保留原名/签名/返回类型 → 现有调用与测试零改动)。
- **不动 `D.features` 路径**(`eval_backfill.py`/`diag_score_sign.py`/`_eval_trend.py` inline):不同 qlib API,本轮不收编(留作后续;`_normalize` 已可复用于它们)。
- 不改复权口径(沿用各 expr 的 `$open/$close/Ref(...)`)。

## 架构 / 数据流
调用方(领域命名薄 wrapper)→ `load_features`/`load_series` → `_normalize`(归一)→ 上游 `QlibDataLoader`。purelib fixup 与 `init_qlib_from_config` 移入本 module 一次。

## 验证 & 成功判据
- `_normalize` 纯单测:喂合成帧(MultiIndex `(instrument,datetime)` 索引、表达式名/多级列),断言输出 `(datetime,instrument)` 索引 + `names` 列、排序正确;单列与多列都覆盖;列数≠names 长度时报清晰错误。
- **回归锚**:迁移后 `production/tests/test_backtest_data.py`、`test_market.py`、`test_executability.py`、`test_exec_backtest.py` 全部仍过(行为不变);新 `test_qlib_features.py` 过。
- 合并到 main 时跑一次数据 smoke(`_eval_user_exec` / `load_entry_ohlc` 真数据)确认真实 qlib 输出仍正确。

## 风险(诚实)
- QlibDataLoader 每 field 返回一列 → `df.columns = names`(按位置)成立;`_normalize` 对长度不匹配显式报错以早暴露。
- `load_fwd_returns` 等的 dropna/rename 细节必须逐一保留(行为锚靠既有测试守住)。

## 文件清单
- 新增:`production/qlib_features.py`(+ `production/tests/test_qlib_features.py`)
- 改:`backtest/data.py`、`backtest/market.py`、`backtest/executability.py`、`intraday/exec_backtest.py`、`research/_eval_user_exec.py`(各 loader 委托)
