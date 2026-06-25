# per-period metrics 统一 · 设计 (2026-06-25,arch #3)

## 背景与动机
"把每周期收益序列 → 年化 CAGR / maxDD / Calmar / 胜率"的同一段数学被**复制两份**:`intraday/exec_backtest.py::simulate`(`eq^(252/(period·n))`)与 `research/_eval_user_exec.py::_metrics`(`eq^(252/(hold·n))`)。两份口径其实一致(都是 per-period 序列、bars_per_period=持有天数),但各写各的 → 漂移风险 + 无单测。(`metrics_net.net_metrics` 是**日频** ledger 的 `252/n`,输入形状不同,**不在本次合并范围**。)

## 目标(本轮,scoped)
在 `production/backtest/metrics_net.py` 加一个**纯函数** `period_metrics(returns, *, bars_per_period=1, periods_per_year=252) -> dict`,返回 `{net_cagr, calmar, max_dd, win, n_periods}`(键与两个 sim 调用方一致 → 调用方零接口改动)。把上述**两处** per-period 数学改为调用它。

## 非目标
- **不动 `net_metrics`/`tail_stats`**(日频 ledger 路径,形状不同;强行合并会让接口和实现一样宽 = 反 deep)。这是评审里把 #3 标 Speculative 的原因 —— 只做"统一 per-period 数学"这个安全增量,不强造单一 ledger。
- 不改任何数值/输出(只换实现)。
- `engine.run_backtest` 的日频 ledger 不动。

## 验证 & 成功判据
- `test_metrics_net.py` 加 `period_metrics` 纯单测:按 bars_per_period 年化正确、maxDD/Calmar、胜率、空序列。
- **回归(数值不变)**:控制者跑 `_eval_user_exec`(用 `_metrics`→`period_metrics`)与 `_eval_am30_entry`(用 `exec_backtest.simulate`→`period_metrics`),输出须与迁移前逐格一致(_eval_user_exec:hold3 close +21.75%/Cal0.34、hold5 open +22.11%;_eval_am30_entry:open +22.1%、am30_vwap −1.85%、Δ−24pp)。

## 文件清单
- 改:`production/backtest/metrics_net.py`(+ `period_metrics`)、`production/tests/test_metrics_net.py`(+ 测)
- 改:`production/intraday/exec_backtest.py::simulate`、`production/research/_eval_user_exec.py`(各自的 per-period 数学 → `period_metrics`)
