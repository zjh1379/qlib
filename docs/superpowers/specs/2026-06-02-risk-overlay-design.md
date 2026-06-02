# P3 设计：市场择时暴露层（Risk Overlay）

> 状态：设计已与用户对齐，待写实现计划。日期：2026-06-02。
> 前置：P2a 已证明 LGBM+ALSTM（持5只/5天换，¥10万）**净 CAGR +19%、net_ir 0.67、各 regime 为正**，但**最大回撤 −56.7%**（A股熊市拖累）。

## 1. 背景与目标
- **问题**：长线 long-only top-K 在 A 股熊市（2021–2024）被整体拖累 → −57% 回撤，对真人不可交易。
- **目标**：在现有选股组合之上叠加**市场择时暴露层**，把最大回撤压到 **~−25%**，同时尽量保住 +19% 的大部分收益。
- **用户决策**：目标回撤 ~−25%（平衡）；**允许减仓/空仓持现金**。
- **关键约束**：完全复用 P0/P1 回测引擎与 P2a 的 2 模型 OOF，**零模型重训**；严格无前视。

## 2. 目标 / 非目标
- **做**：大盘代理 + 趋势/波动暴露信号 + 引擎暴露应用 + 参数扫描 + 长窗口评估。
- **不做（YAGNI）**：拉取真大盘指数（先用 universe 等权代理）、个股止损、行业上限、模型重训。后续如需再议。

## 3. 架构（隔离、可独立测试）
```
production/backtest/
  market.py    # load_market_proxy(start,end,config) -> 大盘代理收盘序列
  regime.py    # compute_exposure(market_close, method, **params) -> Series[0..1]（无前视）
  engine.py    # run_backtest(..., exposure: Series|None)：目标权重整体 × e_lagged
  run.py       # --regime/--ma-window/--band/--min-exposure/--vol-target
  sweep.py     # 增加 regime 维度扫描（命中 −25% 回撤下最高净收益）
```

## 4. 大盘代理（market.py）
- `load_market_proxy(start, end, config_path)`：取 universe 全体的**等权日收益均值**（按 datetime 对 1 日开→开收益求均值）→ `cumprod(1+r)` 得到合成大盘指数。无需额外数据，复用 `data.load_fwd_returns` 的同源价格。
- 理由：策略 long 的是 universe 的 top-K，universe 等权即其天然基准；避免引入真指数数据依赖。真指数拉取列为可选后续。

## 5. 暴露信号（regime.py）
`compute_exposure(market_close, method="trend_ma", ma_window=120, band=0.10, min_exposure=0.0, vol_target=None, vol_window=20) -> pd.Series`（值域 [min_exposure, 1]，索引为 date）：
- **trend_ma（主）**：`raw = (close / Mean(close, ma_window) − 1) / band + 0.5`；`e = clip(raw, min_exposure, 1.0)`。大盘在均线上方→趋满投，跌破→按幅度线性降到 `min_exposure`（可设 0=可完全空仓）。分级、平滑，避免二元开关的来回打脸。
- **vol_target（可选叠加）**：`e *= min(1.0, vol_target / realized_vol(close, vol_window))`（年化）。高波动再压一档。
- **无前视**：`e_t` 仅用 `close` 在 `≤t` 的值（MA/vol 均为尾部窗口）；应用时再滞后一天（见 §6）。
- 暖机期（不足 ma_window）→ `e = 1.0`（或 min_exposure，可配；默认 1.0 不误伤早期）。

## 6. 引擎集成（最小改动）
`run_backtest(scores, fwd_ret, policy, cost, capital, score_col, exposure=None)`：
- 决策日 `d`：`target = policy.target_weights(...)`；若 `exposure` 提供，取 **`e = exposure.asof(d_前一交易日)`（滞后一天，无前视）**，令 `target = target * e`。
- 权重和 = `e`（其余 `1−e` 即现金，计 0 收益）。
- **现有 delta/换手/成本逻辑天然同时覆盖"换股"与"加减仓"成本**（因为目标权重已被 e 缩放，delta 自然包含暴露变化）——只加权重缩放一行，其余不动。
- `gross = Σ(scaled_w · fwd_ret) = e · 组合收益`；`net = gross − cost_frac`；`nav *= (1+net)`。

## 7. 数据流
universe 等权日收益 → 累计大盘代理 → `compute_exposure` → 暴露序列 → 引擎滞后应用到 2 模型 OOF（`oof_2model_2021_2026.pkl`）→ 净收益/回撤/各 regime → `sweep` 扫 regime 参数 → 命中 ~−25% 回撤下的最高净收益配置。

## 8. 测试
- 🔴 **无前视**：`compute_exposure` 的 `e_t` 只依赖 `≤t` 数据（断言：改未来不改当前 e）；引擎应用滞后一天。
- **趋势逻辑**：大盘持续上行 → e→1；持续跌破均线 → e→min_exposure；分级 clip 边界正确。
- **vol_target 数学**：高波动按 `min(1, target/realized)` 压缩。
- **引擎缩放**：`exposure=常数c` 时 `gross = c × 无overlay的gross`（线性）；加减仓成本随 |Δe| 计入。
- **回撤下降（核心）**：合成"先涨后熊"序列上，overlay 的 maxDD 显著小于无 overlay（断言 `DD_overlay > DD_base`，即更接近 0）。
- 合成数据 golden test。

## 9. 验收（对 2 模型 OOF, 2020–2026, ¥10万, 持5只/5天换）
1. 最优 overlay 配置的**最大回撤 ≈ −25%（−20% ~ −30%）**。
2. 净 CAGR 尽量保住（**目标仍 > +10%**；记录 vs 无 overlay 的 +19%）。
3. **各 regime 净 IR 仍为正**。
4. **Calmar（净CAGR/|maxDD|）明显改善**（无 overlay ≈ 0.33 → 目标 > 0.5）。
5. 无前视测试 + 成本测试**全过**；现有 backtest 套件无回归。

## 10. 后续
- 可选：拉真大盘指数（000300/000905/000852）替代 universe 代理做对照。
- 可选：个股止损/行业上限（P3.1）。
- 把"持5只/5天换 + 暴露层"落进产品（独立任务）。
