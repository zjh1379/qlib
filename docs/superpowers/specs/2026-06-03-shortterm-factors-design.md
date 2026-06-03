# P2b 设计：短线因子层（加厚 alpha，无需补数据）

> 状态：设计已对齐，待写实现计划。日期：2026-06-03。
> 实现 2026-05-29 主 spec §5 的因子分析；执行决策：**不补数据**（只用现有 OHLCV+vwap 可算的因子；真换手率 `turn` 留后续）。
> 前置：P2a 给出 LGBM+ALSTM 净 +19%、各 regime 为正，但 alpha 偏薄（Calmar ~0.39）。本期试图**加厚 LGBM 侧 alpha**。

## 1. 目标
往 LGBM 的 Alpha158 之上加一批**非冗余、A股特异**的短线因子（全部用现有数据可算），在 2021–2026 长窗口上检验是否抬升净收益/IC，并看 (LGBM+因子)+ALSTM 是否超过现有 +19%。**零补数据、LGBM-only 回填（快、安全）。**

## 2. 因子集（全部纯 qlib 表达式，现有数据可算）
去冗余原则见主 spec §5.1（已剔除与 Alpha158 重复的：日内动量=KMID、振幅=KLEN、短期反转=ROC5、量价背离=CORR、量能=VMA、影线≈KUP/KLOW）。入选：

| 因子 | 表达式 | 为何非冗余 |
|---|---|---|
| OVNGAP | `$open/Ref($close,1)-1` | Alpha158 无"今开 vs 昨收"；隔夜跳空 |
| OVNGAP_MA5 | `Mean($open/Ref($close,1)-1, 5)` | 隔夜跳空趋势 |
| OVNGAP_STD5 | `Std($open/Ref($close,1)-1, 5)` | 隔夜波动 |
| AMT_SURGE | `($vwap*$volume)/Mean($vwap*$volume,20)` | 资金额突增（vwap 已有）；Alpha158 仅 volume 比率 |
| LIMITUP10_CNT20 | `Sum(Gt($close/Ref($close,1)-1, 0.095), 20)` | 近20日~10%涨停触板数；Alpha158 无涨跌停概念 |
| LIMITUP20_CNT20 | `Sum(Gt($close/Ref($close,1)-1, 0.19), 20)` | ~20%板（创业/科创）触板数 |

板块无关近似（用 9.5%/19% 双阈值）规避 per-instrument 板块常量；精确 board-aware 版留 fast-follow。特异波动率(vs 大盘)与真换手率留后续（需更多数据/复杂度）。

> 若 `Gt`/`Sum` 在本机 qlib 版本表达式引擎不可用，实现期测试会暴露 → 回退为 Python 计算的因子合并（同 P0/P1 limit 方案）。

## 3. 集成
- 新 handler `production/factors/short_term_handler.py::AlphaShortTerm`：继承 `Alpha158_OpenH`，`get_feature_config` 在 158 之上**追加**上述因子块（表达式+名称）。
- 仅喂 **LGBM**（表格路径）；神经网络注入短线因子留 P3+。
- 纯函数 `short_term_factor_config() -> (exprs, names)` 便于单测。

## 4. 评估
1. **LGBM+因子长回填**：`backfill_longwindow`/`rolling_train backfill` 用 AlphaShortTerm，2021→2026 半年折（~10 折，**LGBM-only，~2h，watchdog 下**）。
2. `pool_range` → `oof_lgbmfac_2021_2026.pkl`。
3. **对比**（用 P0/P1 引擎，fixed/k5/p5/¥10万/小成本）：
   - LGBM+因子 vs LGBM-baseline（`oof_lgbm_2021_2026.pkl`）：净 IC/CAGR/IR 是否抬升？
   - (LGBM+因子)+ALSTM vs 现有 LGBM+ALSTM 的 +19%：3 输入是否更好？
   - 叠加 P3 暴露层后回撤/Calmar 是否改善。

## 5. 测试
- `short_term_factor_config` 表达式/名称正确、无 forward Ref（无前视：只用 Ref(...,+k) 尾部）。
- handler 在小日期范围实例化产出含新列且无 NaN 爆炸（冒烟）。
- 因子值合理性（OVNGAP 在已知合成数据上数值正确）。

## 6. 验收
- 加因子后 LGBM 的**净 IC 或净 CAGR 较 baseline 有可测提升**（报告 delta；薄信号不强设阈值）。
- (LGBM+因子)+ALSTM 的净 CAGR ≥ 现有 +19%（或在同回撤下更优）。
- 无前视/现有 backtest 套件无回归。
- 诚实记录：若因子无显著增量，如实报告（负结果也是结论）。

## 7. 非目标 / 后续
- 真换手率（需补 turn/流通股本）、特异波动率、board-aware 精确涨停、神经网络因子注入 → 后续。
- 算力：仅 LGBM-only 回填（轻），watchdog 下；不碰神经，无卡死风险。
