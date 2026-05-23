# 模型优化方案

> 评估日期: 2026-05-23
> 评估工具: `production/eval_recorder.py` (in-process service from `app/evaluation/`)
> 数据范围: 当前 `examples/mlruns` 中的所有 recorder

## 1. 已有模型清单

| recorder_id | experiment | run_name | pred_start | pred_end | rows |
|---|---|---|---|---|---|
| `73a9ee6c99f6` | rolling_v2_ensemble | lgbm_20d_2026-05-10 | 2026-05-06 | 2026-05-08 | 900 |
| `bb768505a935` | rolling_v2_ensemble | lgbm_5d_2026-05-10  | 2026-05-06 | 2026-05-08 | 900 |
| `a6e5b8243464` | rolling_v2_ensemble | lgbm_1d_2026-05-10  | 2026-05-06 | 2026-05-08 | 900 |
| `f29f042f7263` | daily_cn_fresh      | mlflow_recorder     | 2025-01-02 | 2026-05-08 | 96772 |

注：`daily_cn_fresh` 是 P1 阶段单模型 LightGBM (Alpha158 特征, 1d horizon)，目前 Picks 页和图表的预测都来自它。
`rolling_v2_ensemble` 下三个 recorder 是 β phase 仅 LightGBM 部分滚动训练的产出（ALSTM/TRA/Ridge stacking 尚未跑完），且预测窗口仅 3 个交易日 — 还不构成"可对比"的候选模型。

## 2. daily_cn_fresh 评估结果 (现行 Picks 数据源)

- **recorder_id**: `f29f042f72634226aa0dc7782d4873d9`
- **window**: 2025-01-02 → 2026-05-08（96772 行 (date, symbol) 对）
- **TopK** = 30  |  **cost** = 10 bps

### 8 指标记分卡

| 指标 | 取值 | 阈值 | 通过 |
|---|---|---|---|
| IC mean | +0.0263 | ≥ 0.030 | FAIL |
| RIC mean | +0.0166 | — | — |
| ICIR | +0.1978 | ≥ 0.40 | FAIL |
| Top-Bottom Spread (monthly) | +1.05% | ≥ 1.5% | FAIL |
| Annual Excess Return | +41.91% | ≥ +15% | PASS |
| IR (cost-adjusted) | +1.5594 | ≥ 2.5 | FAIL |
| Max Drawdown | -20.40% | ≥ -15% | FAIL |
| Daily Turnover | 48.95% | ≤ 20% | FAIL |

### Regime 分段

| 段 | 开始 | 结束 | IR | IC | 样本数 |
|---|---|---|---|---|---|
| Recent (full window) | 2025-01-02 | 2026-05-08 | +1.5594 | +0.0263 | 96772 |

（当前 regime 切分器只产出一个全窗口段；待数据跨度更长或出现明显 regime shift 后会再细分。）

### 验收结论: **FAIL**

未通过的判据：
- `ic_mean` FAIL  — 真实 IC=+0.0263，比阈值 0.030 低约 12%
- `ir` FAIL — 真实 IR=+1.5594，比阈值 2.5 低约 38%
- `max_drawdown` FAIL — 真实 -20.40%，比阈值 -15% 还低约 5.4 个百分点
- `daily_turnover` FAIL — 真实 48.95%，是阈值 20% 的 2.4 倍
- `regimes_all_positive` PASS — 全窗口 IR 为正

## 3. rolling_v2_ensemble (β 阶段) 评估尝试

挑了 `bb768505a935` (lgbm_5d_2026-05-10) 来评估：

- **window**: 2026-05-06 → 2026-05-08（**仅 3 个交易日 / 900 行**）
- 评估**形式上执行成功**，但数字几乎不可信（详见下面"为何不能轻信"）。

### 数字（仅供调试参考，勿用作模型决策）

| 指标 | 取值 | 阈值 | 通过 |
|---|---|---|---|
| IC mean | +0.0561 | ≥ 0.030 | PASS |
| RIC mean | +0.0381 | — | — |
| ICIR | +0.7551 | ≥ 0.40 | PASS |
| Top-Bottom Spread (monthly) | +1.22% | ≥ 1.5% | FAIL |
| Annual Excess Return | +224.15% | ≥ +15% | PASS (annualized from 3 days — 高度膨胀) |
| IR (cost-adjusted) | +14.8051 | ≥ 2.5 | PASS (3 天年化后膨胀严重) |
| Max Drawdown | 0.00% | ≥ -15% | PASS (3 天还没机会回撤) |
| Daily Turnover | 57.78% | ≤ 20% | FAIL |

### 为何不能轻信

3 个交易日的窗口里：
- ICIR 的分母（IC 标准差）极不稳定 — 任何一天的 IC 极端值都会主导
- IR / annual return 是按日均收益 × √252 年化的 — 在 3 天里若刚好赶上一波上涨，年化数字会被放大 70~80 倍
- Max DD = 0% 仅说明这 3 天里没有任何一段累计净值低于历史最高 — 不是模型抗回撤能力的证据

**这次评估是工具链 smoke test 的成功**：流程能跑通、记分卡能产出、JSON/Markdown 报告能落盘；但要把 β 模型真正纳入"候选"，至少需要 60 个交易日（约一个季度）的样本外预测。

## 4. 当前问题诊断

基于上述测量，目前 Picks 页给出的所有 rank 都来自 `daily_cn_fresh`，而该模型：

- **IC = +0.0263**（相比阈值 0.030 低 12%）— 信号方向勉强为正
- **ICIR = +0.1978**（相比阈值 0.40 低 50%）— IC 在不同日子之间波动很大，稳定性差
- **IR = +1.5594**（相比阈值 2.5 低 38%）— 扣完 10 bps 成本后年化收益/波动比一般
- **Max DD = -20.40%**（比阈值 -15% 还差 5.4 pp）— 持仓有过实质性回撤
- **Daily Turnover = 48.95%**（是阈值 20% 的 2.4 倍）— 每天换近一半，交易成本压得很重
- **Annual Excess = +41.91% PASS** — 这是 8 项里唯一过线的，但和高换手叠加后实际收益打折严重

换句话说，**当前模型有微弱的方向性信号（IC 不到 0.03），但稳定性、回撤、换手率都不达标，扣成本后 IR 不到 2.5 — 离"可信"还差一截，离"可用"勉强算个候选池**。Annual Excess +41.91% 看起来漂亮，但它没扣换手成本（10 bps × 49% × 252 ≈ 12% 一年），扣完后实际增益会小很多。

## 5. 优化方案

### 短期 (1-2 天)

1. **暴露问题给用户** — 在 Picks 页顶部加一个 banner（已有 Settings 入口，可以做成"模型状态"折叠条）：
   > 「当前模型 IC=0.026 (验收阈值 0.030)，预测信号弱。建议尽快训练 β 阶段集成模型。」

2. **明确数据源切换路径** — Settings 已有 `default_experiment` 字段（T5/T20 已完成）。一旦 β 阶段训完，用户能一键从 `daily_cn_fresh` 切到 `rolling_v2_ensemble`，无需重启后端。

### 中期 (4-6 小时 GPU 时间)

3. **完整跑一次 β phase 滚动训练** — 按 spec `docs/superpowers/specs/2026-05-21-rolling-ensemble-algorithm-design.md` 执行 1 次 `production/rolling_train.py run-once`。预期产出:
   - LightGBM × 3 horizons (CPU, ~15 min) — **3 个已存在，但预测窗口仅 3 天**，需要扩到 60+ 天
   - ALSTM × 3 horizons (GPU, ~20-60 min)
   - TRA × 3 horizons (GPU, ~45-120 min)
   - Ridge stacking 整合 9 个 base preds
   - 一份 ensemble pred.pkl + 写入 mlruns

4. **用同样工具评估 β 集成模型**:
   ```
   python -m production.eval_recorder list
   python -m production.eval_recorder eval <new_ensemble_recorder_id>
   python -m production.eval_recorder compare f29f042f72634226aa0dc7782d4873d9 <new_ensemble_id>
   ```
   核对 IC/IR/MDD 是否过 spec §11 验收门槛（IC ≥ 0.030, IR ≥ 2.5, MDD ≥ -15%, daily turnover ≤ 20%, 各 regime 段 IR > 0）。Compare 的 paired t-test 会给出统计显著性 p 值，避免"看起来涨了一点"的伪进步。

5. **如果通过验收**: 在 Settings UI 改 `default_experiment = "rolling_v2_ensemble"` → Picks 页和图表都会自动切换数据源（前端已有 T20 view selector + consensus column 支持）。

### 长期 (Tier 2/3/4 数据补充)

6. 接 baostock 季频财报 → 加 PE/PB/ROE 筛选维度（Tier 2）
7. 接 akshare 资金流 → 加主力净流入维度（Tier 3）
8. 启用 spec §8 的 shadow paper trading 框架，新模型 4 周对照后才接管 Picks（Tier 4，T19 的 shadow tracker 和 auto-rollback 已实现，只待数据通路接上）

## 6. 现行交易行动建议

在 β 模型未完成训练之前：

- **不要 100% 依赖 Picks 页的排名做决策** — 当前模型 IC=0.026, ICIR=0.20，本质上是"略好于随机"的弱信号
- **可以用作"候选池"** — Top 30 不算精挑细选，但至少不是从全市场盲选，比掷骰子强
- **重视 consensus 列** — 如果将来 β 模型上线，consensus 列才会真正有意义（现在 view selector 只能看 `daily_cn_fresh` 单源，consensus 全是 0）
- **使用 Tier 1 筛选作为主要筛选手段** — 价格区间 / 板块 / 创新高 / 成交额 / 排除 ST 这些是**确定性指标**，不依赖弱预测；模型排名只用作"在筛完的候选里再排一次序"，不要倒过来用
- **关注高换手警示** — 当前模型 daily_turnover ≈ 49%，意味着如果每天都跟着 Top 30 换仓，光交易成本一年要吞掉约 12% 收益。建议**至少 5 个交易日才调一次仓**，或只保留新进入 Top 30 且预测分数明显高出旧持仓的标的
- **关注 -20% 回撤风险** — 历史窗口里出现过 20% 级别的回撤段，应预设单笔仓位上限（建议 ≤ 5%）和总仓位上限（建议 ≤ 60%）以控总回撤
