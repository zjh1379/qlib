# 模型迭代 · 交接报告 (2026-06-03)

目标:迭代更好的**短线(1–5天)选股模型**,以**扣费后净收益**为先。本轮把"能不能信"和"能不能赚到手"打通,并量化了真实战绩。

## 一、做到哪一步了(全部已合入 main)

| 阶段 | 内容 | 关键结果 |
|---|---|---|
| **P0** 可信回测 | 成本感知(含最低5元)、持仓周期感知、多regime真回测引擎(`production/backtest/`) | 戳穿旧 `metrics.py` 的"年化54%"假象(它把多日收益当单日重复计数) |
| **P1** 降换手 | Daily/FixedPeriod/**Banded** 策略 + 中性化 + 扫描 | **日频调仓=资金归零;持5只/5天换是生存线** |
| **P2a** 长窗口可信评估 | 粗粒度可断点续回填(`rolling_train --step-weeks/--test-weeks`)+ `pool_range` + 编排;回填 LGBM(11折)+ALSTM(5折) 2021–2026 | **LGBM+ALSTM = 净CAGR +19%、net_ir 0.67、各regime为正**(¥10万,持5只/5天)。LGBM-only 仅 +6%。**P1的+27%是行情运气** |
| **P3** 风控暴露层 | 大盘代理(`market.py`)+趋势暴露信号(`regime.py`)+引擎暴露(`--regime`) | **回撤 −57%→−28%**,保住净CAGR +11%,各regime正。波动率目标(vt)无效已弃。备选 ma200/min0.2=+16.6%/−37%/Calmar0.45 |
| **P2b** 短线因子(代码) | 6个非冗余因子(隔夜跳空/换手代理/触板计数)+`AlphaShortTerm` handler +`--features shortterm` | **代码已合并+测试;增量评估(T4回填)尚未跑** ← 开放线索 |

## 二、当前"最好的模型"(可交付)
**LGBM+ALSTM 集成,持5只/5天换,可叠 P3 暴露层(ma60/min0)。**
- 净CAGR +19%(无风控)→ 加暴露层 +11%/回撤−28%。
- 数据:qlib daily(baostock),CSI800,2018-01起;已更新到 **2026-06-02**。

## 三、关键结论(诚实)
1. 评估基础现在**可信**(以前是坏指标)。
2. **换手纪律 > 选股**:低换手是正负收益的分水岭。
3. **ALSTM 真实加厚 alpha**(6%→19%),神经模型值得留。
4. **头号短板:alpha 偏薄、Calmar 仅 ~0.39。**
5. **模型有反转偏好** → top 排名里多是超跌的下跌趋势票(实盘需趋势过滤,否则接飞刀)。
6. **TRA 在长窗口是数据死胡同**(早期折训练窗早于2018数据起点→空训练集);它属于"每周生产管线",不属于长回填。
7. 小资金(¥3300+最低5元)→ 集中度不可避免,只能精选1只+严格止损。

## 四、未完成 / 开放线索
- **P2b-T4**:短线因子的增量评估没跑(代码就绪)。`docs/.../plans/2026-06-03-shortterm-factors.md` T4 runbook。
- **趋势感知**:模型反转偏好导致 top 名单多为空头排列(本轮实盘已证实);缺一个趋势门/趋势因子。
- **更厚 alpha**:核心缺口未解。
- **产品化**:"持5只/5天 + 暴露层 + 趋势过滤"尚未落进 app/Picks(页面仍是原始排名)。

## 五、下一步计划(按 ROI 排序;★=建议立即做)
1. **★ 跑 P2b-T4 因子增量评估**(~2h LGBM-only 回填,代码就绪)。判定短线因子到底加不加厚 alpha;加则纳入,不加则记负结果。**最便宜、最该先做。**
2. **趋势感知排名**:给模型加"趋势门"或趋势因子(如 close>MA20、动量),让**可交付的 top 名单不再是空头飞刀**——直接提升实盘可用性,大概率改善 Calmar。可做成新因子(进 P2b)或选股后处理过滤。
3. **更厚 alpha(核心)**,三选一/组合:
   (a) 排序目标 LambdaRank 替代 MSE 回归;
   (b) 更多/更强短线因子(资金流、行业动量、北向——部分需补数据);
   (c) **P4 日内5分钟数据**(此前推迟的真正新杠杆,微结构 alpha)。
4. **风控/组合精修**:个股止损、行业/波动中性 → 把 Calmar 从 0.39 往 0.6+ 推。
5. **产品化**:把验证过的配方(持5只/5天 + 暴露层 + 趋势过滤)接进后端/Picks,日常用得上。
6. **TRA**(可选):若要补第三模型,放到每周生产管线修(非长回填)。

**建议路径:先做 1(验证因子)→ 2(趋势感知,实盘最痛)→ 再决定 3 走 LambdaRank 还是日内数据。**

## 六、文档与产物
- 设计/结果 specs:`docs/superpowers/specs/2026-05-29…/2026-05-31…/2026-06-01…/2026-06-02…/2026-06-03…`
- 计划:`docs/superpowers/plans/2026-05-29-net-return-backtest-p0-p1.md`、`2026-05-31-p2a…`、`2026-06-02-risk-overlay.md`、`2026-06-03-shortterm-factors.md`
- 引擎/因子代码:`production/backtest/`、`production/{market,regime,neutralize,fetch_industry,factors/short_term}.py`、`production/custom_handler.py::AlphaShortTerm`、`rolling_train.py(--features/--step-weeks)`
- OOF/评估产物:`production/reports/oof_*.pkl`、`long_*`、`risk_overlay_result.json`
- 最新预测:`examples/mlruns/pred_2026-06-02.pkl`(LGBM+ALSTM)
- 一次性脚本(未提交,可删):`production/_pick_*.py`、`_t5_overlay_sweep.py`
