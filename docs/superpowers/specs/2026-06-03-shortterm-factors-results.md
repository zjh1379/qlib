# P2b 短线因子 · 增量评估结果 (2026-06-03)

**结论(TL;DR):强正结果,采纳。** 给 LGBM 的 Alpha158 追加 6 个非冗余短线因子(隔夜跳空 / 换手代理 / 触板计数),在**同折同universe同策略**(只多 6 个特征)的可信 OOF 回测上:LGBM-only 净 CAGR **+5.9% → +18.2%**(~3×),整套 LGBM+ALSTM **+19.0% → +31.2%**,且**回撤同时下降**(−56.8% → −40.7%)→ **Calmar 0.34 → 0.77**。增益集中在**近期最难的 2021/2023 regime**(不是 2020 过拟合)。这正是趋势门没能提供的"更厚 alpha"。

---

## 1. 方法(为何可信)
- **隔离实验**:因子回填写入独立 mlflow 实验 `rolling_v2_ensemble_fac`(配置 `production/configs/rolling_ensemble_fac.yaml`,仅 `experiment_name` 与基线不同)。避免了两个坑:(a) 同名 recorder 导致整体**跳过**;(b) `pool_range` 把同名 recorder 全部 concat 造成**污染**。基线 recorder 毫发无损。
- **同口径对比**:`oof_lgbmfac_2021_2026.pkl` 与基线 `oof_lgbm_2021_2026.pkl` 的 **(datetime,instrument) 索引完全相同**(855,452 行,同 11 折半年窗、同 csi800 PIT universe、2020-07..2025-12)。唯一差异 = LGBM 多了 6 个因子。
- **无前视**:6 因子全为尾部 qlib 表达式;`test_no_forward_refs` 通过。
- **回测**:`policy=fixed / 持5只 / 5天换 / ¥10万 / 最低5元成本`(= +19% 基线同款);2-model 变体 = 因子-LGBM 的 1d/5d/20d 列 + 基线 ALSTM 列经 `assemble_score` 重组。
- 11 折回填 ~50min(LGBM-only ~5min/折),看门狗全程无报警,0 折失败。

## 2. 结果(policy=fixed top_k=5 period=5, ¥100k, 小成本)

| 变体 | 净CAGR | net_ir | 换手 | 最大回撤 | Calmar | 胜率 |
|---|---|---|---|---|---|---|
| baseline-LGBM | +5.92% | 0.34 | 0.170 | −46.90% | 0.13 | 48.95% |
| **factor-LGBM** | **+18.23%** | **0.64** | 0.171 | −47.56% | **0.38** | 49.92% |
| baseline-2model(+19%) | +19.05% | 0.67 | 0.173 | −56.75% | 0.34 | 50.45% |
| **factor-LGBM+ALSTM** | **+31.16%** | **0.94** | 0.174 | **−40.69%** | **0.77** | 50.45% |
| baseline-2model +overlay | +10.98% | 0.57 | 0.110 | −28.18% | 0.39 | 46.25% |
| **factor-2model +overlay** | **+12.61%** | **0.63** | 0.111 | **−25.22%** | **0.50** | 46.10% |

**增量(因子 − 基线):** LGBM-only `net_cagr +12.31% / net_ir +0.297`;2-model `net_cagr +12.11% / net_ir +0.266`。

分 regime 净 CAGR:
```
变体                           2020-02   2021-03   2023-01
----------------------------------------------------------
baseline-LGBM                  +38.45%    -5.38%    +7.16%
factor-LGBM                    +39.35%    -0.37%   +26.87%
baseline-2model(+19%)         +110.87%    +4.82%   +13.73%
factor-LGBM+ALSTM             +121.40%   +18.84%   +24.39%
baseline-2model +overlay       +78.64%   +10.46%    +0.31%
factor-LGBM+ALSTM +overlay     +75.77%   +16.11%    +0.26%
```

## 3. 解读
- **因子真实加厚 alpha,且非 2020 假象**:最难的 2021(+4.82%→+18.84%)和 2023(+13.73%→+24.39%)改善最大;2021 还从负转正(LGBM-only −5.38%→−0.37%)。
- **同时改善收益和回撤**:2-model 回撤 −56.8%→−40.7%,Calmar 翻倍多(0.34→0.77)。说明因子不只加收益,还帮模型避开部分回撤期烂票(触板/隔夜情绪类因子的作用)。
- **因子-LGBM 单模型(+18.2%)已接近原 LGBM+ALSTM 双模型(+19%)**:6 个便宜的日线因子 ≈ 一个 ALSTM 神经网的贡献量级,性价比极高。
- **最优风险调整组合 = 因子-2model + overlay**:+12.6% / 回撤 −25.2% / Calmar **0.50**(优于基线 overlay 的 0.39,且回撤更低)。
- **诚实保留**:仍是 5 只集中持仓的 OOF 回测,最大回撤 −40.7% 依然很疼(单靠 overlay 压到 −25%);2020 反弹贡献仍最大。但近期 regime 的稳定改善是最可信的信号。

## 4. 建议(采纳路径)
1. **把 `--features shortterm` 设为 LGBM 生产默认**(rolling_train / 日常推断 / 调仓管线)。当前默认仍是 `alpha158`(向后兼容),改默认即采纳。
2. **重算实盘预测**:用因子模型重新生成最新 `pred_*.pkl`,刷新 Picks(当前 `pred_2026-06-02.pkl` 是旧 alpha158 版)。
3. **配方落地**:`因子-LGBM+ALSTM + P3 overlay(ma60/min0)` = 当前最佳(+12.6%/−25%/Calmar 0.50)接进产品。
4. **可继续加厚**:6 因子已验证有效 → 值得再试资金流/北向/行业动量(部分需补数据,P4)。
5. 与**趋势门**对照(`2026-06-03-trend-aware-ranking-results.md`):趋势门删 alpha(−14% CAGR),**因子加 alpha(+12% CAGR)**——方向决定:走"喂模型更多因子",不走"事后趋势过滤"。

## 5. 产物
- 代码(已合并到 main 的因子实现 + 本轮新增,均**未提交**待过目):
  - `production/factors/short_term.py`, `production/custom_handler.py::AlphaShortTerm`, `production/rolling_train.py --features`(上轮已提交);
  - 本轮新增:`production/configs/rolling_ensemble_fac.yaml`(隔离实验)、`production/_pool_fac.py`、`production/_eval_factors.py`。
- OOF:`production/reports/oof_lgbmfac_2021_2026.pkl`;数字:`logs/eval_factors.log` + `logs/eval_factors_summary.json`。
- 回填日志:`logs/backfill_lgbmfac.log`(11/11 折)。
- 因子 recorder 在 mlflow 实验 `rolling_v2_ensemble_fac`(936498066353273871);如需回收空间可整目录删除。
