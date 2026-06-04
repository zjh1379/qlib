# LambdaRank 排序目标 · 结果 (2026-06-04)

**结论(TL;DR):负结果,否决。** 把 LGBM 目标从回归(MSE)换成排序(`objective=lambdarank`,同因子特征、同结构超参),在同一可信回测(fixed/持5/5天,¥10万,小成本,OOF 2020-07→2025-12)上**远差于 MSE 冠军**:rank-2model **+5.2% / Calmar 0.08**,MSE factor-2model **+31.2% / Calmar 0.77**。MSE 完胜。

## 动机
稳健性检验(`2026-06-04-robustness-results.md`)证明 alpha 高度集中在 top-5。假设:用 LambdaRank 直接优化"排名顶端"(NDCG@5/10)→ 比 MSE 回归更准。

## 做法
- 新增 `production/lgbm_rank.py::LGBRankModel`(qlib 兼容):把连续 forward return 按**每日横截面分位**转成 0..15 的相关性等级,按**每日 query group**喂 `lgb.train(objective="lambdarank", metric="ndcg", eval_at=[5,10])`;`predict` 返回打分 Series(与 LGBModel 同形,可picklable、可被 daily_inference 调用)。单测 `test_lgbm_rank.py`(等级/分组逻辑,5 项全过)。
- `rolling_train` 加 `--objective {mse,lambdarank}`(默认 mse,向后兼容;`run_split` 不传 → 生产不变)。
- 同因子特征(shortterm)+ 隔离实验 `rolling_v2_ensemble_rank`,LGBM-only 回填 11 折(1 折 2022-07-01 首次因多进程 MemoryError 失败,补跑成功)→ pool → 与 MSE 因子模型同口径对比。

## 结果(fixed/持5/5天,¥10万,小成本)
| 模型 | 净CAGR | net_ir | 回撤 | Calmar | 去-2020 | 负年 |
|---|---|---|---|---|---|---|
| baseline-2model | +19.0% | 0.67 | −56.7% | 0.34 | +15.1% | 2 |
| **factor-2model(MSE,冠军)** | **+31.2%** | **0.94** | −40.7% | **0.77** | +27.4% | 2 |
| rank-2model(lambdarank) | +5.2% | 0.32 | −67.6% | 0.08 | +7.6% | 4 |
| factor-LGBM(MSE) | +18.2% | 0.64 | −47.6% | 0.38 | +13.6% | 1 |
| rank-LGBM(lambdarank) | +3.2% | 0.29 | −51.5% | 0.06 | +5.6% | 3 |

逐年 净CAGR(factor MSE vs rank lambdarank):
```
              2020     2021     2022     2023     2024     2025
factor(MSE)  +75.0%  +87.6%   -7.4%   -5.8%  +27.9%  +60.0%
rank(λrank)  -15.8%  +65.5%  -27.6%  -13.6%  -14.8%  +63.8%
```
配对 t 检验(日净,treatment − control):
- rank-2model vs factor-2model:**−21.1%/yr,t=−1.75,p=0.08** —— lambdarank 显著更差。
- rank-LGBM vs factor-LGBM:−18.4%/yr,t=−0.96,ns。

## 解读
- **NDCG 排序丢掉了收益"幅度",MSE 保留了它。** 本策略的反转 edge 恰恰在幅度上(rank-average 融合 + top-5 选择需要"涨多少"而非仅"谁前谁后")。lambdarank 把幅度信息压成序数等级 → 丢了 alpha。
- **2020 崩盘(−16% vs +75%)**最刺眼:lambdarank 完全没抓住那波幅度驱动的反转。
- 诚实保留:用的是 MSE 调好的超参(lambda_l1/l2=205/580)+ 固定 16 等级 + NDCG@5/10,**未对 lambdarank 单独调参**。但 −26pp 的差距 + 逐年全面落后 + 2020 崩塌,说明是**目标函数与本任务的根本不匹配**(幅度重要),调参极不可能翻盘到 +31% 之上。

## 建议(否决后的方向)
1. **保留 MSE**,factor-2model(+31%/Calmar 0.77,top_k=5)仍是冠军。
2. 便宜的日线/目标层杠杆(趋势门、分散、排序目标)已**三连否决** → 剩下的真杠杆要么**新数据**(更多因子:资金流/北向/行业动量,需补;或 P4 日内5min),要么**风控**(个股止损平滑负年、overlay 实盘化)。
3. 下一步性价比最高:**个股止损**(便宜、直接治 2022/2023 负年 + 压回撤),或决定投入补数据做新因子。

## 产物
- 代码:`production/lgbm_rank.py`(+`tests/test_lgbm_rank.py`)、`rolling_train.py --objective`、`configs/rolling_ensemble_rank.yaml`、`_pool_rank.py`、`_eval_rank.py`。
- 顺带修复:`backfill_pool.pool_range` 现按 (model,horizon,fold) 保留最新 recorder(re-run 去重),修了本轮"非唯一索引"的 pool 崩溃。
- 数:`production/reports/oof_lgbmrank_2021_2026.pkl`、`logs/eval_rank.log`。隔离实验 `rolling_v2_ensemble_rank`(404411371341101781)可整目录删除回收空间;不影响线上(线上读 `rolling_v2_ensemble`)。
