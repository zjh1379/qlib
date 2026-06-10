# 资金流因子 (money-flow) · 设计 (2026-06-10)

## 背景与动机
- 本仓统一结论:模型 edge = **日频均值反转(超跌反弹)**;5 个常规日线/风控/执行杠杆已全部因"对抗或偏离反转"被否决。天花板只随**新数据 / 新信号**抬升。
- 已验证的因子采纳(+12pp:factor-2model +31.2% vs 基线 +19%)用的是 **qlib 表达式因子**(OVNGAP/AMT_SURGE/LIMITUP_CNT20 等),全部由现有 OHLCV 衍生——没有引入新数据。
- 本设计引入**真·新数据:逐个股资金流(主力/大单净流入)**——订单流而非价格行生,赌它与价格反转 edge **正交**,从而在 +31.2% 冠军之上再加 alpha。

## 目标(已确认)
给 **factor-LGBM** 增加逐个股资金流特征,重训,在规范扣费回测下检验 factor-**MF**-2model 能否**稳健跑赢** factor-2model 冠军(+31.2%)。**判定标准与 +12pp 因子采纳同尺**:净 CAGR 稳健为正增量(逐年基本不输、配对 t 不更差、回撤不恶化)才采纳;否则诚实记负、不采纳。

## 非目标 (out of scope)
- 不动 ALSTM(Alpha360)、不动 TRA。
- 不引入北向资金个股(2024-08 起交易所停披露个股实时持股→实盘不可得)。
- 不做 sidecar 合并(选定 qlib 原生数据层方案 A)、不做"仅大盘/行业覆盖层"方案(C 与现有 P3 重叠)。
- 不在本 spec 内做 P2/实时盘中(资金流为 EOD 日频)。

## 路线(已确认)
**方案 A:qlib 原生新数据层。** 资金流 dump 成新 qlib 字段 → handler 里用表达式因子 → 重训 factor-LGBM → 增量评估。复用 +12pp 因子范式,**服务端不动**(后端照常读 qlib),多日窗口经 qlib 表达式(`Mean`/`Ref`)免费拿。**历史深度先 spike 探底再定跨度**(Step 0 阻断门)。

## 分阶段架构(Step-0 阻断门先行,沿用日内 P1 纪律)

### Step 0 — 数据 spike(阻断门)
- 实拉 akshare 个股资金流(候选:`stock_individual_fund_flow(stock, market)` 逐日历史;`stock_individual_fund_flow_rank` 当日截面)。确认:字段(主力/超大/大/中/小单净额与净占比)、**历史能回到哪年**、更新时点(EOD)、代码格式映射(akshare `600519`/`sh` ↔ 本仓 `SH600519`)。
- **据 spike 定训练+评估跨度**:回到 ~2018 → 回补全 OOF(2020-07→2025-12);只有近期(如 ≥2022/2024)→ 资金流因子的训练+评估**限在覆盖窗口内**做增量验证(并在结果里标注统计力受限)。
- 阻断判定:拉不到/字段缺失/连近 1 年都没有 → 停止,改评估 tushare pro(若用户有 token)或退回大盘/行业资金流覆盖层;不强行继续。

### Step 1 — 拉取 + 缓存 `production/factors/moneyflow_fetch.py`
- 按 **CSI800 PIT universe 逐日**(防生存偏差,复用 `pit_constituents`)拉逐日资金流;鲁棒:校验返回、失败重试、**仅成功落缓存**(可断点续跑)——复用 baostock 5min 层的经验。
- 缓存为 parquet,产出 tidy 表:`(datetime, instrument, mf_main_net, mf_main_ratio, ...)`。

### Step 2 — dump 成 qlib 字段
- 对齐 `cn_data_bs` 的 day 日历 + instruments;把资金流列 dump 成新 qlib 字段——**主力/超大/大/中/小单净额** `$mf_main_net` / `$mf_xl_net` / `$mf_l_net` / `$mf_m_net` / `$mf_s_net`(+ 可选净占比 `$mf_main_ratio`)+ 可得性掩码 `$mf_avail`——并入 `cn_data_bs` provider(或并行 provider,加载时合并)。Step 3 的因子表达式只引用这组已 dump 的字段。
- 缺口处理:某票某日无资金流(停牌/未覆盖)→ 填 0 并加一个 `$mf_avail` 掩码位(让模型可区分"真 0 流入"与"无数据")。
- 验证:`D.features(['$mf_main_net'])` 能读回、与 `$close` 同 (datetime,instrument) 对齐、日期不串位。

### Step 3 — 资金流因子(handler)`production/factors/moneyflow.py`
- `moneyflow_factor_config() -> (fields, names)`:qlib 表达式因子,起步**最小集(YAGNI,3–5 个)**,例如:
  - 主力净流入额对成交额归一化:`$mf_main_net / ($amount + 1e-12)`(当日;若 `cn_data_bs` 无 `$amount` 字段则用 `$close*$volume` 代理,Step 0/2 确认);
  - 5 日累计归一化主力净流入:`Mean($mf_main_net, 5) / (Mean($amount,5)+1e-12)`;
  - 大/小单失衡:`($mf_xl_net + $mf_l_net) - ($mf_m_net + $mf_s_net)` 归一化;
  - 资金流动量:`Mean($mf_main_net,5) - Mean($mf_main_net,20)` 归一化;
  - 可得性掩码 `$mf_avail`。
- 新 handler `AlphaShortTermMF(AlphaShortTerm)`:`get_feature_config()` 在 158+短期因子基础上再追加 `moneyflow_factor_config()`。
- TDD:能纯函数化的(归一化辅助、因子配置名/数量/无重复)写单测;表达式正确性在 Step 2 的 round-trip + 小样本数值核对里验证。

### Step 4 — 重训 factor-LGBM + pool
- 用 `AlphaShortTermMF` 跑因子 backfill(Step-0 定的跨度)→ OOF `production/reports/oof_lgbmmf_*.pkl`。
- ALSTM 不变;经 `production/score_utils.rebuild_2model(oof_lgbmmf, oof_2model)` 重建 factor-MF-2model。
- 算力:仅 LGBM 重训(与 +12pp 因子采纳同量级,可行);ALSTM/TRA 不重训。

### Step 5 — 增量评估 `production/research/_eval_moneyflow.py`
- 规范回测 `build_report`(fixed/持5/5天/top_k=5,¥10万,小成本):
  - factor-2model(冠军,+31.2%)vs factor-**MF**-2model;各自 ±overlay(`regime={'method':'trend_ma','ma_window':60,'band':0.10}`)。
  - **同索引配对**(对齐到两者共同的 datetime×instrument)保证公平;逐年(尤其 2022/2023);配对 t 稳健性(复用 `_eval_robustness._paired_t`)。
- 复用 `score_utils` + 已加深的 `build_report(regime=...)`(一行拿净指标)。
- 诚实判定写结果 spec;达标→Step 6,否则止于 Step 5(记负)。

### Step 6(达标才做)— 产品化
- 翻 handler 默认到 MF(或新增 `--features shortterm_mf`)、重生服务 pred.pkl(`_regen_live_factor` 同款)。
- 把资金流拉取接进 `daily_inference` 的 data-refresh(EOD 追加当日资金流到 qlib 字段)→ 服务口径与训练一致、保持最新。
- 改了管线/服务 → **按标准指令重启前后端供查看**。

## 组件清单(新增/改动)
- `production/factors/moneyflow_fetch.py`(akshare 拉取 + PIT + 缓存)
- `production/factors/moneyflow.py`(`moneyflow_factor_config` 表达式因子)
- `production/factors/__init__.py`(若无)
- `production/custom_handler.py`(`AlphaShortTermMF` 子类)
- 资金流 dump 脚本(扩展现有 dump_bin 工具 → `$mf_*` 字段)
- `production/research/_eval_moneyflow.py`(增量评估,复用 `score_utils` + `build_report`)
- 测试:`production/tests/test_moneyflow.py`
- 结果:`docs/superpowers/specs/2026-06-10-moneyflow-factor-results.md`(Step 5 跑完写)

## 数据流
akshare 资金流 → `moneyflow_fetch`(PIT 逐日 + 缓存)→ 对齐 + dump → qlib `$mf_*` 字段 → `AlphaShortTermMF.get_feature_config` → factor-LGBM 训练 → OOF → `rebuild_2model` → `build_report` 增量评估 →(达标)产品化 + 接入 daily-refresh。

## 实盘一致性(关键)
训练与服务**同源同口径**:akshare 资金流每日 EOD 更新;`daily_inference` 在推理前把当日资金流追加进 qlib `$mf_*` 字段,再跑推理。避免训练/实盘 skew。源选择与 EOD 时点在 Step 0 一并确认。

## 错误处理
- Step-0 阻断门:无/过浅数据即停,不硬上。
- 拉取鲁棒:akshare 偶发失败 → 重试 + 仅成功缓存(断点续跑)。
- 对齐缺口:无资金流的票/日 → 填 0 + `$mf_avail` 掩码。
- 生存偏差:按 PIT 逐日 universe 拉取与训练。
- dump 校验失败(读回错位/缺字段)→ 阻断 Step 4。

## 测试
- TDD 纯函数:资金流归一化辅助、`moneyflow_factor_config` 的名字/数量/无重复、akshare 代码↔qlib 代码映射。
- dump round-trip:`$mf_*` 读回与 OHLCV 对齐、小样本数值与原始 akshare 一致。
- 增量评估是经验阻断门(净 CAGR delta + 逐年 + 配对 t),非单测。
- `open` 类回归:Step 4 重训后,**不带 MF 因子**应仍复现 factor-2model +31.2%(确保新数据层未污染既有特征)。

## 成功判据
factor-MF-2model 在规范扣费回测净 CAGR 上**稳健跑赢** factor-2model(+31.2%):逐年基本不输、配对 t 不更差、maxDD 不恶化、±overlay 一致改善 → **采纳**。否则诚实记负、不采纳(与前 5 个杠杆同等诚实)。

## 风险(诚实)
1. **免费历史浅** → 评估窗口可能短、统计力弱(Step-0 量化,结果里标注)。
2. **正交性存疑**:东财"主力净流入"由大单价×量推出 → 可能与价格部分相关,增量未必显著——增量 eval 给答案(可能是第 6 个负结果)。
3. **实盘口径 skew**:训练源 vs 实盘快照口径差异(Step 0/6 校准)。
4. **dump 复杂度 / 对齐**:新字段并入 provider 的工程量与日历对齐风险。
