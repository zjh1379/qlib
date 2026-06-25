# 小资金操作点诊断 (A1) · 设计 (2026-06-25)

## 背景与动机
- 冠军回测 **+31.2% 净CAGR / Calmar 0.77 / maxDD −41%**(factor-2model,canonical `fixed/top_k=5/持5天`,**¥100k**,small cost)。但这是 **¥100k + 分数权重(fractional)** 的模拟。
- 用户真实账户:**≈¥10k**,实际操作 = **单只 2–3 手 或 买 ETF**,一年内才到 ¥3–5 万。即真正交易的是 **top-1/top-2**,不是 top-5;回测那个"5 名集中组合"在 ¥10k 上**不可实现**。
- **真金样本外证据(关键):** 用户已按当前模型 pick 下单 **2 笔(环保、黄金),买后即暴跌**;¥6k 本金亏 ¥1k+(≈ −17%)。n=2 不能说明均值,但"买后就跌 + 都是板块题材票"这个形态命中三个**可测的结构性问题**:
  1. **涨停选择性偏差**(最可疑):模型最猛的赢家开盘奔涨停→**买不进**;实际成交系统性偏向没起涨的弱票。回测按开盘价把涨停赢家算进收益,实盘只接到输家 → 这正是"回测正 / 实盘亏"的标准机制。
  2. **top-1 反转左尾**:无分散,接飞刀没接住时 100% 吃整段下跌。
  3. **板块/宏观污染**:黄金跟金价宏观走,非 5 日均值回归。
- 既有研究结论:模型层"便宜杠杆"已榨干(因子采纳 +12pp;趋势门/分散/LambdaRank/止损/日内择时 5 杠杆均被实测否决);+31% **统计上边缘显著**(配对 t p≈0.10)。**结论:实盘能否赚钱的瓶颈在"操作点 + 执行落差",不在再做一个模型。**

## 目标(已确认)
做一个 **"¥10k 操作点"忠实诊断**(A1):在用户真实交易的规模与约束下,诚实测出当前模型还剩多少**可实现**净收益,并产出一句话裁决——**在 ¥10k 下,靠哪条路(单票 top-1/2 还是 ETF 择时)有正期望且可成交?** 零重训、零新 alpha。

四块,大头复用现有引擎(`backtest.run.build_report` / `engine.run_backtest` / `score_utils.rebuild_2model` / `_eval_topk_sweep`)。

## 非目标 (out of scope)
- 不重训、不产生新信号、不改选股模型(那是另一条线)。
- **A2 执行/前向测试层仅草图**(见末节),其详细设计取决于 A1 结论,另起 spec。
- ST 涨跌停(±5%)v1 不特判(CSI800 基本不含 ST,影响小,列为已知局限)。
- 出场可成交性(跌停卖不掉)v1 只标记、不建模优化(沿用 open 出场)。

## A1 架构(四块)

### 1. 操作点下扫 — 扩展 `production/research/_eval_topk_sweep.py`
- `TOP_KS` 增加 **{1, 2, 3}**(现为 {5,10,15,20,30});**新增 ¥10k 一遍**(现为 ¥100k),两个资金档并排,看最低佣金地板在小资金的拖累 delta。
- 新增**逐笔收益分布诊断**(top-1/2 方差是核心问题):每周期(5日块)净收益的 **左尾(最差 5%/10%)、std、负周期占比**——量化"top-1 有多惨烈",并给用户那 2 笔亏损一个分布上的定位。
- 保留现有 net_metrics(净CAGR/IR/maxDD/Calmar/win/cost_drag/逐年)+ 配对 t vs baseline。
- 产出:扩展 `logs/eval_topk_sweep_summary.json` + 一张 markdown 表。

### 2. 涨停可成交性 + 选择性偏差 — 新模块 `production/backtest/executability.py`
- **新 loader**(同 `data.py` 口径,对齐决策日 d、进场在 d+1):
  - `entry_open = Ref($open,-1)`,`entry_high=Ref($high,-1)`,`entry_low=Ref($low,-1)`,`prev_close=$close`(= d 日收盘,涨停基准)。
  - `limit_price = round(prev_close×(1+limit), 2)`;`unbuyable_at_open = entry_open ≥ limit_price`;`一字板 = entry_open==entry_high==entry_low 且达涨停`。
  - 板块涨停幅按代码前缀(复用 intraday spec 约定):主板 10%、创业(SZ300)/科创(SH688)20%、北交所(BJ8)30%。
- **(a) 门控回测**:选 top_k 时**跳过进场开盘买不进的票,顺延下一名** → 净收益相对未门控掉多少。实现 = 在 `_equal_top_k` 前按可成交 mask 过滤 scores(薄包装,不改引擎)。
- **(b) 选择性偏差分解**(冒烟枪):对未门控的 top_k picks,按"开盘买得到 / 买不到(跳空)"分两组,比较各组**实现 open→open 收益分布**。若买不到组系统性更高 → 实锤"赢家都被错过"→ 直接解释"买的总跌"。
- 产出:门控 vs 未门控净指标 + 两组收益分布 + picks 中"买不进"占比。

### 3. 复盘两笔真实交易 — 新脚本 `production/research/_reconcile_live.py` + 用户填的 `production/reports/live_trades.csv`
- 输入 CSV(用户填,可增量):`[trade_date, instrument, side, fill_price, shares, entry_timing, note]`,`entry_timing ∈ {open, intraday}`。先填环保/黄金那两笔。
- 对每笔:取**模型当日给该票的排名/分数**(从 OOF),**回测假设的进场**(d+1 开盘)与其 **open→open 实现收益**,**用户实际 fill_price**,以及从 fill 起的真实前向收益。把亏损拆成:
  - **信号成分**:模型是否真把它排进 top_k?回测那期对它的 open→open 收益是正是负?(也负 → 信号/方差)
  - **执行成分**:fill_price vs 回测进场开盘(高于开盘 → 追高);是否涨停跳空(门控模型里属"买不进")。
- 产出:逐笔分解表 + 一段叙述,校准"回测对你的真实体验有多大代表性"。
- 注:即便两笔细节不全,脚本对**今后所有记录的真金交易**复用(= A2 前向日志的种子)。

### 4. ETF 对照臂 — 新变体 `production/research/_eval_etf_timing.py`
- 策略:持有宽基 ETF,用 **P3 趋势 overlay**(`regime trend_ma, ma_window=60, band=0.10` → `regime.compute_exposure`)择时进出(MA 带上=持有,带下=空仓),对比 buy&hold。
- 实现:**微型专用 sim**(不走横截面选股引擎)—— `net(d) = etf_ret(d)×exposure(d) − cost(翻仓时)`,结果序列复用 `metrics_net`。仅在 exposure 0↔1 翻转时计一次 ETF 成本。
- ETF 代理:v1 用 `market.py::load_market_proxy` 的**合成等权宽基**(零新数据);精化 = 拉真实 ETF/指数 NAV(baostock `sh.000300` / 510300、510500)。
- 成本:`costs.py` 新增 `etf` profile(**stamp_bps=0、transfer_bps=0**、commission 万2.5/min¥5、slippage)。¥10k 单只 ETF + 低频翻转 → 成本可忽略、**完全可实现**。
- 产出:ETF 择时在 ¥10k 下的净CAGR/Calmar/maxDD vs Block 1 的单票 top-1/2 → **单票 vs ETF 裁决**。

## 数据流
OOF(`oof_lgbmfac_2021_2026.pkl` + `oof_2model_2021_2026.pkl`)→ `rebuild_2model` 得分 →
- B1: `build_report` 在 top_k∈{1,2,3,5} × capital∈{1e4,1e5} 扫描 → 逐笔分布;
- B2: `executability` mask → 门控选股 + 两组分布;
- B3: `live_trades.csv` + OOF 排名 + 实现收益 → 分解;
- B4: market proxy + trend overlay + etf cost → ETF 净序列。
→ 汇总成 **「¥10k 下 单票 top-1/2/3(门控&未门控) vs ETF」对照表 + 一句话裁决**。

## A股微结构 / 成交判定(关键正确性)
- **涨停买不进**:`entry_open ≥ limit_price`(尤其一字板)→ 该名不可买 → 门控跳过/顺延;偏差分解中归"买不到"组。阈值按板块前缀。
- **跌停卖不掉**:v1 只在出场标记(不优化),记负面占比。
- **停牌/缺 OHLC/新股**:标记 no-data,单列占比(过高则结论存疑)。
- 涨停判定用 `round(prev_close×(1+limit),2)` 真实限价口径(含 0.01 取整)。

## 验证 & 成功判据
- **主裁决**:在 ¥10k、门控可成交、扣费后,**单票 top-1/2 是否仍正期望**?对照 **ETF 择时**谁更高(净CAGR/Calmar/maxDD/逐笔左尾)。
- **机制确认**:Block 2 必须量化"+31% 里有多大比例活在买不进的涨停名里";Block 3 必须把用户 −¥1k 拆成 信号 vs 执行。
- **诚实负结果照常存档**:若 top-1/2 门控后转负或被涨停掏空 → 明确写"单票在 ¥10k 不可行,应转 ETF 择时为主线"(与前 5 个被否杠杆同等诚实)。
- 辅助:"买不进"占比、no-data 占比——过高则削弱可信度。

## 测试
- TDD 纯函数:`executability` —— 合成 OHLC,断言主板/创业/科创/北交所涨停阈值、一字板、限价取整、`unbuyable_at_open` 正确;缺数据→标记回退。
- 回归锚:门控包装在"无任何名涨停"的合成数据上必须**精确复现**未门控基线;`_eval_topk_sweep` 扩展后 top_k=5/¥100k 一档须**复现既有 +31.2%**。
- Block 3/4 小样本端到端跑通 + 与手算一致。

## 风险(诚实)
- n=2 真金亏损本身不证伪模型;A1 的价值在**机制归因**而非"因为亏了所以差"。
- 合成宽基 ETF 代理 ≠ 真实 ETF NAV(权重/跟踪差);v1 给方向,真实 NAV 为精化项。
- top-1 方差极大:即便期望为正,maxDD 可能远深于 −41%,需如实呈现给用户做风险取舍。
- 涨停门控可能把 +31% 砍到很低——但这正是要测的真相;砍得狠 = 单票路线在小资金本就不成立。
- 复权口径:涨停判定与 `$open/$close` 必须同一复权口径(沿用 `data.py` 的 baostock 日线口径)。

## 文件清单
**新增**
- `production/backtest/executability.py`(+ `production/tests/test_executability.py`)
- `production/research/_reconcile_live.py`
- `production/research/_eval_etf_timing.py`
- `production/reports/live_trades.csv`(用户填,gitignore)
- 结果:`docs/superpowers/specs/2026-06-25-small-capital-execution-diagnostic-results.md`(A1 跑完写)

**修改**
- `production/research/_eval_topk_sweep.py`(加 top_k {1,2,3} + ¥10k 档 + 逐笔分布)
- `production/backtest/costs.py`(加 `etf` profile)

## Phase 2 (A2) 草图 —— 仅当 A1 判定有路可走
- **若单票 top-1/2 门控后存活**:薄执行层 —— 整手取整到 2–3 手、下单时跳涨停、每日 pick 落盘 + 隔日对账真实成交(复用 Block 3 脚本),边涨到 ¥3–5 万边攒真 OOS。
- **若 ETF 择时胜出**:把 ETF 择时做成服务主线(前端给"持有/空仓 + ETF 代码"信号)。
- 届时若改前后端 → 按用户标准指令重启前后端供查看。A2 另起 spec + plan。
