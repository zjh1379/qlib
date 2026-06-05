# 日内执行择时 · 设计 (2026-06-04)

## 背景与动机
- 用户痛点:**信息滞后**——当前信号建于昨日收盘、次日开盘"盲买",感觉慢半拍。
- 硬约束:**算力不支持重训**;只能做推理/规则/轻量在线统计。
- 既有结论(本仓研究):模型 edge 是**日频均值反转**;4 个常规杠杆(趋势门/分散/LambdaRank/止损)均因"对抗反转"被否决;冠军 = factor-2model(MSE)+ P3 overlay,top_k=5。
- 现状数据:qlib `cn_data_bs` 为**纯日线**(只有 `day` 日历)。日内是全新并行数据层。

## 目标(已确认)
**执行择时**:**保留** factor-2model 选出的当日 top-5(选股不变),仅用日内信息改善**进场**——即把"次日开盘买入"换成"日内择时价买入/或不买"。**零重训**。

**路线(已确认):方案 A 两阶段——先离线验证(P1),证明有用再上实时(P2)。** 本 spec 聚焦 **P1**;P2 仅草图,待 P1 结论决定是否启动。

## 非目标 (out of scope)
- 不训练任何日内模型、不产生新 alpha 信号(那是 P4/重训范畴)。
- 不改选股(只改进场执行)。
- 出场择时:P1 默认沿用 open 出场;出场规则为可选扩展(见 §验证)。
- P2 实时基础设施不在本 spec(仅草图)。

## Phase 1 架构(离线模拟器)

三个聚焦小组件 + 复用现有 `metrics_net`:

### 1. `production/intraday/fetch_5min.py` — 日内数据获取(按需+缓存)
- 源:baostock `query_history_k_data_plus(..., frequency="5")`,字段 `time,open,high,low,close,volume,amount`。**复权口径在 Step0 与日线 `$open` 对齐校准**(baostock min 默认不复权;进出场跨日的比值受复权影响,必须与回测口径一致)。
- 只拉**策略真正进场的 (instrument, entry_date)** 集合(由 P1 模拟器从 picks 推出):≈ 270 个调仓期 × 5 票 ≈ 一千多 name-day,极小。
- 缓存为 parquet:`production/intraday/cache/<instrument>_<YYYYMM>.parquet`(按月分片,可增量)。
- **P1-Step0 数据 spike**:先实测 baostock 5min 能登录、能拉、字段/复权/时区正确;拉不到则回退 akshare 历史 min(`stock_zh_a_hist_min_em`)。spike 失败即阻断,先解决数据再继续。

### 2. `production/intraday/entry_rules.py` — 进场价规则(纯函数,TDD)
输入:某票某进场日的 5min bars(DataFrame)。输出:该规则下的**有效进场价**(或"不可成交"标记)。规则集(P1 全测):
- `open`(基线):当日开盘价(= 现行口径)。
- `vwap`:全日 VWAP(∑amount/∑volume);变体 `vwap_am`(上午 VWAP)。
- `low_band(k)`:若日内最低 ≤ open×(1−k) 则按 open×(1−k) 成交,否则按收盘补(没接到)。k∈{0.5%,1%,2%}。
- `gap_cond(g)`:低开(open<prevclose)→ 开盘买(反转更超跌=好);高开 ≥ +g → **不追**(跳过/或收盘补)。
- `first30_low`:前 30 分钟(前 6 根)最低价。
所有价格基于已拉的 5min bars;无未来信息(只用进场日当天的盘中,决策点在进场动作内,符合"当日执行")。

### 3. `production/intraday/exec_backtest.py` — 执行模拟器
- 取**冠军 factor-2model**(= `_rebuild_2model(oof_lgbmfac, oof_2model)`)在 `fixed/持5/5天/top_k=5` 下的 picks。**口径对齐现引擎**:决策在 D 收盘 → 进场 = 下一交易日开盘所在 session(在此 session 内用日内择时价替代 open)→ 出场 = 持有期末 session 开盘。枚举每笔"交易"=(进场日, 出场日, 票)。
- 基线每笔收益 = `open(出场)/open(进场) − 1`;规则每笔收益 = `open(出场)/规则进场价 − 1`。
- 按 5 票等权 + 周期(5日块)聚合成**周期净收益序列** → 复用 `metrics_net`(净CAGR/Calmar/胜率/逐年)→ 各规则 vs open 基线对比。
- 同时报告**每笔进场价改善分布**(规则价 vs open 的中位/均值 bp)。

**数据流:** OOF picks →(进场日,票)集合 → `fetch_5min`(缓存)→ `entry_rules` 算规则价 →(应用微结构成交判定)→ `exec_backtest` 聚合 → 净指标 delta + 进场价改善分布。

## A股微结构 / 成交判定(关键正确性)
模拟器对每笔进场必须判"**能不能真的按这个价成交**":
- **一字/盘中涨停**:若进场日 open 已涨停(open ≥ prevclose×1.1,创业板/科创 ×1.2)或全天封板 → 反转票冲高 → **不可买入 → 该笔跳过**(不计入,避免虚高收益)。
- **跌停**:影响**出场**(卖不掉)——P1 出场沿用 open,若 open 跌停则标记/顺延(简化:标记为不可出,记负面)。
- **停牌/无 5min/新股不足**:回退 open 价 + 标记(no-timing),单列统计占比。
- ST/退市:沿用现有 universe 过滤,不额外处理。
涨跌停阈值按板块(主板10%、创业/科创20%、北交所30%)用代码前缀判定。

## 验证 & 成功判据
- 主指标:某规则相对 `open` 基线的**净CAGR / Calmar / 胜率** delta(逐年也看,尤其 2022/2023)。
- 辅指标:进场价改善中位 bp;不可成交/回退占比(太高则结论不可信)。
- **成功 = 至少一条规则在扣费后稳健为正**(且改善幅度盖过滑点/摩擦量级)。**负结果照常记录并存档**(与前 4 个杠杆同等诚实)。
- 与反转理论一致性检查:预期 `gap_cond`(低开多买)/`low_band` 对反转票更友好;若全规则无效,说明日频反转的执行价不敏感,记负结果、不进 P2。

## 测试
- TDD:`entry_rules` 纯函数单测——给合成 5min bars,断言 VWAP/low_band/gap/first30 价正确;**涨停→不可成交**;停牌/缺失→回退 open。
- 模拟器集成测试:小样本 picks + 合成日内 → 端到端跑通 + 与手算一致;`open` 规则必须**精确复现**现有 open 基线(回归锚)。

## Phase 2 草图(仅当 P1 为正)
- 盘中:后端会话期轮询服务用 akshare 准实时快照(`stock_zh_a_spot_em`)取当日 top-5 现价/开盘/最高低/量 → 按 P1 验证过的规则算"可买 / 等回踩 / 已反包别追 + 建议价位带"。
- 前端:Picks 增加日内执行小组件(会话内自适应轮询,复用 `useActiveJobs` 模式)。
- 届时改动管线/前后端 → 按用户标准指令重启前后端供查看。

## 风险(诚实)
- 反转 edge 是日频的;日内择时可能只挪几十 bp,需看 delta 是否真盖过摩擦/滑点。
- baostock 5min 为盘后历史(P1 足够);P2 实时快照精度低于真 tick。
- T+1 + 涨跌停可能吃掉进场改善——微结构建模是成败关键,故列为一等公民。
- 数据 spike(Step0)是前置阻断点:拉不到日内数据则整体重估。

## 文件清单(新增)
- `production/intraday/__init__.py`
- `production/intraday/fetch_5min.py`
- `production/intraday/entry_rules.py`(+ `production/tests/test_entry_rules.py`)
- `production/intraday/exec_backtest.py`(+ `production/tests/test_exec_backtest.py`)
- `production/intraday/cache/`(parquet 缓存,gitignore)
- 结果:`docs/superpowers/specs/2026-06-04-intraday-execution-results.md`(P1 跑完写)
