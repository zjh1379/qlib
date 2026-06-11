# AI 分析层 (解读 + 风险旗标) · 设计 (2026-06-11)

## 背景与动机
- 本仓已是一个能跑的 qlib 量化 app:`production/` 出 1–5d 选股信号,`backend/` 服务,`frontend/` 展示 Picks/Charts/Eval/Portfolio。量化核心(均值反转 edge)已充分迭代,常规日线杠杆基本耗尽。
- 用户希望在**不推翻 qlib** 的前提下,叠加一层 **LLM 定性分析**:让大模型读个股新闻/公告,给量化选出的票做"二次解读 + 利空预警",辅助人最后拍板。
- 定位明确:这是 **serving 侧的决策辅助增强**,不是新的 alpha,**不改 qlib 排名**。

## 目标(已确认)
对每天 top-N 的 picks,LLM 产出两样东西:
1. **一句话解读** —— 结合反转信号 + 近期消息,说"为什么这只票现在值得关注"。
2. **风险旗标** —— 命中立案/退市/商誉/解禁/业绩预警/诉讼等利空就高亮,每条引用来源标题+日期。

结构化、可缓存、可审计;全程 fail-soft;成本可预测(top-N × 每天一次)。

## 非目标 (out of scope / 留给 v2)
- 不接 **基本面/估值** 与 **资金面(北向/龙虎榜/资金流)** —— 用户已决定这两块完善后单独出一版开发计划。
- 不做非候选票的**按需分析**(方案 C 那条在线路径)。
- 不做 ChartPage 面板、不做多 agent 辩论。
- **不用分析去过滤/重排 qlib 结果** —— v1 纯辅助,排名只由量化决定。
- 不在本 spec 内做盘中/实时(批量挂在 EOD 的 daily_inference 后)。

## 路线(已确认)
**方案 A:批量,挂在 `daily_inference` 之后。** daily_inference 算完当天 picks → 触发后端为 top-N 拉新闻+公告 → 调 Claude → 把"解读+风险旗标"按 `(symbol, as_of_date)` 存进 `app.db` → 失效缓存。前端**秒开**,看到的是预先算好的结果。

**LLM 编排、API key、app.db、serving 全部留在后端一处**(production 只负责触发),与"backend 管 serving+app.db,production 管模型计算"的现有分工一致。后端跑在 qlib conda 环境,akshare 可直接懒导入使用。

## 分阶段架构(Step-0 阻断门先行,沿用资金流因子/日内 P1 的纪律)

### Step 0 — 数据 + 模型 spike(阻断门)
- **akshare 取数**:实拉 1–2 只票确认接口形状与可用性 ——
  - 新闻:`stock_news_em(symbol)`(东财个股新闻);确认字段(标题/摘要/时间/来源)、能回溯多近、代码格式映射(akshare `600519` ↔ 本仓 `SH600519`)。
  - 公告:候选 `stock_notice_report` / 巨潮 cninfo 披露接口;确认能否**按个股**拿到近期公告标题+类型+日期(利空旗标的关键依据)。
- **LLM 输出**:用 1 只票真实调一次 Claude,确认强制结构化 JSON(工具调用约束)能稳定解析成 `AiAnalysis`。
- **阻断判定**:新闻/公告**拉不到或按个股不可得** → 停,先只做"基于现有上下文(分数/涨跌幅)的解读、暂无风险旗标",或退回评估 tushare(若用户有 token);不强行继续。

### Step 1 — 数据源封装 `backend/app/analysis/sources.py`
- akshare 个股新闻 + 公告的薄封装:**懒导入** akshare(与 models/service 懒导入一致),同步调用经 `run_in_threadpool`。
- 产出归一化的 `NewsItem` / `NoticeItem`(标题、日期、来源、可选摘要),按时间倒序、截断到近 ~10–20 条,控制 prompt 体积。
- 鲁棒:单票取数失败 → 返回空 + 标记,不抛到批次外。

### Step 2 — 分析服务 + 存储 `analysis/service.py` `analysis/orm.py` `analysis/schemas.py`
- 表 `ai_analysis(symbol, as_of_date, note, risk_flags_json, stance, model, status, created_at)`,主键 `(symbol, as_of_date)`(沿用 `portfolio`/`scheduling` 的 ORM 写法)。
- `analyze_picks(as_of_date, symbols)`:每票 取数(线程池)→ 拼 prompt → 调 Claude(anthropic 异步 SDK)→ 解析 → upsert;`asyncio.Semaphore` 限并发(3–5),对 akshare 与 API 都温和。
- 幂等:`(symbol, as_of_date)` 已存在且 `status=ok` 则跳过(成本闸 + 可断点续跑)。
- schemas:`AiAnalysis`(note/risk_flags/stance/model/as_of_date/status)、`RiskFlag`(type/severity/reason/source/source_date)。

### Step 3 — API 面 + 触发钩子 `analysis/router.py` + `production/daily_inference.py`
- `POST /api/internal/analysis/refresh {as_of_date, symbols}` —— localhost-only,镜像现有 `/api/internal/cache/invalidate`;起后台 job 跑 `analyze_picks`,完成后再 `invalidate` 一次缓存。
- `POST /api/analysis/run-now` —— 手动重算(re-infer 后或想要更新时用)。
- `GET /api/analysis/{symbol}?date=` —— 详情/单票取用。
- `/api/analysis/{active/peek,status,jobs/{id}}` —— 复用你的 inference job 模式。
- `daily_inference` 成功后,在现有 `cache/invalidate` 旁多 POST 一个 `analysis/refresh`,**直接把它已算出的 top-N symbols 传过去**(后端不必重算 picks)。

### Step 4 — Serving 挂载 `models/schemas.py` `models/service.py`
- `ScreenItem` 加 `ai_analysis: AiAnalysis | None`。
- `/api/models/candidates` 按 `(symbol, latest_date)` LEFT-JOIN 挂上分析(沿用"一次取全、客户端过滤"偏好;无分析则为 `None`)。

### Step 5 — 前端 `frontend/src/pages/picks/`
- `RiskFlagBadge.tsx`:行内红旗 🚩(命中数 + 最高 severity 配色)。
- `AiNotePanel.tsx`:展开行/弹层 —— 一句话解读(stance 配色)+ 旗标明细(type chip + reason + 来源标题/日期)+ "数据截至 {as_of_date} · 模型 {model}" 脚注。
- `useActiveJobs` 加第 5 个 job 类型(analysis);header 徽标 + Picks 显示"AI 解读生成中"。
- 重新生成 API 类型:`npm run gen:api`。

### Step 6 — 重启前后端供查看
- 改了服务/UI → **按标准指令重启 backend + frontend** 供用户查看。

## 组件清单(新增/改动)
- 新增:`backend/app/analysis/{__init__,router,service,schemas,orm,sources}.py` + `analysis/tests/`
- 改动:`backend/app/models/{schemas,service}.py`(挂 `ai_analysis`)
- 改动:`backend/app/core/config.py`(新配置项)
- 改动:`production/daily_inference.py`(触发 `analysis/refresh`)
- 新增:`frontend/src/pages/picks/{RiskFlagBadge,AiNotePanel}.tsx` + `useActiveJobs` 加 analysis job

## 数据流
```
刷新数据 → daily_inference 算完 picks
  → POST /api/internal/cache/invalidate              (现有)
  → POST /api/internal/analysis/refresh {as_of_date, top-N symbols}   (新增)
        → 后端后台 job:每票 akshare 新闻+公告(线程池) → Claude → 解析 → upsert app.db
        → 完成后再 invalidate 缓存
前端轮询 analysis job(useActiveJobs);完成后 /api/models/candidates 自带 ai_analysis → Picks 渲染
```

## LLM 契约(关键:防幻觉)
- 输入 = 该票近 ~10–20 条新闻标题/摘要 + 近期公告 + ScreenItem 已有上下文(分数/涨跌幅/board/ST)。
- 强制结构化输出(Claude 工具调用约束 JSON):
  ```json
  {
    "interpretation": "一句话:为什么现在值得关注(结合反转信号+近期消息)",
    "risk_flags": [{"type":"立案|退市|商誉|解禁|业绩预警|诉讼|其他","severity":"high|medium|low",
                    "reason":"简述","source":"公告或新闻标题","source_date":"2026-06-09"}],
    "stance": "favorable|neutral|caution"
  }
  ```
- **硬规则写进 prompt**:风险旗标**只能基于提供的新闻/公告**,每条必须引用来源标题+日期,**不准凭模型先验推断**;`stance` 是参考倾向、**不是交易信号**,不改 qlib 排名。
- 模型:Claude API **Sonnet 一档**(摘要+抽取够用又便宜);确切 model id 与每日成本估算在实现计划阶段查 `claude-api` 参考钉死。

## 错误处理(全程 fail-soft,沿用 calibration 容错风格)
- 无 API key / `ai_analysis_enabled=false` → 照常返回 picks,`ai_analysis=None`,UI 不显示徽标。
- 某票 akshare 取数失败 → 标 `status=partial`(只基于上下文给解读)或跳过,不拖垮整批。
- LLM 失败/超时 → 重试一次,仍失败标 `status=failed`,其余正常。
- 陈旧:分析按 `as_of_date` 存;re-infer 到新日期后旧分析不匹配 → UI 显示"无当日解读",`run-now` 可补。
- 成本闸:硬上限 `ai_analysis_top_n`;`(symbol,date)` 幂等 upsert,已分析则跳过。
- 内存:纯 I/O(网络),无大模型加载 → 后台 task 进程内即可,不触发 OOM 看门狗(无需像训练那样开子进程)。

## 配置
- `QLIB_COMPANION_ANTHROPIC_API_KEY`(env,不入库不提交)
- `ai_model`(默认 Sonnet 一档)
- `ai_analysis_top_n`(默认 10)
- `ai_analysis_enabled`(默认 false;设了 key 再开)

## 测试
- **Step-0 spike(门禁,先跑)**:真实 akshare 拉 1–2 票新闻/公告确认接口形状 + 真实 Claude 调 1 票确认输出可解析。通过才往下建。
- TDD 单元(mock 掉 anthropic + akshare):prompt 构造器、输出解析器(JSON→`AiAnalysis`,含畸形/缺字段降级)、upsert/去重、akshare↔本仓代码映射、serving JOIN。
- 集成:确定性假 LLM + 假 akshare fixtures 跑通 `analyze_picks`→app.db→`/api/models/candidates` 挂载。
- fail-soft 用例:无 key / akshare 报错 / LLM 报错 / 部分票失败。
- 回归:未开启 AI 时 candidates 形状不变(`ai_analysis=None`),既有 Picks 测试不破。

## 成功判据
- 开启后,top-N picks 在 Picks 表带上**可信的一句话解读**与**有来源的风险旗标**;命中真实利空(如近期立案/解禁)能被标出。
- 关闭/失败时**完全不影响**既有选股展示(fail-soft 验证通过)。
- 成本可预测(top-N × 每天一次),无 per-view 成本意外。

## 风险(诚实)
1. **公告按个股可得性**:akshare 个股公告接口若覆盖不全/不稳 → 风险旗标召回受限(Step-0 量化,必要时降级为"仅解读")。
2. **LLM 幻觉**:即便有"只依据来源"硬规则,仍可能误读 → 旗标强制带来源标题+日期供人核对;stance 明确非交易信号。
3. **新闻噪声/时效**:个股新闻含大量行情类水文 → prompt 里要求优先公告与实质性事件,弱化纯行情复述。
4. **akshare 稳定性**:第三方接口偶发变更/限流 → 薄封装隔离 + 重试 + fail-soft,接口变更只改 `sources.py` 一处。
