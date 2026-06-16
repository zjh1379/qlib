# 筛选重算 UX：两层筛选 + 可见进度

**Date**: 2026-06-16
**Author**: brainstorming session (用户 + Claude)
**Status**: Approved

---

## 1. Context & Problem Statement

### 现状

选股工作台（`Picks.tsx`）的筛选目前混了两种成本截然不同的操作，但 UX 不加区分：

- **后端调用**：`useCandidates({ top, days, min_top, view, models })`（[hooks.ts](../../../frontend/src/models/hooks.ts)）。其 React Query 缓存键含 `view` 与 `models`，**一改就立即重新请求后端**。后端 `_candidates_cached`（[service.py:238](../../../backend/app/models/service.py)）是一次性整体计算：`load_pred` → 对所选列做 rank-avg → EWMA 平滑 → 取 ~300 只行情指标 → 多周期富集 + 校准。读本地数据，**约 5s（首次冷启动 30–60s）**。
- **纯前端**：价格 / 涨跌幅 / 振幅 / 量比 / 创新高 / 板块 / ST / 共识 / 排序（[filter.ts](../../../frontend/src/pages/picks/filter.ts) + `sort.ts`），瞬时。

### 三个核心问题

1. **模型组合每勾一个框就单独触发一次 5s 重算**（`ModelSelector` → `onChange({ models })` 无防抖），连勾 3 个 = 连续 3 次 5s 请求，体感"卡死"。
2. **几乎没有反馈**：重算时只把旧表格变淡（`isFetching` → `opacity-60`），只有"首次加载"才有文字提示（[Picks.tsx:151](../../../frontend/src/pages/Picks.tsx)）。
3. **`窗口天数` / `最少进topN` / `Top N` 是死控件**：`Picks.tsx` 把候选池硬编码成 `POOL_SIZE=300 / WINDOW_DAYS=5 / MIN_TOP=0`，这三个框改了不起作用，却标着"immediate update"。

### 关键代码事实

- `_build_screen_items`（[service.py:111](../../../backend/app/models/service.py)）里 `top` 一参三用：① `days_in_top = (rank <= top).sum()` 的阈值；② 持续性过滤 `days_in_top >= min_top`；③ 最终池 `.head(top)`。所以 `Top N` 与 `最少进topN` **耦合**，`Top N` 不能在保留 `min_top` 语义的同时做成纯前端显示限制。
- `_candidates_cached` 由 `@functools.lru_cache(maxsize=32)` 缓存，键 = `(recorder_id, exp, view, top, days, min_top, models_key)`。命中即时返回。
- `get_filter_metrics` / `get_latest_close_prices`（[qlib_adapter.py](../../../backend/app/core/qlib_adapter.py)）读本地 qlib bin（`D.features`），**非网络**；是重算里最大且天然可分批的一段。
- 现成可复用：`inference` 的 job 模式（in-memory `_JOBS` OrderedDict + `threading.Thread` + `_ACTIVE_JOB_ID` 守卫，[inference/service.py](../../../backend/app/inference/service.py)）；`data` 的 `ProgressInfo{phase,current,total,message}`（[data/schemas.py](../../../backend/app/data/schemas.py)）；前端 `useJobPolling` / `useActiveJobs`（[jobs/](../../../frontend/src/jobs/)）。

### 用户决策（brainstorming 已确认）

| 维度 | 选择 |
|---|---|
| 进度形式 | **真正的百分比进度条**（非时间估算的假进度） |
| 死控件处理 | **连进做活** |
| 控件归属（发现耦合后） | **A：三个都变即时** —— 后端返回每股每日排名+分数，`窗口天数/Top N/最少进topN` 全部浏览器即时算；重算层只剩 `视图+模型组合` |
| 进度实现 | 轮询式后台任务（复用现有 job 模式），**非** SSE、**非**假进度 |

---

## 2. Goals

1. **重算层与即时层分离**：`视图 + 模型组合` 进草稿态，点「重新计算」按钮才请求后端；其余筛选（含做活后的 `窗口天数/Top N/最少进topN`）保持实时。
2. **真实可见的进度**：重算过程显示百分比进度条（阶段文案 + 已用秒数），进度来自实际计算阶段而非估时动画。其中最耗时的"取行情指标"按批上报，进度平滑且诚实。
3. **三个死控件做活且即时**：基于后端返回的每股近 K 日排名/分数数组，在浏览器即时计算持续性过滤与窗口排序。
4. **缓存命中即时完成**：已算过的 `(view, models)` 组合点重算瞬间完成（lru_cache 命中，计算体不跑）。
5. **首次冷加载（30–60s）也走同一进度条**：最慢的等待最该有进度。
6. **失败不污染**：重算失败保留当前生效池，提示并可重试。

## 3. Non-Goals

- 不改打分算法 / 模型 / 数据源（重算逻辑与现状一致，仅包装成可上报进度的任务 + 扩展 payload）。
- 不引入 SSE / WebSocket（沿用项目既有的轮询 job 模式）。
- 不动旧的 `/api/models/screen` 服务端筛选端点（Picks 用的是 `candidates`；`screen` 保持现状）。
- 不做重算的取消功能（~5s 任务，YAGNI；如未来常误点再加）。
- 不做进度的持久化 / 跨重启恢复（in-memory job 足够，与 inference 一致）。

---

## 4. 架构：两层筛选

```
用户操作
├── 即时层（纯客户端，实时）──────────────────────────────
│     价格 · 涨跌幅 · 振幅 · 量比 · 创新高 · 板块 · ST · 共识 · 排序
│     窗口天数(D) · Top N · 最少进topN(M)
│        ↑ 全部基于"当前生效池"在浏览器计算，零网络
│
└── 重算层（草稿态 + 「重新计算」按钮 + 进度条）────────────
      视图(view) · 模型组合(models)
         ↓ 点按钮
      POST /recompute → 后台线程跑 candidates()（warm lru_cache，上报进度）
         ↓ 轮询 GET /recompute/{job_id} 渲染进度条
      done → 提交 draft→applied → GET /candidates（命中缓存，瞬时）→ 换池
```

**不变量**：草稿（未应用的 view/models）存在时，即时层始终作用于**当前生效池**；只有点「重新计算」成功后才换池。

---

## 5. 后端设计（`app/models/`）

### 5.1 候选池 payload 扩展（支撑即时层）

让前端能即时算 `days_in_top / score_avg / 窗口 / Top N / min_top`，后端需返回每股的近 K 日时序：

- 常量 `WINDOW_K = 20`（覆盖 UI 窗口上限；短线工具 20 个交易日 ≈ 1 个月足够。可配置，增大仅增 payload）。
- `CandidatesResponse` 新增 `window_dates: list[str]`（升序的最近 K 个交易日 ISO 日期）。
- `ScreenItem` 新增：
  - `daily_ranks: list[int | None]`（与 `window_dates` 对齐；该股当日在**全universe**按 score 的排名，无数据为 `None`）。
  - `daily_scores: list[float | None]`（同上，当日 score）。
- `_build_screen_items` 改造：
  - 仍按固定 `top=POOL_CAP(300) / days=WINDOW_K(20) / min_top=0` 选出宽松池（即"近 20 日 score_avg 最高的 ~300 只"），保证客户端筛选有余量。
  - 对每只返回股，额外导出其在 `window_df` 内的 `daily_ranks` / `daily_scores`（按 `window_dates` 对齐填充）。
  - 后端仍可保留 `rank_avg / days_in_top` 字段（按池参数算），但**前端不依赖**它们做用户筛选——前端从数组重算。

**接受的近似**：池按 20 日 score_avg 取 top ~300；若某股在更短窗口（如 3 日）能进 top-30 却不在这 300 内，会被漏掉。300/~800 universe 下边缘影响可忽略，记录在案。

### 5.2 重算任务（复用 inference job 模式）

- **进度注入用 contextvar，不进 lru_cache 参数**（否则每次回调对象不同会破坏缓存键）：
  - 新增模块级 `ContextVar` `_progress_sink`（默认 `None`）。
  - `_candidates_cached` 内部在各阶段调用 `_emit_progress(phase, current, total, message)`，该函数读 `_progress_sink`，为 `None` 时是 no-op。
  - **缓存命中时函数体不执行 → 不 emit → 任务瞬间 done**，符合"已算过的瞬时完成"。
- **阶段与权重**（总进度 = 阶段基准 + 阶段内 current/total 插值；权重实测微调）：

  | phase | 内容 | 权重 |
  |---|---|---|
  | `load` | `load_pred` 读 pred.pkl | ~15% |
  | `score` | rank-avg 重算 + EWMA | ~15% |
  | `metrics` | `get_filter_metrics` + 价格，**按 50 只一批**，每批 emit | ~55% |
  | `enrich` | 多周期富集 + 校准 | ~15% |

  - `metrics` 分批：在 `_candidates_cached` 内把 symbols 切成 50 一组，逐组调 `get_filter_metrics(chunk)` 并 emit `current=已完成只数, total=总只数, message="正在取行情指标 150/300"`。多次 `D.features` 调用的额外开销可忽略。
- **Job 注册表 / 线程**：照搬 `inference/service.py` 结构——in-memory `OrderedDict` `_RECOMPUTE_JOBS`、`_remember_job`、`get_job`、`threading.Thread(target=_run_recompute, ...)`、`_ACTIVE_RECOMPUTE_ID` 守卫（进行中再 POST 返回 `already_running` 复用同 job）。
- **线程体 `_run_recompute(job_id, view, models)`**：set `_progress_sink`（写该 job 的 progress）→ try `candidates(view=view, models=models)`（固定 top/days/min_top 常量）→ 成功置 `done(100%)`，失败置 `failed` + error → finally 清 `_progress_sink` 与 `_ACTIVE_RECOMPUTE_ID`。

### 5.3 Schemas（`app/models/schemas.py`）

- 复用进度结构：把 `ProgressInfo` 提升为共享（`app/core/schemas.py` 或直接在 models 内定义同形 `RecomputeProgress`，避免 `models → data` 依赖）。**决定**：在 `app/core/schemas.py` 新建共享 `ProgressInfo`，`data` 与 `models` 都引用（`data` 侧保持向后兼容的 re-export）。
- `RecomputeJob { job_id, status: "running"|"done"|"failed", started_at, finished_at?, error?, view, models, progress: ProgressInfo | None }`。
- `RecomputeTriggerResponse { status: "started"|"already_running", job_id? }`。

### 5.4 Endpoints（`app/models/router.py`）

- `POST /api/models/candidates/recompute` body `{ view: str, models: list[str] }` → `RecomputeTriggerResponse`。
- `GET  /api/models/candidates/recompute/{job_id}` → `RecomputeJob`。
- `GET  /api/models/candidates/recompute/active` → 最近/进行中的 `RecomputeJob | null`（供 `useActiveJobs` 聚合 + 头部徽标，与 inference 一致）。
- `GET  /api/models/candidates`（现有）不变，仍走 lru_cache，命中即时返回。

### 5.5 缓存影响

`top/days/min_top` 不再随 UI 变（固定常量），lru_cache 键实际只随 `(recorder, exp, view, models_key)` 变 → 组合更少、命中率更高。

---

## 6. 前端设计

### 6.1 草稿态 / applied 拆分

- `view`、`models` 改为 **draft 局部 state**（`Picks.tsx` 内 `useState`），不再直接驱动 `useCandidates`。
- `useCandidates` 的查询参数 = **applied** 的 `{ view, models }`（仍落 URL，保持可分享/书签）。
- `useFilterParams` 继续管理即时层参数（含做活后的 `top/days/min_top/显示条数`）+ applied 的 `view/models`。draft 与 applied 的 diff 决定按钮 dirty 态。

### 6.2 FilterBar 重组

- 新增"模型 / 视图 · 需重新计算"分区，含 `视图` 下拉 + `模型组合` 网格 + **「重新计算」按钮**（draft≠applied 时高亮并提示有改动；相等时禁用）。重算进行中按钮显示进度/禁用。
- `窗口天数 / Top N / 最少进topN` 移入即时层分组，去掉"死控件"。`窗口天数` 上限由 60 改为 `WINDOW_K=20`（短线足够；记录此变更）。

### 6.3 重算流程

点「重新计算」：
1. `POST /recompute {draft.view, draft.models}` → `job_id`。
2. `useJobPolling('recompute', () => api.models.recomputeStatus(job_id), interval)` 轮询 → 渲染**进度条**（百分比 + `progress.message` 阶段文案 + 已用秒数）。
3. `status==='done'`：把 draft 提交为 applied（`update({ view, models })` 落 URL）→ `useCandidates` 键变 → `GET /candidates` 命中后端缓存 → 瞬时换池 → 隐藏进度条。
4. `status==='failed'`：显示 `error`，保留当前生效池，按钮恢复可重试。

### 6.4 首次冷加载统一走进度条

- 用 `enabled` 把 `useCandidates`（`GET`）**门控**在"该 `(view, models)` 组合本会话已 warm"之后——确保 30–60s 的重计算永远发生在带进度的 job 内，而非裸 `GET` 阻塞（否则进度条形同虚设）。
- `Picks.tsx` 挂载 / applied 组合变化时，若该组合本会话未 warm，则先发 `POST /recompute` → 进度条 → done 标记 warm → `GET`（命中缓存瞬时）。
- 会话内已 warm（React Query `staleTime: Infinity` 命中）则直接显示，不起 job。

### 6.5 即时层客户端计算（`filter.ts` / 新 helper）

基于 `data.items[].daily_ranks/daily_scores` 与 `window_dates`：
- 取最近 `D = 窗口天数` 个 `window_dates`。
- `days_in_top = count(daily_ranks[最近D] <= TopN)`；过滤 `days_in_top >= min_top`。
- `score_avg = mean(daily_scores[最近D], skip None)` → 作为默认排序键；据此重算客户端 `rank`。
- 其余即时过滤（价格等）不变。
- 最后按 `Top N` 截断显示。**`Top N` 一控双用**：既是持续性阈值（`rank ≤ TopN`），又是结果显示上限（过滤+排序后取前 N 行）——匹配"我的前 N 只票"心智，不新增控件。比旧逻辑更优：旧 `head(top)` 先截再被价格等筛掉常不足 N，新逻辑先筛后截能填满 N。

### 6.6 进度条组件

- 新建小组件 `pages/picks/RecomputeProgress.tsx`（百分比条 + 阶段文案 + 已用秒数），或复用 data refresh 的进度展示样式。`useActiveJobs` 增加 `recompute` kind，头部徽标显示"重算中"。

---

## 7. Data Flow（端到端）

```
[改 视图/模型组合] → draft state（按钮高亮，未发请求）
         │  改 价格/窗口/TopN/min_top/排序/显示条数 → 即时层重算（浏览器，零网络）
         ▼ 点「重新计算」
POST /api/models/candidates/recompute {view, models}
         │   _ACTIVE 守卫 → 起线程 _run_recompute
         │   线程 set _progress_sink → candidates() →（缓存未命中）
         │     _candidates_cached emit: load→score→metrics(分批)→enrich
         ▼   （命中则瞬间 done）
GET /api/models/candidates/recompute/{job_id}（轮询）→ progress% → 进度条
         ▼ done
draft→applied(URL) → GET /api/models/candidates（lru_cache 命中，瞬时）
         ▼
新候选池（含 daily_ranks/scores）→ 即时层 + 排序 → 表格
```

---

## 8. Error Handling & Edge Cases

- **重算失败**：job `failed` + error；前端保留当前生效池，提示可重试。
- **缓存命中**：线程内 `candidates()` 瞬间返回 → job 直接 `done(100%)`，进度条一闪或跳过。
- **并发点击**：`_ACTIVE_RECOMPUTE_ID` 守卫，进行中再 POST 返回 `already_running` 复用同 job_id。
- **草稿未应用**：即时层作用于当前生效池；切换页面/重置时草稿同步回 applied。
- **某股窗口内缺数据**：`daily_ranks/scores` 对应位置 `None`，`days_in_top` 不计、`score_avg` skip。
- **池近似**：见 5.1，边缘漏选可忽略并记录。
- **`window_dates` 不足 K**：新 recorder 历史短则返回实际可得天数，前端按实际长度算。

---

## 9. Testing

**后端**
- `recompute` job：起停、`already_running` 守卫、`done/failed` 状态流转。
- 进度上报：缓存未命中时各阶段 emit 单调递增且收敛到 100%；`metrics` 分批 `current/total` 正确。
- 缓存命中：不 emit、瞬间 done。
- payload：`window_dates` 升序、长度 ≤ K；`daily_ranks/daily_scores` 与之对齐、缺数据为 `None`。
- `ProgressInfo` 共享后 `data` 与 `models` 均可用（回归现有 data 测试）。

**前端**
- draft↔applied：改 view/models 进草稿、按钮 dirty/disabled、应用后落 URL。
- 进度轮询渲染：running→done 隐藏、failed 报错保留旧池。
- 即时层客户端计算：`days_in_top/min_top/窗口/TopN/score_avg/rank/显示条数` 与旧后端语义对齐（用固定 fixture 对照）。
- 首次冷加载：无缓存起 job、有缓存直显。

---

## 10. Build Sequence（高层；细节交给 writing-plans）

1. 后端 `ProgressInfo` 提升至 `core/schemas` + 共享。
2. 后端 payload 扩展（`window_dates` + 每股 `daily_ranks/scores`）+ `_build_screen_items` 改造 + schema/类型回归。
3. 后端 `_emit_progress` contextvar + `_candidates_cached` 分阶段埋点 + `metrics` 分批。
4. 后端 recompute job（注册表/线程/守卫）+ 3 个 endpoints + schemas。
5. 前端 `npm run gen:api` 重生类型；`filter.ts` 即时层客户端计算 + helper。
6. 前端 draft 态 + FilterBar 重组 + 「重新计算」按钮 + 进度条组件 + `useActiveJobs` 接入。
7. 前端首次冷加载走进度条。
8. 实测各阶段耗时，微调进度权重。
9. 重启前后端供用户验收（用户标准指令）。

---

## 11. Open Questions / Risks

- **进度权重需实测**：`load/score/metrics/enrich` 实际占比要 profile 后定，避免进度条"卡某段再跳"。
- **`窗口天数` 上限 60→20**：短线工具合理，但属行为变更，需在 UI 注明或与用户确认（已含于 K 可配置）。
- **池近似**：见 5.1，可接受。
