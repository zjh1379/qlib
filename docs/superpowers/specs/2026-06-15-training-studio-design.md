# 训练工作台 (Model Studio) · 设计 (2026-06-15)

## 背景与动机
- 当前"模型训练界面"其实只是设置页里的一个排程小组件 [`RetrainScheduleEditor.tsx`](frontend/src/pages/settings/RetrainScheduleEditor.tsx):只有「周排程 + 立即运行 + 回滚上一周」。
- 用户重训时**看不到任何进度**:后端 [`scheduling/service.py`](backend/app/scheduling/service.py) 把 `python -m production.rolling_train run-once` 的 stdout 打到后端日志,但前端只拿到 `running → done`,没有阶段/百分比/损失 → 看起来像卡死。
- 同时:**历史模型、评估、不同算法**这些没有集中呈现 → 用户判定"界面几乎不可用",要求做一次大更新。
- 摸底后的事实(重要,影响设计):
  - **历史 + 评估其实已具备**:`models/service.py::version_info()`(当前 + 上一版 + 上上版,带 IC/IR)、独立的"评估对比页"、`models/service.py::rollback_to()`。它们与训练界面**割裂**,所以训练界面看不到。
  - **三个算法本就在分别训练**:管线每次跑 **LGBM + ALSTM + TRA** × **1d/5d/20d**,Ridge 融合成一个分;`pred.pkl` 已存每个模型的分(`lgbm_1d`…`tra_20d`)。UI 只是把它们压成一个融合分,且不可单独对比/训练。
- 本设计的本质 = 把这些**串成一个统一「训练工作台」**,补上**结构化实时进度**,并把**按算法训练/选择**做出来。

## 目标(已确认)
1. **实时训练进度**:阶段文本(在训哪个模型/周期)+ 深度模型(ALSTM/TRA)逐 epoch train/valid 损失曲线;LGBM 显示 iteration/best-score。← 治本。
2. **单算法按需训练**:只训 LGBM / ALSTM / TRA 之一,产出**可对比、可上线**的候选模型。
3. **历史模型列表**:每版的训练时间、范围、关键指标、状态。
4. **多版本对比 + 上线/回滚**:总览 / 分周期 / 各底层模型单独 IC,一键 promote/rollback。
5. **底层模型纳入开关**:勾选哪些模型进融合(现 3 个,设计上做成可插拔注册表,留扩展位)。

确认参数:**GPU**、全量重训 **~15min**(单算法更快)→ 进度可"实时观看";**现有 3 模型为主、留扩展位**;**一步到位含单训**(整体设计,分期落地)。

## 非目标 (out of scope)
- **不新增算法类型**(XGBoost/GRU/Transformer/线性…):只做可插拔注册表**留位**,本期不接新 trainer。
- **不做全超参调优 UI**:只在"高级"折叠暴露 end-date 等少数项。
- **不引 SSE/WebSocket**:用轮询(Windows 上更稳,训练才十几分钟,复用现有 ActiveJobsBadge 轮询风格)。
- **不改 qlib 排名/选股逻辑**;**不改净成本回测口径**(对比区复用评估口径)。
- 不做分布式/多机训练、不做训练队列(一次一个,沿用并发锁)。
- 不做盘中训练(交易时段守卫沿用,force 可越过)。

## 总体方案(已确认:方案 A · 统一工作台)
- **A 统一工作台(采纳)**:新建独立页面收拢 训练/实时进度/历史/对比/上线回滚。正面回应"大更新";后端评估/版本/回滚已具备,主要是串起来 + 补结构化进度;每块可独立测、分期合并。
- B 就地增强设置页:快但局促,撑不起"可视化/对比",大概率返工 —— 不符合"大更新"。
- C 历史/对比挂 MLflow UI:对比 UI 近零开发,但**不展示本仓净成本回测口径**、另开端口、体验割裂成两个 App —— 弃。

## 架构

### 前端:新页面「训练工作台」(四区)
导航新增一项;现有设置页的排程小组件并入本页"训练 > 排程"子区。
- **① 训练**:范围选择(全量集成 / 单算法:LGBM·ALSTM·TRA 单选)+ **底层模型纳入开关** + 立即训练(交易时段守卫/force 沿用)+ 排程子区。`高级`折叠放 end-date 等少数项。
- **② 进行中(治本核心)**:当前任务卡(总进度 +「正在训练 ALSTM·5d·epoch 12/50」)+ 深度模型实时 **train/valid 损失曲线** + 阶段时间线(各 模型×周期 待训/训练中/完成/失败)+ 可展开日志尾巴。
- **③ 历史模型**:表格,每版一行 = 训练时间 / 范围(全量 or 单算法+哪些)/ IC·IR / 净成本回测年化超额 / 最大回撤 / 状态(当前·候选·归档)/ 训练窗口 / git sha。
- **④ 对比与上线**:勾 2+ 版本并排 —— 总览 + 分周期 + **各底层模型单独 IC/贡献** + 净成本回测曲线叠加(复用评估页口径);「设为当前(promote)」「回滚」复用现有 rollback 机制。
> 现"评估对比页"内容并入 ③④,避免两套对比 UI。

### 进度治本机制
- 训练进程在 `production/runs/<job_id>/progress.jsonl` 逐帧写结构化事件:
  ```json
  {"ts":"…","stage":"train","model":"alstm","horizon":"5d","epoch":12,"total_epochs":50,"train_loss":0.013,"valid_loss":0.015,"pct":0.42}
  ```
  以及里程碑 `{"event":"model_done","model":"lgbm","metrics":{…}}`、`{"event":"run_done","recorder_id":"…","status":"ok"}`。
- 后端扩展 job 状态(在 `SchedulerManager`/新 training service):除现有 status,新增 `progress`(最新帧)+ `phases`(各阶段状态)+ `recorder_id`。后端 **tail** `progress.jsonl`(沿用数据刷新 checkpoint 的读法)。
- 传输:前端**轮询** `~1–2s` 取最新帧画进度/曲线。**不引 SSE**。
- **无侵入发射**:新增 `production/progress.py`,`emit(...)` **仅当环境变量 `QLIB_PROGRESS_FILE` 存在时才写**;单测/命令行直跑训练脚本不受影响。

### 单算法训练语义(关键)
- 选"只训 LGBM":**只重训 LGBM**;其余已启用模型**复用上一版 recorder 里已存的预测列**(`alstm_*`/`tra_*`),重新跑融合 → 产出一个**完整的候选 recorder**(语义与全量一致,可直接对比/上线);同时单独算该算法的 IC/IR 进对比区。
- 好处:单训快(只训 1 个),产物统一(仍是一个完整候选),复用现有评估/上线路径。
- **边界**:若"其余模型无上一版预测"(从没跑过全量)→ 明确报错,提示先跑一次全量。
- ⚠️ **实现细节(落地前定)**:再融合时 Ridge stacker 两条路 —— (a) **复用上一版权重**(快、轻微近似);(b) **持久化 OOF 预测后重拟合**(更准)。倾向 (b);落地前先确认 `pred.pkl`/recorder 是否已存 OOF(valid 窗 out-of-fold 预测),若没存,P3 顺带把 OOF 持久化下来再重拟合。

### 候选 vs 当前(promote 语义,消歧)
引入"候选 → promote"后,需明确"训完是否自动上线":
- **手动训练(立即训练、尤其单算法)→ 产出候选,不自动上线**,需在 ④ 显式 promote 才成为当前。这样能"先评估再上线",符合用户要的对比决策。
- **排程(cron)自动重训 → 沿用现状 = 训完即当前**(无人值守,不能等人 promote)。
- 回滚机制不变(当前 ↔ 上一版);promote 即"把某候选设为当前",与回滚共用底层 recorder 切换逻辑。

### 后端:新增垂直切片 `backend/app/training/`
呼应现有 `analysis/`、`scheduling/` 切片风格:`schemas.py · service.py · router.py · orm.py · registry.py` + tests。
- **复用而非重写**:进度任务复用并扩展 `scheduling/service.py::SchedulerManager`(让它能带训练参数、tail 进度文件);评估/分周期复用 `evaluation/`;版本/回滚复用 `models/service.py`。

### 数据模型
1. **训练运行表(sqlite,alembic 0005)** `training_runs`:
   `job_id · kind(cron/manual) · scope(full/single) · models(json,如 ["lgbm"]) · status · started_at · finished_at · recorder_id · error · metrics_json · git_sha`。
   理由:历史区要排序/查询,且**失败/进行中**的 run 没有 recorder 也要能列出来;与现有 alembic 体系(AI 分析加过 0004)一致。
2. **recorder 元数据**:训练完把 `{scope, models, train_window, git_sha, created_at}` 写进 recorder 的 MLflow tags + `artifacts/meta.json`,历史区据此显示"这版训了什么"。
3. **模型注册表** `production/model_registry.py`:`id → {config 路径, trainer 入口, enabled}`,现登记 LGBM/ALSTM/TRA。前端"底层模型"开关读它;`enabled` 落回 [`rolling_ensemble.yaml`](production/configs/rolling_ensemble.yaml) 的 `models[]`。加新模型 = 加一条 + 一个 trainer(扩展位)。

### 端点(`/api/training`)
| 方法 | 路径 | 作用 |
|---|---|---|
| GET | `/api/training/registry` | 可用算法 + 各自 enabled |
| PUT | `/api/training/registry` | 改 enabled(写回 yaml) |
| POST | `/api/training/run` | `{scope, models[], end_date?, force?}` → `job_id` |
| GET | `/api/training/jobs/{job_id}` | 进度(最新帧 + phases) |
| GET | `/api/training/jobs/active` | 进行中任务 |
| GET | `/api/training/runs` | 历史(join `training_runs` + 评估缓存) |
| POST | `/api/training/promote` | 候选 → 当前(复用 rollback 机制) |

评估/分周期/单模型 IC 优先复用 `evaluation` + `models` 端点;缺的(per-model 单独 IC)再补。

### Production 改动
- [`rolling_train.py`](production/rolling_train.py) 加 `run-once --only lgbm[,alstm]`(默认 all = 全量):`--only` 时只训指定模型,其余复用上一版预测列重跑融合 → 新候选 recorder。
- 新增 `production/progress.py`(`emit(...)` 仅在 `QLIB_PROGRESS_FILE` 存在时写),在 LGBM callback / ALSTM·TRA 每 epoch / 融合·评估阶段插桩。
- P3 顺带:OOF 预测持久化(供单算法再融合时重拟合 stacker)。

## 数据流
立即训练:前端 `POST /api/training/run` → training service 起子进程(注入 `QLIB_PROGRESS_FILE`、`--only` 等)→ 训练进程逐阶段写 `progress.jsonl` + 完成写 `meta.json` + MLflow recorder + 落 `training_runs` → 前端轮询 `jobs/{id}` 接口(后端 tail 文件返回最新帧)→ 画进度/曲线 → 完成后刷新历史表 → 用户在 ④ promote。

## 错误处理 / 边界
- 子进程崩溃 → `job=failed` + 末尾 progress/log;`training_runs` 记 error。
- 单算法但"其余模型无上一版预测" → 明确报错(提示先跑全量)。
- 交易时段守卫 + 并发锁(一次一个训练)沿用。
- `progress.jsonl` 与 `production/runs/` 按现有 `keep_weeks` housekeeping 清理。

## 测试
- `progress.jsonl` 解析单测;`--only` 复用预测再融合单测(喂假 `pred.pkl`);registry 读写;端点契约;前端进度组件渲染;promote/rollback 回归。
- 约定:`F:/Tools/Anaconda/envs/qlib` python、**从主仓跑**、`-X utf8`、MultiIndex 列注意(沿用本仓既有踩坑)。

## 分期(各期独立可用、可单独合并)
- **P1 治本进度**:`production/progress.py` 插桩 + 后端解析进度 + 前端"进行中"实时进度/损失曲线(全量训练即可看到进度)。← 先解决最痛的"看不到进度"。
- **P2 历史 + 对比**:`training_runs` 表(alembic 0005)+ recorder 元数据 + 历史表 + 对比/promote(多数复用后端已有)。
- **P3 单算法 + 注册表**:`rolling_train --only` 重训再融合 + OOF 持久化 + 底层模型纳入开关 + `model_registry`。

## 开放问题 / 待落地确认
- stacker 再融合:复用旧权重 vs 重拟合 OOF —— 倾向重拟合,P3 落地前先确认 OOF 是否已存盘。
- 历史区数据源:`training_runs` 表与评估缓存的 join 字段对齐(recorder_id 为锚)。
- 进度轮询频率与"任务结束后保留进度文件供回看"的清理窗口(默认沿用 `keep_weeks`)。
