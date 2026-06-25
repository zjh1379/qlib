# 资源治理:双档预算 + GPU 卸载 + 崩溃兜底 · 设计 (2026-06-25)

## 背景与动机
用户反馈:在**训练模型 / 选股跑模型**界面操作时,电脑频繁**卡死**,甚至 **Windows 崩溃重启**——怀疑资源占用过于贪婪。希望能限制占用(可接受变慢、可后台/每天执行),或给一个"固定的虚拟环境"。

### 摸底事实(影响设计,实测于 2026-06-25)
项目**已有相当完整的资源保护设施,但大多没在运行 / 按错误的前提校准**:

- **计算如何启动**(子进程隔离,已具备):
  - 训练:[`scheduling/service.py`](backend/app/scheduling/service.py) 用 `asyncio.create_subprocess_exec` 跑 `python -m production.rolling_train run-once`;[`run_split.py`](production/run_split.py) 再把 LGBM/ALSTM/TRA 拆成 **3 个串行子进程**,每个带 `max_rss_gb=10`、`max_commit_pct=85` 的内部看门狗。
  - 推理:[`inference/service.py`](backend/app/inference/service.py) 用 `subprocess.Popen`(守护线程内)跑 `python -m production.daily_inference`,超时 600s。
  - **并发锁已具备**:训练 `asyncio.Lock`、推理 `threading.Lock`,**同一时刻只允许 1 训练 + 1 推理**,不会叠加。
- **现有资源旋钮**:LGBM [`num_threads: 20`](production/configs/lgbm_alpha158_multi.yaml)、`num_leaves: 210`;ALSTM [`batch_size: 1024`、`GPU: 0`](production/configs/alstm_alpha360.yaml)(=CPU);**`OMP/MKL/OPENBLAS_NUM_THREADS` 完全没设** → BLAS/numpy 默认吃满全部逻辑核。
- **看门狗**:[`safety_watchdog.py`](production/safety_watchdog.py) 逻辑扎实(2 秒采样 commit,>92% 杀最重训练子进程,白名单保护 uvicorn/vite/chrome),但**①是独立脚本、没接进后端**(用户从界面点训练时它几乎肯定没开),**②阈值按 60GB commit 上限校准,而实测只有 35.7GB**。
- **配置面**:[`core/config.py`](backend/app/core/config.py) pydantic Settings,有 `retrain_python_path`,但**无任何资源限制字段**。

### 机器实测(2026-06-25)
| 项 | 代码/注释假设 | **实测** |
|---|---|---|
| 物理内存 | 32GB | 31.8GB |
| 页面文件 | 28GB | **3.9GB(系统托管,懒扩)** |
| **commit 上限** | **60GB** | **35.7GB**(空载已用 22.6GB / 63%) |
| CPU | — | i7-13700KF,16 核 / **24 逻辑** |
| GPU | "CPU-only" | **RTX 3080 Ti 12GB,空闲 10.7GB**,torch `2.5.1+cu121`、CUDA 可用 |
| C: 空闲 | — | 115GB(够建大页面文件) |

### 两个独立病灶(必须分开治)
1. **卡顿**(界面无反应但能恢复)= **CPU 饱和**:LGBM 20 线程 + 不限的 BLAS 线程占满 24 逻辑核,前台抢不到 CPU。
2. **崩溃/重启**(真死机)= **commit 上限击穿**:真实上限仅 35.7GB、空载已占 63%,页面文件懒扩跟不上 ALSTM 秒涨几 GB 的尖峰 → Event 2004 低虚拟内存 → Event 41 内核电源硬重启(正是 `safety_watchdog.py` 注释里描述的死亡链)。

## 目标(已确认)
1. **双档可切换的资源预算**:保守档(白天/手动,保前台流畅)与放开档(夜间/调度,求快);可接受变慢。
2. **物理兜底(治崩溃主力)**:扩页面文件(commit 35.7GB → ~80GB)+ 把 watchdog 接进后端常驻。
3. **GPU 卸载**:神经网络 ALSTM/TRA 移到 3080 Ti,主要为**提速 + 给 CPU 让路**(部分省内存,非崩溃根治,见关键认知);**附回测验收门**保护量化基线。
4. **夜间自动调度**:每天凌晨放开档跑推理、每周放开档重训,按序不叠加。

## 非目标 (out of scope)
- ❌ **Job Object 硬上限 / WSL2 / Docker 容器**:用户已选「预算级隔离」,本期不做 OS 级硬墙或容器。
- ⏸ **UI 实时资源仪表盘**(commit/显存可视化):留作 P2。
- ⏸ **手动档位覆盖下拉**:本期档位**自动按触发来源**选择,手动覆盖留作易加的后续。
- ❌ 不改 qlib 排名/选股逻辑、不改净成本回测口径(GPU 卸载只用回测**验收**,不改口径)。
- ❌ 不引入新算法、不做分布式/多机训练、不动并发锁(沿用一次一个)。

## 总体方案:三层防御
```
第①层 预算注入(治卡顿) ── 父进程给子进程套预算
       线程上限(OMP/MKL/LGBM) + 进程优先级 + CPU 亲和性
第②层 物理兜底(治崩溃·主力) ── 扩页面文件 + watchdog 接进后端常驻
       commit 上限 35.7GB → ~80GB;watchdog 永远在岗、加绝对余量地板
第③层 GPU 卸载(提速 + 治卡顿 + 部分省内存) ── ALSTM/TRA → 3080 Ti
       搬走模型张量(CPU 让路、加速);但 handler 数据内存仍在 RAM(见下)
调度 ── 夜间放开档按序跑(数据→推理→(每周)训练),峰值不叠加
```
**崩溃的主力解是第②层**(扩页面文件 + 常驻 watchdog),不是 GPU。为什么这套而非容器:预算注入 + 扩页面文件 + 常驻 watchdog **改动小、零额外依赖、不动 akshare/qlib 数据路径、立刻见效**;容器在单机日跑场景属过度工程(GPU 透传、网络、挂载都要重搭)。

### ⚠️ 关键认知:17GB 大概率是数据 handler,不是模型
`safety_watchdog.py` 注释原文 "grows to 17+ GB **due to handler accumulation**" 表明峰值内存主要来自 **qlib Alpha360 数据 handler**(把整个特征矩阵物化在系统内存),而非神经网络张量。因此:
- **GPU 卸载只搬走模型 + batch 张量(估计 1–3GB)**,handler 的大头**仍在系统内存** → GPU 对内存崩溃只是**部分缓解**,不是根治。
- **崩溃根治靠第②层**(扩页面文件把 commit 抬到 ~80GB + watchdog 兜底)。
- **须早验证(P0 里就做)**:实测一次训练的 RSS 构成(handler vs 模型),确认 17GB 的来源,据此判断 GPU 卸载到底能省多少内存,以及是否需要后续专门优化 handler 内存(分块/及时释放,本期之外、视测量结果决定是否开 P4)。

## 架构

### 1. 资源档位(核心)——`backend/app/core/resources.py`(新建)
两档定义为唯一真相源,父进程启动子进程时注入,子进程脚本开头读环境变量自配。

| 旋钮 | 保守档 conservative | 放开档 aggressive | 注入方式 |
|---|---|---|---|
| `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS` | 4 | 8 | 子进程 `env=` |
| LGBM `num_threads` | 6 | 16 | 子进程读 `QLIB_RES_PROFILE` 后覆盖 yaml |
| 进程优先级 | BELOW_NORMAL | NORMAL | Popen `creationflags` / psutil `nice()` |
| CPU 亲和性 | 绑 12 逻辑核(留半给前台) | 全部 24 核 | spawn 后 psutil `cpu_affinity()` |
| 单进程内存软上限 | 8GB | 12GB | 传给 watchdog / run_split |
| 神经网络设备 | GPU | GPU | (见第 2 节,两档都用 GPU) |

> 数字为**起始值、可调**。

**模块接口(单一职责,可独立测):**
- `ResourceProfile`(dataclass/pydantic):上表字段。
- `PROFILES: dict[str, ResourceProfile]`:`{"conservative":…, "aggressive":…}`。
- `popen_env(profile) -> dict[str,str]`:产出线程上限 + `QLIB_RES_PROFILE=<name>` 的环境变量(纯函数,易测)。
- `popen_creationflags(profile) -> int`:Windows 优先级标志(非 Windows 返回 0)。
- `apply_post_spawn(pid, profile)`:spawn 后用 psutil 设 `nice()` + `cpu_affinity()`,fail-soft(拿不到进程/无权限不报错)。
- `bootstrap_from_env()`:**子进程侧**,训练/推理脚本入口调用,读 `QLIB_RES_PROFILE` → 设 LGBM 线程、torch 设备等;无该变量时退回默认(命令行直跑、单测不受影响)。

**接入点(改造而非重写):**
- [`scheduling/service.py`](backend/app/scheduling/service.py) 起训练子进程处:`env={**os.environ, **popen_env(profile)}` + `creationflags` + spawn 后 `apply_post_spawn`。
- [`inference/service.py`](backend/app/inference/service.py) 起推理子进程处:同上。

**档位选择:自动按触发来源**——界面手动触发 = `conservative`;调度(cron/夜间)= `aggressive`。

### 2. GPU 卸载(核心·提速 + 治卡顿 + 部分省内存)
- **设备配置**:改 [`alstm_alpha360.yaml`](production/configs/alstm_alpha360.yaml) 与 [`tra_alpha360.yaml`](production/configs/tra_alpha360.yaml) 的设备字段使用 CUDA。
  - ⚠️ **实现首步须核实**:qlib `pytorch_alstm` / `pytorch_tra` 对 `GPU` 字段的语义(是设备 id 还是布尔?`GPU: 0` 到底是"用 cuda:0"还是"不用 GPU"?)。读 qlib 源确认后再定改法,避免误配。
- **回测验收门(硬性)**:GPU 重训后跑一次 canonical **net-of-cost** 回测(复用评估口径),IC / 年化超额须落在 CPU 基线**容差内**(容差值落地前定,如 IC 漂移 < 1e-3、年化超额不劣化)才采纳;否则保留 CPU、记录差异并排查。**保护既有量化基线。**
- **降级保护**:子进程侧 `bootstrap_from_env` 设设备前先 `torch.cuda.is_available()` + 查可用显存;CUDA 不可用 / 空闲显存不足(阈值如 < 4GB,防 LM Studio 占用)→ **自动回退 CPU**(回退时仍受第①③层保护),并在进度/日志里标注"已回退 CPU"。
- **VRAM 预检**:启动神经训练前查一次空闲显存,不足则告警 + 回退。

### 3. 扩页面文件(核心·最便宜的崩溃杀手)
- C: 设**固定**页面文件 48GB(initial = max = 49152 MB),commit 上限 35.7GB → ~80GB,消除懒扩 race。
- 一次性操作,需**管理员 + 重启**。提供确切命令(`Set-CimInstance`/`wmic pagefileset` + 关闭系统托管),由用户执行并重启(重启须用户来)。
- C: 115GB 空闲,占用 48GB 后仍余 67GB,安全。

### 4. 常驻 watchdog(核心·补缺口)——改造 [`safety_watchdog.py`](production/safety_watchdog.py)
- **接进 FastAPI lifespan**:后端启动即作为受管子进程常驻(2 秒一采,极廉价),退出时收回。
- **加绝对余量地板**:除百分比外,可用 commit < 阈值(如 4GB)即视为危险——百分比在小上限下反应太迟。
- **白名单/猎杀名单补全**:`KILLABLE_TOKENS` 增补 `train_alstm`、`train_tra`、`walk_forward`(目前只有 rolling_train/run_split/daily_inference/incremental_refresh/backfill);继续保护 uvicorn/vite/chrome/claude/explorer。
- **杀进程事件回传 job tracker**:被杀的任务在 UI 显示"因内存保护被中止(OOM 防护)"而非神秘失败。
- **阈值随真实上限重校**:扩页面文件后 92% of ~80GB ≈ 74GB,余量充足;地板值守住绝对底线。

### 5. 夜间调度(核心)——复用 [`scheduling/service.py`](backend/app/scheduling/service.py) 的 APScheduler
- 已有:每周重训 cron + 交易时段护栏。
- 新增:**每天凌晨(起始 02:00,可配)放开档跑 `daily_inference`**;每周重训归入夜间放开档。
- 已有单任务锁保证 数据→推理→训练 **按序不叠加**;调度触发的任务自动用 `aggressive` 档。
- 时间、开关可配(进 Settings)。

## 数据流
```
[手动] UI 点训练/选股
  → service 选 conservative 档 → popen_env+creationflags 起子进程 → apply_post_spawn 设优先级/亲和性
  → 子进程 bootstrap_from_env:设线程数 + (神经网络)GPU/回退
  → watchdog(常驻)守 commit;超线触发地板/百分比 → 杀最重训练子进程 → 事件回传 job tracker

[夜间] APScheduler 02:00
  → 选 aggressive 档 → 同上链路,CPU 放开、全核
  → 数据刷新 → 推理 →(每周)重训,锁保证按序
```

## 错误处理
- **GPU 不可用/显存不足** → 回退 CPU,标注,不中断。
- **watchdog 杀进程** → job 标 `killed_oom`,UI 明示原因,可重试。
- **页面文件未扩**(用户没执行)→ 系统仍可跑,但 watchdog 地板会更早介入(更易被杀),日志提示"建议扩页面文件"。
- **psutil 设亲和性/优先级失败** → fail-soft,降级为仅线程上限,不报错。
- **`QLIB_RES_PROFILE` 缺失**(命令行直跑/单测)→ 退回默认行为,脚本照常工作。

## 测试策略
- **`resources.py` 纯函数**:`popen_env`/`popen_creationflags` 给定档位输出确定 → 直接断言(单测,无需起进程)。
- **`apply_post_spawn`**:mock psutil,断言调用了 `nice`/`cpu_affinity` 且异常被吞。
- **`bootstrap_from_env`**:设/不设 `QLIB_RES_PROFILE`,断言线程数与设备选择;mock `torch.cuda.is_available` 验证回退。
- **watchdog 改造**:构造假 commit 读数,断言地板/百分比触发逻辑与猎杀名单匹配(沿用现有测试风格)。
- **调度**:断言夜间 job 用 aggressive 档、受锁串行。
- **GPU 回测验收**:作为**计划里的人工/脚本验收里程碑**(跑真实回测对比基线),非单测。
- 后端测试从 `backend/` cwd 跑(避免 worktree 根目录 qlib 源码遮蔽编译版),`-X utf8`。

## 落地分期(建议)
- **P0 兜底(最高优先,改动最小)**:扩页面文件(用户执行)+ watchdog 接进 lifespan + 阈值/地板/猎杀名单重校 + **实测 17GB 内存构成**(handler vs 模型)。← 立刻止崩溃 + 为 P2 定调。
- **P1 预算注入**:`resources.py` + 两档 + 接入训练/推理 spawn。← 止卡顿、双档成形。
- **P2 GPU 卸载**:设备配置 + 回退保护 + **回测验收门**。← 加速 + 给 CPU 让路(省内存幅度据 P0 测量而定)。
- **P3 夜间调度**:每日推理 + 每周重训归夜间放开档。← 无人值守。

## 关联
- 量化基线/回测口径:[[quant_pipeline]];训练界面:[`2026-06-15-training-studio-design.md`](docs/superpowers/specs/2026-06-15-training-studio-design.md)。
- 既有兜底脚本:[`safety_watchdog.py`](production/safety_watchdog.py)、[`run_split.py`](production/run_split.py)。
