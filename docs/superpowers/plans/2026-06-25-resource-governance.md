# 资源治理实施计划(P0 兜底 → P1 双档预算 → P2 夜间调度)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让训练 / 推理在有界资源预算里跑,机器不再卡死或崩溃;双档可切换、夜间自动跑,并补上常驻内存兜底。

**Architecture:** 三层防御 —— ①父进程给子进程注入预算(线程上限/优先级/亲和性)治卡顿;②扩页面文件 + 把 `safety_watchdog` 接进后端常驻治崩溃;③复用 APScheduler 夜间放开档按序跑。神经模型**已在 GPU**,只加显存预检回退小补丁。详见 [`spec`](docs/superpowers/specs/2026-06-25-resource-governance-design.md)。

**Tech Stack:** Python 3 / FastAPI / pydantic-settings / APScheduler / psutil / pytest;Windows(PowerShell + `creationflags`)。

**前置约定(所有命令):**
- 假设 qlib conda 环境已激活;若未激活,把命令里的 `python` 换成 `F:/Tools/Anaconda/envs/qlib/python.exe`。
- **后端测试从 `backend/` 目录跑**(避免 worktree 根目录的 `qlib/` 源码遮蔽已编译的 qlib),加 `-X utf8`。
- **production/ 测试从仓库根目录跑**(这些测试不 import qlib,无遮蔽问题)。
- 频繁提交:每个 Task 末尾提交一次。

---

## 阶段 P0 — 兜底:止崩溃 + GPU 小补丁 + 测量

> P0 完成即可**立刻止住崩溃**(扩页面文件 + 常驻 watchdog),并量清 17GB 内存真相,与 P1/P2 无依赖。

### Task P0.0:扩页面文件(用户手动执行,无代码)

**这是一次性系统配置,需管理员 + 重启,必须由用户执行。** 把下列内容交给用户,确认执行并重启后再继续。

- [ ] **Step 1: 以管理员打开 PowerShell,关闭"系统托管"并设固定 48GB 页面文件**

```powershell
# 关闭"自动管理所有驱动器的分页文件大小"
$cs = Get-CimInstance Win32_ComputerSystem
if ($cs.AutomaticManagedPagefile) {
  Set-CimInstance $cs -Property @{ AutomaticManagedPagefile = $false }
}
# 设 C: 固定 48GB(initial = maximum = 49152 MB)
$pf = Get-CimInstance Win32_PageFileSetting -Filter "Name='C:\\pagefile.sys'"
if ($pf) {
  Set-CimInstance $pf -Property @{ InitialSize = 49152; MaximumSize = 49152 }
} else {
  New-CimInstance -ClassName Win32_PageFileSetting -Property @{ Name='C:\pagefile.sys'; InitialSize=49152; MaximumSize=49152 }
}
```

- [ ] **Step 2: 重启电脑**(页面文件改动需重启生效)。

- [ ] **Step 3: 重启后验证 commit 上限已抬升**

Run:
```powershell
$os = Get-CimInstance Win32_OperatingSystem
"Commit limit (GB): {0:N1}" -f ($os.TotalVirtualMemorySize/1MB)
```
Expected: 约 **80 GB**(31.8 RAM + 48 pagefile),而非之前的 35.7 GB。

> 无需提交(纯系统配置)。

---

### Task P0.1:watchdog 决策纯函数 + 绝对余量地板

**Files:**
- Modify: `production/safety_watchdog.py`
- Test: `production/tests/test_safety_watchdog.py`

- [ ] **Step 1: 写失败测试**

在 `production/tests/test_safety_watchdog.py` 新增(文件不存在则创建,顶部加 `from production.safety_watchdog import decide_action`):

```python
from production.safety_watchdog import decide_action


def test_decide_action_ok_when_low():
    assert decide_action(50.0, 18.0, 80.0, warn_pct=80.0, kill_pct=92.0, floor_gb=4.0) == "ok"


def test_decide_action_warn_at_warn_pct():
    assert decide_action(85.0, 68.0, 80.0, warn_pct=80.0, kill_pct=92.0, floor_gb=4.0) == "warn"


def test_decide_action_kill_at_kill_pct():
    assert decide_action(93.0, 74.0, 80.0, warn_pct=80.0, kill_pct=92.0, floor_gb=4.0) == "kill"


def test_decide_action_kill_on_absolute_floor_even_if_pct_low():
    # free = 80 - 77 = 3GB < 4GB floor -> kill, even though pct (96%) ... here pct is 96 but
    # the point is the floor catches thin headroom regardless of pct threshold:
    assert decide_action(70.0, 77.0, 80.0, warn_pct=80.0, kill_pct=92.0, floor_gb=4.0) == "kill"
```

- [ ] **Step 2: 运行,确认失败**

Run: `python -m pytest production/tests/test_safety_watchdog.py -v`
Expected: FAIL — `ImportError: cannot import name 'decide_action'`.

- [ ] **Step 3: 实现 `decide_action`**

在 `production/safety_watchdog.py` 的 `_commit_pct` 之后插入:

```python
def decide_action(
    pct: float,
    used_gb: float,
    total_gb: float,
    *,
    warn_pct: float,
    kill_pct: float,
    floor_gb: float,
) -> str:
    """Pure decision: 'kill' | 'warn' | 'ok'.

    Kills when commit pct crosses kill_pct OR when *absolute* free commit
    (total - used) drops below floor_gb. The floor matters because on a small
    commit ceiling a percentage threshold reacts too late — by the time pct is
    high the few remaining GB vanish within one poll interval.
    """
    free_gb = total_gb - used_gb
    if pct >= kill_pct or free_gb < floor_gb:
        return "kill"
    if pct >= warn_pct:
        return "warn"
    return "ok"
```

- [ ] **Step 4: 重构 `main()` 循环用 `decide_action`**

在 `main()` 里把 `args` 解析后(`_setup_logging` 之前或之后)取到 `floor_gb`(下个 Task 加 CLI;此处先用默认 `4.0`),并把循环体替换为:

```python
        while True:
            try:
                pct, used_gb, total_gb = _commit_pct()
                now = time.time()
                action = decide_action(
                    pct, used_gb, total_gb,
                    warn_pct=args.warn_pct, kill_pct=args.kill_pct,
                    floor_gb=getattr(args, "floor_gb", 4.0),
                )
                if action == "kill":
                    target = _find_heaviest_killable()
                    if target:
                        _kill_with_grace(
                            target,
                            reason=f"commit {pct:.1f}% / free {total_gb-used_gb:.1f}GB crossed limits",
                        )
                        consecutive_kill_attempts += 1
                        if consecutive_kill_attempts > 3:
                            log.error("too many kill attempts — pausing 30s to let OS recover")
                            time.sleep(30)
                            consecutive_kill_attempts = 0
                    else:
                        if now - last_warn_at > 30:
                            log.warning("commit %.1f%% CRITICAL but no killable training proc", pct)
                            last_warn_at = now
                elif action == "warn":
                    consecutive_kill_attempts = 0
                    if now - last_warn_at > 30:
                        log.warning("commit %.1f%% (%.1f / %.1f GB) approaching limit", pct, used_gb, total_gb)
                        last_warn_at = now
                else:
                    consecutive_kill_attempts = 0
                time.sleep(args.interval)
            except Exception as exc:
                log.exception("watchdog loop error: %s", exc)
                time.sleep(args.interval)
```

- [ ] **Step 5: 运行,确认通过**

Run: `python -m pytest production/tests/test_safety_watchdog.py -v`
Expected: PASS(4 passed)。

- [ ] **Step 6: 提交**

```bash
git add production/safety_watchdog.py production/tests/test_safety_watchdog.py
git commit -m "feat(watchdog): pure decide_action + absolute free-commit floor"
```

---

### Task P0.2:猎杀名单判定函数 + 补全 token

**Files:**
- Modify: `production/safety_watchdog.py`
- Test: `production/tests/test_safety_watchdog.py`

- [ ] **Step 1: 写失败测试**

追加:

```python
from production.safety_watchdog import is_killable_cmd


def test_is_killable_matches_training_tokens():
    assert is_killable_cmd("python -m production.train_alstm --end-date 2026-06-20")
    assert is_killable_cmd("python -m production.rolling_train run-once")
    assert is_killable_cmd("python -m production.run_split --end-date 2026-06-20")
    assert is_killable_cmd("python -m production.train_tra ...")
    assert is_killable_cmd("python -m production.walk_forward ...")


def test_is_killable_protects_infra():
    assert not is_killable_cmd("uvicorn app.main:app --port 8000")
    assert not is_killable_cmd("node vite")
    assert not is_killable_cmd("chrome.exe --type=renderer")
```

- [ ] **Step 2: 运行,确认失败**

Run: `python -m pytest production/tests/test_safety_watchdog.py -k is_killable -v`
Expected: FAIL — `ImportError: cannot import name 'is_killable_cmd'`.

- [ ] **Step 3: 实现 + 补 token**

把 `KILLABLE_TOKENS` 改为(增补 3 个):

```python
KILLABLE_TOKENS = (
    "rolling_train",
    "production.run_split",
    "production.daily_inference",
    "production.incremental_refresh",
    "production.backfill_calibration",
    "production.train_alstm",
    "production.train_tra",
    "production.walk_forward",
)


def is_killable_cmd(cmd: str) -> bool:
    """True iff the command line matches a training process we are allowed to
    kill. Infra (uvicorn/vite/chrome/claude/explorer) never matches."""
    return any(tok in cmd for tok in KILLABLE_TOKENS)
```

然后把 `_find_heaviest_killable` 里的判断改为用它:

```python
            cmd = " ".join(p.info.get("cmdline") or [])
            if not is_killable_cmd(cmd):
                continue
```

- [ ] **Step 4: 运行,确认通过**

Run: `python -m pytest production/tests/test_safety_watchdog.py -v`
Expected: PASS(全部)。

- [ ] **Step 5: 提交**

```bash
git add production/safety_watchdog.py production/tests/test_safety_watchdog.py
git commit -m "feat(watchdog): is_killable_cmd + cover train_alstm/train_tra/walk_forward"
```

---

### Task P0.3:杀进程写 kills.jsonl

**Files:**
- Modify: `production/safety_watchdog.py`
- Test: `production/tests/test_safety_watchdog.py`

- [ ] **Step 1: 写失败测试**

追加:

```python
import json
from production.safety_watchdog import record_kill


def test_record_kill_appends_json_line(tmp_path):
    p = tmp_path / "watchdog_kills.jsonl"
    record_kill(p, {"pid": 123, "rss_gb": 11.2, "cmd": "python -m production.train_alstm", "reason": "floor"})
    record_kill(p, {"pid": 456, "rss_gb": 9.0, "cmd": "x", "reason": "pct"})
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert rec["pid"] == 123 and rec["reason"] == "floor"
    assert "ts" in rec  # record_kill stamps time


def test_record_kill_none_path_is_noop():
    # passing None must not raise (watchdog may run without a kills path)
    record_kill(None, {"pid": 1})
```

- [ ] **Step 2: 运行,确认失败**

Run: `python -m pytest production/tests/test_safety_watchdog.py -k record_kill -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: 实现 `record_kill`,并在 `_kill_with_grace` 里调用**

在 import 区确保有 `import json` 和 `from pathlib import Path`(`Path` 若未导入则加),实现:

```python
def record_kill(path, record: dict) -> None:
    """Append one JSON line describing a watchdog kill. Fail-soft; path=None
    skips (watchdog can run without a kills file). The backend tails this file
    to surface 'killed by OOM guard' in the UI instead of a mystery failure."""
    if path is None:
        return
    try:
        import time as _t
        rec = {"ts": _t.strftime("%Y-%m-%dT%H:%M:%S"), **record}
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass
```

改 `_kill_with_grace` 签名加 `kills_path=None`,并在拿到 reason 后记录:

```python
def _kill_with_grace(p: psutil.Process, reason: str, kills_path=None) -> None:
    try:
        rss_gb = p.memory_info().rss / 2**30
        cmd = " ".join(p.cmdline())[:200]
        log.error("KILLING pid=%d rss=%.1fGB cmd=%s — %s", p.pid, rss_gb, cmd, reason)
        record_kill(kills_path, {"pid": p.pid, "rss_gb": round(rss_gb, 2), "cmd": cmd, "reason": reason})
        # Kill children first so they don't reparent to init
        for ch in p.children(recursive=True):
            try:
                ch.terminate()
            except Exception:
                pass
        p.terminate()
        gone, alive = psutil.wait_procs([p], timeout=GRACE_SECONDS)
        for survivor in alive:
            log.error("force-killing pid=%d (didn't terminate gracefully)", survivor.pid)
            try:
                survivor.kill()
            except Exception:
                pass
    except Exception as exc:
        log.exception("kill failed pid=%d: %s", p.pid, exc)
```

在 `main()` 调用处把 kills_path 传进去(`KILLS` 由下个 Task 的 CLI 提供;此处先 `getattr(args, "kills_path", None)`):

```python
                        _kill_with_grace(
                            target,
                            reason=f"commit {pct:.1f}% / free {total_gb-used_gb:.1f}GB crossed limits",
                            kills_path=getattr(args, "kills_path", None),
                        )
```

- [ ] **Step 4: 运行,确认通过**

Run: `python -m pytest production/tests/test_safety_watchdog.py -v`
Expected: PASS(全部)。

- [ ] **Step 5: 提交**

```bash
git add production/safety_watchdog.py production/tests/test_safety_watchdog.py
git commit -m "feat(watchdog): record_kill -> kills.jsonl for UI surfacing"
```

---

### Task P0.4:watchdog CLI 加 --floor-gb / --kills-path

**Files:**
- Modify: `production/safety_watchdog.py`(仅 `main()` 的 argparse 段)

- [ ] **Step 1: 加两个参数**

在 `main()` 的 `parser.add_argument(... --log-level ...)` 之后追加:

```python
    parser.add_argument("--floor-gb", type=float, default=4.0,
                        help="kill heaviest training proc if free commit < this many GB (default 4)")
    parser.add_argument("--kills-path", default=None,
                        help="append JSON kill records here (e.g. logs/watchdog_kills.jsonl)")
```

- [ ] **Step 2: 冒烟验证(无单测,人工)**

Run: `python -m production.safety_watchdog --warn-pct 99 --kill-pct 99.9 --floor-gb 0.1 --interval 5`
Expected: 打印 `watchdog started warn=99.0% kill=99.9% ...`,稳定轮询不报错;`Ctrl-C` 退出。

- [ ] **Step 3: 提交**

```bash
git add production/safety_watchdog.py
git commit -m "feat(watchdog): --floor-gb and --kills-path CLI options"
```

---

### Task P0.5:watchdog 监督器(后端起停)

**Files:**
- Create: `backend/app/core/watchdog_supervisor.py`
- Test: `backend/app/core/tests/test_watchdog_supervisor.py`

- [ ] **Step 1: 写失败测试**

`backend/app/core/tests/test_watchdog_supervisor.py`(目录不存在则创建):

```python
from pathlib import Path
from app.core.watchdog_supervisor import watchdog_cmd


def test_watchdog_cmd_builds_expected_args():
    cmd = watchdog_cmd(
        python_path="C:/py/python.exe",
        repo_root=Path("E:/Projects/qlib"),
        floor_gb=4.0,
        kill_pct=92.0,
        kills_path=Path("E:/Projects/qlib/logs/watchdog_kills.jsonl"),
    )
    assert cmd[0] == "C:/py/python.exe"
    assert cmd[1:4] == ["-m", "production.safety_watchdog", "--floor-gb"]
    assert "4.0" in cmd
    assert "--kill-pct" in cmd and "92.0" in cmd
    assert "--kills-path" in cmd
```

- [ ] **Step 2: 运行,确认失败**

Run: `cd backend && python -X utf8 -m pytest app/core/tests/test_watchdog_supervisor.py -v`
Expected: FAIL — `ModuleNotFoundError: app.core.watchdog_supervisor`.

- [ ] **Step 3: 实现监督器**

`backend/app/core/watchdog_supervisor.py`:

```python
"""Start/stop the system commit-charge watchdog (production.safety_watchdog)
as a managed child process for the FastAPI lifespan. The watchdog is the
last-line OOM defense; before this it was a standalone script users forgot
to run. Fail-soft: a failed watchdog start must never block app startup."""
from __future__ import annotations

import subprocess
from pathlib import Path

from app.core.logging import get_logger

_log = get_logger("watchdog")


def watchdog_cmd(python_path: str, repo_root: Path, *, floor_gb: float,
                 kill_pct: float, kills_path: Path) -> list[str]:
    """Pure: build the watchdog subprocess argv."""
    return [
        python_path, "-m", "production.safety_watchdog",
        "--floor-gb", str(floor_gb),
        "--kill-pct", str(kill_pct),
        "--kills-path", str(kills_path),
    ]


def start_watchdog(python_path: str, repo_root: Path, *, floor_gb: float = 4.0,
                   kill_pct: float = 92.0) -> subprocess.Popen | None:
    """Launch the watchdog detached-ish (own log file). Returns the Popen or
    None on failure (never raises)."""
    try:
        logs = repo_root / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        kills_path = logs / "watchdog_kills.jsonl"
        cmd = watchdog_cmd(python_path, repo_root, floor_gb=floor_gb,
                           kill_pct=kill_pct, kills_path=kills_path)
        logf = (logs / "watchdog.log").open("ab")
        proc = subprocess.Popen(cmd, cwd=str(repo_root), stdout=logf, stderr=subprocess.STDOUT)
        _log.info("watchdog_started", pid=proc.pid, cmd=" ".join(cmd))
        return proc
    except Exception as exc:
        _log.warning("watchdog_start_failed", error=str(exc))
        return None


def stop_watchdog(proc: subprocess.Popen | None) -> None:
    """Terminate the watchdog child. Fail-soft."""
    if proc is None:
        return
    try:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
        _log.info("watchdog_stopped")
    except Exception as exc:
        _log.warning("watchdog_stop_failed", error=str(exc))
```

- [ ] **Step 4: 运行,确认通过**

Run: `cd backend && python -X utf8 -m pytest app/core/tests/test_watchdog_supervisor.py -v`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add backend/app/core/watchdog_supervisor.py backend/app/core/tests/test_watchdog_supervisor.py
git commit -m "feat(watchdog): backend supervisor (start/stop managed child)"
```

---

### Task P0.6:接进 FastAPI lifespan

**Files:**
- Modify: `backend/app/main.py:28-62`(lifespan)

- [ ] **Step 1: 在 lifespan 启动段起 watchdog**

在 `backend/app/main.py` 顶部 import 区加:

```python
from app.core.watchdog_supervisor import start_watchdog, stop_watchdog
```

在 `lifespan` 里、`init_qlib_once` 之后、scheduler 段之前插入:

```python
    # Always-on OOM watchdog (last line of defense against commit-charge
    # exhaustion -> Windows hard freeze). Fail-soft: never blocks startup.
    watchdog_proc = start_watchdog(settings.retrain_python_path, repo_root)
```

> 注意 `repo_root` 在原代码里是在 scheduler 段才定义(`repo_root = Path(__file__).resolve().parent.parent.parent`)。把这行**上移**到 `init_qlib_once` 之后、`start_watchdog` 之前,使两处都能用。

在 `yield` 之后的关闭段(`await manager.stop()` 附近)加:

```python
    stop_watchdog(watchdog_proc)
```

- [ ] **Step 2: 人工验证启动起停**

Run(前台):`cd backend && python -X utf8 -m uvicorn app.main:app --port 8011`
Expected: 启动日志含 `watchdog_started pid=...`;`logs/watchdog.log` 出现 `watchdog started warn=... kill=...`。`Ctrl-C` 后日志含 `watchdog_stopped`,且 `production.safety_watchdog` 进程消失(`tasklist | findstr python` 核对)。

- [ ] **Step 3: 提交**

```bash
git add backend/app/main.py
git commit -m "feat(watchdog): wire always-on watchdog into FastAPI lifespan"
```

---

### Task P0.7:GPU 显存预检回退 helper

**Files:**
- Create: `production/gpu_guard.py`
- Test: `production/tests/test_gpu_guard.py`

- [ ] **Step 1: 写失败测试**

`production/tests/test_gpu_guard.py`:

```python
from production.gpu_guard import effective_gpu


def test_keeps_gpu_when_enough_free_vram():
    # probe returns (free_bytes, total_bytes)
    probe = lambda: (8 * 2**30, 12 * 2**30)
    assert effective_gpu(0, min_free_gb=4.0, probe=probe) == 0


def test_falls_back_to_cpu_when_low_vram():
    probe = lambda: (2 * 2**30, 12 * 2**30)  # 2GB free < 4GB
    assert effective_gpu(0, min_free_gb=4.0, probe=probe) == -1


def test_falls_back_when_probe_raises():
    def probe():
        raise RuntimeError("no cuda")
    assert effective_gpu(0, min_free_gb=4.0, probe=probe) == -1


def test_already_cpu_stays_cpu():
    assert effective_gpu(-1, min_free_gb=4.0, probe=lambda: (8 * 2**30, 12 * 2**30)) == -1
```

- [ ] **Step 2: 运行,确认失败**

Run: `python -m pytest production/tests/test_gpu_guard.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: 实现**

`production/gpu_guard.py`:

```python
"""GPU VRAM pre-check + CPU fallback for the neural trainers.

The ALSTM/TRA models already run on cuda:0 (pytorch_alstm.py:73 maps
GPU>=0 + cuda available -> cuda:0). This guard only prevents a CUDA OOM
when something else (e.g. LM Studio) has eaten the VRAM: if free VRAM is
below the threshold, force CPU by returning GPU id -1 (which pytorch_alstm
interprets as cpu)."""
from __future__ import annotations

import logging

_log = logging.getLogger("gpu_guard")


def _default_probe():
    """Return (free_bytes, total_bytes) for the current CUDA device."""
    import torch
    return torch.cuda.mem_get_info()  # raises if no CUDA


def effective_gpu(requested_gpu: int, *, min_free_gb: float = 4.0, probe=_default_probe) -> int:
    """Return the GPU id to actually use. -1 means CPU.

    requested_gpu < 0 -> already CPU, keep it.
    Otherwise probe free VRAM; if < min_free_gb (or probe fails) -> -1 (CPU)."""
    if requested_gpu < 0:
        return -1
    try:
        free_bytes, _total = probe()
        free_gb = free_bytes / 2**30
        if free_gb < min_free_gb:
            _log.warning("gpu_low_vram free=%.1fGB < %.1fGB -> falling back to CPU", free_gb, min_free_gb)
            return -1
        _log.info("gpu_ok free=%.1fGB -> using cuda:%d", free_gb, requested_gpu)
        return requested_gpu
    except Exception as exc:
        _log.warning("gpu_probe_failed (%s) -> falling back to CPU", exc)
        return -1
```

- [ ] **Step 4: 运行,确认通过**

Run: `python -m pytest production/tests/test_gpu_guard.py -v`
Expected: PASS(4 passed)。

- [ ] **Step 5: 提交**

```bash
git add production/gpu_guard.py production/tests/test_gpu_guard.py
git commit -m "feat(gpu): effective_gpu VRAM pre-check + CPU fallback"
```

---

### Task P0.8:接 GPU helper + 设备日志到 ALSTM/TRA 训练

**Files:**
- Modify: `production/train_alstm.py:139-153`(`train_alstm_multihead`)
- Modify: `production/train_tra.py:192-242`(`train_tra_multihead`)

- [ ] **Step 1: ALSTM —— 用 `effective_gpu` 覆盖 GPU kwarg + 打印设备**

`train_alstm.py:139` 建 `model_kwargs`,`:153` 是 `model = ALSTM(**model_kwargs)`。在两者**之间**(line 139 之后、line 153 之前)插入:

```python
    from production.gpu_guard import effective_gpu
    _req = int(model_kwargs.get("GPU", 0))
    model_kwargs["GPU"] = effective_gpu(_req)   # -1 (CPU) if VRAM too low (e.g. LM Studio)
    print(f"PROGRESS-DEVICE alstm requested_gpu={_req} effective_gpu={model_kwargs['GPU']}", flush=True)
```

(缩进对齐 `train_alstm_multihead` 函数体的 4 空格。)

- [ ] **Step 2: TRA —— 显存不足时强制 CPU + 打印设备**

`pytorch_tra.py:30` 是模块级 `device = "cuda" if torch.cuda.is_available() else "cpu"`。`train_tra.py:192` 是 `from qlib.contrib.model.pytorch_tra import TRAModel`(在 `train_tra_multihead` 内),`:242` 是 `model = TRAModel(**tra_yaml["model"]["kwargs"])`。在 line 192 **之后**插入(让模块级 device 在实例化前被改):

```python
    from production.gpu_guard import effective_gpu
    import qlib.contrib.model.pytorch_tra as _tra_mod
    if effective_gpu(0) < 0:
        _tra_mod.device = "cpu"
    print(f"PROGRESS-DEVICE tra device={_tra_mod.device}", flush=True)
```

(缩进对齐 `train_tra_multihead` 函数体的 4 空格。)

- [ ] **Step 3: 人工冒烟(可选,需数据)**

Run: `python -m production.train_alstm --help`(确认 import 不炸)
Expected: 正常打印 help;`from production.gpu_guard import effective_gpu` 不报错。

> 真实训练时日志应出现 `PROGRESS-DEVICE alstm ... effective_gpu=0`(显存充足)或 `=-1`(被占用回退)。

- [ ] **Step 4: 提交**

```bash
git add production/train_alstm.py production/train_tra.py
git commit -m "feat(gpu): VRAM-guarded device selection + device log in alstm/tra trainers"
```

---

### Task P0.9:实测 17GB 内存构成(测量,为 P3 定调)

**Files:**
- Create: `production/mem_probe.py`(一次性测量脚本)

- [ ] **Step 1: 写采样脚本**

`production/mem_probe.py`:

```python
"""One-off: sample a training subprocess's RSS over time + tag phases, so we
know how much of the ~17GB peak is the Alpha360 data handler vs everything
else. Run alongside a real single-model train. Writes production/cache/mem_probe.csv."""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import psutil

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True, help="training python.exe pid to watch")
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--out", default=str(REPO_ROOT / "production" / "cache" / "mem_probe.csv"))
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    p = psutil.Process(args.pid)
    peak = 0.0
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ts", "rss_gb", "children_rss_gb", "commit_pct"])
        while p.is_running():
            try:
                rss = p.memory_info().rss / 2**30
                ch = sum((c.memory_info().rss for c in p.children(recursive=True)), 0) / 2**30
                vm, sw = psutil.virtual_memory(), psutil.swap_memory()
                commit_pct = 100.0 * (vm.used + sw.used) / (vm.total + sw.total)
                peak = max(peak, rss + ch)
                w.writerow([time.strftime("%H:%M:%S"), f"{rss:.2f}", f"{ch:.2f}", f"{commit_pct:.1f}"])
                f.flush()
                time.sleep(args.interval)
            except psutil.NoSuchProcess:
                break
    print(f"PEAK total RSS = {peak:.2f} GB  (csv: {out})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: 实测一次单模型训练(人工,需数据 + 大约十几分钟)**

1. 起一次单模型训练(后台):`python -m production.rolling_train run-once --only-models alstm --skip-pool --end-date <最近交易日>`
2. 立刻拿到其 pid(`tasklist | findstr python` 或训练日志),另开一个终端:`python -m production.mem_probe --pid <PID>`
3. 训练结束后看 `PEAK total RSS` + `production/cache/mem_probe.csv`。

- [ ] **Step 3: 记录结论**

把峰值 + handler 占比写入 spec 的「关键认知」节(追加一行实测数据),据此决定是否开 P3(handler 内存优化 / API 训练改走 run_split)。

- [ ] **Step 4: 提交**

```bash
git add production/mem_probe.py docs/superpowers/specs/2026-06-25-resource-governance-design.md
git commit -m "chore(mem): one-off RSS probe + record 17GB breakdown finding"
```

---

## 阶段 P1 — 双档预算(止卡顿)

> 给训练/推理子进程注入「保守/放开」两档资源预算。手动触发=保守,调度=放开(P2 用)。

### Task P1.1:`resources.py` —— ResourceProfile + PROFILES

**Files:**
- Create: `backend/app/core/resources.py`
- Test: `backend/app/core/tests/test_resources.py`

- [ ] **Step 1: 写失败测试**

`backend/app/core/tests/test_resources.py`:

```python
from app.core.resources import PROFILES, ResourceProfile


def test_two_profiles_exist():
    assert set(PROFILES) == {"conservative", "aggressive"}
    assert isinstance(PROFILES["conservative"], ResourceProfile)


def test_conservative_is_lighter_than_aggressive():
    c, a = PROFILES["conservative"], PROFILES["aggressive"]
    assert c.blas_threads < a.blas_threads
    assert c.lgbm_threads < a.lgbm_threads
    assert c.below_normal is True and a.below_normal is False
    assert c.affinity_cores is not None       # reserves cores for foreground
    assert a.affinity_cores is None            # all cores
```

- [ ] **Step 2: 运行,确认失败**

Run: `cd backend && python -X utf8 -m pytest app/core/tests/test_resources.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: 实现**

`backend/app/core/resources.py`:

```python
"""Resource budget profiles injected into heavy subprocesses (training,
inference). Two profiles: 'conservative' (daytime/manual — keep the desktop
responsive) and 'aggressive' (nightly/scheduled — go fast). The PARENT
(backend) injects these via subprocess env + creationflags + psutil affinity.
The CHILD needs no special code: OMP_*/MKL_* env vars cap BLAS automatically;
only LightGBM's explicit num_threads is overridden via QLIB_RES_LGBM_THREADS
(read at the lgbm build site in rolling_train)."""
from __future__ import annotations

import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class ResourceProfile:
    name: str
    blas_threads: int        # OMP/MKL/OPENBLAS/NUMEXPR
    lgbm_threads: int        # LightGBM num_threads override
    affinity_cores: int | None   # None = all logical cores; else bind to first N
    below_normal: bool       # lower process priority
    mem_soft_gb: float       # advisory; surfaced to watchdog tuning later


PROFILES: dict[str, ResourceProfile] = {
    "conservative": ResourceProfile(
        name="conservative", blas_threads=4, lgbm_threads=6,
        affinity_cores=12, below_normal=True, mem_soft_gb=8.0,
    ),
    "aggressive": ResourceProfile(
        name="aggressive", blas_threads=8, lgbm_threads=16,
        affinity_cores=None, below_normal=False, mem_soft_gb=12.0,
    ),
}
```

- [ ] **Step 4: 运行,确认通过**

Run: `cd backend && python -X utf8 -m pytest app/core/tests/test_resources.py -v`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add backend/app/core/resources.py backend/app/core/tests/test_resources.py
git commit -m "feat(resources): ResourceProfile + conservative/aggressive profiles"
```

---

### Task P1.2:`popen_env` —— 线程上限 + profile 环境变量

**Files:**
- Modify: `backend/app/core/resources.py`
- Test: `backend/app/core/tests/test_resources.py`

- [ ] **Step 1: 写失败测试**

追加:

```python
from app.core.resources import popen_env


def test_popen_env_caps_blas_and_sets_profile():
    env = popen_env(PROFILES["conservative"])
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        assert env[k] == "4"
    assert env["QLIB_RES_PROFILE"] == "conservative"
    assert env["QLIB_RES_LGBM_THREADS"] == "6"


def test_popen_env_aggressive_values():
    env = popen_env(PROFILES["aggressive"])
    assert env["OMP_NUM_THREADS"] == "8"
    assert env["QLIB_RES_LGBM_THREADS"] == "16"
```

- [ ] **Step 2: 运行,确认失败**

Run: `cd backend && python -X utf8 -m pytest app/core/tests/test_resources.py -k popen_env -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: 实现**

在 `resources.py` 追加:

```python
def popen_env(profile: ResourceProfile) -> dict[str, str]:
    """Env vars to merge into a heavy subprocess's environment. The BLAS libs
    read these at import, so no child code is needed to cap CPU threads."""
    n = str(profile.blas_threads)
    return {
        "OMP_NUM_THREADS": n,
        "MKL_NUM_THREADS": n,
        "OPENBLAS_NUM_THREADS": n,
        "NUMEXPR_NUM_THREADS": n,
        "QLIB_RES_PROFILE": profile.name,
        "QLIB_RES_LGBM_THREADS": str(profile.lgbm_threads),
    }
```

- [ ] **Step 4: 运行,确认通过**

Run: `cd backend && python -X utf8 -m pytest app/core/tests/test_resources.py -k popen_env -v`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add backend/app/core/resources.py backend/app/core/tests/test_resources.py
git commit -m "feat(resources): popen_env (BLAS thread caps + profile env)"
```

---

### Task P1.3:`popen_creationflags` —— Windows 优先级

**Files:**
- Modify: `backend/app/core/resources.py`
- Test: `backend/app/core/tests/test_resources.py`

- [ ] **Step 1: 写失败测试**

追加:

```python
import sys as _sys
from app.core.resources import popen_creationflags


def test_creationflags_below_normal_on_windows():
    flags = popen_creationflags(PROFILES["conservative"])
    if _sys.platform.startswith("win"):
        assert flags == 0x00004000  # BELOW_NORMAL_PRIORITY_CLASS
    else:
        assert flags == 0


def test_creationflags_zero_for_aggressive():
    assert popen_creationflags(PROFILES["aggressive"]) == 0
```

- [ ] **Step 2: 运行,确认失败**

Run: `cd backend && python -X utf8 -m pytest app/core/tests/test_resources.py -k creationflags -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: 实现**

追加:

```python
def popen_creationflags(profile: ResourceProfile) -> int:
    """Windows process-priority creation flag. 0 elsewhere / for aggressive."""
    if profile.below_normal and sys.platform.startswith("win"):
        return 0x00004000  # BELOW_NORMAL_PRIORITY_CLASS
    return 0
```

- [ ] **Step 4: 运行,确认通过**

Run: `cd backend && python -X utf8 -m pytest app/core/tests/test_resources.py -k creationflags -v`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add backend/app/core/resources.py backend/app/core/tests/test_resources.py
git commit -m "feat(resources): popen_creationflags (BELOW_NORMAL on Windows)"
```

---

### Task P1.4:`apply_post_spawn` —— 亲和性 + 优先级(psutil)

**Files:**
- Modify: `backend/app/core/resources.py`
- Test: `backend/app/core/tests/test_resources.py`

- [ ] **Step 1: 写失败测试**(mock psutil)

追加:

```python
from unittest.mock import MagicMock, patch
from app.core.resources import apply_post_spawn


def test_apply_post_spawn_sets_affinity_for_conservative():
    fake_proc = MagicMock()
    with patch("app.core.resources.psutil") as ps:
        ps.Process.return_value = fake_proc
        ps.cpu_count.return_value = 24
        apply_post_spawn(1234, PROFILES["conservative"])
        fake_proc.cpu_affinity.assert_called_once()
        bound = fake_proc.cpu_affinity.call_args[0][0]
        assert len(bound) == 12


def test_apply_post_spawn_aggressive_skips_affinity():
    fake_proc = MagicMock()
    with patch("app.core.resources.psutil") as ps:
        ps.Process.return_value = fake_proc
        ps.cpu_count.return_value = 24
        apply_post_spawn(1234, PROFILES["aggressive"])
        fake_proc.cpu_affinity.assert_not_called()


def test_apply_post_spawn_swallows_errors():
    with patch("app.core.resources.psutil") as ps:
        ps.Process.side_effect = RuntimeError("gone")
        # must not raise
        apply_post_spawn(999999, PROFILES["conservative"])
```

> 注意:实现里需 `import psutil`(模块级),测试才能 patch `app.core.resources.psutil`。

- [ ] **Step 2: 运行,确认失败**

Run: `cd backend && python -X utf8 -m pytest app/core/tests/test_resources.py -k post_spawn -v`
Expected: FAIL — `ImportError` 或 `AttributeError`。

- [ ] **Step 3: 实现**

在 `resources.py` 顶部 import 区加 `import psutil`,然后追加:

```python
def apply_post_spawn(pid: int, profile: ResourceProfile) -> None:
    """After spawning, set CPU affinity + priority on the child. Fail-soft —
    any psutil/permission error degrades to 'thread caps only', never raises."""
    try:
        p = psutil.Process(pid)
        if profile.affinity_cores is not None:
            ncpu = psutil.cpu_count() or 1
            cores = list(range(ncpu))[: profile.affinity_cores]
            if cores:
                p.cpu_affinity(cores)
        if profile.below_normal:
            if sys.platform.startswith("win"):
                p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            else:
                p.nice(10)
    except Exception:
        pass
```

- [ ] **Step 4: 运行,确认通过**

Run: `cd backend && python -X utf8 -m pytest app/core/tests/test_resources.py -v`
Expected: PASS(全部)。

- [ ] **Step 5: 提交**

```bash
git add backend/app/core/resources.py backend/app/core/tests/test_resources.py
git commit -m "feat(resources): apply_post_spawn (affinity + priority, fail-soft)"
```

---

### Task P1.5:接入推理 spawn(手动=保守档)

**Files:**
- Modify: `backend/app/inference/service.py:108-134`

- [ ] **Step 1: 在 import 区加**

```python
import os
from app.core.resources import PROFILES, popen_env, popen_creationflags, apply_post_spawn
```

- [ ] **Step 2: 改 `_run_subprocess` 的 Popen 调用注入预算**

把现有:

```python
        with log_path.open("wb") as logf:
            proc = subprocess.Popen(
                cmd, cwd=str(REPO_ROOT),
                stdout=logf, stderr=subprocess.STDOUT,
            )
```

替换为:

```python
        profile = PROFILES["conservative"]  # manual/UI inference keeps desktop responsive
        with log_path.open("wb") as logf:
            proc = subprocess.Popen(
                cmd, cwd=str(REPO_ROOT),
                stdout=logf, stderr=subprocess.STDOUT,
                env={**os.environ, **popen_env(profile)},
                creationflags=popen_creationflags(profile),
            )
            apply_post_spawn(proc.pid, profile)
```

- [ ] **Step 3: 人工冒烟**

Run(若有数据):从 UI 或 `curl -X POST http://127.0.0.1:8000/api/inference/trigger` 触发一次推理,确认 `logs/inference_*.log` 正常、进程优先级为「低于正常」(任务管理器 → 详细信息 → 基本优先级列)。
Expected: 推理成功;子进程优先级 = 低于正常。

- [ ] **Step 4: 提交**

```bash
git add backend/app/inference/service.py
git commit -m "feat(resources): inject conservative budget into manual inference subprocess"
```

---

### Task P1.6:接入训练 spawn + 按来源选档

**Files:**
- Modify: `backend/app/scheduling/service.py`(`make_subprocess_retrain_job` + `_gated_job_fn`)

- [ ] **Step 1: import 区加**

```python
import os
from app.core.resources import PROFILES, popen_env, popen_creationflags, apply_post_spawn
```

- [ ] **Step 2: `make_subprocess_retrain_job` 的 `_job` 加 `profile` 形参并注入**

把 `_job` 签名改为 `async def _job(job_id: str, log_path: Path, profile_name: str = "conservative") -> None:`,并把 `create_subprocess_exec` 改为:

```python
        prof = PROFILES.get(profile_name, PROFILES["conservative"])
        proc = await asyncio.create_subprocess_exec(
            python_path,
            "-m", "production.rolling_train", "run-once",
            cwd=str(repo_root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env={**os.environ, **popen_env(prof)},
            creationflags=popen_creationflags(prof),
        )
        apply_post_spawn(proc.pid, prof)
```

> `JobCallable` 类型别名相应放宽:`JobCallable = Callable[..., Awaitable[None]]`。

- [ ] **Step 3: `_gated_job_fn` 按 kind 选档并传入**

在 `_gated_job_fn` 里、调用 `self._raw_job_fn(...)` 处,先算 profile:

```python
                    # cron/nightly = aggressive (unattended, go fast);
                    # manual/run-now = conservative (user is at the machine).
                    kind = (entry or {}).get("kind", "manual")
                    profile_name = "aggressive" if kind == "cron" else "conservative"
                    await self._raw_job_fn(_tracked_job_id, self._log_path_for(_tracked_job_id), profile_name)
```

(把原来的 `await self._raw_job_fn(_tracked_job_id, self._log_path_for(_tracked_job_id))` 替换为上面两行/三行。)

- [ ] **Step 4: 跑现有调度测试确认没破坏**

Run: `cd backend && python -X utf8 -m pytest app/scheduling/tests -v`
Expected: PASS(若有测试用旧的 2 参 `_raw_job_fn` mock,放宽 mock 或更新签名)。

- [ ] **Step 5: 提交**

```bash
git add backend/app/scheduling/service.py
git commit -m "feat(resources): inject budget into retrain; cron=aggressive manual=conservative"
```

---

### Task P1.7:LGBM num_threads 读 env

**Files:**
- Modify: `production/rolling_train.py:287`

- [ ] **Step 1: 在 `_lgbm_kwargs = dict(...)` 后覆盖 num_threads**

把 `production/rolling_train.py:287` 的:

```python
    _lgbm_kwargs = dict(lgbm_yaml["model"]["kwargs"])
```

改为:

```python
    _lgbm_kwargs = dict(lgbm_yaml["model"]["kwargs"])
    # Resource-budget override: parent injects QLIB_RES_LGBM_THREADS per profile
    # (conservative=6 / aggressive=16). Falls back to the yaml value when unset
    # (command-line / tests unaffected).
    import os as _os
    _lt = _os.environ.get("QLIB_RES_LGBM_THREADS")
    if _lt:
        try:
            _lgbm_kwargs["num_threads"] = int(_lt)
        except ValueError:
            pass
```

- [ ] **Step 2: 人工验证 env 覆盖生效**

Run:
```bash
python -c "import os; os.environ['QLIB_RES_LGBM_THREADS']='6'; \
import yaml; d=yaml.safe_load(open('production/configs/lgbm_alpha158_multi.yaml')); \
k=dict(d['model']['kwargs']); lt=os.environ.get('QLIB_RES_LGBM_THREADS'); k['num_threads']=int(lt); \
print('num_threads ->', k['num_threads'])"
```
Expected: `num_threads -> 6`(证明覆盖逻辑正确)。

- [ ] **Step 3: 提交**

```bash
git add production/rolling_train.py
git commit -m "feat(resources): LGBM num_threads honors QLIB_RES_LGBM_THREADS"
```

---

## 阶段 P2 — 夜间调度(无人值守)

> 每天凌晨放开档自动跑推理;复用现有 APScheduler + 交易时段护栏 + 并发锁。

### Task P2.1:Settings 加夜间推理开关

**Files:**
- Modify: `backend/app/core/config.py`
- Test: `backend/app/core/tests/test_config_nightly.py`

- [ ] **Step 1: 写失败测试**

`backend/app/core/tests/test_config_nightly.py`:

```python
from app.core.config import Settings


def test_nightly_defaults_off():
    s = Settings()
    assert s.nightly_inference_enabled is False
    assert 0 <= s.nightly_inference_hour <= 23
```

- [ ] **Step 2: 运行,确认失败**

Run: `cd backend && python -X utf8 -m pytest app/core/tests/test_config_nightly.py -v`
Expected: FAIL — `AttributeError: ... has no attribute 'nightly_inference_enabled'`.

- [ ] **Step 3: 实现**

在 `backend/app/core/config.py` 的 `Settings` 里(`ai_analysis_enabled` 行后)追加:

```python
    # Nightly unattended inference (off by default). Runs daily_inference under
    # the 'aggressive' resource profile at the chosen local hour, guarded by
    # the existing trading-hours check + single-run lock.
    nightly_inference_enabled: bool = False
    nightly_inference_hour: int = 2     # 02:00 local
```

- [ ] **Step 4: 运行,确认通过**

Run: `cd backend && python -X utf8 -m pytest app/core/tests/test_config_nightly.py -v`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add backend/app/core/config.py backend/app/core/tests/test_config_nightly.py
git commit -m "feat(scheduling): nightly_inference_enabled/hour settings (off by default)"
```

---

### Task P2.2:trigger_inference 加 profile 形参

**Files:**
- Modify: `backend/app/inference/service.py`

- [ ] **Step 1: 给 `trigger_inference` 和 `_run_subprocess` 加 `profile_name`**

`trigger_inference` 签名加 `profile_name: str = "conservative"`,并把它透传到线程:

```python
def trigger_inference(
    force: bool = False,
    end_date: date | None = None,
    reason: str = "manual",
    profile_name: str = "conservative",
) -> TriggerResponse:
    ...
    thread = threading.Thread(
        target=_run_subprocess,
        args=(job_id, end_date, force, reason, profile_name),
        daemon=True,
    )
```

`_run_subprocess` 签名加 `profile_name: str = "conservative"`,并把 Task P1.5 里写死的那行改为按名取档:

```python
        profile = PROFILES.get(profile_name, PROFILES["conservative"])
```

- [ ] **Step 2: 跑推理测试确认没破坏**

Run: `cd backend && python -X utf8 -m pytest app/inference/tests -v`
Expected: PASS(已有调用用默认 conservative,不受影响)。

- [ ] **Step 3: 提交**

```bash
git add backend/app/inference/service.py
git commit -m "feat(scheduling): trigger_inference accepts profile_name (default conservative)"
```

---

### Task P2.3:装夜间推理 cron(放开档)

**Files:**
- Modify: `backend/app/scheduling/service.py`(`SchedulerManager`)
- Modify: `backend/app/main.py`(把 settings 传给 manager)
- Test: `backend/app/scheduling/tests/test_nightly.py`

- [ ] **Step 1: 写失败测试**

`backend/app/scheduling/tests/test_nightly.py`:

```python
from unittest.mock import MagicMock
from app.scheduling.service import SchedulerManager


def test_install_nightly_inference_adds_cron_job():
    mgr = SchedulerManager(job_fn=MagicMock())
    mgr._scheduler = MagicMock()
    mgr.install_nightly_inference(enabled=True, hour=2)
    # one cron job added with our id
    assert mgr._scheduler.add_job.called
    kwargs = mgr._scheduler.add_job.call_args.kwargs
    assert kwargs.get("id") == "nightly_inference"


def test_install_nightly_inference_disabled_is_noop():
    mgr = SchedulerManager(job_fn=MagicMock())
    mgr._scheduler = MagicMock()
    mgr.install_nightly_inference(enabled=False, hour=2)
    mgr._scheduler.add_job.assert_not_called()
```

- [ ] **Step 2: 运行,确认失败**

Run: `cd backend && python -X utf8 -m pytest app/scheduling/tests/test_nightly.py -v`
Expected: FAIL — `AttributeError: ... has no attribute 'install_nightly_inference'`.

- [ ] **Step 3: 实现 `install_nightly_inference` + job 体**

在 `SchedulerManager` 里加:

```python
    NIGHTLY_JOB_ID = "nightly_inference"

    def install_nightly_inference(self, *, enabled: bool, hour: int) -> None:
        """Install (or skip) a daily cron that runs daily_inference under the
        aggressive profile. Reuses the single-run lock indirectly via the
        inference service's own lock; trading-hours guard applies."""
        try:
            self._scheduler.remove_job(self.NIGHTLY_JOB_ID)
        except JobLookupError:
            pass
        if not enabled:
            return
        self._scheduler.add_job(
            self._run_nightly_inference,
            trigger=CronTrigger(hour=hour, minute=0),
            id=self.NIGHTLY_JOB_ID,
            replace_existing=True,
        )

    async def _run_nightly_inference(self) -> None:
        from app.inference import service as inf
        now = datetime.now(tz=_CST)
        if is_trading_hours_cst(now):
            _log.warning("nightly_inference_skipped_trading_hours")
            return
        _log.info("nightly_inference_trigger")
        inf.trigger_inference(reason="nightly_scheduled", profile_name="aggressive")
```

- [ ] **Step 4: 在 lifespan 调用安装**

`backend/app/main.py` lifespan 里 `await manager.start(session)` 之后加:

```python
    manager.install_nightly_inference(
        enabled=settings.nightly_inference_enabled,
        hour=settings.nightly_inference_hour,
    )
```

- [ ] **Step 5: 运行,确认通过**

Run: `cd backend && python -X utf8 -m pytest app/scheduling/tests/test_nightly.py -v`
Expected: PASS。

- [ ] **Step 6: 提交**

```bash
git add backend/app/scheduling/service.py backend/app/main.py backend/app/scheduling/tests/test_nightly.py
git commit -m "feat(scheduling): nightly daily_inference cron under aggressive profile"
```

---

## 收尾:全量回归

- [ ] **Step 1: 后端全量测试**

Run: `cd backend && python -X utf8 -m pytest -q`
Expected: 全绿(新增测试 + 既有 46 不回归)。

- [ ] **Step 2: production 侧新测试**

Run: `python -m pytest production/tests/test_safety_watchdog.py production/tests/test_gpu_guard.py -v`
Expected: 全绿。

- [ ] **Step 3: 实机冒烟**

启动后端,确认:`watchdog_started` 日志;手动触发推理 = 低优先级子进程;(若装了夜间)`/api/scheduling` 状态正常。

---

## 自审清单(写计划时已核对)
- **Spec 覆盖**:目标 1(双档)→ P1;目标 2(页面文件+watchdog)→ P0.0–P0.6;目标 3(GPU 验证+显存防护)→ P0.7–P0.8;目标 4(夜间调度)→ P2;关键认知(17GB 测量)→ P0.9。✓
- **无占位符**:每个写代码的 Step 都给了完整代码 + 确切文件:行号(含 `train_alstm.py:139-153`、`train_tra.py:192-242` 的精确插入点)。
- **类型/命名一致**:`ResourceProfile` 字段(blas_threads/lgbm_threads/affinity_cores/below_normal/mem_soft_gb)在 P1.1–P1.6、popen_env/creationflags/apply_post_spawn 全程一致;`profile_name` 在 inference/scheduling 一致;`decide_action/is_killable_cmd/record_kill` 命名贯穿 watchdog 各 Task。
- **依赖顺序**:P0 自足;P1 依赖 resources.py(P1.1 先于注入);P2 依赖 P1 的 PROFILES + P2.2 的 profile 形参。
