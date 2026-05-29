from fastapi import APIRouter

from app.core.qlib_adapter import get_calendar_end, init_qlib_once
from app.ops.schemas import HealthResponse

router = APIRouter()

APP_VERSION = "0.1.0"


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    qlib_ready = False
    cal_end = None
    try:
        init_qlib_once()
        cal_end = str(get_calendar_end())
        qlib_ready = True
    except Exception:
        pass
    return HealthResponse(
        status="ok",
        version=APP_VERSION,
        qlib_ready=qlib_ready,
        calendar_end=cal_end,
    )


@router.get("/memory")
def memory() -> dict:
    """System + per-process memory snapshot.

    Added 2026-05-29 after diagnosing Windows commit-charge exhaustion
    causing hard freezes. Reports:
      - System: RAM total/used, pagefile total/used, commit %
      - Project processes: RSS/VMS for backend / vite / training subprocs
      - Heaviest other python.exe / chrome.exe / claude.exe for context

    Frontend `ActiveJobsBadge` polls this and shows a red banner when
    commit > 85%, so the user can react before another freeze.
    """
    import psutil

    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    commit_used = vm.used + sw.used
    commit_total = vm.total + sw.total
    commit_pct = 100.0 * commit_used / commit_total if commit_total else 0.0

    project_keywords = (
        "uvicorn", "app.main", "vite", "rolling_train", "run_split",
        "daily_inference", "incremental_refresh", "backfill_calibration",
    )
    project: list[dict] = []
    others: list[dict] = []
    me = psutil.Process()
    for p in psutil.process_iter(["pid", "name", "cmdline", "memory_info"]):
        try:
            mi = p.info.get("memory_info")
            if mi is None:
                continue
            rss = mi.rss
            if rss < 50 * 2**20:  # <50MB irrelevant
                continue
            cmd = " ".join(p.info.get("cmdline") or [])
            row = {
                "pid": p.info["pid"],
                "name": p.info["name"],
                "rss_mb": int(rss / 2**20),
                "vms_mb": int(mi.vms / 2**20),
                "is_self": p.info["pid"] == me.pid,
            }
            if any(k in cmd for k in project_keywords):
                row["cmdline"] = cmd[:200]
                project.append(row)
            else:
                others.append(row)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    project.sort(key=lambda r: r["rss_mb"], reverse=True)
    others.sort(key=lambda r: r["rss_mb"], reverse=True)

    # In-process state sizes (catches my-fixes regression)
    sizes: dict[str, int] = {}
    try:
        from app.inference.service import _JOBS as _inf_jobs
        sizes["inference_jobs"] = len(_inf_jobs)
    except Exception:
        pass
    try:
        from app.data.service import _refresh_jobs as _ref_jobs
        sizes["refresh_jobs"] = len(_ref_jobs)
    except Exception:
        pass
    try:
        from app.models.service import _candidates_cached
        sizes["candidates_lru"] = _candidates_cached.cache_info().currsize
    except Exception:
        pass

    return {
        "system": {
            "ram_total_gb": round(vm.total / 2**30, 2),
            "ram_used_gb": round(vm.used / 2**30, 2),
            "ram_available_gb": round(vm.available / 2**30, 2),
            "pagefile_total_gb": round(sw.total / 2**30, 2),
            "pagefile_used_gb": round(sw.used / 2**30, 2),
            "commit_used_gb": round(commit_used / 2**30, 2),
            "commit_total_gb": round(commit_total / 2**30, 2),
            "commit_pct": round(commit_pct, 1),
            "warning": commit_pct > 85.0,
            "critical": commit_pct > 92.0,
        },
        "project_processes": project,
        "other_processes_top10": others[:10],
        "in_memory_state": sizes,
    }
