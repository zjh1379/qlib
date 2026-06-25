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

import psutil


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


def popen_creationflags(profile: ResourceProfile) -> int:
    """Windows process-priority creation flag. 0 elsewhere / for aggressive."""
    if profile.below_normal and sys.platform.startswith("win"):
        return 0x00004000  # BELOW_NORMAL_PRIORITY_CLASS
    return 0


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
