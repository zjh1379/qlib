"""Tests for app.core.resources — ResourceProfile, popen_env,
popen_creationflags, apply_post_spawn. TDD: P1.1 → P1.4."""
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
