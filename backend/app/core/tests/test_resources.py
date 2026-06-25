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
        apply_post_spawn(999999, PROFILES["conservative"])  # must not raise
