from production.gpu_guard import effective_gpu


def test_keeps_gpu_when_enough_free_vram():
    probe = lambda: (8 * 2**30, 12 * 2**30)   # (free_bytes, total_bytes)
    assert effective_gpu(0, min_free_gb=4.0, probe=probe) == 0


def test_falls_back_to_cpu_when_low_vram():
    probe = lambda: (2 * 2**30, 12 * 2**30)   # 2GB free < 4GB
    assert effective_gpu(0, min_free_gb=4.0, probe=probe) == -1


def test_falls_back_when_probe_raises():
    def probe():
        raise RuntimeError("no cuda")
    assert effective_gpu(0, min_free_gb=4.0, probe=probe) == -1


def test_already_cpu_stays_cpu():
    assert effective_gpu(-1, min_free_gb=4.0, probe=lambda: (8 * 2**30, 12 * 2**30)) == -1
