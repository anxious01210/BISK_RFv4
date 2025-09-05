# apps/scheduler/resources_cpu.py
import os, math, psutil


def _is_cgv2():
    try:
        with open("/proc/self/mountinfo", "r") as f:
            return any(" - cgroup2 " in line for line in f)
    except Exception:
        return False


def apply_cpu_quota_percent(percent: int) -> bool:
    """Try to set cgroup v2 cpu.max for current process group; return True on success."""
    p = max(1, min(100, int(percent)))
    if not _is_cgv2():
        return False
    cpu_max_path = "/sys/fs/cgroup/cpu.max"
    try:
        period = 100000  # 100ms
        n_cores = psutil.cpu_count() or 1
        quota_us = max(1000, int(period * n_cores * (p / 100.0)))
        with open(cpu_max_path, "w") as f:
            f.write(f"{quota_us} {period}\n")
        return True
    except Exception:
        return False


def approximate_quota_with_affinity(percent: int):
    """Fallback: shrink allowed cores proportional to percent."""
    p = max(1, min(100, int(percent)))
    proc = psutil.Process()
    all_cores = list(range(psutil.cpu_count() or 1))
    want = max(1, math.floor(len(all_cores) * (p / 100.0)))
    try:
        proc.cpu_affinity(all_cores[:want])
    except Exception:
        pass
