#!/usr/bin/env python3
import os, sys, time, json, io
import psutil
from contextlib import suppress

# Ensure Django is importable
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "bisk.settings")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import django

django.setup()

from apps.scheduler.models import RunningProcess

NVML_OK = False
try:
    import pynvml

    pynvml.nvmlInit()
    NVML_OK = True
except Exception:
    NVML_OK = False


def _gpu_pids_util_map():
    """Return {pid: util%} (best-effort, per-process)"""
    mp = {}
    if not NVML_OK:
        return mp
    try:
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            with suppress(Exception):
                procs = pynvml.nvmlDeviceGetGraphicsRunningProcesses_v3(h)
                for p in procs:
                    # p.smUtil is only in newer NVMLs; fall back to None
                    util = getattr(p, "smUtil", None)
                    mp[int(p.pid)] = util if util is not None else None
    except Exception:
        pass
    return mp


def _has_libcuda(proc: psutil.Process) -> bool:
    """Heuristic: does the process map libcuda (likely using CUDA)?"""
    with suppress(Exception):
        for m in proc.memory_maps():
            p = (m.path or "").lower()
            if "libcuda" in p or "nvidia/libcuda" in p:
                return True
    return False


def _cmdline(proc: psutil.Process):
    with suppress(Exception):
        return " ".join(proc.cmdline())
    return ""


def audit_runner(proc: psutil.Process):
    info = {
        "pid": proc.pid,
        "name": proc.name(),
        "cmd": _cmdline(proc),
        "cpu_samples": [],
        "ffmpeg_cpu_samples": [],
        "ffmpeg_hwaccel": None,
        "has_libcuda": False,
        "gpu_util_ppid": None,
        "gpu_util_children": {},
    }

    # find ffmpeg child (if any)
    ffchild = None
    with suppress(Exception):
        for c in proc.children(recursive=True):
            if (c.name() or "").lower().startswith("ffmpeg"):
                ffchild = c
                break

    # sample CPU%
    for _ in range(5):  # 5 seconds total
        with suppress(Exception):
            info["cpu_samples"].append(proc.cpu_percent(interval=1.0))
        if ffchild:
            with suppress(Exception):
                info["ffmpeg_cpu_samples"].append(ffchild.cpu_percent(interval=None))
        else:
            info["ffmpeg_cpu_samples"].append(None)

    # check ffmpeg hwaccel from cmdline
    if ffchild:
        cmd = _cmdline(ffchild)
        if " -hwaccel cuda" in f" {cmd} ":
            info["ffmpeg_hwaccel"] = "cuda"
        elif " -hwaccel " in f" {cmd} ":
            # some other hwaccel (or none)
            with suppress(Exception):
                ix = cmd.index("-hwaccel")
                frag = cmd[ix:].split(" ", 2)[:2]
                info["ffmpeg_hwaccel"] = " ".join(frag)
        else:
            info["ffmpeg_hwaccel"] = "none"

    # libcuda presence (heuristic for CUDA EP loaded)
    info["has_libcuda"] = _has_libcuda(proc)

    # GPU per-process util
    pmap = _gpu_pids_util_map()
    info["gpu_util_ppid"] = pmap.get(proc.pid)
    if ffchild:
        info["gpu_util_children"][ffchild.pid] = pmap.get(ffchild.pid)

    # summarize
    avg_cpu = round(sum([x or 0 for x in info["cpu_samples"]]) / max(1, len(info["cpu_samples"])), 1)
    avg_ff = round(sum([x or 0 for x in info["ffmpeg_cpu_samples"] if x is not None]) / max(1, len([x for x in info[
        "ffmpeg_cpu_samples"] if x is not None]) or [1]), 1)

    # derive verdicts
    recog_on_gpu = bool(info["has_libcuda"] or (info["gpu_util_ppid"] and info["gpu_util_ppid"] > 0))
    decode_on_gpu = (info["ffmpeg_hwaccel"] == "cuda") or any(
        [(v or 0) > 0 for v in info["gpu_util_children"].values()])

    # consider “excessive CPU” if > 150% on multi-core (tune as you like)
    excessive_runner_cpu = avg_cpu > 150.0
    excessive_ffmpeg_cpu = avg_ff > 150.0

    print("=" * 80)
    print(f"Runner PID {info['pid']}  name={info['name']}")
    print(f"CMD: {info['cmd'][:200]}{'...' if len(info['cmd']) > 200 else ''}")
    print(f"CPU avg: runner={avg_cpu}%  ffmpeg={avg_ff}%")
    print(f"FFmpeg hwaccel: {info['ffmpeg_hwaccel']}")
    print(f"CUDA loaded (libcuda mapped): {info['has_libcuda']}")
    print(f"GPU util (runner pid): {info['gpu_util_ppid']}")
    print(f"GPU util (children): {json.dumps(info['gpu_util_children'])}")
    print(f"Recognition on GPU? {'YES' if recog_on_gpu else 'NO'}")
    print(f"Decode on GPU (NVDEC)? {'YES' if decode_on_gpu else 'NO'}")
    if excessive_runner_cpu or excessive_ffmpeg_cpu:
        print(f"CPU usage excessive? runner={excessive_runner_cpu} ffmpeg={excessive_ffmpeg_cpu}")
    print("=" * 80)
    return info


def _candidate_pids_from_db(limit=50):
    try:
        qs = RunningProcess.objects.order_by("-id")[:limit]
        return [int(r.pid) for r in qs if getattr(r, "pid", None)]
    except Exception:
        return []

def _candidate_pids_from_ps():
    # Fallback: scan processes for our runner names
    want = {"recognize_runner_ffmpeg.py", "recognize_runner_all_ffmpeg.py", "recognize_ffmpeg_live.py"}
    pids = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmd = " ".join(p.info.get("cmdline") or [])
            if any(w in cmd for w in want):
                pids.append(p.info["pid"])
        except Exception:
            continue
    return pids

def main():
    # 1) try DB, 2) fall back to process scan, 3) de-dupe and validate
    pids = _candidate_pids_from_db() or _candidate_pids_from_ps()
    pids = sorted(set(int(x) for x in pids if x))

    # keep only alive processes
    pids = [pid for pid in pids if psutil.pid_exists(pid)]
    if not pids:
        print("No candidate runner processes found (DB or ps).")
        return

    print(f"Auditing PIDs: {pids}")
    for pid in pids:
        try:
            p = psutil.Process(pid)
        except Exception:
            continue
        audit_runner(p)



if __name__ == "__main__":
    main()
