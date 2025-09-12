# !/usr/bin/env python3
# Audit variant of the FFmpeg runner.
#
# It parses the SAME args as extras/recognize_runner_all_ffmpeg.py,
# reuses the enforcer's policy/resources merge, and prints:
#
# - cli_args          : what was passed on the command line
# - env_seen          : BISK_* vars seen by this process
# - effective         : resolved values (caps applied, final det/fps, quality knobs, placement)
# - derived           : convenience summaries (providers order, gpu placement, etc.)
#
# It does NOT run ffmpeg or recognition.

import os, sys, json, argparse
from pathlib import Path

# Make Django imports work
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "bisk.settings")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import django

django.setup()

# Reuse the SAME logic the enforcer uses to compute effective values
from apps.scheduler.services.enforcer import (
    _choose_policy,
    _choose_resources,
    _clamp_and_finalize,
)
from apps.cameras.models import Camera
from apps.scheduler.models import StreamProfile


def _env_int(name, default=None):
    v = os.getenv(name)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except Exception:
        return default


def _providers_for(device):
    return {
        "auto": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "cpu": ["CPUExecutionProvider"],
    }.get(device or "auto", ["CUDAExecutionProvider", "CPUExecutionProvider"])


def build_parser():
    p = argparse.ArgumentParser("recognize_runner_all_ffmpeg_audit")
    # core runner flags (keep in sync with the real runner)
    p.add_argument("--camera", required=True, type=int)
    p.add_argument("--profile", required=True, type=int)
    p.add_argument("--rtsp", required=False, default=None)
    p.add_argument("--hb", required=False, default=None)
    p.add_argument("--hb_key", required=False, default=None)
    p.add_argument("--snapshots", required=False, default=None)
    p.add_argument("--ffmpeg", default="/usr/bin/ffmpeg")
    p.add_argument("--ffprobe", default="/usr/bin/ffprobe")
    p.add_argument("--hb_interval", type=int, default=_env_int("BISK_HB_INTERVAL", 10))
    p.add_argument("--snapshot_every", type=int, default=_env_int("BISK_SNAPSHOT_EVERY", 3))
    p.add_argument("--rtsp_transport", choices=["auto", "tcp", "udp"],
                   default=os.getenv("BISK_RTSP_TRANSPORT") or "auto")
    p.add_argument("--hwaccel", choices=["none", "nvdec"],
                   default=os.getenv("BISK_PLACE_HWACCEL") or os.getenv("BISK_HWACCEL") or "none")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"],
                   default=os.getenv("BISK_PLACE_DEVICE") or os.getenv("BISK_DEVICE") or "auto")
    p.add_argument("--gpu_index", type=int, default=_env_int("BISK_PLACE_GPU_INDEX", 0))
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--fps", type=float, default=float(os.getenv("BISK_CAP_MAX_FPS") or 6.0))
    p.add_argument("--det_set", default=os.getenv("BISK_CAP_DET_SET_MAX") or "auto")  # requested detector size

    # quality/perf knobs
    p.add_argument("--model", default=(os.getenv("BISK_MODEL", "").strip() or None))
    p.add_argument("--pipe_q", type=int, default=_env_int("BISK_PIPE_MJPEG_Q", 2))
    p.add_argument("--crop_format", choices=["jpg", "png"], default=(os.getenv("BISK_CROP_FMT") or "jpg"))
    p.add_argument("--crop_quality", type=int, default=_env_int("BISK_CROP_JPEG_Q", 90))
    p.add_argument("--min_face_px", type=int, default=_env_int("BISK_MIN_FACE_PX", 0))
    p.add_argument("--quality_version", type=int, default=_env_int("BISK_QUALITY_VERSION", 0))
    p.add_argument("--save_debug_unmatched", action="store_true",
                   default=(os.getenv("BISK_SAVE_DEBUG_UNMATCHED") == "1"))
    return p


def main():
    parser = build_parser()
    args, _ = parser.parse_known_args()

    cam = Camera.objects.get(pk=args.camera)
    prof = StreamProfile.objects.get(pk=args.profile)

    # Use the *same* logic the enforcer uses to compute what the runner gets
    policy_req = _choose_policy(cam, prof)
    res_eff = _choose_resources(cam)
    det_final, fps_final = _clamp_and_finalize(policy_req, res_eff)

    # Build the env snapshot exactly like enforcer does (subset relevant here)
    env_snapshot = {
        "BISK_PLACE_DEVICE": str(res_eff.get("device") or ""),
        "BISK_PLACE_GPU_INDEX": str(res_eff.get("gpu_index") or ""),
        "BISK_PLACE_HWACCEL": str(res_eff.get("hwaccel") or ""),
        "BISK_CAP_MAX_FPS": str(res_eff.get("max_fps_cap") or ""),
        "BISK_CAP_DET_SET_MAX": str(res_eff.get("det_set_max") or ""),
        "BISK_MODEL": str(res_eff.get("model_tag") or (args.model or "")),
        "BISK_PIPE_MJPEG_Q": str(res_eff.get("pipe_mjpeg_q") or args.pipe_q or ""),
        "BISK_CROP_FMT": str(res_eff.get("crop_format") or args.crop_format or ""),
        "BISK_CROP_JPEG_Q": str(res_eff.get("crop_jpeg_quality") or args.crop_quality or ""),
        "BISK_MIN_FACE_PX": str(
            (res_eff.get("min_face_px") if res_eff.get("min_face_px") not in (None, "", 0) else res_eff.get(
                "min_face_px_default"))
            or (args.min_face_px or "")
        ),
        "BISK_QUALITY_VERSION": str(res_eff.get("quality_version") or args.quality_version or ""),
        "BISK_SAVE_DEBUG_UNMATCHED": (
            "1" if res_eff.get("save_debug_unmatched") is True
            else ("0" if res_eff.get("save_debug_unmatched") is False else ("1" if args.save_debug_unmatched else ""))
        ),
        # convenience echoes for admin
        "_det_set": det_final,
        "_fps": str(fps_final if fps_final is not None else ""),
    }

    # Effective view grouped
    effective = {
        "caps": {
            "detector": {"requested": policy_req.get("det_req"), "cap": res_eff.get("det_set_max"),
                         "effective": det_final},
            "fps": {"requested": policy_req.get("fps_req"), "cap": res_eff.get("max_fps_cap"), "effective": fps_final},
        },
        "placement": {
            "device": res_eff.get("device") or "auto",
            "gpu_index": res_eff.get("gpu_index"),
            "hwaccel": res_eff.get("hwaccel") or "none",
            "providers_order": _providers_for(res_eff.get("device") or "auto"),
        },
        "model": res_eff.get("model_tag") or (args.model or "buffalo_l"),
        "pipe": {"mjpeg_q": int(env_snapshot["BISK_PIPE_MJPEG_Q"] or 2)},
        "crops": {
            "format": env_snapshot["BISK_CROP_FMT"] or "jpg",
            "jpeg_quality": int(env_snapshot["BISK_CROP_JPEG_Q"] or 90),
            "save_unmatched": env_snapshot["BISK_SAVE_DEBUG_UNMATCHED"] == "1",
        },
        "face_gate": {"min_face_px": int(env_snapshot["BISK_MIN_FACE_PX"] or 0)},
        "quality_version": int(env_snapshot["BISK_QUALITY_VERSION"] or 0),
        "threshold": args.threshold,
    }

    out = {
        "app": "recognize_runner_all_ffmpeg_audit",
        "cli_args": vars(args),
        "derived": {"hb_interval": policy_req.get("hb_interval"), "snapshot_every": policy_req.get("snapshot_every")},
        "effective": effective,
        "effective_os_resources": {
            "cpu_affinity": res_eff.get("cpu_affinity"),
            "cpu_nice": res_eff.get("cpu_nice"),
            "cpu_quota_percent": res_eff.get("cpu_quota_percent"),
            "gpu_memory_fraction": res_eff.get("gpu_memory_fraction"),
            "gpu_target_util_percent": res_eff.get("gpu_target_util_percent"),
            "gpu_visible_devices": str(res_eff.get("gpu_index") or ""),
            "max_fps_cap": res_eff.get("max_fps_cap"),
            "det_set_max_cap": res_eff.get("det_set_max"),
        },
        "env_seen": {k: v for k, v in os.environ.items() if k.startswith("BISK_")},
        "env_would_export": env_snapshot,
        "camera_id": cam.id,
        "profile_id": prof.id,
    }

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
