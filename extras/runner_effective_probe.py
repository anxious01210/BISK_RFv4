#!/usr/bin/env python3
# extras/runner_effective_probe.py
# Print the effective config that extras/recognize_runner_all_ffmpeg.py would use.
# It reads the same CLI flags AND falls back to the same BISK_* env vars the enforcer exports.

import argparse, os, json

def _env_int(name, default=None):
    v = os.getenv(name)
    if v is None or v == "": return default
    try: return int(v)
    except Exception: return default

def main():
    p = argparse.ArgumentParser("Probe effective config for recognize_runner_all_ffmpeg")

    # --- core flags (mirror the runner) ---
    p.add_argument("--camera", type=int, help="Camera ID")
    p.add_argument("--profile", type=int, help="Profile/StreamProfile ID")
    p.add_argument("--rtsp", default=None, help="RTSP URL (optional)")

    p.add_argument("--rtsp_transport", choices=["auto","tcp","udp"],
                   default=os.getenv("BISK_RTSP_TRANSPORT") or "auto")
    p.add_argument("--hwaccel", choices=["none","nvdec"],
                   default=os.getenv("BISK_HWACCEL") or "none")
    p.add_argument("--device", choices=["auto","cuda","cpu"],
                   default=os.getenv("BISK_DEVICE") or "auto")
    p.add_argument("--gpu_index", type=int,
                   default=_env_int("BISK_PLACE_GPU_INDEX", 0))
    p.add_argument("--threshold", type=float, default=None,
                   help="If None, runner will use RecognitionSettings at runtime")
    p.add_argument("--fps", type=float,
                   default=float(os.getenv("BISK_CAP_MAX_FPS") or 6.0))

    # Detector request (runner may also get this via policy CLI)
    p.add_argument("--det_set", default=os.getenv("BISK_CAP_DET_SET_MAX") or "auto",
                   help="Requested detector size in px (e.g., 640/800/1024/1600) or 'auto'")

    # --- quality/perf knobs we added to the runner ---
    p.add_argument("--model", default=os.getenv("BISK_MODEL") or None)
    p.add_argument("--pipe_q", type=int, default=_env_int("BISK_PIPE_MJPEG_Q", 2))
    p.add_argument("--crop_format", choices=["jpg","png"], default=os.getenv("BISK_CROP_FMT") or "jpg")
    p.add_argument("--crop_quality", type=int, default=_env_int("BISK_CROP_JPEG_Q", 90))
    p.add_argument("--min_face_px", type=int, default=_env_int("BISK_MIN_FACE_PX", 0))
    p.add_argument("--quality_version", type=int, default=_env_int("BISK_QUALITY_VERSION", 0))
    p.add_argument("--save_debug_unmatched", action="store_true",
                   default=(os.getenv("BISK_SAVE_DEBUG_UNMATCHED") == "1"))

    args = p.parse_args()

    # --- derive "effective" values like the runner would ---
    # detector: requested vs cap → effective
    cap_det = _env_int("BISK_CAP_DET_SET_MAX", None)
    try:
        req_det = None if (args.det_set in (None, "", "auto")) else int(args.det_set)
    except Exception:
        req_det = None
    eff_det = (req_det or 1024)
    if cap_det: eff_det = min(eff_det, int(cap_det))

    # fps: requested vs cap → effective
    cap_fps = _env_int("BISK_CAP_MAX_FPS", None)
    eff_fps = float(args.fps or 6.0)
    if cap_fps: eff_fps = min(eff_fps, float(cap_fps))

    providers = {
        "auto": ["CUDAExecutionProvider","CPUExecutionProvider"],
        "cuda": ["CUDAExecutionProvider","CPUExecutionProvider"],
        "cpu":  ["CPUExecutionProvider"],
    }[args.device]

    out = {
        "app": "recognize_runner_all_ffmpeg",
        "camera_id": args.camera,
        "profile_id": args.profile,
        "rtsp_transport": args.rtsp_transport,
        "hwaccel": args.hwaccel,
        "device": args.device,
        "gpu_index": args.gpu_index,
        "providers_order": providers,
        "model": args.model or "buffalo_l",
        "threshold": args.threshold,  # None → runner will use RecognitionSettings
        "detector": {"requested": (req_det or "auto/1024"), "cap": cap_det, "effective": eff_det},
        "fps": {"requested": args.fps, "cap": cap_fps, "effective": eff_fps},
        "pipe": {"mjpeg_q": args.pipe_q},
        "face_gate": {"min_face_px": args.min_face_px},
        "crops": {"format": args.crop_format, "jpeg_quality": args.crop_quality,
                  "save_unmatched": bool(args.save_debug_unmatched)},
        "quality_version": args.quality_version,
        "env_seen": {k: v for k, v in os.environ.items() if k.startswith("BISK_")},
    }
    print(json.dumps(out, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
