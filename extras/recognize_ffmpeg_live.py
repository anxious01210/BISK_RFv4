#!/usr/bin/env python3
"""
extras/recognize_ffmpeg_live.py
FFmpeg snapshots + heartbeats  AND  a live recognition thread.

- Opens RTSP with OpenCV (CAP_FFMPEG), runs InsightFace.FaceAnalysis (GPU-first),
  loads gallery from media/embeddings/*.npy (stem=H-CODE), computes cosine similarity,
  then calls apps.attendance.services.record_recognition(student, score, camera, ts, crop_path).
- Debounce per-student using RecognitionSettings.re_register_window_sec to avoid event spam.
- Crops are saved to media/attendance_crops/YYYY/MM/DD/<hcode>_<ts>.jpg (DIRS['ATTN_CROPS']).

Matches your models: AttendanceEvent/AttendanceRecord and PeriodOccurrence(start_dt/end_dt).
"""

import argparse, json, os, signal, subprocess, threading, time
from pathlib import Path
from typing import Optional, Dict, List, Tuple

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "bisk.settings")
# If you sometimes run the script outside the project root, also add this (optional but robust):
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import django

django.setup()

# --- Project imports (after django.setup) ---
from django.utils import timezone
from django.conf import settings

from apps.scheduler.resources import resolve_effective
from apps.cameras.models import Camera
from apps.attendance.models import Student, RecognitionSettings
from apps.attendance.services import record_recognition
from apps.attendance.utils.media_paths import DIRS, ensure_media_tree

import cv2
import numpy as np
from insightface.app import FaceAnalysis

# Debug knobs (put near other globals)
SAVE_DEBUG_UNMATCHED = True  # save crops even when below threshold (for diagnostics)
HB_COUNTS = {"detected": 0, "matched": 0}  # optional: used in heartbeat payload


# ------------------------------
# Helpers
# ------------------------------
def l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n <= eps:
        return v.astype(np.float32)
    return (v / n).astype(np.float32)


def load_gallery_from_npy() -> Tuple[np.ndarray, List[Tuple[int, str, str]]]:
    """Return (M, meta) where M=(N,512) float32 L2-normalized; meta=[(student_id,h_code,full_name)]"""
    emb_dir = Path(DIRS.get("EMBEDDINGS", Path(settings.MEDIA_ROOT) / "embeddings"))
    paths = sorted(p for p in emb_dir.glob("*.npy") if p.is_file())
    if not paths:
        raise RuntimeError(f"No embeddings found in {emb_dir} (*.npy)")

    vecs: List[np.ndarray] = []
    metas: List[Tuple[int, str, str]] = []
    for p in paths:
        h = p.stem  # H230002, ...
        try:
            s = Student.objects.get(h_code=h, is_active=True)
        except Student.DoesNotExist:
            continue
        v = np.load(str(p)).reshape(-1).astype(np.float32)
        if v.shape[0] != 512:
            continue
        vecs.append(l2_normalize(v))
        metas.append((int(s.id), h, s.full_name))
    if not vecs:
        raise RuntimeError("No valid 512-D vectors were loaded.")
    return np.vstack(vecs).astype(np.float32), metas  # M, meta


# InsightFace singleton
_APP: Optional[FaceAnalysis] = None
_APP_DET = (640, 640)


def get_app(det_size: Optional[int], device: str) -> FaceAnalysis:
    """GPU-first with CPU fallback; re-prepare on det_size changes."""
    global _APP, _APP_DET
    if _APP is None:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        _APP = FaceAnalysis(name="buffalo_l", providers=providers)
        _APP.prepare(ctx_id=(0 if device == "cuda" else -1), det_size=(det_size or 640, det_size or 640))
        _APP_DET = (det_size or 640, det_size or 640)
        return _APP
    # Re-prepare if detector size changed
    want = (det_size or _APP_DET[0], det_size or _APP_DET[1])
    if want != _APP_DET:
        _APP.prepare(ctx_id=(0 if device == "cuda" else -1), det_size=want)
        _APP_DET = want
    return _APP


# ------------------------------
# FFmpeg snapshot process (reuse your existing pattern)
# ------------------------------
def ffmpeg_cmd(ffmpeg_bin, rtsp_url, out_jpg, args):
    cmd = [ffmpeg_bin, "-nostdin", "-hide_banner", "-loglevel", "warning"]
    if getattr(args, "rtsp_transport", None) and args.rtsp_transport != "auto":
        cmd += ["-rtsp_transport", args.rtsp_transport]
    if getattr(args, "hwaccel", None) == "nvdec":
        cmd += ["-hwaccel", "cuda"]
    cmd += ["-i", rtsp_url]
    snap_every = max(1, int(getattr(args, "snapshot_every", 3)))
    cmd += ["-vf", f"fps=1/{snap_every}", "-q:v", "2", "-f", "image2", "-update", "1", "-y", out_jpg]
    return cmd


def spawn_ffmpeg(args, rtsp_url, out_jpg):
    return subprocess.Popen(
        ffmpeg_cmd(args.ffmpeg, rtsp_url, out_jpg, args),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,  # read for error mapping
        close_fds=True,
    )


# ------------------------------
# HTTP heartbeat (same payload fields you already use)
# ------------------------------
def post_json(url: str, payload: dict, key: str, timeout: float = 3.0) -> None:
    import urllib.request
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", "X-BISK-KEY": key})
    urllib.request.urlopen(req, timeout=timeout).read()


# ------------------------------
# Recognition loop
# ------------------------------
def recognition_loop(*, args, eff, gal_M: np.ndarray, gal_meta, camera_obj: Camera, rs_cfg: RecognitionSettings):
    # Fresher frames: use TCP, short timeout, tiny buffers (OpenCV FFmpeg)
    opts = "rtsp_transport;tcp|stimeout;5000000|reorder_queue_size;0|buffer_size;1024000"
    if args.rtsp_transport and args.rtsp_transport != "auto":
        # keep user choice but also apply other opts
        opts = f"rtsp_transport;{args.rtsp_transport}|stimeout;5000000|reorder_queue_size;0|buffer_size;1024000"
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = opts

    cap = cv2.VideoCapture(args.rtsp, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError("OpenCV could not open RTSP stream")

    try:
        # Some OpenCV builds ignore this; harmless to try
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    # Target FPS: camera/profile → CLI → default
    target_fps = (eff.max_fps if eff.max_fps is not None else None) or (args.fps if args.fps else None) or 6.0
    dt = 1.0 / max(0.1, float(target_fps))

    # Detector size from EffectiveResources if provided
    det_size = None
    if eff.det_set_max:
        try:
            det_size = int(eff.det_set_max)
        except Exception:
            det_size = None

    device = args.device
    app = get_app(det_size, device=device)

    # Precompute transposed gallery for dot products
    if gal_M.ndim != 2 or gal_M.shape[1] != 512:
        raise RuntimeError(f"Gallery matrix must be (N,512), got {gal_M.shape}")
    Gt = gal_M.T  # (512, N)

    # Runner thresholds / debounce
    threshold = float(args.threshold) if args.threshold is not None else float(
        getattr(rs_cfg, "min_score", 0.80) or 0.80)
    debounce_sec = int(getattr(rs_cfg, "re_register_window_sec", 10) or 10)
    last_seen: Dict[int, float] = {}  # student_id -> monotonic ts

    # Resolve media dirs (with safe fallbacks)
    ensure_media_tree()
    # ATTENDANCE CROPS dir
    try:
        attn_root = Path(DIRS["ATTN_CROPS"])
    except Exception:
        attn_root = Path(settings.MEDIA_ROOT) / "attendance_crops"
    attn_root.mkdir(parents=True, exist_ok=True)
    # DEBUG FACES dir
    try:
        debug_root = Path(DIRS["DEBUG_FACES"])
    except Exception:
        debug_root = Path(settings.MEDIA_ROOT) / "logs" / "debug_faces"
    debug_root.mkdir(parents=True, exist_ok=True)
    dbg_cam_root = debug_root / f"camera_{args.camera}"

    while True:
        t0 = time.monotonic()
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.2)
            continue

        faces = app.get(frame)
        HB_COUNTS["detected"] += len(faces)
        print(f"[faces] detected={len(faces)}", flush=True)

        for f in faces:
            # Extract embedding
            feat = getattr(f, "normed_embedding", None)
            if feat is None:
                feat = getattr(f, "embedding", None)
            if feat is None:
                print("[skip] face without embedding", flush=True)
                continue

            v = l2_normalize(np.asarray(feat, dtype=np.float32).reshape(-1))
            if v.shape[0] != 512:
                print(f"[skip] bad embedding shape={v.shape}", flush=True)
                continue

            # Similarity vs gallery
            sims = (v @ Gt).astype(float)  # (N,)
            j = int(np.argmax(sims))
            score = float(sims[j])
            sid, hcode, _name = gal_meta[j]

            # Print top-5 to understand score distribution
            order = np.argsort(sims)[-5:][::-1]
            print("Top5:", [(gal_meta[k][1], float(sims[k])) for k in order], flush=True)

            # Crop region now (we may save unmatched crops for debugging)
            x1, y1, x2, y2 = map(int, f.bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1] - 1, x2), min(frame.shape[0] - 1, y2)
            crop = frame[y1:y2, x1:x2].copy()

            # Below threshold → optionally save unmatched crop, then continue
            if score < threshold:
                if SAVE_DEBUG_UNMATCHED:
                    today = timezone.localdate()
                    dbg_dir = dbg_cam_root / f"{today:%Y/%m/%d}"
                    dbg_dir.mkdir(parents=True, exist_ok=True)
                    dbg_path = dbg_dir / f"unmatched_{int(time.time() * 1000)}.jpg"
                    try:
                        cv2.imwrite(str(dbg_path), crop)
                        print(f"[UNMATCHED] score={score:.3f} saved={dbg_path}", flush=True)
                    except Exception as e:
                        print(f"[UNMATCHED] score={score:.3f} save-failed: {e}", flush=True)
                continue

            # Debounce per-student to avoid spamming
            now_m = time.monotonic()
            if now_m - last_seen.get(sid, 0.0) < debounce_sec:
                print(f"[debounce] {hcode} score={score:.3f}", flush=True)
                continue
            last_seen[sid] = now_m

            # Save matched crop to attendance_crops/YYYY/MM/DD/
            today = timezone.localdate()
            crop_dir = attn_root / f"{today:%Y/%m/%d}"
            crop_dir.mkdir(parents=True, exist_ok=True)
            crop_path = str(crop_dir / f"{hcode}_{int(time.time())}.jpg")
            try:
                ok2, enc = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                if ok2:
                    enc.tofile(crop_path)
                    print(f"[MATCH] {hcode} score={score:.3f} -> crop={crop_path}", flush=True)
                else:
                    crop_path = ""
                    print(f"[MATCH] {hcode} score={score:.3f} -> crop-encode-failed", flush=True)
            except Exception as e:
                crop_path = ""
                print(f"[MATCH] {hcode} score={score:.3f} -> crop-save-error: {e}", flush=True)

            HB_COUNTS["matched"] += 1

            # ORM logging (creates/updates attendance record if period active)
            try:
                student = Student.objects.get(pk=sid)
                record_recognition(student=student, score=score, camera=camera_obj, crop_path=(crop_path or None))
                print(f"[LOGGED] {hcode} score={score:.3f}", flush=True)
            except Exception as e:
                print(f"[LOG ERROR] {hcode} score={score:.3f}: {e}", flush=True)

        # Throttle loop
        elapsed = time.monotonic() - t0
        sleep_for = dt - elapsed
        if sleep_for > 0:
            time.sleep(sleep_for)


# ------------------------------
# Main
# ------------------------------
def main():
    p = argparse.ArgumentParser(description="FFmpeg snapshots + heartbeats + LIVE recognition")
    p.add_argument("--camera", required=True)
    p.add_argument("--profile", required=True)
    p.add_argument("--rtsp", required=True)
    p.add_argument("--hb", required=True)
    p.add_argument("--hb_key", required=True)
    p.add_argument("--snapshots", required=True)
    p.add_argument("--ffmpeg", default="/usr/bin/ffmpeg")
    p.add_argument("--ffprobe", default="/usr/bin/ffprobe")
    p.add_argument("--hb_interval", type=int, default=10)
    p.add_argument("--snapshot_every", type=int, default=3)
    p.add_argument("--rtsp_transport", choices=["auto", "tcp", "udp"], default="auto")
    p.add_argument("--hwaccel", choices=["none", "nvdec"], default="none")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--gpu_index", type=int, default=0)
    p.add_argument("--threshold", type=float, default=None)  # default pulls from RecognitionSettings

    args, _ = p.parse_known_args()

    # Resolve resource caps and niceness/affinity BEFORE GPU frameworks init
    eff = resolve_effective(camera_id=args.camera)
    if eff.gpu_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = eff.gpu_visible_devices
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        if eff.cpu_nice is not None: proc.nice(eff.cpu_nice)
        if eff.cpu_affinity: proc.cpu_affinity(eff.cpu_affinity)
    except Exception:
        pass

    # Prepare snapshot path + spawn ffmpeg
    cam_id = int(args.camera)
    prof_id = int(args.profile)
    snap_dir = Path(args.snapshots);
    snap_dir.mkdir(parents=True, exist_ok=True)
    snap_path = snap_dir / f"{cam_id}.jpg"
    ff_proc = spawn_ffmpeg(args, args.rtsp, str(snap_path))

    # Load gallery
    gal_M, gal_meta = load_gallery_from_npy()

    # Fetch camera object and settings for recognition
    try:
        camera_obj = Camera.objects.get(pk=cam_id)
    except Camera.DoesNotExist:
        camera_obj = None
    rs_cfg = RecognitionSettings.get_solo()

    # Start recognition thread
    t_rec = threading.Thread(
        target=recognition_loop,
        args=(),
        kwargs=dict(args=args, eff=eff, gal_M=gal_M, gal_meta=gal_meta, camera_obj=camera_obj, rs_cfg=rs_cfg),
        daemon=True,
    )
    t_rec.start()

    # ffmpeg stderr reader to map last_error
    last_error = ""

    def _map_err(line: str) -> Optional[str]:
        L = line.lower()
        if "401" in L and ("unauthorized" in L or "authorization" in L): return "401 unauthorized"
        if "describe" in L and "404" in L: return "rtsp describe 404"
        if "connection refused" in L: return "connection refused"
        if "timed out" in L or "timeout" in L: return "connection timeout"
        if "unrecognized option" in L or "option not found" in L: return "ffmpeg option not found"
        if "unsupported transport" in L or "461" in L: return "transport mismatch"
        if "invalid data found" in L: return "invalid input data"
        if "server returned 5" in L: return "server 5xx error"
        return None

    def _stderr_reader(proc):
        nonlocal last_error
        if not proc.stderr: return
        for raw in iter(proc.stderr.readline, b""):
            try:
                line = raw.decode("utf-8", "ignore").strip()
            except Exception:
                continue
            mapped = _map_err(line)
            if mapped: last_error = mapped

    threading.Thread(target=_stderr_reader, args=(ff_proc,), daemon=True).start()

    # Heartbeat loop (compatible with your current /api/runner/heartbeat/ payload)
    running = True

    def _stop(*_):
        nonlocal running;
        running = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    interval = max(1.0, float(args.hb_interval))
    next_tick = time.monotonic()
    # We can’t measure processed_fps trivially here; reuse ffprobe if desired later.
    camera_fps_val = 0.0

    while running:
        payload = {
            "camera_id": cam_id,
            "profile_id": prof_id,
            "fps": float(camera_fps_val),  # keep legacy field
            "detected": 0,
            "matched": 0,
            # "detected": int(HB_COUNTS["detected"]),
            # "matched": int(HB_COUNTS["matched"]),
            "latency_ms": 0.0,
            "pid": os.getpid(),
            "last_error": (last_error or "")[:200],
        }
        try:
            post_json(args.hb, payload, args.hb_key, timeout=3.0)
        except Exception:
            pass

        next_tick += interval
        sleep_for = next_tick - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            next_tick = time.monotonic()

    try:
        if ff_proc and ff_proc.poll() is None:
            ff_proc.terminate()
            try:
                ff_proc.wait(timeout=3)
            except Exception:
                ff_proc.kill()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
