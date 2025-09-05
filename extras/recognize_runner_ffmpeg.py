#!/usr/bin/env python3
# extras/recognize_runner_ffmpeg.py
# GPU-first (auto) with CPU fallback, FFmpeg snapshots + OpenCV-FFMPEG live read,
# verbose logging, reconnects on read failures, saves matched/unmatched crops,
# and sends heartbeats with detected/matched counters.

import argparse, json, os, signal, subprocess, threading, time, sys, pathlib
from pathlib import Path
from typing import Optional, Dict, List, Tuple

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "bisk.settings")
# Ensure project root is importable when running from extras/
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

SAVE_DEBUG_UNMATCHED = True
HB_COUNTS = {"detected": 0, "matched": 0}


def log(msg: str):
    print(msg, flush=True)


# ---------- math helpers ----------
def l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n <= eps:
        return v.astype(np.float32)
    return (v / n).astype(np.float32)


# ---------- gallery ----------
def load_gallery_from_npy() -> Tuple[np.ndarray, List[Tuple[int, str, str]]]:
    emb_dir = Path(DIRS.get("EMBEDDINGS", Path(settings.MEDIA_ROOT) / "embeddings"))
    paths = sorted(p for p in emb_dir.glob("*.npy") if p.is_file())
    if not paths:
        raise RuntimeError(f"No embeddings found in {emb_dir} (*.npy)")

    vecs: List[np.ndarray] = []
    metas: List[Tuple[int, str, str]] = []
    for p in paths:
        h = p.stem
        try:
            s = Student.objects.get(h_code=h, is_active=True)
        except Student.DoesNotExist:
            continue
        try:
            v = np.load(str(p)).reshape(-1).astype(np.float32)
        except Exception:
            continue
        if v.shape[0] != 512:
            continue
        vecs.append(l2_normalize(v))
        name = getattr(s, "full_name", None)
        if callable(name):
            name = s.full_name()
        elif isinstance(name, str):
            pass
        else:
            name = s.h_code
        metas.append((int(s.id), h, name))
    if not vecs:
        raise RuntimeError("No valid 512-D vectors were loaded.")
    M = np.vstack(vecs).astype(np.float32)
    log(f"[gallery] loaded={M.shape[0]} vectors from {emb_dir}")
    return M, metas


# ---------- InsightFace singleton ----------
_APP: Optional[FaceAnalysis] = None
# _APP_DET = (640, 640)
_APP_DET = (800, 800)
# _APP_DET = (1024, 1024)


def _providers_for(device: str) -> List[str]:
    if device == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif device == "cpu":
        return ["CPUExecutionProvider"]
    else:  # auto
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]


def get_app(det_size: Optional[int], device: str) -> FaceAnalysis:
    """Create once. If CUDA init fails and device=auto, retry with CPU. Re-prepare when det_size changes."""
    global _APP, _APP_DET
    want_det = (det_size or _APP_DET[0], det_size or _APP_DET[1])
    if _APP is None:
        providers = _providers_for(device)
        try:
            _APP = FaceAnalysis(name="buffalo_l", providers=providers)
            ctx = 0 if ("CUDAExecutionProvider" in providers) else -1
            _APP.prepare(ctx_id=ctx, det_size=want_det)
            _APP_DET = want_det
        except Exception as e:
            if device == "auto":
                log(f"[insightface] CUDA init failed: {e}; falling back to CPU")
                _APP = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
                _APP.prepare(ctx_id=-1, det_size=want_det)
                _APP_DET = want_det
            else:
                raise
        # Log active providers
        try:
            provs = []
            for m in getattr(_APP, "models", {}).values():
                sess = getattr(m, "session", None)
                if sess is not None and hasattr(sess, "get_providers"):
                    provs = sess.get_providers();
                    break
            log(f"Applied providers: {provs if provs else _providers_for(device)}")
        except Exception:
            pass
        return _APP

    if want_det != _APP_DET:
        ctx = 0 if device == "cuda" else (-1 if device == "cpu" else 0)
        try:
            _APP.prepare(ctx_id=ctx, det_size=want_det)
            _APP_DET = want_det
        except Exception as e:
            if device == "auto":
                log(f"[insightface] re-prepare on CUDA failed: {e}; CPU retry")
                _APP.prepare(ctx_id=-1, det_size=want_det)
                _APP_DET = want_det
            else:
                raise
    return _APP


# ---------- FFmpeg snapshots ----------
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
    log(f"[ffmpeg] spawning snapshotter → {out_jpg}")
    return subprocess.Popen(
        ffmpeg_cmd(args.ffmpeg, rtsp_url, out_jpg, args),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        close_fds=True,
    )


# ---------- HTTP heartbeat ----------
def post_json(url: str, payload: dict, key: str, timeout: float = 3.0) -> None:
    import urllib.request
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", "X-BISK-KEY": key})
    urllib.request.urlopen(req, timeout=timeout).read()


# ---------- Recognition loop ----------
def _build_cv_capture(rtsp_url: str, transport: str) -> cv2.VideoCapture:
    # Keep the options minimal: some OpenCV builds are strict about the allowed keys.
    opts = "stimeout;5000000"  # 5s
    if transport and transport != "auto":
        opts = f"rtsp_transport;{transport}|{opts}"
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = opts
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    return cap


def recognition_loop(*, args, eff, gal_M: np.ndarray, gal_meta, camera_obj: Camera, rs_cfg: RecognitionSettings):
    cap = _build_cv_capture(args.rtsp, args.rtsp_transport)
    opened = cap.isOpened()
    log(f"[cv] rtsp open = {opened}")
    if not opened:
        raise RuntimeError("OpenCV could not open RTSP stream")

    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    target_fps = (eff.max_fps if eff.max_fps is not None else None) or (args.fps if args.fps else None) or 6.0
    dt = 1.0 / max(0.1, float(target_fps))

    det_size = None
    if eff.det_set_max:
        try:
            det_size = int(eff.det_set_max)
        except Exception:
            det_size = None

    device = args.device
    app = get_app(det_size, device=device)

    if gal_M.ndim != 2 or gal_M.shape[1] != 512:
        raise RuntimeError(f"Gallery matrix must be (N,512), got {gal_M.shape}")
    Gt = gal_M.T

    threshold = float(args.threshold) if args.threshold is not None else float(
        getattr(rs_cfg, "min_score", 0.80) or 0.80)
    debounce_sec = int(getattr(rs_cfg, "re_register_window_sec", 10) or 10)
    last_seen: Dict[int, float] = {}

    ensure_media_tree()
    try:
        attn_root = Path(DIRS["ATTN_CROPS"])
    except Exception:
        attn_root = Path(settings.MEDIA_ROOT) / "attendance_crops"
    attn_root.mkdir(parents=True, exist_ok=True)
    try:
        debug_root = Path(DIRS["LOGS_DEBUG_FACES"])
    except Exception:
        debug_root = Path(settings.MEDIA_ROOT) / "logs" / "debug_faces"
    debug_root.mkdir(parents=True, exist_ok=True)
    dbg_cam_root = debug_root / f"camera_{args.camera}"

    fail_reads = 0
    last_log = time.monotonic()

    while True:
        t0 = time.monotonic()
        ok, frame = cap.read()
        if not ok or frame is None:
            fail_reads += 1
            if time.monotonic() - last_log > 5.0:
                log(f"[cv] read fail count={fail_reads} (will reconnect after 50)")
                last_log = time.monotonic()
            if fail_reads >= 50:
                log("[cv] reconnecting RTSP...")
                try:
                    cap.release()
                except Exception:
                    pass
                time.sleep(0.5)
                cap = _build_cv_capture(args.rtsp, args.rtsp_transport)
                log(f"[cv] rtsp open = {cap.isOpened()}")
                fail_reads = 0
            time.sleep(0.05)
            continue
        fail_reads = 0

        faces = app.get(frame)

        # --- GPU pacing (env-driven) ---
        pacer = getattr(args, "_pacer", None)
        if pacer:
            pacer.maybe_sleep()


        HB_COUNTS["detected"] += len(faces)
        log(f"[faces] detected={len(faces)}")

        for f in faces:
            feat = getattr(f, "normed_embedding", None)
            if feat is None:
                feat = getattr(f, "embedding", None)
            if feat is None:
                log("[skip] face without embedding")
                continue

            v = l2_normalize(np.asarray(feat, dtype=np.float32))
            if v.shape[0] != 512:
                log(f"[skip] bad embedding shape={v.shape}")
                continue

            sims = (v @ Gt).astype(float)
            j = int(np.argmax(sims))
            score = float(sims[j])
            sid, hcode, _name = gal_meta[j]

            order = np.argsort(sims)[-3:][::-1]
            log("Top3: " + ", ".join([f"{gal_meta[k][1]}={float(sims[k]):.3f}" for k in order]))

            x1, y1, x2, y2 = map(int, f.bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1] - 1, x2), min(frame.shape[0] - 1, y2)
            crop = frame[y1:y2, x1:x2].copy()

            if score < threshold:
                if SAVE_DEBUG_UNMATCHED:
                    today = timezone.localdate()
                    dbg_dir = dbg_cam_root / f"{today:%Y/%m/%d}"
                    dbg_dir.mkdir(parents=True, exist_ok=True)
                    dbg_path = dbg_dir / f"unmatched_{int(time.time() * 1000)}.jpg"
                    try:
                        cv2.imwrite(str(dbg_path), crop)
                        log(f"[UNMATCHED] score={score:.3f} saved={dbg_path}")
                    except Exception as e:
                        log(f"[UNMATCHED] score={score:.3f} save-failed: {e}")
                continue

            now_m = time.monotonic()
            if now_m - last_seen.get(sid, 0.0) < debounce_sec:
                log(f"[debounce] {hcode} score={score:.3f}")
                continue
            last_seen[sid] = now_m

            today = timezone.localdate()
            crop_dir = attn_root / f"{today:%Y/%m/%d}"
            crop_dir.mkdir(parents=True, exist_ok=True)
            crop_path = str(crop_dir / f"{hcode}_{int(time.time())}.jpg")
            try:
                ok2, enc = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                if ok2:
                    enc.tofile(crop_path)
                    log(f"[MATCH] {hcode} score={score:.3f} -> crop={crop_path}")
                else:
                    crop_path = ""
                    log(f"[MATCH] {hcode} score={score:.3f} -> crop-encode-failed")
            except Exception as e:
                crop_path = ""
                log(f"[MATCH] {hcode} score={score:.3f} -> crop-save-error: {e}")

            HB_COUNTS["matched"] += 1

            try:
                student = Student.objects.get(pk=sid)
                record_recognition(student=student, score=score, camera=camera_obj, crop_path=(crop_path or None))
                log(f"[LOGGED] {hcode} score={score:.3f}")
            except Exception as e:
                log(f"[LOG ERROR] {hcode} score={score:.3f}: {e}")

        elapsed = time.monotonic() - t0
        sleep_for = dt - elapsed

        pacer = getattr(args, "_pacer", None)
        if pacer:
            pacer.maybe_sleep()

        if sleep_for > 0:
            time.sleep(sleep_for)


# ---------- main ----------
def main():
    p = argparse.ArgumentParser(description="FFmpeg snapshots + HB + LIVE recognition (GPU-first fallback)")
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
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--gpu_index", type=int, default=0)
    p.add_argument("--threshold", type=float, default=None)  # default pulls from RecognitionSettings
    p.add_argument("--fps", type=float, default=6.0)

    args, _ = p.parse_known_args()

    from apps.scheduler.resources_cpu import apply_cpu_quota_percent, approximate_quota_with_affinity
    CPU_QUOTA = int(os.getenv("BISK_CPU_QUOTA_PERCENT", "0") or "0")
    if CPU_QUOTA > 0:
        ok = apply_cpu_quota_percent(CPU_QUOTA)
        if not ok:
            approximate_quota_with_affinity(CPU_QUOTA)

    from apps.scheduler.resources_gpu import GpuPacer
    GPU_TARGET = int(os.getenv("BISK_GPU_TARGET_UTIL", "0") or "0")
    GPU_WIN_MS = int(os.getenv("BISK_GPU_UTIL_WINDOW_MS", "1500") or "1500")
    GPU_INDEX = int(os.getenv("BISK_PLACE_GPU_INDEX", str(getattr(args, "gpu_index", 0))))
    args._pacer = GpuPacer(GPU_INDEX, GPU_TARGET, GPU_WIN_MS) if GPU_TARGET > 0 else None

    # Resolve resource policy BEFORE GPU frameworks init
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

    # Heartbeat loop
    running = True

    def _stop(*_):
        nonlocal running;
        running = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    interval = max(1.0, float(args.hb_interval))
    next_tick = time.monotonic()
    camera_fps_val = 0.0

    log("[runner] started; sending heartbeats...")
    while running:
        payload = {
            "camera_id": cam_id,
            "profile_id": prof_id,
            "fps": float(camera_fps_val),
            "detected": int(HB_COUNTS["detected"]),
            "matched": int(HB_COUNTS["matched"]),
            "latency_ms": 0.0,
            "last_error": last_error or "",
        }
        try:
            post_json(args.hb, payload, args.hb_key, timeout=3.0)
            log(f"[hb] {payload}")
        except Exception as e:
            log(f"[hb] send failed: {e}")

        # wait next tick or stop early
        while running and time.monotonic() < next_tick + interval:
            time.sleep(0.2)
        next_tick += interval

    # shutdown
    log("[runner] stopping...")
    try:
        ff_proc.terminate()
        try:
            ff_proc.wait(timeout=2.0)
        except Exception:
            ff_proc.kill()
    except Exception:
        pass


if __name__ == "__main__":
    main()
