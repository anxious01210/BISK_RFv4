#!/usr/bin/env python3
# extras/recognize_runner_all_ffmpeg.py
# All-FFmpeg RTSP reader (no OpenCV capture). Uses FFmpeg for both snapshots and live frame piping.
# Decodes JPEG frames with Pillow; runs InsightFace; logs heartbeats; saves crops (Pillow).

import argparse, json, os, signal, subprocess, threading, time, sys, pathlib, random
import contextlib
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Generator

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
from apps.scheduler.models import StreamProfile
import numpy as np
from insightface.app import FaceAnalysis
from PIL import Image
import io

SAVE_DEBUG_UNMATCHED = True
HB_COUNTS = {"detected": 0, "matched": 0}


def map_ffmpeg_err(line: str) -> Optional[str]:
    L = line.lower()
    if "401" in L and ("unauthorized" in L or "authorization" in L): return "401 unauthorized"
    if "describe" in L and "404" in L: return "rtsp describe 404"
    if "connection refused" in L: return "connection refused"
    if "timed out" in L or "timeout" in L: return "connection timeout"
    if "unrecognized option" in L or "option not found" in L: return "ffmpeg option not found"
    if "unsupported transport" in L or "461" in L: return "transport mismatch"
    if "invalid data found" in L: return "invalid input data"
    if "server returned 5" in L or "5xx" in L: return "server 5xx error"
    return None


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
# _APP_DET = (800, 800)
# _APP_DET = (864, 864)
_APP_DET = (1024, 1024)


# _APP_DET = (1600, 1600)
# _APP_DET = (2048, 2048)


def _providers_for(device: str) -> List[str]:
    if device == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif device == "cpu":
        return ["CPUExecutionProvider"]
    else:  # auto
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]


def get_app(det_size: Optional[int], device: str) -> FaceAnalysis:
    # Create once. If CUDA init fails and device=auto, retry with CPU. Re-prepare when det_size changes.
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


# ---------- FFmpeg helpers ----------
def ffmpeg_snapshot_cmd(ffmpeg_bin: str, rtsp_url: str, out_jpg: str, args) -> list:
    cmd = [ffmpeg_bin, "-nostdin", "-hide_banner", "-loglevel", "warning"]
    if getattr(args, "rtsp_transport", None) and args.rtsp_transport != "auto":
        cmd += ["-rtsp_transport", args.rtsp_transport]
    if getattr(args, "hwaccel", None) == "nvdec":
        cmd += ["-hwaccel", "cuda"]
    cmd += ["-stimeout", "5000000"]
    cmd += ["-i", rtsp_url]
    snap_every = max(1, int(getattr(args, "snapshot_every", 3)))
    cmd += ["-vf", f"fps=1/{snap_every}", "-q:v", "2", "-f", "image2", "-update", "1", "-y", out_jpg]
    return cmd


def spawn_snapshotter(args, rtsp_url: str, out_jpg: str) -> subprocess.Popen:
    log(f"[ffmpeg] spawning snapshotter → {out_jpg}")
    return subprocess.Popen(
        ffmpeg_snapshot_cmd(args.ffmpeg, rtsp_url, out_jpg, args),
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, close_fds=True,
    )


# --- drop-in replacement ---
def ffmpeg_pipe_cmd(ffmpeg_bin: str, rtsp_url: str, fps: float,
                    rtsp_transport: str = None, hwaccel: str = "none"):
    """
    Build the FFmpeg command that transcodes H264->MJPEG to stdout (image2pipe).
    Adds a sane RTSP socket timeout to avoid hanging forever on SETUP/PLAY.
    """
    cmd = [ffmpeg_bin, "-nostdin", "-hide_banner", "-loglevel", "warning"]

    if rtsp_transport:
        cmd += ["-rtsp_transport", rtsp_transport]

    # 5s socket timeout (microseconds). Prevents indefinite waits in libavformat.
    cmd += ["-stimeout", "5000000"]

    if hwaccel.lower() in ("nvdec", "cuda"):
        # decoding with CUDA; still producing MJPEG on CPU unless you later add -hwaccel_output_format
        cmd += ["-hwaccel", "cuda"]

    cmd += ["-i", rtsp_url]

    # decimate to target fps early to reduce compute
    cmd += ["-vf", f"fps={float(fps):.3f}", "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1"]
    return cmd


# --- drop-in replacement ---
def iter_mjpeg_frames(rtsp_url: str,
                      ffmpeg: str,
                      ffprobe: str,
                      fps: float,
                      rtsp_transport: str = None,
                      hwaccel: str = "none",
                      stall_timeout: float = 8.0,
                      reconnect_backoff: float = 2.0):
    """
    Yield RGB frames produced by FFmpeg (H264->MJPEG->RGB) WITHOUT blocking forever.
    - Non-blocking read using select.select()
    - Stderr reader thread maps common RTSP/FFmpeg errors
    - Restarts FFmpeg on stall or EOF with a small backoff
    """
    import select, os, threading

    def spawn():
        proc = subprocess.Popen(
            ffmpeg_pipe_cmd(ffmpeg, rtsp_url, fps=fps,
                            rtsp_transport=rtsp_transport, hwaccel=hwaccel),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0
        )

        last_err_msg = {"msg": ""}

        def _stderr_reader(p):
            if not p.stderr:
                return
            for raw in iter(p.stderr.readline, b""):
                try:
                    line = raw.decode("utf-8", "ignore").strip()
                except Exception:
                    continue
                mapped = map_ffmpeg_err(line)
                if mapped:
                    last_err_msg["msg"] = mapped

        threading.Thread(target=_stderr_reader, args=(proc,), daemon=True).start()
        return proc, last_err_msg

    # Outer loop: keep the pipe alive and respawn on stalls
    while True:
        proc, last_err = spawn()
        stdout = proc.stdout
        buf = bytearray()
        last_rx = time.monotonic()

        # non-blocking readiness + watchdog
        try:
            while True:
                # if process died, stop and respawn
                if proc.poll() is not None:
                    break

                r, _, _ = select.select([stdout], [], [], 0.5)
                if r:
                    # read what's currently available without blocking
                    chunk = stdout.read1(65536) if hasattr(stdout, "read1") else stdout.read(65536)
                    if not chunk:
                        # EOF: let outer loop respawn
                        break
                    buf.extend(chunk)
                    last_rx = time.monotonic()

                    # parse JPEG frames out of the buffer
                    while True:
                        j_start = buf.find(b"\xff\xd8")
                        if j_start < 0:
                            # keep tail only
                            if len(buf) > 2:
                                del buf[:-2]
                            break
                        j_end = buf.find(b"\xff\xd9", j_start + 2)
                        if j_end < 0:
                            # wait for the rest of the frame
                            if j_start > 0:
                                del buf[:j_start]
                            break
                        jpeg = bytes(buf[j_start:j_end + 2])
                        del buf[:j_end + 2]
                        try:
                            frame = Image.open(io.BytesIO(jpeg)).convert("RGB")
                            yield np.array(frame)
                        except Exception:
                            # corrupt frame — skip
                            continue
                else:
                    # nothing ready — check for stall
                    if time.monotonic() - last_rx > stall_timeout:
                        # surface the stall as a last_error for the heartbeat thread (if any)
                        HB_COUNTS["last_error"] = last_err["msg"] or "no frames (stall)"
                        break
        finally:
            with contextlib.suppress(Exception):
                proc.kill()
            with contextlib.suppress(Exception):
                proc.wait(timeout=1.5)

        time.sleep(reconnect_backoff)


# ---------- HTTP heartbeat ----------
def post_json(url: str, payload: dict, key: str, timeout: float = 3.0) -> None:
    import urllib.request
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", "X-BISK-KEY": key})
    urllib.request.urlopen(req, timeout=timeout).read()


# ---------- Recognition loop ----------
def recognition_loop(*, args, eff, gal_M: np.ndarray, gal_meta, camera_obj: Camera, rs_cfg: RecognitionSettings,
                     hb_stats: dict):
    # Target fps preference
    target_fps = (eff.max_fps if eff.max_fps is not None else None) or (args.fps if args.fps else None) or 6.0
    det_size = None
    # NEW: fallback to StreamProfile.detection_set if not provided by eff
    if det_size is None:
        try:
            prof = StreamProfile.objects.get(pk=int(args.profile))
            if getattr(prof, "detection_set", None) and str(prof.detection_set).isdigit():
                det_size = int(prof.detection_set)
        except Exception:
            pass

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

    # FFmpeg frame iterator (all-ffmpeg; no cv2.VideoCapture)
    frame_iter = iter_mjpeg_frames(
        ffmpeg=args.ffmpeg,
        rtsp_url=args.rtsp,
        transport=args.rtsp_transport,
        fps=target_fps,
        hwaccel=args.hwaccel,
    )

    for frame_rgb in frame_iter:
        hb_stats["frames"] += 1  # count frames for actual FPS
        # frame_rgb is RGB HxWx3 uint8
        faces = app.get(frame_rgb)  # InsightFace expects RGB by default
        HB_COUNTS["detected"] += len(faces)
        # log(f"[faces] detected={len(faces)}")

        for f in faces:
            feat = getattr(f, "normed_embedding", None)
            if feat is None:
                feat = getattr(f, "embedding", None)
            if feat is None:
                # log("[skip] face without embedding")
                continue

            v = l2_normalize(np.asarray(feat, dtype=np.float32))
            if v.shape[0] != 512:
                # log(f"[skip] bad embedding shape={v.shape}")
                continue

            sims = (v @ Gt).astype(float)
            j = int(np.argmax(sims))
            score = float(sims[j])
            sid, hcode, _name = gal_meta[j]

            # crop using bbox
            x1, y1, x2, y2 = map(int, f.bbox)
            H, W = frame_rgb.shape[0], frame_rgb.shape[1]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W - 1, x2), min(H - 1, y2)
            crop_rgb = frame_rgb[y1:y2, x1:x2].copy()

            if score < threshold:
                if SAVE_DEBUG_UNMATCHED:
                    today = timezone.localdate()
                    dbg_dir = dbg_cam_root / f"{today:%Y/%m/%d}"
                    dbg_dir.mkdir(parents=True, exist_ok=True)
                    dbg_path = dbg_dir / f"unmatched_{int(time.time() * 1000)}.jpg"
                    try:
                        Image.fromarray(crop_rgb).save(str(dbg_path), format="JPEG", quality=90)
                        # log(f"[UNMATCHED] score={score:.3f} saved={dbg_path}")
                    except Exception as e:
                        # log(f"[UNMATCHED] score={score:.3f} save-failed: {e}")
                        pass
                continue

            now_m = time.monotonic()
            if now_m - last_seen.get(sid, 0.0) < debounce_sec:
                # log(f"[debounce] {hcode} score={score:.3f}")
                continue
            last_seen[sid] = now_m

            # Save crop
            today = timezone.localdate()
            crop_dir = attn_root / f"{today:%Y/%m/%d}"
            crop_dir.mkdir(parents=True, exist_ok=True)
            crop_path = str(crop_dir / f"{hcode}_{int(time.time())}.jpg")
            try:
                Image.fromarray(crop_rgb).save(crop_path, format="JPEG", quality=90)
                # log(f"[MATCH] {hcode} score={score:.3f} -> crop={crop_path}")
            except Exception as e:
                crop_path = ""
                # log(f"[MATCH] {hcode} score={score:.3f} -> crop-save-error: {e}")

            HB_COUNTS["matched"] += 1

            try:
                student = Student.objects.get(pk=sid)
                record_recognition(student=student, score=score, camera=camera_obj, crop_path=(crop_path or None))
                # log(f"[LOGGED] {hcode} score={score:.3f}")
            except Exception as e:
                # log(f"[LOG ERROR] {hcode} score={score:.3f}: {e}")
                pass


# ---------- main ----------
def main():
    p = argparse.ArgumentParser(description="All-FFmpeg runner: snapshots + HB + LIVE recognition (GPU-first fallback)")
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
    ff_proc = spawn_snapshotter(args, args.rtsp, str(snap_path))

    # Shared state for heartbeats
    hb_stats = {"frames": 0}
    hb_shared = {"last_error": ""}

    # Load gallery
    gal_M, gal_meta = load_gallery_from_npy()

    # Fetch camera object and settings for recognition
    try:
        camera_obj = Camera.objects.get(pk=cam_id)
    except Camera.DoesNotExist:
        camera_obj = None
    rs_cfg = RecognitionSettings.get_solo()

    def _stderr_reader(proc):
        if not proc.stderr: return
        for raw in iter(proc.stderr.readline, b""):
            try:
                line = raw.decode("utf-8", "ignore").strip()
            except Exception:
                continue
            mapped = map_ffmpeg_err(line)
            if mapped: hb_shared["last_error"] = mapped

    threading.Thread(target=_stderr_reader, args=(ff_proc,), daemon=True).start()

    # Compute target_fps preference (for reporting)
    target_fps_pref = (eff.max_fps if eff.max_fps is not None else None) or (args.fps if args.fps else None) or 6.0
    snapshot_every = int(args.snapshot_every or 3)

    # Heartbeat loop (periodic, with small jitter)
    hb_stop = threading.Event()

    def _hb_loop():
        # small initial delay so Admin flips to Online quickly
        time.sleep(1.0)
        prev_frames = 0
        prev_t = time.monotonic()
        while not hb_stop.is_set():
            now = time.monotonic()
            frames = hb_stats.get("frames", 0)
            df = max(0, frames - prev_frames)
            dt = max(1e-3, now - prev_t)
            fps_actual = float(df) / dt
            prev_frames = frames
            prev_t = now
            payload = {
                "camera_id": cam_id,
                "profile_id": prof_id,
                "fps": round(fps_actual, 3),
                "detected": int(HB_COUNTS["detected"]),
                "matched": int(HB_COUNTS["matched"]),
                "latency_ms": 0.0,
                "last_error": hb_shared.get("last_error", ""),
                # helpful meta (server may ignore)
                "target_fps": float(target_fps_pref),
                "snapshot_every": int(snapshot_every),
            }
            try:
                post_json(args.hb, payload, args.hb_key, timeout=3.0)
                # log(f"[hb] {payload}")
            except Exception as e:
                # log(f"[hb] send failed: {e}")
                pass
            # jitter ±10%
            interval = max(1.0, float(args.hb_interval or 10))
            jitter = 0.9 + (random.random() * 0.2)
            hb_stop.wait(interval * jitter)

    hb_thread = threading.Thread(target=_hb_loop, daemon=True)
    hb_thread.start()

    # Start recognition loop (same process)
    try:
        recognition_loop(args=args, eff=eff, gal_M=gal_M, gal_meta=gal_meta, camera_obj=camera_obj, rs_cfg=rs_cfg,
                         hb_stats=hb_stats)
    finally:
        log("[runner] stopping...")
        hb_stop.set()
        try:
            hb_thread.join(timeout=2.0)
        except Exception:
            pass
        try:
            ff_proc.terminate()
            try:
                ff_proc.wait(timeout=2.0)
            except Exception:
                ff_proc.kill()
        except Exception:
            pass

    # Send one final heartbeat on exit
    # Use the last measured fps window if available
    payload = {
        "camera_id": cam_id,
        "profile_id": prof_id,
        "fps": float(0.0),  # final fps not meaningful; periodic HBs already sent
        "detected": int(HB_COUNTS["detected"]),
        "matched": int(HB_COUNTS["matched"]),
        "latency_ms": 0.0,
        "last_error": hb_shared.get("last_error", ""),
    }
    try:
        post_json(args.hb, payload, args.hb_key, timeout=3.0)
        log(f"[hb-final] {payload}")
    except Exception as e:
        log(f"[hb-final] send failed: {e}")


if __name__ == "__main__":
    main()
