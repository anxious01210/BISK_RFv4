#!/usr/bin/env python3
# extras/recognize_runner_all_ffmpeg.py
# All-FFmpeg RTSP reader (no OpenCV capture). Uses FFmpeg for both snapshots and live frame piping.
# Decodes JPEG frames with Pillow; runs InsightFace; logs heartbeats; saves crops (Pillow).

import argparse, json, os, signal, subprocess, threading, time, sys, pathlib
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
from apps.attendance.models import Student, RecognitionSettings, Student, FaceEmbedding
from apps.attendance.services import record_recognition
from apps.attendance.utils.media_paths import DIRS, ensure_media_tree
from apps.attendance.utils.media_paths import student_capture_dir  # new
from apps.attendance.utils.capture_curate import curate_top_crops

import numpy as np
from insightface.app import FaceAnalysis
from PIL import Image
import io


SAVE_DEBUG_UNMATCHED = True
HB_COUNTS = {"detected": 0, "matched": 0}
HB_EXTRAS = {}  # values to ship with every heartbeat (e.g., min_face_px)


# ---------- config helpers ----------
def _effective_min_face_px(args, rs_cfg):
    """
    Resolve the final min_face_px once:
    CLI (--min_face_px) > CameraResourceOverride > GlobalResourceSettings > RecognitionSettings > fallback 40
    """
    # CLI flag takes top precedence
    if getattr(args, "min_face_px", None):
        return int(args.min_face_px)
    # Enforcer already exports BISK_MIN_FACE_PX into env
    env_v = os.getenv("BISK_MIN_FACE_PX")
    if env_v not in (None, "", "None"):
        try:
            return int(env_v)
        except Exception:
            pass
    # DB settings fallback
    return int(getattr(rs_cfg, "min_face_px", 40) or 40)


class FPSMeter:
    """Exponential moving average FPS meter using inter-arrival time."""

    def __init__(self, alpha: float = 0.2):
        self.alpha = float(alpha)
        self._ema = 0.0
        self._last = None

    def tick(self):
        now = time.monotonic()
        if self._last is not None:
            dt = now - self._last
            if dt > 0:
                inst = 1.0 / dt
                self._ema = (1 - self.alpha) * self._ema + self.alpha * inst if self._ema > 0 else inst
        self._last = now

    def value(self) -> float:
        return float(self._ema)


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
        model_name = (getattr(globals().get("args", object()), "model", None)  # argparse value if visible
                      or os.getenv("BISK_MODEL", "").strip()
                      or "buffalo_l")
        try:
            _APP = FaceAnalysis(name=model_name, providers=providers)
            ctx = 0 if ("CUDAExecutionProvider" in providers) else -1
            _APP.prepare(ctx_id=ctx, det_size=want_det)
            global HB_EXTRAS
            HB_EXTRAS["model_pack"] = model_name
            log(f"[insightface] model={model_name}")
            _APP_DET = want_det
        except Exception as e:
            if device == "auto":
                log(f"[insightface] init failed on {model_name} ({e}); CPU retry")
                try:
                    _APP = FaceAnalysis(name=model_name, providers=["CPUExecutionProvider"])
                    _APP.prepare(ctx_id=-1, det_size=want_det)
                    _APP_DET = want_det
                except Exception as e2:
                    if model_name != "buffalo_l":
                        log(f"[insightface] retrying with buffalo_l on CPU due to: {e2}")
                        _APP = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
                        _APP.prepare(ctx_id=-1, det_size=want_det)
                        _APP_DET = want_det
                    else:
                        raise
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
        cmd += ["-rtsp_transport", args.rtsp_transport] + ["-rtsp_flags", "prefer_tcp"]
    if getattr(args, "hwaccel", None) == "nvdec":
        cmd += ["-hwaccel", "cuda"]
    cmd += ["-i", rtsp_url]
    snap_every = max(1, int(getattr(args, "snapshot_every", 3)))
    q = getattr(args, "pipe_q", None)
    cmd += ["-vf", f"fps=1/{snap_every}", "-q:v", str(q if q else 2),
            "-f", "image2", "-update", "1", "-y", out_jpg]

    return cmd


def spawn_snapshotter(args, rtsp_url: str, out_jpg: str) -> subprocess.Popen:
    log(f"[ffmpeg] spawning snapshotter → {out_jpg}")
    return subprocess.Popen(
        ffmpeg_snapshot_cmd(args.ffmpeg, rtsp_url, out_jpg, args),
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, close_fds=True,
    )


def ffmpeg_pipe_cmd(ffmpeg_bin: str, rtsp_url: str, *, transport: str,
                    fps: Optional[float], hwaccel: str) -> list:
    cmd = [ffmpeg_bin, "-nostdin", "-hide_banner", "-loglevel", "warning"]
    if transport and transport != "auto":
        cmd += ["-rtsp_transport", transport]
    if hwaccel == "nvdec":
        cmd += ["-hwaccel", "cuda"]
    cmd += ["-i", rtsp_url]
    # Only downsample if an fps was explicitly provided
    if fps and float(fps) > 0:
        cmd += ["-vf", f"fps={max(0.1, float(fps))}"]
    # cmd += ["-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1"]
    # High-quality JPEGs for the pipe to avoid blockiness/blur (tunable via --pipe_q)
    q = getattr(globals().get("args", object()), "pipe_q", None)
    cmd += ["-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", str(q if q else 2), "pipe:1"]

    return cmd


def iter_mjpeg_frames(*, ffmpeg: str, rtsp_url: str, transport: str, fps: Optional[float], hwaccel: str):
    args = ffmpeg_pipe_cmd(ffmpeg, rtsp_url, transport=transport, fps=fps, hwaccel=hwaccel)

    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    buf = bytearray()
    SOI, EOI = b"\xff\xd8", b"\xff\xd9"

    def _kill():
        try:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except Exception:
                proc.kill()
        except Exception:
            pass

    try:
        while True:
            chunk = proc.stdout.read(4096)
            if not chunk:
                # If FFmpeg ended, stop iteration
                break
            buf += chunk
            while True:
                i = buf.find(SOI)
                if i < 0:
                    # keep last up to 1MB to avoid unbounded growth
                    if len(buf) > 1024 * 1024:
                        del buf[:-1024]
                    break
                j = buf.find(EOI, i + 2)
                if j < 0:
                    # need more data
                    if i > 0:
                        del buf[:i]  # discard garbage before SOI
                    break
                # complete JPEG
                j_end = j + 2
                frame_bytes = bytes(buf[i:j_end])
                del buf[:j_end]
                try:
                    im = Image.open(io.BytesIO(frame_bytes))
                    im = im.convert("RGB")
                    yield np.array(im, dtype=np.uint8)  # HxWx3 RGB
                except Exception as e:
                    # corrupted frame; continue
                    continue
    finally:
        _kill()


# # ---------- HTTP heartbeat ----------
# def post_json(url: str, payload: dict, key: str, timeout: float = 3.0) -> None:
#     import urllib.request
#     data = json.dumps(payload).encode("utf-8")
#     req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", "X-BISK-KEY": key})
#     urllib.request.urlopen(req, timeout=timeout).read()

def post_json(url: str, payload: dict, key: str, timeout: float = 3.0) -> None:
    import urllib.request
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json", "X-BISK-KEY": key}
    )
    # LOG before sending
    print(f"[HB->] {url} cam={payload.get('camera_id') or payload.get('camera')} "
          f"pid={payload.get('pid')} fps={payload.get('camera_fps') or payload.get('fps')} "
          f"snap={payload.get('snapshot_every')}", flush=True)
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        body = resp.read()
        # LOG after success
        print(f"[HB<-] {resp.getcode()} bytes={len(body)}", flush=True)
    except Exception as e:
        # LOG on failure
        print(f"[HB!!] send failed: {e}", flush=True)
        raise


def hb_worker(stop_evt, args, cam_id: int, prof_id: int, get_metrics=None):
    """Posts a heartbeat every args.hb_interval seconds.
    get_metrics() may return (camera_fps, processed_fps), else we send zeros."""
    while not stop_evt.is_set():
        try:
            cam_fps, proc_fps = (0.0, 0.0)
            if callable(get_metrics):
                try:
                    cam_fps, proc_fps = get_metrics()
                except Exception:
                    pass
            payload = {
                "camera_id": cam_id,
                "profile_id": prof_id,
                "pid": os.getpid(),
                "camera_fps": float(cam_fps),
                "processed_fps": float(proc_fps),
                "target_fps": float(getattr(args, "_computed_target_fps", 0) or 0) or None,
                "snapshot_every": int(getattr(args, "snapshot_every", 10) or 10),
                "detected": int(HB_COUNTS.get("detected", 0)),
                "matched": int(HB_COUNTS.get("matched", 0)),
                "latency_ms": 0.0,
            }
            payload.update(HB_EXTRAS)
            post_json(args.hb, payload, args.hb_key, timeout=3.0)
        except Exception:
            # post_json already logs; keep loop alive
            pass
        # Sleep, but wake early if stopping
        stop_evt.wait(float(getattr(args, "hb_interval", 10) or 10))


# ---------- Recognition loop ----------
def recognition_loop(*, args, eff, gal_M: np.ndarray, gal_meta, camera_obj: Camera, rs_cfg: RecognitionSettings,
                     cam_meter: FPSMeter, proc_meter: FPSMeter):
    # Target fps preference
    target_fps = (eff.max_fps if eff.max_fps is not None else None) or (args.fps if args.fps else None) or 6.0
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

    MIN_FACE_PX = _effective_min_face_px(args, rs_cfg)
    log(f"[cfg] init: min_score={threshold:.2f} debounce={debounce_sec}s min_face_px={MIN_FACE_PX}")
    HB_EXTRAS["min_face_px"] = int(MIN_FACE_PX)

    # Track settings changes for hot-reload
    rs_changed_at = getattr(rs_cfg, "changed_at", None)

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
        fps=None,  # <— no downsample; measure true ingest
        # fps=target_fps,           # <— this was downsampling the pipe
        hwaccel=args.hwaccel,
    )

    for frame_rgb in frame_iter:
        # --- Hot-reload RecognitionSettings every ~2s ---
        if not hasattr(recognition_loop, "_last_rs_check"):
            recognition_loop._last_rs_check = 0.0
        now_mono = time.monotonic()
        if now_mono - recognition_loop._last_rs_check > 2.0:
            recognition_loop._last_rs_check = now_mono
            try:
                # lightweight probe: only fetch the change marker
                new_changed = (RecognitionSettings.objects
                               .only("changed_at")
                               .get(pk=rs_cfg.pk).changed_at)
                if new_changed != rs_changed_at:
                    # re-load full settings row
                    rs_cfg = RecognitionSettings.get_solo()
                    rs_changed_at = getattr(rs_cfg, "changed_at", new_changed)
                    # re-compute derived values (respect CLI override for --threshold)
                    if args.threshold is None:
                        threshold = float(getattr(rs_cfg, "min_score", 0.80) or 0.80)
                    debounce_sec = int(getattr(rs_cfg, "re_register_window_sec", 10) or 10)
                    MIN_FACE_PX = _effective_min_face_px(args, rs_cfg)
                    log(f"[cfg] reloaded: min_score={threshold:.2f}  debounce={debounce_sec}s  min_face_px={MIN_FACE_PX}")
                    HB_EXTRAS["min_face_px"] = int(MIN_FACE_PX)
            except Exception:
                pass
        # --- end hot-reload ---

        # Ingest FPS (per-frame arrival)
        try:
            cam_meter.tick()
        except Exception:
            pass

        # -*-*- throttle: keep processed FPS at target_fps (or below) ---
        tgt = float(getattr(args, "_computed_target_fps", 0) or 0.0)
        if tgt > 0.0:
            # Process at most 1 / tgt seconds
            # Keep a 'next_due' timestamp on the function object so it persists across frames
            if not hasattr(recognition_loop, "_next_due"):
                recognition_loop._next_due = time.monotonic()  # first call
            now = time.monotonic()
            interval = 1.0 / tgt

            # If we're early for the next slot, skip this frame (cheap and keeps latency low)
            if now < recognition_loop._next_due - 0.002:  # 2ms slack
                continue

            # We are due; schedule the next slot.
            # If we were late, don't accumulate debt—just move the window forward from 'now'.
            recognition_loop._next_due = max(recognition_loop._next_due + interval, now + interval)
        # -*-*- end throttle ---

        # frame_rgb is RGB HxWx3 uint8
        faces = app.get(frame_rgb)  # InsightFace expects RGB by default
        # Processed FPS (after detection step finishes for this frame)
        try:
            proc_meter.tick()
        except Exception:
            pass

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

            # crop using bbox
            x1, y1, x2, y2 = map(int, f.bbox)
            H, W = frame_rgb.shape[0], frame_rgb.shape[1]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W - 1, x2), min(H - 1, y2)

            # NEW: size gate from RecognitionSettings (default 40px)
            min_side = min(x2 - x1, y2 - y1)
            # Prefer explicit override; else use RecognitionSettings; else 40
            min_px = int(MIN_FACE_PX)
            HB_EXTRAS["min_face_px"] = int(min_px)

            if min_side < min_px:
                # (optional) keep a debug for tuning, same pattern used for unmatched crops
                if SAVE_DEBUG_UNMATCHED:
                    today = timezone.localdate()
                    dbg_dir = dbg_cam_root / f"{today:%Y/%m/%d}"
                    dbg_dir.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(frame_rgb[y1:y2, x1:x2]).save(
                        dbg_dir / f"small_{int(time.time() * 1000)}.jpg", quality=90
                    )
                continue

            crop_rgb = frame_rgb[y1:y2, x1:x2].copy()

            if score < threshold:
                if SAVE_DEBUG_UNMATCHED:
                    today = timezone.localdate()
                    dbg_dir = dbg_cam_root / f"{today:%Y/%m/%d}"
                    dbg_dir.mkdir(parents=True, exist_ok=True)
                    dbg_path = dbg_dir / f"unmatched_{int(time.time() * 1000)}.jpg"
                    fmt = (getattr(args, "crop_format", None) or "jpg").lower()
                    q = int(getattr(args, "crop_quality", None) or 90)
                    try:
                        if fmt == "png":
                            Image.fromarray(crop_rgb).save(str(dbg_path.with_suffix(".png")), format="PNG")
                        else:
                            Image.fromarray(crop_rgb).save(str(dbg_path.with_suffix(".jpg")), format="JPEG",
                                                           quality=max(1, min(100, q)))
                        log(f"[UNMATCHED] score={score:.3f} saved={dbg_path}")
                    except Exception as e:
                        log(f"[UNMATCHED] score={score:.3f} save-failed: {e}")
                continue

            now_m = time.monotonic()
            if now_m - last_seen.get(sid, 0.0) < debounce_sec:
                log(f"[debounce] {hcode} score={score:.3f}")
                continue
            last_seen[sid] = now_m

            # Save crop
            today = timezone.localdate()
            crop_dir = attn_root / f"{today:%Y/%m/%d}"
            crop_dir.mkdir(parents=True, exist_ok=True)
            fmt = (getattr(args, "crop_format", None) or "jpg").lower()
            q = int(getattr(args, "crop_quality", None) or 90)
            ext = ".png" if fmt == "png" else ".jpg"
            crop_path = str(crop_dir / f"{hcode}_{int(time.time())}{ext}")
            try:
                if fmt == "png":
                    Image.fromarray(crop_rgb).save(crop_path, format="PNG")
                else:
                    Image.fromarray(crop_rgb).save(crop_path, format="JPEG", quality=max(1, min(100, q)))
                log(f"[MATCH] {hcode} score={score:.3f} -> crop={crop_path}")

            except Exception as e:
                crop_path = ""
                log(f"[MATCH] {hcode} score={score:.3f} -> crop-save-error: {e}")

            # --- DB-driven per-student archiving + optional curation ---
            try:
                # 1) read global policy once per write (cheap)
                rs = RecognitionSettings.objects.order_by("-changed_at",
                                                          "-id").first() or RecognitionSettings.objects.first()
                if rs and rs.crops_enabled:
                    # 2) check scope: all students OR per-student opt-in
                    do_student = bool(rs.crops_apply_all_students)
                    if not do_student:
                        try:
                            fe_active = FaceEmbedding.objects.filter(student=student, is_active=True).first()
                            do_student = bool(fe_active and fe_active.crops_opt_in)
                        except Exception:
                            do_student = False

                    if do_student:
                        # 3) threshold gate (independent from recognition min_score)
                        save_thr = float(getattr(rs, "crops_save_threshold", 0.65) or 0.65)
                        if score >= save_thr:
                            cam_name = str(camera_obj.name)
                            # TODO: wire a real period id if you want it here; None becomes "_none_"
                            # resolve period occurrence for this recognition
                            period_id = None
                            try:
                                from apps.attendance.services import resolve_active_occurrence
                                occ = resolve_active_occurrence(student)
                                if occ:
                                    period_id = occ.id
                            except Exception:
                                pass

                            cap_dir = Path(student_capture_dir(
                                hcode, cam_name, period_id, when=timezone.now(),
                                subdir=getattr(rs, "crops_subdir", "captures"),
                                include_period=bool(getattr(rs, "crops_include_period", True)),
                            ))
                            cap_dir.mkdir(parents=True, exist_ok=True)

                            fmt = str(getattr(rs, "crops_format", "png")).lower()
                            ext = ".png" if fmt == "png" else ".jpg"
                            # camera id + name
                            cam_label = f"cam{camera_obj.id}-{camera_obj.name}"
                            # period id + name (if resolved)
                            per_label = f"per{period_id}" if period_id else "perNone"
                            try:
                                if period_id:
                                    from apps.attendance.models import PeriodOccurrence
                                    occ = PeriodOccurrence.objects.filter(id=period_id).first()
                                    if occ:
                                        per_label = f"per{period_id}-{occ.template.name}"
                            except Exception:
                                pass

                            # ts = timezone.now().strftime("%Y%m%d-%H%M%S")
                            ts = timezone.localtime(timezone.now()).strftime("%Y%m%d-%H%M%S")
                            # score_str = f"{score:.4f}".replace(".", "_")
                            score_str = f"{score:.4f}"
                            if per_label == "perNone":
                                out_path = cap_dir / f"{hcode}_S{score_str}_{cam_label}_tzta-{ts}{ext}"
                            else:
                                out_path = cap_dir / f"{hcode}_S{score_str}_{cam_label}_{per_label}_{ts}{ext}"

                            if fmt == "png":
                                Image.fromarray(crop_rgb).save(str(out_path), format="PNG")
                            else:
                                q = int(getattr(rs, "crops_quality", 95) or 95)
                                Image.fromarray(crop_rgb).save(str(out_path), format="JPEG",
                                                               quality=max(1, min(100, q)))

                            log(f"[captures] {hcode} score={score:.3f} saved {out_path}")

                            # 4) light curation (keep top-N) when folder grows
                            keep_n = int(getattr(rs, "crops_keep_n", 5) or 5)
                            imgs_now = [p for p in cap_dir.iterdir()
                                        if
                                        p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")]
                            if keep_n > 0 and len(imgs_now) > (2 * keep_n):
                                stats = curate_top_crops(cap_dir, keep_n=keep_n)
                                log(f"[captures] curated {cap_dir.name}: {stats}")
                        else:
                            log(f"[captures] {hcode} score={score:.3f} < crops_save_threshold -> skip")
            except Exception as e:
                log(f"[captures] DB-driven archive failed: {e}")

            HB_COUNTS["matched"] += 1

            try:
                student = Student.objects.get(pk=sid)
                record_recognition(student=student, score=score, camera=camera_obj, crop_path=(crop_path or None))
                log(f"[LOGGED] {hcode} score={score:.3f}")
            except Exception as e:
                log(f"[LOG ERROR] {hcode} score={score:.3f}: {e}")


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
    # Quality / model knobs (env fallbacks are provided so enforcer env works too)
    p.add_argument("--model", default=(os.getenv("BISK_MODEL", "").strip() or None),
                   help="InsightFace model pack (e.g., buffalo_l, antelopev2).")
    p.add_argument("--pipe_q", type=int, default=(int(os.getenv("BISK_PIPE_MJPEG_Q") or "0") or None),
                   help="FFmpeg MJPEG -q:v for pipe & snapshots (1=best, 31=worst).")
    p.add_argument("--crop_format", default=((os.getenv("BISK_CROP_FMT") or "").strip() or None),
                   choices=["jpg", "png", None], help="Format for saved face crops.")
    p.add_argument("--crop_quality", type=int, default=(int(os.getenv("BISK_CROP_JPEG_Q") or "0") or None),
                   help="JPEG quality for saved crops (1–100). Ignored for PNG.")
    p.add_argument("--min_face_px", type=int, default=(int(os.getenv("BISK_MIN_FACE_PX") or "0") or None),
                   help="Minimum face box size (pixels) to consider.")
    p.add_argument("--quality_version", type=int, default=(int(os.getenv("BISK_QUALITY_VERSION") or "0") or None),
                   help="Optional tuning profile switch (1/2/3…).")
    p.add_argument("--save_debug_unmatched", action="store_true",
                   default=(os.getenv("BISK_SAVE_DEBUG_UNMATCHED") == "1"),
                   help="If set, save unmatched crops for diagnostics.")

    args, _ = p.parse_known_args()
    # make the module-level toggle follow the resolved CLI/env default
    global SAVE_DEBUG_UNMATCHED
    SAVE_DEBUG_UNMATCHED = bool(getattr(args, "save_debug_unmatched", False))

    # ---- CPU quota (set before heavy threads start)
    from apps.scheduler.resources_cpu import apply_cpu_quota_percent, approximate_quota_with_affinity
    CPU_QUOTA = int(os.getenv("BISK_CPU_QUOTA_PERCENT", "0") or "0")
    if CPU_QUOTA > 0:
        ok = apply_cpu_quota_percent(CPU_QUOTA)
        if not ok:
            approximate_quota_with_affinity(CPU_QUOTA)

    # ---- GPU pacer
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

    # Determine effective target FPS once (profile / args fallback)
    computed_target_fps = (
                              eff.max_fps if eff.max_fps is not None else None
                          ) or (
                              args.fps if args.fps else None
                          ) or 6.0
    setattr(args, "_computed_target_fps", float(computed_target_fps))
    cam_id = int(args.camera)
    prof_id = int(args.profile)
    snap_dir = Path(args.snapshots);
    snap_dir.mkdir(parents=True, exist_ok=True)
    snap_path = snap_dir / f"{cam_id}.jpg"
    ff_proc = spawn_snapshotter(args, args.rtsp, str(snap_path))

    # ---- START HB THREAD ----
    cam_meter = FPSMeter(alpha=0.2)
    proc_meter = FPSMeter(alpha=0.2)

    def _get_metrics():
        # Return current EMA values; hb_worker handles float/None
        return (cam_meter.value() or 0.0, proc_meter.value() or 0.0)

    stop_hb = threading.Event()
    hb_thread = threading.Thread(
        target=hb_worker,
        # args=(stop_hb, args, cam_id, prof_id, None),  # no metrics yet
        args=(stop_hb, args, cam_id, prof_id, _get_metrics),
        daemon=True,
    )
    hb_thread.start()
    # -------------------------

    # Load gallery
    gal_M, gal_meta = load_gallery_from_npy()

    # Fetch camera object and settings for recognition
    try:
        camera_obj = Camera.objects.get(pk=cam_id)
    except Camera.DoesNotExist:
        camera_obj = None
    rs_cfg = RecognitionSettings.get_solo()

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

    # Start recognition loop (same process)
    try:
        recognition_loop(args=args, eff=eff, gal_M=gal_M, gal_meta=gal_meta, camera_obj=camera_obj, rs_cfg=rs_cfg,
                         cam_meter=cam_meter, proc_meter=proc_meter)
    finally:
        log("[runner] stopping...")
        try:
            ff_proc.terminate()
            try:
                ff_proc.wait(timeout=2.0)
            except Exception:
                ff_proc.kill()
        except Exception:
            pass
        # ---- STOP HB THREAD ----
        try:
            stop_hb.set()
            hb_thread.join(timeout=2.0)
        except Exception:
            pass
        # ------------------------

    # Send one final heartbeat on exit
    try:
        cam_fps_final, proc_fps_final = _get_metrics()
    except Exception:
        cam_fps_final, proc_fps_final = (0.0, 0.0)
    payload = {
        "camera_id": cam_id,
        "profile_id": prof_id,
        "pid": os.getpid(),
        "camera_fps": float(cam_fps_final),
        "processed_fps": float(proc_fps_final),
        "target_fps": float(getattr(args, "_computed_target_fps", 0) or 0) or None,
        "snapshot_every": int(getattr(args, "snapshot_every", 10) or 10),
        "detected": int(HB_COUNTS["detected"]),
        "matched": int(HB_COUNTS["matched"]),
        "latency_ms": 0.0,
        "last_error": last_error or "",
    }
    payload.update(HB_EXTRAS)
    try:
        post_json(args.hb, payload, args.hb_key, timeout=3.0)
        log(f"[hb-final] {payload}")
    except Exception as e:
        log(f"[hb-final] send failed: {e}")


if __name__ == "__main__":
    main()
