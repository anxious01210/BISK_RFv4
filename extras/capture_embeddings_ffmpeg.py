#!/usr/bin/env python3
# extras/capture_embeddings_ffmpeg.py
import argparse, os, sys, subprocess, io, time
from pathlib import Path
from typing import List, Tuple, Dict, Iterator, Optional, TypedDict, cast
import numpy as np
from PIL import Image, ImageOps, ImageStat
from django.conf import settings
from insightface.app import FaceAnalysis

# --- Optional live preview uplink (MJPEG) ------------------------------------
try:
    # when running the script directly (python extras/capture_embeddings_ffmpeg.py)
    from stream_uplink_publisher import StreamUplinkPublisher
except Exception:
    try:
        # when imported from elsewhere as a package
        from extras.stream_uplink_publisher import StreamUplinkPublisher
    except Exception:
        StreamUplinkPublisher = None  # graceful fallback

MEDIA_ROOT_FALLBACK = Path("media").absolute()
TIMEZONE_LOCAL = None


def _bootstrap_django():
    global TIMEZONE_LOCAL
    try:
        import django  # type: ignore
        from django.conf import settings as dj_settings  # type: ignore
        from django.utils import timezone as dj_tz  # type: ignore
        if not dj_settings.configured:
            os.environ.setdefault("DJANGO_SETTINGS_MODULE", "bisk.settings")
            django.setup()
        TIMEZONE_LOCAL = dj_tz
        media_root = Path(getattr(dj_settings, "MEDIA_ROOT", MEDIA_ROOT_FALLBACK))
        return media_root, dj_settings
    except Exception:
        class _TZ:
            @staticmethod
            def now():
                import datetime as _dt
                return _dt.datetime.now()

            @staticmethod
            def localdate():
                import datetime as _dt
                return _dt.date.today()

        TIMEZONE_LOCAL = _TZ

        class _Dummy:
            MEDIA_ROOT = str(MEDIA_ROOT_FALLBACK)

        return MEDIA_ROOT_FALLBACK, _Dummy()


MEDIA_ROOT, DJANGO_SETTINGS = _bootstrap_django()


def now_localdate_str() -> str:
    try:
        d = TIMEZONE_LOCAL.localdate()  # type: ignore[attr-defined]
    except Exception:
        import datetime as _dt
        d = _dt.date.today()
    return f"{d:%Y/%m/%d}"


class Sample(TypedDict, total=False):
    emb: np.ndarray
    qual: float
    crop: np.ndarray


def parse_device(device: str) -> Tuple[str, int]:
    ds = (device or "auto").lower().strip()
    if ds == "cpu":
        return "cpu", -1
    if ds.startswith("cuda"):
        try:
            idx = int(ds.replace("cuda", "") or 0)
        except Exception:
            idx = 0
        return "cuda", idx
    if ds == "auto":
        return "auto", 0
    return "auto", 0


def get_app(det: int, model: str = "buffalo_l", device: str = "auto") -> FaceAnalysis:
    runtime, ctx_id = parse_device(device)
    if runtime == "cpu":
        providers = ["CPUExecutionProvider"];
        real_ctx = -1
    elif runtime == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"];
        real_ctx = ctx_id
    else:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"];
        real_ctx = 0
    try:
        app = FaceAnalysis(name=model, providers=providers)
        app.prepare(ctx_id=real_ctx, det_size=(int(det), int(det)))
    except Exception:
        app = FaceAnalysis(name=model, providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(int(det), int(det)))
    return app


def laplacian_var_numpy(rgb_crop: np.ndarray) -> float:
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    gray = np.dot(rgb_crop[..., :3].astype(np.float32), [0.299, 0.587, 0.114])
    from numpy.lib.stride_tricks import sliding_window_view as swv
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    patches = swv(gray, (3, 3))
    out = (patches * k).sum(axis=(-1, -2))
    return float(max(np.var(out), 0.0))


def brightness_score(rgb_crop: np.ndarray) -> float:
    pil = Image.fromarray(rgb_crop);
    gs = ImageOps.grayscale(pil)
    stat = ImageStat.Stat(gs);
    mean = stat.mean[0] if stat.mean else 0.0
    return float(mean / 255.0)


def l2norm(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return (v if n <= eps else v / n).astype(np.float32)


def l2cos(a: np.ndarray, b: np.ndarray) -> float: return float(np.dot(a, b))


def face_quality(rgb_crop: np.ndarray, frame_wh, bbox, w_sharp, w_bright, w_size) -> float:
    H, W = frame_wh[1], frame_wh[0]
    x1, y1, x2, y2 = bbox
    area = max(0, (x2 - x1)) * max(0, (y2 - y1))
    size = area / max(1.0, float(H * W))
    sharp = laplacian_var_numpy(rgb_crop) / 2000.0
    bright = brightness_score(rgb_crop)
    return float(w_sharp * sharp + w_bright * bright + w_size * size)


def greedy_clusters(vecs: List[np.ndarray], sim_thresh: float = 0.70) -> List[List[int]]:
    centers: List[np.ndarray] = [];
    groups: List[List[int]] = []
    for idx, v in enumerate(vecs):
        placed = False
        for ci, c in enumerate(centers):
            if l2cos(v, c) >= float(sim_thresh):
                groups[ci].append(idx)
                centers[ci] = l2norm((c * (len(groups[ci]) - 1) + v) / len(groups[ci]))
                placed = True;
                break
        if not placed:
            centers.append(v.copy());
            groups.append([idx])
    return groups if groups else [[]]


def build_ffmpeg_cmd(src: str, fps: int, transport: str, hwaccel: str,
                     pipe_mjpeg_q: Optional[int],
                     pipe_size: Optional[str] = None,
                     webcam_input_format: str = "auto",
                     webcam_size: Optional[str] = None) -> List[str]:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if src.startswith("rtsp://"):
        if transport and transport != "auto": cmd += ["-rtsp_transport", transport]
        ha = (hwaccel or "none").lower()
        if ha in ("nvdec", "cuda", "cuvid"):
            cmd += ["-hwaccel", "cuda"]
        elif ha == "vaapi":
            cmd += ["-hwaccel", "vaapi"]
        cmd += ["-i", src]
    else:
        dev = src
        if dev.isdigit(): dev = f"/dev/video{dev}"
        cmd += ["-f", "v4l2"]
        if webcam_input_format and webcam_input_format != "auto":
            cmd += ["-input_format", webcam_input_format]
        if webcam_size: cmd += ["-video_size", webcam_size]
        cmd += ["-i", dev]
    # if int(fps) > 0: cmd += ["-vf", f"fps={int(fps)}"]

    # Build filter chain (fps and/or scaling)
    vf = []
    if int(fps) > 0:
        vf.append(f"fps={int(fps)}")
    # pipe_size: 'WIDTHxHEIGHT' or a single HEIGHT (keep aspect)
    if pipe_size:
        s = str(pipe_size).strip().lower()
        if "x" in s:
            # explicit WxH; force even dims, good scaler
            w, h = s.split("x", 1)
            vf.append(f"scale={int(w)}:{int(h)}:flags=lanczos")
        else:
            # height only, keep aspect; width divisible by 2
            vf.append(f"scale=-2:{int(s)}:flags=lanczos")
    if vf:
        cmd += ["-vf", ",".join(vf)]

    cmd += ["-an", "-f", "image2pipe", "-vcodec", "mjpeg"]
    if pipe_mjpeg_q is not None: cmd += ["-q:v", str(int(pipe_mjpeg_q))]
    cmd += ["pipe:1"];
    return cmd


def iter_jpeg_frames(proc: subprocess.Popen) -> Iterator[bytes]:
    stdout = proc.stdout
    if stdout is None: return
    buf = bytearray()
    while True:
        chunk = stdout.read(4096)
        if not chunk: break
        buf.extend(chunk)
        while True:
            idx = buf.find(b"\xff\xd9")
            if idx == -1: break
            frame = bytes(buf[:idx + 2]);
            del buf[:idx + 2];
            yield frame


def iter_frames(src: str, fps: int, transport: str, hwaccel: str,
                pipe_mjpeg_q: Optional[int], duration: int,
                webcam_input_format: str = "auto", webcam_size: Optional[str] = None,
                pipe_size: Optional[str] = None) -> Iterator[np.ndarray]:
    cmd = build_ffmpeg_cmd(src, fps=fps, transport=transport, hwaccel=hwaccel,
                           pipe_mjpeg_q=pipe_mjpeg_q, pipe_size=pipe_size,
                           webcam_input_format=webcam_input_format,
                           webcam_size=webcam_size)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    t0 = time.time()
    for jpeg in iter_jpeg_frames(proc):
        try:
            img = Image.open(io.BytesIO(jpeg));
            rgb = np.array(img.convert("RGB"))
        except Exception:
            continue
        yield rgb
        if duration > 0 and (time.time() - t0) >= duration: break
    try:
        proc.kill()
    except Exception:
        pass


def ensure_dirs(path: Path): path.mkdir(parents=True, exist_ok=True)


# def expand_bbox(bbox, W, H, expand: float):
#     if not expand or expand <= 0.0: return bbox
#     x1, y1, x2, y2 = map(float, bbox);
#     w = x2 - x1;
#     h = y2 - y1
#     ex = w * float(expand);
#     ey = h * float(expand)
#     nx1 = max(0, int(round(x1 - ex)));
#     ny1 = max(0, int(round(y1 - ey)))
#     nx2 = min(W - 1, int(round(x2 + ex)));
#     ny2 = min(H - 1, int(round(y2 + ey)))
#     return (nx1, ny1, nx2, ny2)

def expand_bbox(bbox, W, H, expand: float):
    """Expand bbox by 'expand' fraction on each side and clamp to image.
    bbox = (x1,y1,x2,y2) with x2>x1,y2>y1; returns integer, clamped coords."""
    if not expand or expand <= 0.0:
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, min(x1, W))
        x2 = max(0, min(x2, W))
        y1 = max(0, min(y1, H))
        y2 = max(0, min(y2, H))
        return (x1, y1, x2, y2)

    x1, y1, x2, y2 = map(float, bbox)
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    ex = w * float(expand)
    ey = h * float(expand)

    nx1 = int(round(x1 - ex))
    ny1 = int(round(y1 - ey))
    nx2 = int(round(x2 + ex))
    ny2 = int(round(y2 + ey))

    # clamp to [0..W] / [0..H]
    nx1 = max(0, min(nx1, W))
    ny1 = max(0, min(ny1, H))
    nx2 = max(0, min(nx2, W))
    ny2 = max(0, min(ny2, H))

    # ensure at least 1px in both directions
    if nx2 <= nx1: nx2 = min(W, nx1 + 1)
    if ny2 <= ny1: ny2 = min(H, ny1 + 1)
    return (nx1, ny1, nx2, ny2)


def _screen_size_fallback() -> Tuple[int, int]:
    # Try to get real screen size; fallback to 1280x720
    try:
        import tkinter as tk
        root = tk.Tk();
        root.withdraw()
        w = root.winfo_screenwidth();
        h = root.winfo_screenheight()
        root.destroy()
        return int(w), int(h)
    except Exception:
        return 1280, 720


def _scale_for_preview(w: int, h: int, max_w: int, max_h: int, allow_upscale: bool) -> float:
    if max_w <= 0 or max_h <= 0:
        return 1.0
    sw = max_w / float(w);
    sh = max_h / float(h)
    s = min(sw, sh)
    if not allow_upscale:
        s = min(1.0, s)
    return max(1e-6, s)


def main():
    p = argparse.ArgumentParser(
        description="Live-capture FaceEmbeddings to a .npy and/or save crops; designed for Django admin modal usage.")
    p.add_argument("--rtsp", type=str, default=getattr(settings, "DEFAULT_RTSP_RUN_LIVE_CAPTURE", ""))
    p.add_argument("--webcam", type=str, default="")
    p.add_argument("--webcam_input_format", choices=("auto", "mjpeg", "yuyv422"), default="auto")
    p.add_argument("--webcam_size", type=str, default="")
    p.add_argument("--rtsp_transport", choices=("auto", "tcp", "udp"), default="auto")
    p.add_argument("--hwaccel", choices=("none", "nvdec", "vaapi", "cuda"), default="none")
    p.add_argument("--hcode", required=True)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--duration", type=int, default=30)
    p.add_argument("--fps", type=int, default=4)
    p.add_argument("--det_size", "--det_set", type=int, default=1024, dest="det_size")
    p.add_argument("--model", default="buffalo_l")
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "cuda0", "cuda1", "cuda2", "cuda3"), default="auto")
    p.add_argument("--min_face_px", type=int, default=80)
    p.add_argument("--cluster_sim", type=float, default=0.70)
    p.add_argument("--weights", type=str, default="0.6,0.3,0.1")
    p.add_argument("--bbox_expand", type=float, default=0.0)
    p.add_argument("--gallery_root", default="face_gallery")
    p.add_argument("--save_all_crops", action="store_true")
    p.add_argument("--save_top_crops", action="store_true")
    p.add_argument("--crop_fmt", choices=("jpg", "png"), default="jpg")
    p.add_argument("--crop_jpeg_quality", type=int, default=92)
    p.add_argument("--pipe_mjpeg_q", type=int, default=None)
    p.add_argument("--preview", action="store_true")
    # New preview controls
    p.add_argument("--preview_fullscreen", action="store_true", help="Start the preview window in fullscreen.")
    p.add_argument("--preview_max_w", type=int, default=0,
                   help="Max preview width (0 uses screen width or 1280 fallback).")
    p.add_argument("--preview_max_h", type=int, default=0,
                   help="Max preview height (0 uses screen height or 720 fallback).")
    p.add_argument("--preview_allow_upscale", action="store_true",
                   help="Allow scaling up smaller sources to fill preview.")
    # live-capture for .npy --preview flag
    p.add_argument("--preview-session", default=None,
                   help="If set, post annotated preview frames to the MJPEG session id (e.g. cap_xxxxxx)")
    p.add_argument("--uplink-key", default=os.environ.get("STREAM_UPLINK_KEY"),
                   help="Uplink auth key; if omitted and the endpoint requires a key, posts will be 403")
    p.add_argument("--server", default=os.environ.get("BISK_SERVER", "http://127.0.0.1:8000"),
                   help="Base URL of the Django server that hosts /attendance/stream/uplink/")
    p.add_argument("--uplink-maxfps", type=int, default=6, help="Max post rate for preview frames")
    p.add_argument("--uplink-quality", type=int, default=85, help="JPEG quality for preview frames (2..100)")
    p.add_argument("--no-window", action="store_true", help="Disable OpenCV preview window even if --preview is set")
    p.add_argument("--pipe-size", type=str, default="",
                   help="Working frame size before detection. Accepts 'WIDTHxHEIGHT' or a single HEIGHT. "
                        "Examples: '1920x1080' or '1080'. If empty, no scaling is applied.")
    p.add_argument("--write-npy", action="store_true", help="If set, write embeddings/<HCODE>.npy")

    args = p.parse_args()

    publisher = None
    if getattr(args, "preview_session", None) and StreamUplinkPublisher is not None:
        try:
            publisher = StreamUplinkPublisher(
                server=args.server,
                session=args.preview_session,
                key=args.uplink_key,
                max_fps=args.uplink_maxfps,
            )
        except Exception:
            publisher = None  # do not crash capture if preview wiring fails

    def _preview_publish_bgr(frame_bgr):
        """
        Best-effort: send an annotated BGR frame to the MJPEG uplink.
        Safe to call on every frame; internally throttled by --uplink-maxfps.
        """
        if publisher is None or frame_bgr is None:
            return
        try:
            # args is closed over from the outer scope
            publisher.publish_bgr(frame_bgr, quality=int(getattr(args, "uplink_quality", 85)))
        except Exception:
            pass  # never break the capture on preview errors

    w_sharp, w_bright, w_size = [float(x) for x in str(args.weights).split(",")]
    src = args.webcam or args.rtsp
    if not src:
        print("No source provided: pass --rtsp or --webcam.", file=sys.stderr);
        sys.exit(2)

    ensure_dirs(MEDIA_ROOT);
    emb_dir = Path(MEDIA_ROOT) / "embeddings"
    dbg_root = Path(MEDIA_ROOT) / "logs" / "debug_faces" / f"capture_{args.hcode}"
    ensure_dirs(emb_dir);
    ensure_dirs(dbg_root)

    app = get_app(args.det_size, model=args.model, device=args.device)

    samples: List[Sample] = [];
    # has_preview = False
    # if args.preview:
    #     try:
    #         import cv2  # noqa: F401
    #     except Exception:
    #         pass
    #     else:
    #         has_preview = True
    has_preview = False
    if args.preview:
        try:
            import cv2  # noqa: F401
        except Exception:
            pass
        else:
            has_preview = True

    # Split responsibilities:
    # want_window = bool(args.preview) and has_preview  # controls cv2 window
    want_window = bool(args.preview) and has_preview and (not getattr(args, "no_window", False))
    want_uplink = bool(args.preview_session) and (publisher is not None)  # controls MJPEG posting
    has_overlay = bool(want_window or want_uplink)  # draw boxes if either path is active

    # Screen size defaults for auto-fit
    if args.preview_max_w <= 0 or args.preview_max_h <= 0:
        sw, sh = _screen_size_fallback()
        max_w = args.preview_max_w or sw
        max_h = args.preview_max_h or sh
    else:
        max_w, max_h = args.preview_max_w, args.preview_max_h

    fullscreen = bool(args.preview_fullscreen)

    # Prepare window if preview
    # if has_preview:
    if want_window:
        import cv2
        cv2.namedWindow("capture_preview", cv2.WINDOW_NORMAL)
        if fullscreen:
            cv2.setWindowProperty("capture_preview", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty("capture_preview", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

    for rgb in iter_frames(src, fps=args.fps, transport=args.rtsp_transport,
                           hwaccel=args.hwaccel, pipe_mjpeg_q=args.pipe_mjpeg_q,
                           duration=args.duration, pipe_size=(args.pipe_size or None),
                           webcam_input_format=args.webcam_input_format,
                           webcam_size=(args.webcam_size or None)):
        H, W = rgb.shape[0], rgb.shape[1]
        faces = app.get(rgb)
        # disp: Optional[np.ndarray] = rgb.copy() if has_preview else None
        disp: Optional[np.ndarray] = rgb.copy() if has_overlay else None

        for f in faces:
            x1, y1, x2, y2 = [int(t) for t in f.bbox]
            if (x2 - x1) < int(args.min_face_px) or (y2 - y1) < int(args.min_face_px): continue
            ex1, ey1, ex2, ey2 = expand_bbox((x1, y1, x2, y2), W, H, float(args.bbox_expand))

            # (synthetic padding => which adds black margin/border)
            # crop = rgb[max(0, ey1):max(0, ey2), max(0, ex1):max(0, ex2)]

            # clamp to image bounds (no synthetic padding ever)
            x1 = max(0, min(ex1, W))
            x2 = max(0, min(ex2, W))
            y1 = max(0, min(ey1, H))
            y2 = max(0, min(ey2, H))
            crop = rgb[y1:y2, x1:x2]

            if crop.size == 0: continue

            emb = getattr(f, "normed_embedding", None)
            if emb is None: emb = getattr(f, "embedding", None)
            if emb is None: continue

            v = l2norm(np.asarray(emb, dtype=np.float32))
            qual = face_quality(crop, (W, H), (ex1, ey1, ex2, ey2), w_sharp, w_bright, w_size)
            entry: Sample = {"emb": v, "qual": float(qual)}
            if args.save_top_crops: entry["crop"] = crop
            samples.append(entry)

            if args.save_all_crops:
                ddir = dbg_root / now_localdate_str();
                ensure_dirs(ddir)
                outp = ddir / f"keep_{int(time.time() * 1000)}.{args.crop_fmt}"
                if args.crop_fmt == "jpg":
                    Image.fromarray(crop).save(outp, quality=int(args.crop_jpeg_quality))
                else:
                    Image.fromarray(crop).save(outp)

            if disp is not None:
                import cv2
                cv2.rectangle(disp, (ex1, ey1), (ex2, ey2), (0, 255, 0), 2)
                cv2.putText(disp, f"{qual:.2f}", (ex1, max(0, ey1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        if disp is not None:
            import cv2
            # Down/up-scale to fit preview bounds
            scale = _scale_for_preview(W, H, max_w, max_h, bool(args.preview_allow_upscale))
            if scale != 1.0:
                new_w = max(1, int(W * scale))
                new_h = max(1, int(H * scale))
                interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
                disp_scaled = cv2.resize(disp, (new_w, new_h), interpolation=interp)
            else:
                disp_scaled = disp

            # Overlay hint (useful when window is visible)
            hint = "[F]ullscreen  [Q]/ESC to quit"
            cv2.putText(disp_scaled, hint, (10, max(20, disp_scaled.shape[0] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            # Convert for cv2 & uplink
            bgr = disp_scaled[..., ::-1].copy()

            # Post annotated frames to MJPEG session if requested
            if want_uplink:
                _preview_publish_bgr(bgr)

            # Show local window only if requested
            if want_window:
                cv2.imshow("capture_preview", bgr)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):
                    break
                if key in (ord('f'), ord('F')):
                    fullscreen = not fullscreen
                    cv2.setWindowProperty("capture_preview", cv2.WND_PROP_FULLSCREEN,
                                          cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)

    # if has_preview:
    if want_window:
        try:
            import cv2;
            cv2.destroyAllWindows()
        except Exception:
            pass

    if not samples: raise SystemExit("No faces captured.")

    vecs: List[np.ndarray] = [cast(np.ndarray, s["emb"]) for s in samples]
    clusters = greedy_clusters(vecs, sim_thresh=float(args.cluster_sim))
    clusters.sort(key=lambda idxs: len(idxs), reverse=True)
    main_idx = clusters[0]
    chosen: List[Sample] = sorted((samples[i] for i in main_idx), key=lambda s: s["qual"], reverse=True)[
        :max(1, int(args.k))]

    arr = np.stack([cast(np.ndarray, c["emb"]) for c in chosen]).astype(np.float32)
    avg = l2norm(arr.mean(axis=0))
    # ensure_dirs(emb_dir)
    # np.save(emb_dir / f"{args.hcode}.npy", avg)
    if args.write_npy:
        ensure_dirs(emb_dir)
        out_path = emb_dir / f"{args.hcode}.npy"
        np.save(out_path, avg)
        print(f"[OK] wrote embedding: {out_path}")

    if args.save_top_crops:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_dir = Path(MEDIA_ROOT) / str(args.gallery_root) / args.hcode / f"live_{ts}"
        ensure_dirs(out_dir);
        saved = 0
        for i, c in enumerate(chosen, start=1):
            crop_np = cast(Optional[np.ndarray], c.get("crop"))
            if not isinstance(crop_np, np.ndarray): continue
            if args.crop_fmt == "jpg":
                Image.fromarray(crop_np).save(out_dir / f"{args.hcode}_{i:02d}.jpg",
                                              quality=int(args.crop_jpeg_quality))
            else:
                Image.fromarray(crop_np).save(out_dir / f"{args.hcode}_{i:02d}.png")
            saved += 1
        rel_dir = str(out_dir.relative_to(Path(MEDIA_ROOT))).replace("\\\\", "/")
        print(f"[TOP_CROPS] saved={saved} dir={rel_dir}")


if __name__ == "__main__":
    main()

# New: press F to toggle fullscreen (works for any source).
# New flags so the window auto-fits instead of overflowing:
# --preview_max_w and --preview_max_h (default: your screen size, or 1280×720 fallback).
# --preview_allow_upscale (off by default, so small webcams won’t be pixel-stretched unless you want them to).
# --preview_fullscreen to start in fullscreen directly.
# The overlay shows a little hint: “[F]ullscreen [Q]/ESC to quit”.
# For camera (RTSP), the preview is now scaled down to fit by default.
# For webcam, you can maximize with F, or allow upscaling with --preview_allow_upscale.

# Camera (fit-to-screen by default)
# python extras/capture_embeddings_ffmpeg.py \
#   --rtsp 'rtsp://admin:B!sk2025@192.168.137.95:554/Streaming/Channels/101/' \
#   --hcode H123456 --k 10 --duration 30 --fps 6 --det_size 1024 \
#   --model buffalo_l --device cuda0 --rtsp_transport auto --hwaccel nvdec \
#   --min_face_px 80 --weights 0.6,0.3,0.1 --gallery_root face_gallery \
#   --save_top_crops --preview \
#   --preview_max_w 1920 --preview_max_h 1080 \
#   --crop_fmt png --pipe_mjpeg_q 2 --bbox_expand 0.0

# Webcam (maximize available)
# python extras/capture_embeddings_ffmpeg.py \
#   --webcam /dev/video0 --webcam_input_format mjpeg --webcam_size 1280x720 \
#   --hcode H123456 --k 10 --duration 30 --fps 6 --det_size 1024 \
#   --model buffalo_l --device auto \
#   --min_face_px 80 --weights 0.6,0.3,0.1 --gallery_root face_gallery \
#   --save_top_crops --preview --preview_allow_upscale \
#   --crop_fmt jpg --crop_jpeg_quality 92 --pipe_mjpeg_q 2 --bbox_expand 0.0
