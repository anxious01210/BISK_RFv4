#!/usr/bin/env python3
# extras/capture_embeddings_ffmpeg.py
# -----------------------------------------------------------------------------
# FFmpeg-only capture -> InsightFace -> top-K quality -> average -> <HCODE>.npy
# -----------------------------------------------------------------------------
# What this script does
# ---------------------
# 1) Reads frames from RTSP or a webcam using *pure FFmpeg* into an image pipe.
# 2) Detects faces with InsightFace (buffalo_l) at a configurable det-size.
# 3) Scores each detected face by a weighted mix of sharpness, brightness,
#    and face size (relative to the frame).
# 4) Clusters the face embeddings by cosine similarity and chooses the
#    dominant person (largest cluster).
# 5) Averages the top-K faces from that cluster and writes:
#       media/embeddings/<HCODE>.npy
# 6) Optionally saves *all* detected crops for debugging and/or saves the
#    final top-K crops under:
#       media/<gallery_root>/<HCODE>/live_YYYYmmdd_HHMMSS/
#
# Where to put it
# ---------------
# Place this file at:  extras/capture_embeddings_ffmpeg.py
# (For quick tests you can run it from anywhere; it bootstraps Django below.)
#
# Quick examples
# --------------
# # Default (auto-picks the RTSP default below if it looks like an RTSP URL):
# python extras/capture_embeddings_ffmpeg.py \
#   --hcode H123456 --k 3 --duration 40 --fps 4 \
#   --det_size 1024 --device auto --save_all_crops
#
# # Force RTSP and **quote** the URL if it contains special chars like '!':
# python extras/capture_embeddings_ffmpeg.py \
#   --use rtsp \
#   --rtsp 'rtsp://admin:B\!sk2025@192.168.137.95:554/Streaming/Channels/101/' \
#   --hcode H123456 --k 3 --duration 40 --fps 4 --det_size 1024 --device auto \
#   --preview --save_top_crops
#
# # Use webcam 0 instead of RTSP:
# python extras/capture_embeddings_ffmpeg.py \
#   --use webcam --webcam 0 \
#   --hcode H123456 --k 3 --duration 60 --fps 4 \
#   --det_size 1024 --device auto --save_all_crops
#
# Preview window requirements
# ---------------------------
# If you use --preview and you installed opencv-python-headless, uninstall it
# and install opencv-python instead; also install GUI libs:
#   sudo apt-get update
#   sudo apt-get install -y libgtk-3-0 libglib2.0-0 libsm6 libxext6 libxrender1 libgl1
# Make sure $DISPLAY is set (non-headless session).
# -----------------------------------------------------------------------------

import argparse, os, sys, pathlib, subprocess, io, json, time, threading
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from PIL import Image, ImageOps, ImageStat

# ---- Django bootstrap ---------------------------------------------------------
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "bisk.settings")
# ensure project root is on sys.path (this file is expected under extras/)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import django

django.setup()
from django.conf import settings
from django.utils import timezone

# Optional helpers if your project exposes media dirs; otherwise we create them.
try:
    from apps.attendance.utils.media_paths import DIRS, ensure_media_tree
except Exception:
    DIRS = {}


    def ensure_media_tree():
        root = Path(getattr(settings, "MEDIA_ROOT", "media"))
        (root / "embeddings").mkdir(parents=True, exist_ok=True)
        (root / "logs" / "debug_faces").mkdir(parents=True, exist_ok=True)

# ---- InsightFace --------------------------------------------------------------
from insightface.app import FaceAnalysis


def l2norm(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize a vector safely."""
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n <= eps:
        return v.astype(np.float32)
    return (v / n).astype(np.float32)


def get_app(det: int, device: str = "auto") -> FaceAnalysis:
    """Init InsightFace with CPU/CUDA fallback and set detector size."""
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device in ("auto", "cuda") else [
        "CPUExecutionProvider"]
    try:
        app = FaceAnalysis(name="buffalo_l", providers=providers)
        ctx = 0 if (device != "cpu") else -1
        app.prepare(ctx_id=ctx, det_size=(det, det))
    except Exception:
        if device == "auto":
            app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=-1, det_size=(det, det))
        else:
            raise
    return app


# ---- FFmpeg helpers -----------------------------------------------------------
def ffmpeg_cmd(src: str, *, fps: float, transport: str, hwaccel: str) -> List[str]:
    """Build an FFmpeg command that outputs an MJPEG image pipe at a given FPS."""
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "warning"]
    if src.startswith("rtsp://"):
        if transport != "auto":
            cmd += ["-rtsp_transport", transport]  # tcp/udp
        if hwaccel == "nvdec":
            cmd += ["-hwaccel", "cuda"]
    cmd += [
        "-i", src,
        "-vf", f"fps={max(0.1, float(fps))}",
        "-f", "image2pipe",
        "-vcodec", "mjpeg",
        "-q:v", "2",  # quality of the MJPEG stream (smaller is better)
        "pipe:1",
    ]
    return cmd


def iter_frames(src: str, *, fps: float, transport: str = "tcp", hwaccel: str = "none"):
    """Yield PIL RGB frames from FFmpeg's MJPEG pipe."""
    args = ffmpeg_cmd(src, fps=fps, transport=transport, hwaccel=hwaccel)
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    SOI, EOI = b"\\xff\\xd8", b"\\xff\\xd9"
    buf = bytearray()

    def _stderr_reader():
        for raw in iter(proc.stderr.readline, b""):
            line = raw.decode("utf-8", "ignore").strip()
            if line:
                pass  # print("[ffmpeg]", line)

    threading.Thread(target=_stderr_reader, daemon=True).start()

    try:
        while True:
            chunk = proc.stdout.read(4096)
            if not chunk:
                break
            buf += chunk
            while True:
                i = buf.find(SOI)
                if i < 0:
                    if len(buf) > 1024 * 1024:
                        del buf[:-1024]
                    break
                j = buf.find(EOI, i + 2)
                if j < 0:
                    if i > 0:
                        del buf[:i]
                    break
                jpg = bytes(buf[i:j + 2]);
                del buf[:j + 2]
                try:
                    im = Image.open(io.BytesIO(jpg)).convert("RGB")
                    yield im
                except Exception:
                    continue
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=2.0)
        except Exception:
            pass


# ---- Quality scoring & clustering --------------------------------------------
def laplacian_var_numpy(rgb: np.ndarray) -> float:
    """A tiny Laplacian variance (edge energy) as a proxy for sharpness."""
    g = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    pad = np.pad(g, 1, mode="edge")
    out = (
            k[0, 0] * pad[:-2, :-2] + k[0, 1] * pad[:-2, 1:-1] + k[0, 2] * pad[:-2, 2:] +
            k[1, 0] * pad[1:-1, :-2] + k[1, 1] * pad[1:-1, 1:-1] + k[1, 2] * pad[1:-1, 2:] +
            k[2, 0] * pad[2:, :-2] + k[2, 1] * pad[2:, 1:-1] + k[2, 2] * pad[2:, 2:]
    )
    v = float(np.var(out))
    return max(v, 0.0)


def brightness_score(rgb_crop: np.ndarray) -> float:
    """Normalize mean brightness to [0..1]."""
    pil = Image.fromarray(rgb_crop)
    gs = ImageOps.grayscale(pil)
    stat = ImageStat.Stat(gs)
    mean = stat.mean[0] if stat.mean else 0.0
    return float(mean / 255.0)


def face_quality(rgb_crop: np.ndarray, frame_wh: Tuple[int, int], bbox, w_sharp, w_bright, w_size) -> float:
    """Weighted score = w_sharp*sharp + w_bright*bright + w_size*relative face area."""
    H, W = frame_wh[1], frame_wh[0]
    x1, y1, x2, y2 = bbox
    area = max(0, (x2 - x1)) * max(0, (y2 - y1))
    size = area / max(1.0, float(H * W))
    sharp = laplacian_var_numpy(rgb_crop) / 2000.0
    bright = brightness_score(rgb_crop)
    return float(w_sharp * sharp + w_bright * bright + w_size * size)


def l2cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def greedy_clusters(vecs: List[np.ndarray], sim_thresh: float = 0.70) -> List[List[int]]:
    """Very small online clustering by cosine similarity (keeps one center per cluster)."""
    clusters, centers = [], []
    for i, v in enumerate(vecs):
        if not clusters:
            clusters.append([i]);
            centers.append(v.copy());
            continue
        best, bj = -1.0, -1
        for j, c in enumerate(centers):
            s = l2cos(v, c)
            if s > best:
                best, bj = s, j
        if best >= sim_thresh:
            clusters[bj].append(i)
            centers[bj] = l2norm((centers[bj] * len(clusters[bj]) + v) / (len(clusters[bj]) + 1e-6))
        else:
            clusters.append([i]);
            centers.append(v.copy())
    return clusters


# ---- Main ---------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Capture best faces from RTSP/webcam and build <HCODE>.npy")
    # Data source options
    p.add_argument("--rtsp", help="RTSP URL (quote or percent-encode passwords that include ! or @)")
    p.add_argument("--webcam", help="Webcam index or device, e.g., 0 or /dev/video0")
    p.add_argument("--use", choices=["auto", "rtsp", "webcam"], default="auto",
                   help="If neither --rtsp nor --webcam are set, choose the default here.")
    p.add_argument("--default_rtsp", default='rtsp://admin:B\\!sk2025@192.168.137.95:554/Streaming/Channels/101/',
                   help="Fallback RTSP URL used when --use rtsp (remember to escape/quote !)")
    p.add_argument("--default_webcam", default="0", help="Fallback webcam index or device (default: 0)")

    # Enrollment parameters
    p.add_argument("--hcode", required=True, help="Target student H-code, e.g., H123456")
    p.add_argument("--k", type=int, default=3, help="Top-K faces to average")
    p.add_argument("--duration", type=int, default=30, help="Capture duration in seconds")
    p.add_argument("--fps", type=float, default=4.0, help="Sampling fps from the stream")
    p.add_argument("--det_size", type=int, default=1024, help="Detector input size (e.g., 640/1024)")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--rtsp_transport", choices=["auto", "tcp", "udp"], default="tcp")
    p.add_argument("--hwaccel", choices=["none", "nvdec"], default="none")
    p.add_argument("--min_face_px", type=int, default=80, help="Ignore faces smaller than this many pixels")
    p.add_argument("--cluster_sim", type=float, default=0.70,
                   help="Cosine sim threshold for the small online clustering")
    p.add_argument("--weights", type=str, default="0.6,0.3,0.1",
                   help="Quality weights as 'sharp,bright,face_size' (sums do not have to be 1)")
    p.add_argument("--preview", action="store_true", help="Show live preview with detections (press q/ESC to quit)")
    p.add_argument("--save_all_crops", action="store_true", help="Save every detected face crop to debug folder")
    p.add_argument("--save_top_crops", action="store_true",
                   help="Save the final selected top-K crops under face_gallery/<HCODE>/live_YYYYmmdd_HHMMSS/")
    p.add_argument("--gallery_root", default="face_gallery",
                   help="MEDIA subdir where per-student images live (default: face_gallery)")

    args = p.parse_args()
    w_sharp, w_bright, w_size = [float(x) for x in args.weights.split(",")]

    # Pick source
    if args.rtsp:
        src = args.rtsp
    elif args.webcam:
        src = str(args.webcam)
    else:
        src = args.default_rtsp if args.use == "rtsp" else str(args.default_webcam)

    ensure_media_tree()
    media_root = Path(getattr(settings, "MEDIA_ROOT", "media"))
    emb_dir = Path(DIRS.get("EMBEDDINGS", media_root / "embeddings"))
    dbg_root = media_root / "logs" / "debug_faces" / f"capture_{args.hcode}"
    emb_dir.mkdir(parents=True, exist_ok=True)
    dbg_root.mkdir(parents=True, exist_ok=True)

    app = get_app(args.det_size, device=args.device)
    frames = iter_frames(src, fps=args.fps, transport=args.rtsp_transport, hwaccel=args.hwaccel)

    # Preview requires OpenCV (only for the window, not for capture)
    if args.preview:
        import cv2  # noqa: F401

    deadline = time.time() + float(args.duration)
    samples: List[Dict] = []
    total_seen = 0

    while time.time() < deadline:
        try:
            im = next(frames)
        except StopIteration:
            break
        except Exception:
            continue

        total_seen += 1
        W, H = im.size
        rgb = np.array(im, dtype=np.uint8)
        faces = app.get(rgb)

        # Draw preview
        if args.preview:
            import cv2
            disp = rgb.copy()
            if faces:
                for f in faces:
                    x1, y1, x2, y2 = [int(t) for t in f.bbox]
                    x1 = max(0, x1);
                    y1 = max(0, y1);
                    x2 = min(W - 1, x2);
                    y2 = min(H - 1, y2)
                    crop = rgb[y1:y2, x1:x2] if y2 > y1 and x2 > x1 else None
                    qual = face_quality(crop, (W, H), (x1, y1, x2, y2), w_sharp, w_bright,
                                        w_size) if crop is not None else 0.0
                    cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(disp, f"{qual:.2f}", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                                1, cv2.LINE_AA)
            cv2.imshow("capture_embeddings (q/ESC to quit)", cv2.cvtColor(disp, cv2.COLOR_RGB2BGR))
            k = (cv2.waitKey(1) & 0xFF)
            if k in (27, ord('q')):  # ESC or q
                break

        if not faces:
            continue

        for f in faces:
            x1, y1, x2, y2 = [int(t) for t in f.bbox]
            x1 = max(0, x1);
            y1 = max(0, y1);
            x2 = min(W - 1, x2);
            y2 = min(H - 1, y2)
            if min(x2 - x1, y2 - y1) < int(args.min_face_px):
                # still save if requested
                if args.save_all_crops and x2 > x1 and y2 > y1:
                    tdir = dbg_root / f"{timezone.localdate():%Y/%m/%d}"
                    tdir.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(rgb[y1:y2, x1:x2]).save(tdir / f"small_{int(time.time() * 1000)}.jpg", quality=90)
                continue

            crop = rgb[y1:y2, x1:x2].copy()
            qual = face_quality(crop, (W, H), (x1, y1, x2, y2), w_sharp, w_bright, w_size)

            emb = getattr(f, "normed_embedding", None)
            if emb is None:
                emb = getattr(f, "embedding", None)

            if emb is None:
                if args.save_all_crops:
                    tdir = dbg_root / f"{timezone.localdate():%Y/%m/%d}"
                    tdir.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(crop).save(tdir / f"noemb_{int(time.time() * 1000)}.jpg", quality=90)
                continue

            v = l2norm(np.asarray(emb, dtype=np.float32))

            entry = {"emb": v, "qual": float(qual)}
            if args.save_top_crops:
                entry["crop"] = crop  # RGB uint8 (saved later only if chosen among top-K)
            samples.append(entry)

            if args.save_all_crops:
                tdir = dbg_root / f"{timezone.localdate():%Y/%m/%d}"
                tdir.mkdir(parents=True, exist_ok=True)
                Image.fromarray(crop).save(tdir / f"keep_{int(time.time() * 1000)}.jpg", quality=90)

    if args.preview:
        import cv2
        cv2.destroyAllWindows()

    if not samples:
        raise SystemExit(
            "No faces captured. Try longer duration, lower --min_face_px, or use --preview to verify framing.")

    # Cluster and pick dominant person
    vecs = [s["emb"] for s in samples]
    clusters = greedy_clusters(vecs, sim_thresh=float(args.cluster_sim))
    clusters.sort(key=lambda idxs: len(idxs), reverse=True)
    main_idx = clusters[0]
    chosen = sorted((samples[i] for i in main_idx), key=lambda s: s["qual"], reverse=True)[:max(1, int(args.k))]
    avg = l2norm(np.mean([c["emb"] for c in chosen], axis=0))

    out_path = Path(DIRS.get("EMBEDDINGS", media_root / "embeddings")) / f"{args.hcode}.npy"
    np.save(str(out_path), avg)
    print(f"[OK] wrote embedding: {out_path}  (topK={len(chosen)}/{len(samples)} faces used)")

    # Optionally save ONLY the final selected crops into the student's gallery
    if args.save_top_crops:
        ts = timezone.localtime().strftime("live_%Y%m%d_%H%M%S")
        out_dir = media_root / args.gallery_root / args.hcode / ts
        out_dir.mkdir(parents=True, exist_ok=True)
        saved = 0
        for i, c in enumerate(chosen, 1):
            if "crop" not in c:
                continue
            try:
                fname = f"{args.hcode}_{i:02d}.jpg"
                Image.fromarray(c["crop"]).save(out_dir / fname, quality=92)
                saved += 1
            except Exception:
                pass

        # Print a machine-friendly, MEDIA-relative path so callers (Django view)
        # can show a link/toast easily.
        rel_dir = str(out_dir.relative_to(media_root)).replace("\\\\", "/")
        print(f"[TOP_CROPS] saved={saved} dir={rel_dir}")


if __name__ == "__main__":
    main()
