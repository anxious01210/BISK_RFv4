# Create an all-FFmpeg capture script that:
# - Reads frames from RTSP or webcam via FFmpeg -> image2pipe (MJPEG) (no cv2.VideoCapture required)
# - Detects faces with InsightFace at a configurable det-size (default 1024)
# - Scores faces by quality (sharpness + brightness + face size)
# - Clusters embeddings by cosine similarity and picks the dominant cluster
# - Averages the top-K faces from that cluster to produce HCODE.npy in media/embeddings/
# - Saves debug crops + a JSON report for transparency
#
# The script expects to be placed under extras/ in the Django project; it bootstraps Django settings.
# It will still run fine from anywhere as long as the project root is discoverable one level up.

#!/usr/bin/env python3
# extras/capture_embeddings_ffmpeg.py
# Capture best-face embeddings from an RTSP/webcam stream using pure FFmpeg piping.
# Produces media/embeddings/<HCODE>.npy by averaging top-K high-quality faces
# from the dominant cluster.

#!/usr/bin/env python3
# extras/capture_embeddings_ffmpeg.py
# FFmpeg-only capture -> InsightFace -> top-K quality -> average -> <HCODE>.npy
import argparse, os, sys, pathlib, subprocess, io, json, time, threading
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from PIL import Image, ImageOps, ImageStat

# ---- Django bootstrap ---------------------------------------------------------
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "bisk.settings")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import django
django.setup()
from django.conf import settings
from django.utils import timezone

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

def l2norm(v: np.ndarray, eps: float=1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n <= eps: return v.astype(np.float32)
    return (v / n).astype(np.float32)

def get_app(det: int, device: str="auto") -> FaceAnalysis:
    providers = ["CUDAExecutionProvider","CPUExecutionProvider"] if device in ("auto","cuda") else ["CPUExecutionProvider"]
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
    cmd = ["/usr/bin/ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "warning"]
    if src.startswith("rtsp://"):
        if transport != "auto": cmd += ["-rtsp_transport", transport]
        if hwaccel == "nvdec": cmd += ["-hwaccel", "cuda"]
    cmd += ["-i", src, "-vf", f"fps={max(0.1, float(fps))}", "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1"]
    return cmd

def iter_frames(src: str, *, fps: float, transport: str="tcp", hwaccel: str="none"):
    args = ffmpeg_cmd(src, fps=fps, transport=transport, hwaccel=hwaccel)
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    SOI, EOI = b"\xff\xd8", b"\xff\xd9"
    buf = bytearray()

    def _stderr_reader():
        for raw in iter(proc.stderr.readline, b""):
            line = raw.decode("utf-8","ignore").strip()
            if line: pass  # uncomment next line to debug ffmpeg
            # print("[ffmpeg]", line, flush=True)

    threading.Thread(target=_stderr_reader, daemon=True).start()

    try:
        while True:
            chunk = proc.stdout.read(4096)
            if not chunk: break
            buf += chunk
            while True:
                i = buf.find(SOI)
                if i < 0:
                    if len(buf) > 1024*1024: del buf[:-1024]
                    break
                j = buf.find(EOI, i+2)
                if j < 0:
                    if i > 0: del buf[:i]
                    break
                jpg = bytes(buf[i:j+2]); del buf[:j+2]
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
    g = (0.299*rgb[...,0] + 0.587*rgb[...,1] + 0.114*rgb[...,2]).astype(np.float32)
    k = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float32)
    pad = np.pad(g, 1, mode="edge")
    out = (
        k[0,0]*pad[:-2,:-2] + k[0,1]*pad[:-2,1:-1] + k[0,2]*pad[:-2,2:] +
        k[1,0]*pad[1:-1,:-2] + k[1,1]*pad[1:-1,1:-1] + k[1,2]*pad[1:-1,2:] +
        k[2,0]*pad[2:,:-2] + k[2,1]*pad[2:,1:-1] + k[2,2]*pad[2:,2:]
    )
    v = float(np.var(out))
    return max(v, 0.0)

def brightness_score(rgb_crop: np.ndarray) -> float:
    pil = Image.fromarray(rgb_crop)
    gs = ImageOps.grayscale(pil)
    stat = ImageStat.Stat(gs)
    mean = stat.mean[0] if stat.mean else 0.0
    return float(mean/255.0)

def face_quality(rgb_crop: np.ndarray, frame_wh: Tuple[int,int], bbox, w_sharp, w_bright, w_size) -> float:
    H,W = frame_wh[1], frame_wh[0]
    x1,y1,x2,y2 = bbox
    area = max(0,(x2-x1))*max(0,(y2-y1))
    size = area / max(1.0, float(H*W))
    sharp = laplacian_var_numpy(rgb_crop) / 2000.0
    bright = brightness_score(rgb_crop)
    return float(w_sharp*sharp + w_bright*bright + w_size*size)

def l2cos(a: np.ndarray, b: np.ndarray) -> float: return float(np.dot(a,b))

def greedy_clusters(vecs: List[np.ndarray], sim_thresh: float=0.70) -> List[List[int]]:
    clusters, centers = [], []
    for i,v in enumerate(vecs):
        if not clusters:
            clusters.append([i]); centers.append(v.copy()); continue
        best, bj = -1.0, -1
        for j,c in enumerate(centers):
            s = l2cos(v,c)
            if s > best: best, bj = s, j
        if best >= sim_thresh:
            clusters[bj].append(i)
            centers[bj] = l2norm((centers[bj]*len(clusters[bj]) + v)/(len(clusters[bj])+1e-6))
        else:
            clusters.append([i]); centers.append(v.copy())
    return clusters

# ---- Main ---------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Capture best faces from RTSP/webcam and build <HCODE>.npy")
    # explicit sources (optional now)
    p.add_argument("--rtsp", help="RTSP URL (quote or percent-encode passwords with !)")
    p.add_argument("--webcam", help="Webcam index or device, e.g., 0 or /dev/video0")
    # fallbacks & selection
    p.add_argument("--use", choices=["auto","rtsp","webcam"], default="auto",
                   help="Which default to use if --rtsp/--webcam not provided")
    p.add_argument("--default_rtsp", default='rtsp://admin:B!sk2025@192.168.137.95:554/Streaming/Channels/101/',
                   help="Fallback RTSP URL (default: your .95 camera, ! encoded as %21)")
    p.add_argument("--default_webcam", default="0", help="Fallback webcam index or device (default: 0)")

    p.add_argument("--hcode", required=True, help="Target student H-code, e.g., H123456")
    p.add_argument("--k", type=int, default=3, help="Top-K faces to average")
    p.add_argument("--duration", type=int, default=30, help="Capture duration in seconds")
    p.add_argument("--fps", type=float, default=4.0, help="Sampling fps")
    p.add_argument("--det_size", type=int, default=1024)
    p.add_argument("--device", choices=["auto","cuda","cpu"], default="auto")
    p.add_argument("--rtsp_transport", choices=["auto","tcp","udp"], default="tcp")
    p.add_argument("--hwaccel", choices=["none","nvdec"], default="none")
    p.add_argument("--min_face_px", type=int, default=80)
    p.add_argument("--cluster_sim", type=float, default=0.70)
    p.add_argument("--weights", type=str, default="0.6,0.3,0.1",
                   help="Quality weights: sharp,bright,face_size")
    p.add_argument("--preview", action="store_true", help="Show live preview with detections (press q/ESC to quit)")
    p.add_argument("--save_all_crops", action="store_true", help="Save every detected face crop to debug folder")
    args = p.parse_args()

    w_sharp, w_bright, w_size = [float(x) for x in args.weights.split(",")]

    # pick source
    src = None
    if args.rtsp: src = args.rtsp
    elif args.webcam: src = str(args.webcam)
    else:
        if args.use == "rtsp": src = args.default_rtsp
        elif args.use == "webcam": src = str(args.default_webcam)
        else:
            # auto: prefer RTSP default if it looks like an rtsp URL
            src = args.default_rtsp if str(args.default_rtsp).startswith("rtsp://") else str(args.default_webcam)

    ensure_media_tree()
    media_root = Path(getattr(settings, "MEDIA_ROOT", "media"))
    emb_dir = Path(DIRS.get("EMBEDDINGS", media_root / "embeddings"))
    dbg_root = media_root / "logs" / "debug_faces" / f"capture_{args.hcode}"
    emb_dir.mkdir(parents=True, exist_ok=True); dbg_root.mkdir(parents=True, exist_ok=True)

    app = get_app(args.det_size, device=args.device)
    frames = iter_frames(src, fps=args.fps, transport=args.rtsp_transport, hwaccel=args.hwaccel)

    # Optional preview (cv2 for display only)
    if args.preview:
        import cv2

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

        # draw preview
        if args.preview:
            disp = rgb.copy()
            if faces:
                for f in faces:
                    x1,y1,x2,y2 = [int(t) for t in f.bbox]
                    x1=max(0,x1); y1=max(0,y1); x2=min(W-1,x2); y2=min(H-1,y2)
                    crop = rgb[y1:y2, x1:x2] if y2>y1 and x2>x1 else None
                    qual = face_quality(crop, (W,H), (x1,y1,x2,y2), w_sharp,w_bright,w_size) if crop is not None else 0.0
                    import cv2
                    cv2.rectangle(disp, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.putText(disp, f"{qual:.2f}", (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
            import cv2
            cv2.imshow("capture_embeddings (q/ESC to quit)", cv2.cvtColor(disp, cv2.COLOR_RGB2BGR))
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord('q')):  # ESC or q
                break

        if not faces:
            continue

        for f in faces:
            x1,y1,x2,y2 = [int(t) for t in f.bbox]
            x1=max(0,x1); y1=max(0,y1); x2=min(W-1,x2); y2=min(H-1,y2)
            if min(x2-x1, y2-y1) < int(args.min_face_px):
                # still save if requested
                if args.save_all_crops and x2>x1 and y2>y1:
                    tdir = dbg_root / f"{timezone.localdate():%Y/%m/%d}"
                    tdir.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(rgb[y1:y2, x1:x2]).save(tdir / f"small_{int(time.time()*1000)}.jpg", quality=90)
                continue

            crop = rgb[y1:y2, x1:x2].copy()
            qual = face_quality(crop, (W,H), (x1,y1,x2,y2), w_sharp, w_bright, w_size)

            emb = getattr(f, "normed_embedding", None)
            if emb is None:
                emb = getattr(f, "embedding", None)

            if emb is None:
                if args.save_all_crops:
                    tdir = dbg_root / f"{timezone.localdate():%Y/%m/%d}"
                    tdir.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(crop).save(tdir / f"noemb_{int(time.time()*1000)}.jpg", quality=90)
                continue
            v = l2norm(np.asarray(emb, dtype=np.float32))

            samples.append({"emb": v, "qual": float(qual)})

            if args.save_all_crops:
                tdir = dbg_root / f"{timezone.localdate():%Y/%m/%d}"
                tdir.mkdir(parents=True, exist_ok=True)
                Image.fromarray(crop).save(tdir / f"keep_{int(time.time()*1000)}.jpg", quality=90)

    if args.preview:
        import cv2
        cv2.destroyAllWindows()

    if not samples:
        raise SystemExit("No faces captured. Try longer duration, lower --min_face_px, or use --preview to verify framing.")

    # cluster and pick dominant person
    vecs = [s["emb"] for s in samples]
    clusters = greedy_clusters(vecs, sim_thresh=float(args.cluster_sim))
    clusters.sort(key=lambda idxs: len(idxs), reverse=True)
    main_idx = clusters[0]
    chosen = sorted((samples[i] for i in main_idx), key=lambda s: s["qual"], reverse=True)[:max(1,int(args.k))]
    avg = l2norm(np.mean([c["emb"] for c in chosen], axis=0))

    out_path = Path(DIRS.get("EMBEDDINGS", media_root / "embeddings")) / f"{args.hcode}.npy"
    np.save(str(out_path), avg)
    print(f"[OK] wrote embedding: {out_path}  (topK={len(chosen)}/{len(samples)} faces used)")

if __name__ == "__main__":
    main()



# With defaults and auto source selection (prefers RTSP default)
# python extras/capture_embeddings_ffmpeg.py \
#   --hcode H123456 --k 3 --duration 40 --fps 4 --det_size 1024 --device auto \
#   --preview --save_all_crops
# 
# Force the RTSP default (with ! already encoded as %21)
# python extras/capture_embeddings_ffmpeg.py \
#   --use rtsp \
#   --hcode H123456 --k 3 --duration 40 --fps 4 --det_size 1024 --device auto \
#   --preview --save_all_crops
#   
# Override the RTSP on the fly (quoting fixes your bash error)
# python extras/capture_embeddings_ffmpeg.py \
#   --rtsp 'rtsp://admin:B\!sk2025@192.168.137.95:554/Streaming/Channels/101/' \
#   --hcode H123456 --k 3 --duration 40 --fps 4 --det_size 1024 --device auto \
#   --preview --save_all_crops
# 
# Use the webcam default
# python extras/capture_embeddings_ffmpeg.py \
#   --use webcam \
#   --hcode H123456 --k 3 --duration 60 --fps 4 --det_size 1024 --device auto \
#   --min_face_px 40 --preview --save_all_crops

# Option A — Enable OpenCV GUI (recommended if you want windows)
#
# In your venv:
# pip show opencv-python opencv-python-headless
# If you see opencv-python-headless, replace it:
#pip uninstall -y opencv-python-headless
# pip install opencv-python
#
# 2. Install the common system libs on Ubuntu/Debian:
# sudo apt-get update
# sudo apt-get install -y libgtk-3-0 libglib2.0-0 libsm6 libxext6 libxrender1 libgl1
# Make sure you’re not headless: echo $DISPLAY should be non-empty (on your laptop it usually is).
# Re-run with --preview.


# Below Worked:
# python extras/capture_embeddings_ffmpeg.py --use rtsp \
#   --hcode H123456 --k 3 --duration 40 --fps 4 \
#   --det_size 1024 --device auto --save_all_crops