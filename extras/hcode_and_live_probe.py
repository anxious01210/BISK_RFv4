#!/usr/bin/env python3
"""
Embed + Live Probe for a single Hcode (student) with:
- AVG embedding (all/topK with min self-consistency)
- Per-NPY live stats
- Per-Hcode aggregation
- Top-1 distribution + Top-K confusion reports
- Mis-ID events (when --target_hcode provided)

PHASE A (offline):
  Create embeddings next to images, save crops and per-image report,
  compute self-consistency and (optionally) an AVG embedding.

PHASE B (live):
  Stream RTSP via FFmpeg→MJPEG, detect+embed with the same settings,
  match against the gallery (optionally including AVG),
  and write detailed logs/reports for troubleshooting.

Defaults: CUDA-first with CPU fallback, buffalo_l, det_size=1024, min_face_px=40,
RTSP TCP, pipe_q=2, cap_fps=6.

"""

import os, sys, json, csv, time, argparse, subprocess, signal, shutil
from pathlib import Path
from io import BytesIO
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from PIL import Image

try:
    import cv2  # optional for blur metric

    HAVE_CV2 = True
except Exception:
    HAVE_CV2 = False

from insightface.app import FaceAnalysis

# ===================== CONSTANTS (recommended defaults) =====================
MODEL_PACK_DEFAULT = "buffalo_l"
DEVICE_DEFAULT = "auto"  # 'cuda', 'cpu', or 'auto'
DET_SIZE_DEFAULT = 1024
MIN_FACE_PX_DEFAULT = 40
TOPK_DEFAULT = 5

RTSP_TRANSPORT_DEF = "tcp"
PIPE_MJPEG_Q_DEFAULT = 2
CAP_FPS_DEFAULT = 6
FRAME_SKIP_N_DEFAULT = 0
SESSION_SECONDS_DEF = 90
LOG_PREFIX_DEFAULT = "hcode_live_probe"

OVERWRITE_NPY = False
SAVE_CROPS = True
CROP_FORMAT = "jpg"
CROP_JPEG_QUALITY = 95
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# AVG defaults
AVG_MODE_DEFAULT = "topk"  # none | all | topk
AVG_TOPK_DEFAULT = 5
AVG_MIN_SELF_DEFAULT = 0.75


# ===========================================================================
# ---------- FIX: safe getter for face embedding (no boolean eval of numpy arrays) ----------
def face_embedding_from(face):
    """
    Return the embedding array for an InsightFace 'Face' object.
    Prefer `normed_embedding` when available; fall back to `embedding`.
    Avoid boolean evaluation on numpy arrays (never do `a or b`).
    """
    emb = getattr(face, "normed_embedding", None)
    if emb is None:
        emb = getattr(face, "embedding", None)
    return emb

# -------------------- helpers --------------------
def l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v.astype(np.float32) if (not np.isfinite(n) or n <= eps) else (v / n).astype(np.float32)


def providers_for(device: str):
    if device == "cuda": return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if device == "cpu":  return ["CPUExecutionProvider"]
    return ["CUDAExecutionProvider", "CPUExecutionProvider"]


def list_images(d: Path) -> List[Path]:
    return [p for p in sorted(d.glob("*")) if
            p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp") and p.is_file()]


def ensure_dir(d: Path): d.mkdir(parents=True, exist_ok=True)


def variance_of_laplacian_rgb(rgb: np.ndarray) -> Optional[float]:
    if not HAVE_CV2: return None
    try:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        return None


def detect_largest_face(app: FaceAnalysis, rgb: np.ndarray, min_face_px: int):
    faces = app.get(rgb)
    if not faces: return None
    picks = []
    for f in faces:
        x1, y1, x2, y2 = map(int, f.bbox)
        if min(x2 - x1, y2 - y1) >= min_face_px:
            picks.append((f, (x1, y1, x2, y2)))
    if not picks: return None
    picks.sort(key=lambda t: (t[1][2] - t[1][0]) * (t[1][3] - t[1][1]), reverse=True)
    return picks[0]


def img_to_embedding(app: FaceAnalysis, img_path: Path, min_face_px: int):
    try:
        im = Image.open(str(img_path)).convert("RGB")
    except Exception as e:
        return None, None, None, f"open-failed:{e}"
    rgb = np.array(im, dtype=np.uint8)
    pick = detect_largest_face(app, rgb, min_face_px)
    if not pick:
        return rgb, None, None, "no-face"
    f, bbox = pick
    # FIX: do not use `or` with numpy arrays
    emb = face_embedding_from(f)
    if emb is None:
        return rgb, bbox, None, "no-embedding"
    v = l2_normalize(np.asarray(emb, dtype=np.float32))
    if v.shape[0] != 512:
        return rgb, bbox, None, f"bad-dim:{v.shape}"
    return rgb, bbox, v, None


def sidecar_write(npy_path: Path, meta: dict):
    with open(npy_path.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)


def sidecar_model(npy_path: Path) -> Optional[str]:
    js = npy_path.with_suffix(".json")
    if js.exists():
        try:
            return json.loads(js.read_text()).get("model")
        except Exception:
            return None
    return None


def load_gallery_vectors(root: Path, expect_model: str, restrict_hcode: Optional[str] = None):
    gal_dir = root / restrict_hcode if restrict_hcode else root
    if not gal_dir.exists():
        raise SystemExit(f"Gallery dir not found: {gal_dir}")
    npys = sorted(gal_dir.rglob("*.npy"))
    vecs = [];
    metas = []
    for p in npys:
        try:
            v = np.load(str(p)).reshape(-1).astype(np.float32)
        except Exception:
            continue
        if v.shape[0] != 512: continue
        tag = sidecar_model(p)
        if tag and tag != expect_model:  # avoid cross-model mix
            continue
        vecs.append(l2_normalize(v))
        metas.append({"npy": str(p), "hcode": p.parent.name, "stem": p.stem})
    if not vecs:
        raise SystemExit(f"No compatible vectors in {gal_dir} (model={expect_model})")
    G = np.vstack(vecs).astype(np.float32)
    return G, metas


def recommended_det_set(min_side_px: int) -> int:
    if min_side_px is None: return 1024
    if min_side_px < 120: return 1024
    if min_side_px < 200: return 800
    return 640


# FFmpeg/MJPEG helpers
def build_ffmpeg_cmd(rtsp_url: str, transport: str, q: int, fps: float):
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-rtsp_transport", transport, "-i", rtsp_url,
        "-an", "-sn", "-dn",
        "-r", str(fps),
        "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", str(q),
        "pipe:1",
    ]


def iter_mjpeg_frames(proc: subprocess.Popen):
    SOI = b"\xff\xd8";
    EOI = b"\xff\xd9"
    buf = bytearray()
    read = proc.stdout.read
    from PIL import ImageFile;
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    while True:
        chunk = read(4096)
        if not chunk: break
        buf.extend(chunk)
        while True:
            i = buf.find(SOI)
            j = buf.find(EOI, i + 2)
            if i != -1 and j != -1:
                frame = bytes(buf[i:j + 2]);
                del buf[:j + 2]
                try:
                    yield Image.open(BytesIO(frame)).convert("RGB")
                except Exception:
                    continue
            else:
                break


# AVG helpers
def self_consistency_scores(V: np.ndarray) -> np.ndarray:
    N = V.shape[0]
    if N == 1: return np.ones((1,), dtype=np.float32)
    mean_all = l2_normalize(np.mean(V, axis=0))
    scores = np.empty((N,), dtype=np.float32)
    for i in range(N):
        centroid = l2_normalize(((mean_all * N) - V[i]) / max(1, N - 1))
        scores[i] = float(np.dot(V[i], centroid))
    return scores


def build_avg_vector(selected: List[np.ndarray]) -> Optional[np.ndarray]:
    if not selected: return None
    return l2_normalize(np.mean(np.stack(selected, axis=0), axis=0))


# ---------- NEW: sync freshly created/reused npys into gallery/<model>/<hcode> ----------
def sync_to_gallery(student_dir: Path, gallery_root_model: Path, hcode: str):
    """
    Copy all .npy and .json sidecars from the student's folder into
    gallery_root_model/<hcode>/ so the live phase always finds vectors.
    """
    dest = gallery_root_model / hcode
    dest.mkdir(parents=True, exist_ok=True)
    for p in sorted(student_dir.glob("*.npy")):
        tgt = dest / p.name
        tgt.write_bytes(p.read_bytes())
        js = p.with_suffix(".json")
        if js.exists():
            (dest / js.name).write_bytes(js.read_bytes())
    # also copy avg vector if present
    for p in sorted(student_dir.glob("*_avg.npy")):
        tgt = dest / p.name
        tgt.write_bytes(p.read_bytes())
        js = p.with_suffix(".json")
        if js.exists():
            (dest / js.name).write_bytes(js.read_bytes())
    return dest



# ===================== main =====================
def main():
    ap = argparse.ArgumentParser("hcode_and_live_probe")
    # mandatory
    ap.add_argument("--student_dir", required=True, help="Path to Hcode folder with images")
    ap.add_argument("--images", nargs="*", default=None)
    # live stream
    ap.add_argument("--rtsp", default=None, help="RTSP URL (if omitted, only offline phase runs)")
    ap.add_argument("--gallery_root", default="media/embeddings_offline",
                    help="Root gallery base (contains <model>/<hcode>/ ...)")
    ap.add_argument("--target_hcode", default=None,
                    help="Restrict live gallery to this Hcode (default: student_dir name)")
    # model/device/detector
    ap.add_argument("--model", default=MODEL_PACK_DEFAULT)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default=DEVICE_DEFAULT)
    ap.add_argument("--det_size", type=int, default=DET_SIZE_DEFAULT)
    ap.add_argument("--min_face_px", type=int, default=MIN_FACE_PX_DEFAULT)
    ap.add_argument("--topk", type=int, default=TOPK_DEFAULT)
    # live pipe & pacing
    ap.add_argument("--rtsp_transport", choices=["tcp", "udp"], default=RTSP_TRANSPORT_DEF)
    ap.add_argument("--pipe_q", type=int, default=PIPE_MJPEG_Q_DEFAULT)
    ap.add_argument("--cap_fps", type=float, default=CAP_FPS_DEFAULT)
    ap.add_argument("--frame_skip_n", type=int, default=FRAME_SKIP_N_DEFAULT)
    ap.add_argument("--seconds", type=int, default=SESSION_SECONDS_DEF)
    ap.add_argument("--log_prefix", default=LOG_PREFIX_DEFAULT)
    # recognition control
    ap.add_argument("--threshold", type=float, default=None, help="Recognition threshold for 'recognized' (0..1)")
    ap.add_argument("--stop_on_first_above", action="store_true", default=False,
                    help="Stop when best >= threshold for target_hcode (requires --threshold)")
    # embedding writes
    ap.add_argument("--overwrite_npy", action="store_true", default=OVERWRITE_NPY)
    ap.add_argument("--save_crops", action="store_true", default=SAVE_CROPS)
    ap.add_argument("--crop_format", choices=["jpg", "png"], default=CROP_FORMAT)
    ap.add_argument("--crop_quality", type=int, default=CROP_JPEG_QUALITY)
    # AVG controls
    ap.add_argument("--avg_mode", choices=["none", "all", "topk"], default=AVG_MODE_DEFAULT)
    ap.add_argument("--avg_topk", type=int, default=AVG_TOPK_DEFAULT)
    ap.add_argument("--avg_min_self", type=float, default=AVG_MIN_SELF_DEFAULT)
    ap.add_argument("--avg_out_name", default=None, help="filename for avg npy (default: <HCODE>_avg.npy)")
    ap.add_argument("--avg_copy_to_gallery", action="store_true", default=False)
    ap.add_argument("--use_avg_in_live", action="store_true", default=False)
    # NEW: always sync npys to gallery; allow turning off if needed
    ap.add_argument("--no_sync_to_gallery", action="store_true", default=False,
                    help="Disable auto-copy of .npy/.json into gallery_root/<model>/<hcode>/")

    args = ap.parse_args()

    student_dir = Path(args.student_dir)
    if not student_dir.exists(): sys.exit(f"Student dir not found: {student_dir}")
    hcode = student_dir.name
    target_hcode = args.target_hcode or hcode

    # init face app
    provs = providers_for(args.device)
    app = FaceAnalysis(name=args.model, providers=provs)
    ctx = 0 if ("CUDAExecutionProvider" in provs) else -1
    app.prepare(ctx_id=ctx, det_size=(args.det_size, args.det_size))

    # ---------- PHASE A: Offline embed + per-image report ----------
    images = [student_dir / n for n in args.images] if args.images else list_images(student_dir)
    crops_dir = student_dir / "_crops"
    if args.save_crops: ensure_dir(crops_dir)

    report_csv = student_dir / f"{hcode}_image_report.csv"
    rows = []
    created = 0
    emb_bank: Dict[str, np.ndarray] = {}
    bbox_bank: Dict[str, Tuple[int, int, int, int]] = {}
    blur_bank: Dict[str, Optional[float]] = {}
    minside_bank: Dict[str, Optional[int]] = {}

    for img_path in images:
        npy_path = img_path.with_suffix(".npy")
        use_prev = (npy_path.exists() and not args.overwrite_npy)
        v = None;
        bbox = None;
        rgb = None;
        err = None

        if use_prev:
            try:
                v = np.load(str(npy_path)).reshape(-1).astype(np.float32)
                if v.shape[0] != 512: v = None
            except Exception:
                v = None

        if v is None:
            rgb, bbox, v, err = img_to_embedding(app, img_path, args.min_face_px)
            if v is not None:
                np.save(str(npy_path), v)
                meta = {
                    "model": args.model, "det_size": args.det_size, "device": args.device,
                    "src_image": str(img_path.name),
                    "bbox": (None if bbox is None else {"x1": int(bbox[0]), "y1": int(bbox[1]), "x2": int(bbox[2]),
                                                        "y2": int(bbox[3])}),
                    "created_at_ms": int(time.time() * 1000),
                    "min_face_px": args.min_face_px
                }
                sidecar_write(npy_path, meta)
                if args.save_crops and (rgb is not None) and (bbox is not None):
                    x1, y1, x2, y2 = bbox
                    crop = Image.fromarray(rgb[y1:y2, x1:x2])
                    ext = ".png" if args.crop_format == "png" else ".jpg"
                    crop.save(str((crops_dir / img_path.stem).with_suffix(ext)),
                              format=("PNG" if args.crop_format == "png" else "JPEG"),
                              quality=(args.crop_quality if args.crop_format == "jpg" else None))
                created += 1

        # record metrics
        if rgb is None and npy_path.with_suffix(".json").exists():
            try:
                jb = json.loads(npy_path.with_suffix(".json").read_text()).get("bbox")
                if jb: bbox = (int(jb["x1"]), int(jb["y1"]), int(jb["x2"]), int(jb["y2"]))
            except Exception:
                pass
        blur = variance_of_laplacian_rgb(rgb) if rgb is not None else None
        min_side = (min(bbox[2] - bbox[0], bbox[3] - bbox[1]) if bbox is not None else None)
        det_reco = recommended_det_set(min_side)
        rows.append([
            img_path.name,
            (npy_path.name if npy_path.exists() else ""),
            ("" if bbox is None else f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"),
            ("" if min_side is None else str(min_side)),
            ("" if blur is None else f"{blur:.1f}"),
            str(det_reco),
            ("created" if not use_prev and v is not None else (
                "reused" if use_prev and v is not None else (err or "error")))
        ])

        # banks
        if v is not None:
            emb_bank[img_path.name] = l2_normalize(v)
        if bbox is not None:
            bbox_bank[img_path.name] = bbox
            minside_bank[img_path.name] = min_side
        blur_bank[img_path.name] = blur

    with open(report_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["image", "embedding", "bbox(x1,y1,x2,y2)", "min_side_px", "blur(LoG)", "recommended_det_set", "status"])
        w.writerows(rows)
    print(f"[offline] wrote image report → {report_csv} (created {created} embeddings)")

    # ---------- AVG embedding ----------
    def self_consistency_scores(V: np.ndarray) -> np.ndarray:
        N = V.shape[0]
        if N == 1: return np.ones((1,), dtype=np.float32)
        mean_all = l2_normalize(np.mean(V, axis=0))
        scores = np.empty((N,), dtype=np.float32)
        for i in range(N):
            centroid = l2_normalize(((mean_all * N) - V[i]) / max(1, N - 1))
            scores[i] = float(np.dot(V[i], centroid))
        return scores

    def build_avg_vector(selected: List[np.ndarray]) -> Optional[np.ndarray]:
        if not selected: return None
        return l2_normalize(np.mean(np.stack(selected, axis=0), axis=0))

    avg_used = []
    avg_vec = None
    if args.avg_mode != "none" and emb_bank:
        names = sorted(emb_bank.keys())
        V = np.stack([emb_bank[n] for n in names], axis=0).astype(np.float32)
        scons = self_consistency_scores(V)
        order = np.argsort(scons)[::-1]

        if args.avg_mode == "all":
            sel_idx = list(range(len(names)))
        else:
            k = max(1, int(args.avg_topk))
            sel_idx = list(order[:k])
            if args.avg_min_self is not None:
                sel_idx = [i for i in sel_idx if scons[i] >= float(args.avg_min_self)]

        selected = [V[i] for i in sel_idx]
        avg_vec = build_avg_vector(selected)
        avg_used = [{
            "image": names[i],
            "self_score": float(scons[i]),
            "min_side_px": (None if names[i] not in minside_bank else int(minside_bank[names[i]] or 0)),
            "blur_LoG": (None if names[i] not in blur_bank else (
                None if blur_bank[names[i]] is None else float(blur_bank[names[i]])))
        } for i in sel_idx]

        if avg_vec is not None:
            avg_name = args.avg_out_name or f"{hcode}_avg.npy"
            avg_path = student_dir / avg_name
            np.save(str(avg_path), avg_vec.astype(np.float32))
            sidecar_write(avg_path, {
                "model": args.model,
                "det_size": args.det_size,
                "device": args.device,
                "created_at_ms": int(time.time() * 1000),
                "mode": args.avg_mode,
                "topk": (args.avg_topk if args.avg_mode == "topk" else None),
                "min_self": args.avg_min_self,
                "used_count": len(sel_idx),
                "used": avg_used
            })
            print(f"[avg] wrote {avg_path} (used {len(sel_idx)} images)")

            if args.avg_copy_to_gallery:
                gal_dir = Path(args.gallery_root) / args.model / hcode
                ensure_dir(gal_dir)
                dst = gal_dir / avg_name
                shutil.copy2(str(avg_path), str(dst))
                js_src = avg_path.with_suffix(".json")
                js_dst = dst.with_suffix(".json")
                if js_src.exists():
                    shutil.copy2(str(js_src), str(js_dst))
                print(f"[avg] copied to gallery → {dst}")

    # ---------- Build gallery for live ----------
    gallery_root = Path(args.gallery_root) / args.model
    ensure_dir(gallery_root / target_hcode)
    # NEW: auto-sync all npys to gallery unless disabled
    if not args.no_sync_to_gallery:
        synced_dir = sync_to_gallery(student_dir, gallery_root, hcode)
        print(f"[sync] copied vectors to {synced_dir}")
    # After sync, always load from gallery_root/<hcode>
    gal_base = gallery_root
    G, metas = load_gallery_vectors(gal_base, expect_model=args.model, restrict_hcode=target_hcode)

    if args.use_avg_in_live and avg_vec is not None:
        G = np.vstack([G, avg_vec.reshape(1, -1)]).astype(np.float32)
        metas.append({"npy": str((student_dir / (args.avg_out_name or f'{hcode}_avg.npy')).resolve()),
                      "hcode": hcode, "stem": (args.avg_out_name or f"{hcode}_avg")})
    Gt = G.T  # [512, N]

    # ---------- PHASE B: Live probe ----------
    if not args.rtsp:
        print("[live] RTSP not provided; skipping live probe.")
        return

    tstamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = Path(f"{args.log_prefix}_{hcode}_{tstamp}.jsonl")
    summary_path = Path(f"{args.log_prefix}_{hcode}_{tstamp}_summary.json")
    per_npy_csv = Path(f"{args.log_prefix}_{hcode}_{tstamp}_live_per_npy.csv")
    per_hcode_csv = Path(f"{args.log_prefix}_{hcode}_{tstamp}_live_per_hcode.csv")
    top1_csv = Path(f"{args.log_prefix}_{hcode}_{tstamp}_live_top1_distribution.csv")
    confusion_csv = Path(f"{args.log_prefix}_{hcode}_{tstamp}_live_topk_confusion.csv")
    misid_csv = Path(f"{args.log_prefix}_{hcode}_{tstamp}_live_misid_events.csv")
    log_f = open(log_path, "w")

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-rtsp_transport", args.rtsp_transport,
        "-i", args.rtsp,
        "-an", "-sn", "-dn",
        "-r", str(args.cap_fps),
        "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", str(args.pipe_q),
        "pipe:1"
    ]
    print("[ffmpeg]", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)

    stop_at = time.time() + max(1, args.seconds)
    processed = 0
    detections = 0
    best_scores = []
    last_print = 0
    frame_idx = 0

    # per-npy stats
    npy_stats: Dict[str, Dict[str, Any]] = {m["npy"]: {
        "hcode": m["hcode"],
        "best_score": None,
        "mean_score": 0.0,
        "seen_count": 0,
        "times_top1": 0,
        "times_in_topk": 0,
        "best_frame_idx": None,
        "best_ts_ms": None,
    } for m in metas}

    # per-hcode aggregation
    hcode_stats: Dict[str, Dict[str, Any]] = {}

    # NEW: top1 distribution and topK confusion accumulators
    top1_stats: Dict[str, Dict[str, Any]] = {}  # hcode -> {count, sum_score, max_score}
    confusion_pairs: Dict[
        Tuple[str, str], Dict[str, Any]] = {}  # (top1_hcode, other_hcode) -> {count, sum_score, max_score}

    try:
        for im in iter_mjpeg_frames(proc):
            frame_idx += 1
            if args.frame_skip_n and ((frame_idx - 1) % (args.frame_skip_n + 1) != 0):
                continue

            rgb = np.array(im, dtype=np.uint8)
            faces = app.get(rgb)
            picks = []
            for f in faces:
                x1, y1, x2, y2 = map(int, f.bbox)
                if min(x2 - x1, y2 - y1) >= args.min_face_px:
                    picks.append((f, (x1, y1, x2, y2)))
            picks.sort(key=lambda t: (t[1][2] - t[1][0]) * (t[1][3] - t[1][1]), reverse=True)

            now = time.time()
            recs = []
            for f, bbox in picks:
                # FIX: do not use `or` with numpy arrays
                emb = face_embedding_from(f)
                if emb is None: continue
                v = l2_normalize(np.asarray(emb, dtype=np.float32))
                if v.shape[0] != 512: continue

                sims = (v @ Gt).astype(float)  # [N]
                j_top = int(np.argmax(sims))
                best = float(sims[j_top])
                best_scores.append(best)
                detections += 1

                # per-npy updates (all N)
                for j, m in enumerate(metas):
                    s = float(sims[j])
                    st = npy_stats[m["npy"]]
                    st["seen_count"] += 1
                    n = st["seen_count"]
                    st["mean_score"] = st["mean_score"] + (s - st["mean_score"]) / n
                    if (st["best_score"] is None) or (s > st["best_score"]):
                        st["best_score"] = s
                        st["best_frame_idx"] = frame_idx
                        st["best_ts_ms"] = int(now * 1000)

                # top-k details
                order = np.argsort(sims)[-args.topk:][::-1]
                topk_pairs = [({"npy": metas[j]["npy"], "hcode": metas[j]["hcode"]}, float(sims[j])) for j in order]
                for j in order:
                    npy_stats[metas[j]["npy"]]["times_in_topk"] += 1
                npy_stats[metas[j_top]["npy"]]["times_top1"] += 1

                # ----- NEW: top1 distribution & topK confusion -----
                top1_hc = metas[j_top]["hcode"]
                t1 = top1_stats.setdefault(top1_hc, {"count": 0, "sum_score": 0.0, "max_score": None})
                t1["count"] += 1
                t1["sum_score"] += best
                t1["max_score"] = best if (t1["max_score"] is None or best > t1["max_score"]) else t1["max_score"]

                # confusion pairs: for each OTHER item in topK (excluding the top1 itself)
                for j in order:
                    if j == j_top: continue
                    other_hc = metas[j]["hcode"]
                    key = (top1_hc, other_hc)
                    cp = confusion_pairs.setdefault(key, {"count": 0, "sum_score": 0.0, "max_score": None})
                    score = float(sims[j])
                    cp["count"] += 1
                    cp["sum_score"] += score
                    cp["max_score"] = score if (cp["max_score"] is None or score > cp["max_score"]) else cp["max_score"]

                recs.append({
                    "bbox": {"x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3]},
                    "best": {"hcode": top1_hc, "npy": metas[j_top]["npy"], "score": best},
                    "topk": [{"hcode": p[0]["hcode"], "npy": p[0]["npy"], "score": p[1]} for p in topk_pairs]
                })

                # threshold / early stop
                if args.threshold is not None and args.stop_on_first_above:
                    if metas[j_top]["hcode"] == target_hcode and best >= float(args.threshold):
                        print(f"[stop] first match for {target_hcode} above {args.threshold:.3f}: {best:.3f}")
                        raise KeyboardInterrupt

                # Mis-ID event log (only if we know the intended target)
                if args.target_hcode and metas[j_top]["hcode"] != target_hcode:
                    # write one row per detection that did not pick the target
                    with open(misid_csv, "a", newline="") as mf:
                        mw = csv.writer(mf)
                        if mf.tell() == 0:
                            mw.writerow(["ts_ms", "frame_idx", "pred_hcode", "pred_score", "target_hcode", "threshold"])
                        mw.writerow([int(now * 1000), frame_idx, metas[j_top]["hcode"], f"{best:.6f}", target_hcode,
                                     ("" if args.threshold is None else args.threshold)])

            processed += 1

            if recs and (now - last_print) > 0.5:
                s = ", ".join([f'{r["best"]["hcode"]}:{r["best"]["score"]:.3f}' for r in recs])
                print(f"[{processed}] {s}")
                last_print = now

            if recs:
                log_f.write(json.dumps({
                    "ts": int(now * 1000),
                    "frame_idx": frame_idx,
                    "detections": len(recs),
                    "results": recs
                }) + "\n")
                log_f.flush()

            if now >= stop_at:
                break

    except KeyboardInterrupt:
        print("\n[stop] session ended")

    finally:
        try:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=1.5)
            except subprocess.TimeoutExpired:
                proc.kill()
        except Exception:
            pass
        log_f.close()

    # aggregate per-hcode from per-npy stats
    hcode_stats: Dict[str, Dict[str, Any]] = {}
    for npy, st in npy_stats.items():
        hc = st["hcode"]
        g = hcode_stats.setdefault(hc, {
            "hcode": hc,
            "items": 0,
            "best_score_max": None,
            "mean_score_mean": 0.0,
            "seen_count_sum": 0,
            "times_top1_sum": 0,
            "times_in_topk_sum": 0
        })
        g["items"] += 1
        g["seen_count_sum"] += st["seen_count"]
        g["times_top1_sum"] += st["times_top1"]
        g["times_in_topk_sum"] += st["times_in_topk"]
        if st["best_score"] is not None:
            g["best_score_max"] = (
                st["best_score"] if (g["best_score_max"] is None or st["best_score"] > g["best_score_max"]) else g[
                    "best_score_max"])
        n = g["items"]
        g["mean_score_mean"] = g["mean_score_mean"] + (st["mean_score"] - g["mean_score_mean"]) / n

    # write per-npy CSV
    with open(per_npy_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["npy", "hcode", "best_score", "mean_score", "seen_count", "times_top1", "times_in_topk", "best_frame_idx",
             "best_ts_ms"])
        for npy, st in sorted(npy_stats.items()):
            w.writerow([
                npy, st["hcode"],
                ("" if st["best_score"] is None else f"{st['best_score']:.6f}"),
                f"{st['mean_score']:.6f}",
                st["seen_count"], st["times_top1"], st["times_in_topk"],
                ("" if st["best_frame_idx"] is None else st["best_frame_idx"]),
                ("" if st["best_ts_ms"] is None else st["best_ts_ms"])
            ])

    # write per-hcode CSV
    with open(per_hcode_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["hcode", "items", "best_score_max", "mean_score_mean", "seen_count_sum", "times_top1_sum",
                    "times_in_topk_sum"])
        for hc, g in sorted(hcode_stats.items()):
            w.writerow([
                hc, g["items"],
                ("" if g["best_score_max"] is None else f"{g['best_score_max']:.6f}"),
                f"{g['mean_score_mean']:.6f}",
                g["seen_count_sum"], g["times_top1_sum"], g["times_in_topk_sum"]
            ])

    # write top-1 distribution
    with open(top1_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["hcode", "top1_count", "top1_max_score", "top1_mean_score"])
        for hc, st in sorted(top1_stats.items()):
            mean = (st["sum_score"] / st["count"]) if st["count"] else 0.0
            w.writerow([hc, st["count"], ("" if st["max_score"] is None else f"{st['max_score']:.6f}"), f"{mean:.6f}"])

    # write top-K confusion pairs
    with open(confusion_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["top1_hcode", "other_hcode", "count_in_topk", "avg_score", "max_score"])
        for (a, b), st in sorted(confusion_pairs.items()):
            avg = (st["sum_score"] / st["count"]) if st["count"] else 0.0
            w.writerow([a, b, st["count"], f"{avg:.6f}", ("" if st["max_score"] is None else f"{st['max_score']:.6f}")])

    # summary
    summary = {
        "hcode": hcode,
        "model": args.model,
        "device": args.device,
        "providers": provs,
        "det_size": args.det_size,
        "min_face_px": args.min_face_px,
        "rtsp_transport": args.rtsp_transport,
        "pipe_mjpeg_q": args.pipe_q,
        "cap_fps": args.cap_fps,
        "frame_skip_n": args.frame_skip_n,
        "gallery_root": str((Path(args.gallery_root) / args.model).resolve()),
        "gallery_items": int(G.shape[0]),
        "frames_processed": processed,
        "detections": detections,
        "best_score_max": (float(np.max(best_scores)) if best_scores else None),
        "best_score_mean": (float(np.mean(best_scores)) if best_scores else None),
        "best_score_min": (float(np.min(best_scores)) if best_scores else None),
        "log_jsonl": str(log_path.resolve()),
        "per_npy_csv": str(per_npy_csv.resolve()),
        "per_hcode_csv": str(per_hcode_csv.resolve()),
        "top1_csv": str(top1_csv.resolve()),
        "confusion_csv": str(confusion_csv.resolve()),
        "misid_csv": (str(misid_csv.resolve()) if args.target_hcode else None),
        # AVG echo
        "avg_mode": args.avg_mode,
        "avg_topk": (args.avg_topk if args.avg_mode == "topk" else None),
        "avg_min_self": args.avg_min_self,
        "avg_in_live": bool(args.use_avg_in_live),
        # thresholding
        "threshold": args.threshold,
        "stop_on_first_above": args.stop_on_first_above
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"[done] live log: {log_path}")
    print(f"[done] per-npy: {per_npy_csv}")
    print(f"[done] per-hcode: {per_hcode_csv}")
    print(f"[done] top1 distribution: {top1_csv}")
    print(f"[done] topK confusion: {confusion_csv}")
    if args.target_hcode: print(f"[done] mis-id events: {misid_csv}")
    print(f"[done] summary: {summary_path}")
    if detections:
        print(
            f"Scores: max={np.max(best_scores):.4f} mean={np.mean(best_scores):.4f} min={np.min(best_scores):.4f}  detections={detections}  frames={processed}")


if __name__ == "__main__":
    main()
