# apps/attendance/utils/facescore.py
from __future__ import annotations
import os, math, json, traceback
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np
from django.conf import settings
from insightface.app import FaceAnalysis

# Local singleton (kept separate from embeddings.py to not disturb its app)
_PROVIDERS = ("CUDAExecutionProvider", "CPUExecutionProvider")
_APP: Optional[FaceAnalysis] = None
_APP_CTX_ID = 0
_APP_DET_SIZE = (settings.FACE_DET_SIZE_DEFAULT, settings.FACE_DET_SIZE_DEFAULT)


def _get_app(det_size: Tuple[int, int] | None = None) -> FaceAnalysis:
    global _APP, _APP_DET_SIZE
    if _APP is None:
        _APP = FaceAnalysis(name="buffalo_l", providers=_PROVIDERS)
        _APP.prepare(ctx_id=_APP_CTX_ID, det_size=_APP_DET_SIZE)
    if det_size and det_size != _APP_DET_SIZE:
        _APP_DET_SIZE = det_size
        _APP.prepare(ctx_id=_APP_CTX_ID, det_size=_APP_DET_SIZE)
    return _APP


def _provider_short_from_app(app: FaceAnalysis) -> str:
    """
    Try to infer current provider ('cuda' or 'cpu') without assuming a specific InsightFace API.
    1) Look at any loaded model's onnxruntime session providers.
    2) Fallback to onnxruntime.get_device().
    3) Else 'unknown'.
    """
    # 1) peek all model sessions, if available
    try:
        models = getattr(app, "models", None)
        if isinstance(models, dict):
            provs = []
            for m in models.values():
                sess = getattr(m, "session", None)
                if sess and hasattr(sess, "get_providers"):
                    provs.extend(sess.get_providers() or [])
            if provs:
                return "cuda" if any("CUDA" in p for p in provs) else "cpu"
    except Exception:
        pass
    # 2) onnxruntime global device
    try:
        import onnxruntime as ort
        dev = str(getattr(ort, "get_device", lambda: "CPU")()).upper()
        return "cuda" if "GPU" in dev else "cpu"
    except Exception:
        pass
    return "unknown"


def _lap_var(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _mean_intensity(gray: np.ndarray) -> float:
    return float(gray.mean())


def _clip(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def _pad_box(x: int, y: int, w: int, h: int, margin: float, W: int, H: int) -> Tuple[int, int, int, int]:
    if margin <= 0: return x, y, w, h
    dx = int(round(w * margin))
    dy = int(round(h * margin))
    x2 = _clip(x - dx, 0, W)
    y2 = _clip(y - dy, 0, H)
    x3 = _clip(x + w + dx, 0, W)
    y3 = _clip(y + h + dy, 0, H)
    return x2, y2, max(1, x3 - x2), max(1, y3 - y2)


def detect_with_cascade(
        bgr: np.ndarray,
        start_size: int | Tuple[int, int] = None,
        cascade_sizes: List[int] = None,
) -> Tuple[Optional[Tuple[int, int, int, int]], float, int, str]:
    """
    Returns: (bbox(x,y,w,h) or None, conf, det_size_used, provider_str)
    Escalates det_size only if detection looks weak for this image.
    """
    H, W = bgr.shape[:2]
    start = start_size or getattr(settings, "FACE_DET_SIZE_DEFAULT", 640)
    if isinstance(start, int): start = (start, start)
    cascade = list(cascade_sizes or getattr(settings, "FACE_DET_CASCADE_SIZES", []))
    # Build a plan: try start first, then cascade (dedup, keep order)
    sizes: List[Tuple[int, int]] = []
    seen = set()
    for s in [start] + [(s, s) if isinstance(s, int) else tuple(s) for s in cascade]:
        if s not in seen:
            seen.add(s);
            sizes.append(s)

    min_conf = float(getattr(settings, "FACE_DET_MIN_CONF", 0.30))
    min_frac = float(getattr(settings, "FACE_DET_MIN_FRAC", 0.04))
    good_conf = float(getattr(settings, "FACE_DET_GOOD_CONF", 0.45))
    max_steps = int(getattr(settings, "FACE_DET_MAX_ESCL_STEPS", 5))
    allow_cpu = bool(getattr(settings, "FACE_DET_ALLOW_CPU_FALLBACK", True))

    tries = 0
    best = (None, 0.0, sizes[0][0], "unknown")  # bbox, conf, det, provider
    for det_w, det_h in sizes:
        tries += 1
        if tries > max_steps: break

        app = _get_app((det_w, det_h))
        provider = _provider_short_from_app(app)

        faces = app.get(bgr)
        if not faces:
            # If totally empty detection, escalate and continue
            best = (None, 0.0, det_w, provider)
            continue

        # pick largest face
        f = max(faces, key=lambda ff: (ff.bbox[2] - ff.bbox[0]) * (ff.bbox[3] - ff.bbox[1]))
        x1, y1, x2, y2 = [int(round(v)) for v in f.bbox]
        w, h = max(1, x2 - x1), max(1, y2 - y1)
        conf = float(getattr(f, "det_score", 0.0))
        frac = min(w, h) / float(min(W, H))
        best = ((x1, y1, w, h), conf, det_w, provider)

        # Decide whether to stop or escalate
        if conf >= good_conf and frac >= min_frac:
            break  # good enough, stop
        if conf < min_conf or frac < min_frac:
            # escalate to next size
            continue
        # confidence between [min_conf, good_conf): try next, else stop now
        continue

    return best


def score_images(
        paths: List[str],
        use_face_roi: bool | None = None,
        start_size: int | None = None,
        cascade_sizes: List[int] | None = None,
        roi_margin: float | None = None
) -> List[Dict]:
    """
    Returns a list of dicts: {name, path, score, sharp, bright, roi, bbox, det_size, det_conf, provider}
    Min–max normalization is done per batch of paths (i.e., per student gallery).
    """
    use_face_roi = getattr(settings, "FACE_SCORE_USE_FACE_ROI", True) if use_face_roi is None else bool(use_face_roi)
    start_size = start_size or getattr(settings, "FACE_DET_SIZE_DEFAULT", 640)
    cascade_sizes = cascade_sizes or list(getattr(settings, "FACE_DET_CASCADE_SIZES", []))
    roi_margin = roi_margin if roi_margin is not None else float(getattr(settings, "FACE_SCORE_ROI_MARGIN", 0.20))

    ws = getattr(settings, "FACE_SCORE_WEIGHTS", {"sharp": 0.7, "bright": 0.3})
    w_sharp = float(ws.get("sharp", 0.7))
    w_bright = float(ws.get("bright", 0.3))
    norm_mode = str(getattr(settings, "FACE_SCORE_NORMALIZE", "pctl")).lower()
    eps = float(getattr(settings, "FACE_SCORE_EPSILON", 0.05))  # shrink extremes to [eps, 1-eps]


    rows = []
    for p in paths:
        try:
            bgr = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
            if bgr is None: continue
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

            bbox = None;
            conf = 0.0;
            det_used = int(start_size);
            provider = "unknown"
            roi = False;
            crop_gray = gray

            if use_face_roi:
                bbox, conf, det_used, provider = detect_with_cascade(bgr, start_size, cascade_sizes)
                if bbox:
                    x, y, w, h = bbox
                    x2, y2, w2, h2 = _pad_box(x, y, w, h, roi_margin, bgr.shape[1], bgr.shape[0])
                    crop_gray = gray[y2:y2 + h2, x2:x2 + w2]
                    roi = True

            sharp = _lap_var(crop_gray)
            bright = _mean_intensity(crop_gray)
            # pack raw first; we'll normalize later
            rows.append({
                "name": os.path.basename(p),
                "path": p,
                "sharp_raw": sharp,
                "bright_raw": bright,
                "roi": roi,
                "bbox": bbox,
                "det_size": det_used,
                "det_conf": conf,
                "provider": provider,
            })
        except Exception:
            # skip unreadable file
            continue

    if not rows:
        return []

    # robust per-batch normalization (percentiles by default)
    svals = [r["sharp_raw"] for r in rows]
    bvals = [r["bright_raw"] for r in rows]
    if norm_mode == "pctl" and len(rows) >= 4:
        s_lo, s_hi = np.percentile(svals, [10, 90])
        b_lo, b_hi = np.percentile(bvals, [10, 90])
    else:
        s_lo, s_hi = min(svals), max(svals)
        b_lo, b_hi = min(bvals), max(bvals)

    for r in rows:
        s_norm = 0.0 if s_hi == s_lo else (r["sharp_raw"] - s_lo) / (s_hi - s_lo)
        b_norm = 0.0 if b_hi == b_lo else (r["bright_raw"] - b_lo) / (b_hi - b_lo)
        # clamp to [0,1] and soften extremes so tiny galleries don’t yield 0.00/1.00
        s_norm = max(0.0, min(1.0, s_norm))
        b_norm = max(0.0, min(1.0, b_norm))
        if 0.0 <= eps < 0.5:
            s_norm = eps + (1.0 - 2 * eps) * s_norm
            b_norm = eps + (1.0 - 2 * eps) * b_norm
        r["sharp"] = s_norm
        r["bright"] = b_norm
        r["score"] = w_sharp * s_norm + w_bright * b_norm
        # cleanup raw fields to keep JSON lean
        r.pop("sharp_raw", None);
        r.pop("bright_raw", None)

    # sort descending by score (most useful for previews)
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows


def crop_face_to_file(
        src_path: str,
        dst_path: str,
        start_size: int | None = None,
        cascade_sizes: List[int] | None = None,
        margin: float | None = None,
        jpeg_quality: int | None = None,
        enhance: bool | None = None,
) -> Dict:
    """
    Detect face (with cascade) and save a padded crop to dst_path.
    Returns metadata dict and success flag.
    """
    start_size = start_size or getattr(settings, "FACE_DET_SIZE_DEFAULT", 640)
    cascade_sizes = cascade_sizes or list(getattr(settings, "FACE_DET_CASCADE_SIZES", []))
    margin = float(margin if margin is not None else getattr(settings, "FACE_INTAKE_CROP_MARGIN", 0.20))
    jpeg_quality = int(jpeg_quality if jpeg_quality is not None else getattr(settings, "FACE_INTAKE_JPEG_QUALITY", 92))
    enhance = bool(enhance if enhance is not None else getattr(settings, "FACE_INTAKE_ENHANCE", False))

    meta = {"ok": False, "src": src_path, "dst": dst_path}
    bgr = cv2.imdecode(np.fromfile(src_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None: return meta
    bbox, conf, det_used, provider = detect_with_cascade(bgr, start_size, cascade_sizes)
    if not bbox: return meta
    x, y, w, h = bbox
    x2, y2, w2, h2 = _pad_box(x, y, w, h, margin, bgr.shape[1], bgr.shape[0])
    crop = bgr[y2:y2 + h2, x2:x2 + w2]
    # optional face enhancement (mild & safe)
    if enhance:
        crop = _enhance_face_bgr(crop)
    # write JPEG robustly (imencode + manual write)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    ok = _write_jpeg(crop, dst_path, jpeg_quality)
    meta.update({"ok": bool(ok), "bbox": [x, y, w, h], "det_size": det_used, "det_conf": conf, "provider": provider})

    return meta


def _write_jpeg(bgr: np.ndarray, dst: str, quality: int) -> bool:
    try:
        ret, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ret:
            return False
        with open(dst, "wb") as f:
            f.write(buf.tobytes())
        return True
    except Exception:
        return False


def _enhance_face_bgr(bgr: np.ndarray) -> np.ndarray:
    """Light enhancement on ROI crop: normalize brightness, CLAHE on L, mild unsharp."""
    # brightness normalize (linear gain)
    alpha_lo, alpha_hi = getattr(settings, "FACE_ENHANCE_ALPHA_LIMITS", (0.6, 1.8))
    target = float(getattr(settings, "FACE_ENHANCE_BRIGHT_TARGET", 0.60))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    m = max(1e-6, gray.mean() / 255.0)
    alpha = np.clip(target / m, alpha_lo, alpha_hi)
    bgr = cv2.convertScaleAbs(bgr, alpha=alpha, beta=0)

    # CLAHE on L in LAB
    cla = float(getattr(settings, "FACE_ENHANCE_CLAHE_CLIP", 2.0))
    tile = int(getattr(settings, "FACE_ENHANCE_CLAHE_TILE", 8))
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=cla, tileGridSize=(tile, tile))
    L2 = clahe.apply(L)
    bgr = cv2.cvtColor(cv2.merge((L2, A, B)), cv2.COLOR_LAB2BGR)

    # mild unsharp mask
    amount = float(getattr(settings, "FACE_ENHANCE_SHARP_AMOUNT", 0.5))
    sigma = float(getattr(settings, "FACE_ENHANCE_SHARP_SIGMA", 1.0))
    if amount > 0:
        blur = cv2.GaussianBlur(bgr, (0, 0), sigma)
        bgr = cv2.addWeighted(bgr, 1.0 + amount, blur, -amount, 0)
    return bgr
