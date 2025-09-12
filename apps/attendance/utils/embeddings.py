# apps/attendance/utils/embeddings.py
from __future__ import annotations
import os, time, hashlib, json
from typing import Optional, List, Tuple
from django.db import transaction
import cv2
import numpy as np
from django.conf import settings
from django.utils import timezone
from datetime import datetime

from insightface.app import FaceAnalysis
from .facescore import detect_with_cascade  # use same cascade policy as scoring

from .media_paths import DIRS, student_gallery_dir
from .facescore import score_images  # NEW: ROI scorer with cascade
from .embedding_meta import float32_to_bytes, l2_norm, sha256_bytes, build_enroll_notes

from apps.attendance.models import Student, FaceEmbedding
from .media_paths import DIRS
# import csv, datetime
import csv


def _clamp01(x: float) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0


# ---- InsightFace singleton (GPU-first) ----
_APP: Optional[FaceAnalysis] = None
_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
# _DET_SIZE = (640, 640)  # default square
_DET_SIZE = (800, 800)  # start higher (still auto-tunes per image)

def set_det_size(w: int, h: int | None = None):
    """Set global detector size; next _get_app(det_size=...) call will prepare with it."""
    global _DET_SIZE, _APP
    _DET_SIZE = (w, w) if h is None else (w, h)
    # do NOT prepare here; let _get_app handle it on demand

def _get_app(det_size: tuple[int, int] | None = None) -> FaceAnalysis:
    """Return singleton FaceAnalysis; prepare only when det_size changes."""
    global _APP, _DET_SIZE
    if _APP is None:
        _APP = FaceAnalysis(name="buffalo_l", providers=_PROVIDERS)
        use_size = det_size or _DET_SIZE
        _APP.prepare(ctx_id=0, det_size=use_size)
        _DET_SIZE = use_size
        return _APP

    # re-prepare only if caller requests a different det_size
    if det_size and det_size != _DET_SIZE:
        _APP.prepare(ctx_id=0, det_size=det_size)
        _DET_SIZE = det_size
    return _APP

def current_det_size() -> int:
    return int(_DET_SIZE[0])



def current_provider_and_model() -> tuple[str, str]:
    """
    Return (active_provider, model_name). Try to read the *actual* provider from
    an ORT InferenceSession; fall back to the configured preference order.
    """
    app = _get_app()
    model = getattr(app, "name", "buffalo_l")
    provider = "unknown"
    try:
        models = getattr(app, "models", {}) or {}
        for m in models.values():
            sess = getattr(m, "session", None)
            if sess is not None and hasattr(sess, "get_providers"):
                provs = sess.get_providers()
                if provs:
                    provider = provs[0]
                    break
        if provider == "unknown":
            provider = "CUDAExecutionProvider" if "CUDAExecutionProvider" in _PROVIDERS else "CPUExecutionProvider"
    except Exception:
        provider = "CUDAExecutionProvider" if "CUDAExecutionProvider" in _PROVIDERS else "CPUExecutionProvider"
    return provider, model


# ---- Scoring helpers ----
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v) + eps
    return (v / n).astype(np.float32)


def _choose_face(faces):
    if not faces:
        return None
    faces = list(faces)
    faces.sort(
        key=lambda f: ((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), getattr(f, "det_score", 0.0)),
        reverse=True,
    )
    return faces[0]


def compute_embedding_from_image(bgr: np.ndarray, det_size: Optional[int] = None) -> Optional[np.ndarray]:
    app = _get_app((det_size, det_size)) if det_size else _get_app()
    faces = app.get(bgr)
    face = _choose_face(faces)
    if face is None:
        return None
    feat = getattr(face, "normed_embedding", None)
    if feat is None:
        feat = getattr(face, "embedding", None)
        if feat is None:
            return None
        feat = l2_normalize(np.asarray(feat, dtype=np.float32))
    else:
        feat = np.asarray(feat, dtype=np.float32)
    if feat.ndim != 1 or feat.shape[0] != 512:
        return None
    return feat


def load_bgr(path: str) -> Optional[np.ndarray]:
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def image_score(bgr: np.ndarray) -> tuple[float, float, float]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    bright = float(gray.mean())
    return 0.0, sharp, bright


# def score_images(paths: List[str]) -> List[dict]:
#     """
#     Return [{'path', 'name', 'sharp', 'bright', 'score'}] for all loadable images.
#     'score' is the combined normalized value we sort by.
#     """
#     rows = []
#     for p in paths:
#         bgr = load_bgr(p)
#         if bgr is None:
#             continue
#         _, sharp, bright = image_score(bgr)
#         rows.append({"path": p, "name": os.path.basename(p), "sharp": float(sharp), "bright": float(bright)})
#     if not rows:
#         return []
#     sharps = np.array([r["sharp"] for r in rows], dtype=np.float32)
#     brights = np.array([r["bright"] for r in rows], dtype=np.float32)
#     s_min, s_max = float(sharps.min()), float(sharps.max())
#     b_min, b_max = float(brights.min()), float(brights.max())
#     s_rng = max(s_max - s_min, 1e-6)
#     b_rng = max(b_max - b_min, 1e-6)
#     for r in rows:
#         s_n = (r["sharp"] - s_min) / s_rng
#         b_n = (r["bright"] - b_min) / b_rng
#         r["score"] = float(0.7 * s_n + 0.3 * b_n)
#     return rows


def embed_image_ex(bgr: np.ndarray) -> tuple[Optional[np.ndarray], int, float]:
    """
    Like embed_image(), but returns (vec, det_used, det_conf).
    det_used is the detector size the cascade settled on for THIS image.
    det_conf is the detection score from that pass.
    """
    try:
        bbox, conf, det_used, _prov = detect_with_cascade(bgr, start_size=current_det_size())
    except Exception:
        bbox, conf, det_used = None, 0.0, current_det_size()

    app = _get_app((det_used, det_used))
    faces = app.get(bgr)
    if not faces:
        return (None, det_used, conf)

    f = max(faces, key=lambda ff: (ff.bbox[2]-ff.bbox[0])*(ff.bbox[3]-ff.bbox[1]))
    vec = getattr(f, "normed_embedding", None)
    if vec is None:
        vec = getattr(f, "embedding", None)

    if vec is None:
        return (None, det_used, conf)

    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n == 0:
        return (None, det_used, conf)
    if abs(n - 1.0) > 1e-3:
        v = v / n
    return (v, det_used, conf)



def select_topk_scored(paths: List[str], k: int, *, min_score: float = 0.0, strict_top: bool = False) -> tuple[
    List[dict], List[dict]]:
    """
    Returns (top_k_scored, all_scored).
    Each element is {'path','name','sharp','bright','score'}.
    """
    if k <= 0:
        return ([], [])

    # 1) score all images (ROI + cascade per settings) and pick top-K (optionally strict by min_score)
    rows = score_images(
        paths)  # [{'name','path','score','sharp','bright','roi','bbox','det_size','det_conf','provider'}, ...]

    if not rows:
        return ([], [])
    rows_sorted = sorted(rows, key=lambda r: r["score"], reverse=True)
    if strict_top:
        rows_sorted = [r for r in rows_sorted if r["score"] >= float(min_score)]
    rows_sorted = sorted(rows, key=lambda r: r["score"], reverse=True)

    def _truthy(v):
        # Defensive cast for numpy types/arrays; treat non-empty array as True
        try:
            if isinstance(v, np.ndarray):
                return bool(v.size and np.any(v))
            if isinstance(v, (np.bool_, np.generic)):
                return bool(v.item())
            return bool(v)
        except Exception:
            return False

    roi_yes = [r for r in rows_sorted if _truthy(r.get("roi"))]
    roi_no = [r for r in rows_sorted if not _truthy(r.get("roi"))]
    topk = (roi_yes + roi_no)[:k]

    return (topk, rows)


def save_npy_for_student(h_code: str, vec: np.ndarray) -> str:
    out_dir = DIRS["EMBEDDINGS"]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{h_code}.npy")
    np.save(out_path, vec.astype(np.float32))
    return out_path


# ---- Enrollment (writes bytes to FaceEmbedding + npy file + metadata) ----
def enroll_student_from_folder(h_code: str, k: int = 3, force: bool = False, min_score: float = 0.0,
                               strict_top: bool = False) -> dict:
    t0 = time.time()

    folder = student_gallery_dir(h_code)
    if not os.path.isdir(folder):
        return {"ok": False, "reason": f"Folder not found: {folder}"}

    # gather candidates
    cand = []
    per_image = []  # list of {'name','path','stage','reason','det_size','det_conf'}
    for name in os.listdir(folder):
        ext = os.path.splitext(name)[1].lower()
        if ext in IMG_EXTS:
            cand.append(os.path.join(folder, name))
    if not cand:
        return {"ok": False, "reason": f"No images under {folder}"}

    images_considered = len(cand)
    topk_rows, all_rows = select_topk_scored(cand, k=k, min_score=min_score, strict_top=strict_top)
    if not topk_rows:
        return {"ok": False, "reason": "No valid images after scoring"}

    debug = {
        "h_code": h_code,
        "images_considered": images_considered,
        "params": {"k": k, "min_score": float(min_score), "strict_top": bool(strict_top)},
        "all_scored": all_rows,  # already contains det_size/conf/provider/roi/bbox/score/sharp/bright
        "topk_scored": topk_rows,
        "per_image": []  # we will append one entry per attempted image during embedding
    }

    embs, skipped = [], []
    img_embs = []
    sharp_list, bright_list, used_names, score_list = [], [], [], []
    used_detail = []  # list of dicts: name/score/sharp/bright
    fail_reasons = []

    for row in topk_rows:
        p = row["path"]
        bgr = load_bgr(p)
        if bgr is None:
            skipped.append(p)
            per_image.append({"name": row["name"], "path": p, "stage": "load", "reason": "unreadable",
                              "det_size": row.get("det_size"), "det_conf": row.get("det_conf")})
            debug["per_image"].append({"name": row["name"], "path": p, "reason": "unreadable_image"})
            continue

        # vec = compute_embedding_from_image(bgr, det_size=int(row.get("det_size") or current_det_size()))
        # vec = embed_image(bgr)  # <- use cascade-aware embedder
        vec, det_used, det_conf = embed_image_ex(bgr)  # cascade-aware; returns (vec, det_size, det_conf)

        if vec is None:
            skipped.append(p)
            per_image.append({"name": row["name"], "path": p, "stage": "embed", "reason": "no_face_or_no_512",
                              "det_size": int(det_used), "det_conf": float(det_conf),})
            debug["per_image"].append({"name": row["name"], "path": p, "reason": "no_face_or_no_512"})
            continue

        # success path (unchanged, plus we mirror core fields into debug)
        embs.append(vec)
        img_embs.append(vec)
        sharp_list.append(row["sharp"])
        bright_list.append(row["bright"])
        score_list.append(float(row.get("score", 0.0)))
        used_names.append(row["name"])
        used_detail.append({
            "name": row["name"],
            "path": p,
            "score": row.get("score"),
            "sharp": row.get("sharp"),
            "bright": row.get("bright"),
            "roi": bool(row.get("roi")),
            "det_size": int(det_used),  # was: row.get("det_size")
            "det_conf": float(det_conf),  # was: row.get("det_conf")
            "provider": row.get("provider"),
            "bbox": row.get("bbox"),
            "raw01": None,
            "rank": None,
        })
        debug["per_image"].append({"name": row["name"], "path": p, "reason": "ok"})

    if not embs:
        # fallback: try other ROI==True images that weren't in top-K
        extras = [r for r in all_rows if r.get("roi") and r["path"] not in skipped and r["name"] not in used_names]
        for row in extras:
            p = row["path"]
            bgr = load_bgr(p)
            if bgr is None:
                continue
            vec, det_used, det_conf = embed_image_ex(bgr)
            if vec is None:
                per_image.append({
                    "name": row["name"], "path": p, "stage": "embed",
                    "reason": "no_face_or_no_512",
                    "det_size": int(det_used), "det_conf": float(det_conf)  # was row.get(...)
                })
                continue
            embs.append(vec)
            img_embs.append(vec)
            sharp_list.append(row["sharp"])
            bright_list.append(row["bright"])
            score_list.append(float(row.get("score", 0.0)))
            used_names.append(row["name"])
            used_detail.append({
                "name": row["name"], "score": row.get("score"), "sharp": row.get("sharp"),
                "bright": row.get("bright"), "roi": bool(row.get("roi")),
                "det_size": int(det_used), "det_conf": float(det_conf),
                "provider": row.get("provider"), "bbox": row.get("bbox"),
                "raw01": None, "rank": None,
            })
            debug["per_image"].append({"name": row["name"], "path": p, "reason": "ok_fallback"})
            if len(embs) >= 1:  # or up to k, your choice
                break

    if not embs:
        # dump a debug file so you can investigate later
        os.makedirs(DIRS["LOGS_DEBUG_FACES"], exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dbg_path = os.path.join(DIRS["LOGS_DEBUG_FACES"], f"enroll_fail_{h_code}_{ts}.json")
        try:
            with open(dbg_path, "w", encoding="utf-8") as f:
                json.dump(debug, f, ensure_ascii=False, indent=2)
        except Exception:
            dbg_path = None
        reason = "No embeddings computed from top-k images"
        if skipped:
            reason += f" (skipped={len(skipped)})"
        if dbg_path:
            reason += f"; debug={os.path.relpath(dbg_path, settings.MEDIA_ROOT)}"
        return {"ok": False, "reason": reason, "meta": {"k": len(topk_rows), "images_considered": images_considered}}


    # average then normalize
    mean_vec = np.mean(np.stack(embs, axis=0), axis=0)
    vec512 = l2_normalize(mean_vec)

    # --- Compute per-image raw01 similarity to the final vector and assign rank ---
    # Both img_embs[i] and vec512 are L2-normalized; cosine is dot product
    raw_items = []
    for i, img_vec in enumerate(img_embs):
        cos = float(np.dot(img_vec, vec512))
        raw01 = _clamp01(0.5 * (cos + 1.0))  # map [-1,1] -> [0,1]
        # Attach raw01 to the corresponding used_detail entry
        if i < len(used_detail):
            used_detail[i]["raw01"] = raw01
        raw_items.append((i, raw01))

    # Rank by raw01 desc among the USED images
    raw_items.sort(key=lambda t: t[1], reverse=True)
    for rank, (i, _) in enumerate(raw_items, start=1):
        if i < len(used_detail):
            used_detail[i]["rank"] = rank

    # Compute avg over raw01 (selected = used_detail)
    avg_used_score_raw = _clamp01(float(np.mean([t[1] for t in raw_items])) if raw_items else 0.0)

    # write .npy (for PKL builder)
    abs_path = save_npy_for_student(h_code, vec512)
    rel_path = os.path.relpath(abs_path, settings.MEDIA_ROOT)

    # prepare FaceEmbedding payload
    vec_bytes = float32_to_bytes(vec512)
    emb_norm = l2_norm(vec512)
    sha = sha256_bytes(vec_bytes)

    def _as_int(x):
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            if x.size == 0:
                return None
            return int(x.flatten()[0])
        if isinstance(x, (np.bool_, np.number, np.generic)):
            return int(np.int64(x).item())
        if isinstance(x, (list, tuple)):
            return int(x[0]) if x else None
        return int(x)

    ds = []
    for ud in used_detail:
        x = _as_int(ud.get("det_size"))
        if x is not None:
            ds.append(x)

    if ds:
        ds.sort()
        det = int(ds[len(ds) // 2])  # median
    else:
        det = current_det_size()

    provider, model = current_provider_and_model()

    # upsert active embedding for student, robust to multiple actives
    try:
        st = Student.objects.get(h_code=h_code, is_active=True)
    except Student.DoesNotExist:
        return {"ok": False, "reason": f"Active student not found for H_CODE={h_code}"}

    notes = build_enroll_notes(
        k_used=len(embs),
        det_size=det,
        images_considered=images_considered,
        images_used=len(embs),
        provider=provider,
        model=model,
        embedding_norm=emb_norm,
        used_filenames=used_names,
        extras={"min": f"{min_score:.2f}", "strict": int(bool(strict_top))}
    )

    avg_used_score = max(0.0, min(1.0, float(avg_used_score_raw)))   # mean of raw01 in [0..1]

    payload = {
        "dim": 512,
        "vector": vec_bytes,
        "source_path": rel_path,
        "is_active": True,
        "last_enrolled_at": timezone.now(),
        "last_used_k": len(topk_rows),
        "last_used_det_size": det,
        "last_used_min_score": float(min_score),  # <— ADD
        "images_considered": images_considered,
        "images_used": len(embs),
        "avg_sharpness": float(np.mean(sharp_list)) if sharp_list else 0.0,
        "avg_brightness": float(np.mean(bright_list)) if bright_list else 0.0,
        "avg_used_score": avg_used_score,  # now mean(raw01)
        "used_images": used_names,
        "used_images_detail": used_detail,  # now includes raw01 + rank
        "embedding_norm": emb_norm,
        "embedding_sha256": sha,
        "arcface_model": model,
        "provider": provider,
        "enroll_runtime_ms": int((time.time() - t0) * 1000),
        "enroll_notes": notes,
    }

    with transaction.atomic():
        qs_active = (FaceEmbedding.objects
                     .select_for_update()
                     .filter(student=st, is_active=True)
                     .order_by("-created_at", "-id"))

        if force:
            # Deactivate anything active, then create a fresh active row
            if qs_active.exists():
                qs_active.update(is_active=False)
            fe = FaceEmbedding.objects.create(student=st, **payload)
            created = True
        else:
            if qs_active.exists():
                # Update the newest active row in place (no deactivation)
                fe = qs_active.first()
                for k_field, v in payload.items():
                    setattr(fe, k_field, v)
                fe.save(update_fields=list(payload.keys()))
                created = False
            else:
                # No active rows: create one
                fe = FaceEmbedding.objects.create(student=st, **payload)
                created = True

    # Write a lightweight success debug file (optional)
    debug.update({
        "used_detail": used_detail,
        "avg_used_score_raw01": avg_used_score,
        "embedding_norm": emb_norm,
        "det_size_used": det,
        "provider": provider,
        "model": model,
    })
    try:
        os.makedirs(DIRS["LOGS_DEBUG_FACES"], exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dbg_ok = os.path.join(DIRS["LOGS_DEBUG_FACES"], f"enroll_ok_{h_code}_{ts}.json")
        with open(dbg_ok, "w", encoding="utf-8") as f:
            json.dump(debug, f, ensure_ascii=False, indent=2)
    except Exception:
        dbg_ok = None

    # write a CSV log for this student under MEDIA/reports/enroll/
    try:
        os.makedirs(DIRS["REPORTS_ENROLL"], exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_csv = os.path.join(DIRS["REPORTS_ENROLL"], f"{h_code}_{ts}.csv")
        with open(out_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["h_code", "images_considered", "images_used", "k", "det_size", "reason_count"])
            w.writerow([h_code, images_considered, len(embs), len(topk_rows), det, len(per_image)])
            w.writerow([])
            w.writerow(["name", "stage", "reason", "det_size", "det_conf", "path"])
            for r in per_image:
                w.writerow([r["name"], r["stage"], r["reason"], r.get("det_size"), r.get("det_conf"), r["path"]])
    except Exception:
        pass  # never let logging break enrollment

    return {
        "ok": True,
        "created": created,
        "h_code": h_code,
        "used_images": [ud["path"] for ud in used_detail],
        "skipped": skipped,
        "embedding_path": rel_path,
        "face_embedding_id": fe.id,
        "updated_field": "vector+metadata",
        "meta": {
            "k": len(topk_rows),
            "det_size": det,
            "images_considered": images_considered,
            "images_used": len(embs),
            "avg_sharpness": float(np.mean(sharp_list)) if sharp_list else 0.0,
            "avg_brightness": float(np.mean(bright_list)) if bright_list else 0.0,
            "embedding_norm": emb_norm,
            "embedding_sha256": sha,
            "provider": provider,
            "model": model,
            "min_score": float(min_score),
            "strict_top": bool(strict_top),
            "debug_log": os.path.relpath(dbg_ok, settings.MEDIA_ROOT) if dbg_ok else None,
            "log_csv": os.path.relpath(out_csv, settings.MEDIA_ROOT) if 'out_csv' in locals() else None,
        },
    }


def embed_image(bgr: np.ndarray) -> Optional[np.ndarray]:
    """Return a 512-dim L2-normalized embedding for the largest detected face, using det-size cascade."""
    # First, run our cascade to find a reliable det-size for THIS image
    try:
        bbox, conf, det_used, _prov = detect_with_cascade(bgr, start_size=current_det_size())
    except Exception:
        bbox, conf, det_used = None, 0.0, current_det_size()
    # Prepare app at the det-size that worked best (or current as fallback)
    app = _get_app((det_used, det_used))
    faces = app.get(bgr)
    if not faces:
        return None
    f = max(faces, key=lambda ff: (ff.bbox[2] - ff.bbox[0]) * (ff.bbox[3] - ff.bbox[1]))
    vec = getattr(f, "normed_embedding", None)
    if vec is None:
        vec = getattr(f, "embedding", None)

    if vec is None:
        return None
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n == 0:
        return None
    if abs(n - 1.0) > 1e-3:
        v = v / n
    return v

# Notes:
# We save .npy to media/embeddings/H_CODE.npy and write bytes into FaceEmbedding.vector.
# We store the .npy relative path into source_path (purely informational — you can change that if you prefer).
# update_or_create(student=..., is_active=True) keeps one active embedding row per student. If you’d rather append new rows and inactivate the old one, tell me and I’ll switch to a “create + mark previous inactive” flow.
# Make the function robust to any existing state:
#   If --force is passed: deactivate all active rows for that student, then create a fresh active row with the new bytes + metadata.
#   If --force is NOT passed:
#   If there is at least one active row, update the most recent active row in place (no deactivations).
#   If there are none, create a new active row.
#   This avoids crashes and gives --force a clear meaning.
