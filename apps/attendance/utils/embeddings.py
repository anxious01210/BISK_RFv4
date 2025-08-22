# apps/attendance/utils/embeddings.py
from __future__ import annotations
import os, time, hashlib
from typing import Optional, List, Tuple
from django.db import transaction
import cv2
import numpy as np
from django.conf import settings
from django.utils import timezone

from insightface.app import FaceAnalysis

from .media_paths import DIRS, student_gallery_dir
from .embedding_meta import float32_to_bytes, l2_norm, sha256_bytes, build_enroll_notes

from apps.attendance.models import Student, FaceEmbedding

# ---- InsightFace singleton (GPU-first) ----
_APP: Optional[FaceAnalysis] = None
_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
_DET_SIZE = (640, 640)  # default square


def set_det_size(w: int, h: int | None = None):
    """Set global detector size and reset the singleton so the next call re-initializes."""
    global _DET_SIZE, _APP
    _DET_SIZE = (w, w) if h is None else (w, h)
    _APP = None


def _get_app() -> FaceAnalysis:
    global _APP
    if _APP is not None:
        return _APP
    _APP = FaceAnalysis(name="buffalo_l", providers=_PROVIDERS)
    _APP.prepare(ctx_id=0, det_size=_DET_SIZE)
    return _APP


def current_det_size() -> int:
    return _DET_SIZE[0]


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


def compute_embedding_from_image(bgr: np.ndarray) -> Optional[np.ndarray]:
    app = _get_app()
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


def score_images(paths: List[str]) -> List[dict]:
    """
    Return [{'path', 'name', 'sharp', 'bright', 'score'}] for all loadable images.
    'score' is the combined normalized value we sort by.
    """
    rows = []
    for p in paths:
        bgr = load_bgr(p)
        if bgr is None:
            continue
        _, sharp, bright = image_score(bgr)
        rows.append({"path": p, "name": os.path.basename(p), "sharp": float(sharp), "bright": float(bright)})
    if not rows:
        return []
    sharps = np.array([r["sharp"] for r in rows], dtype=np.float32)
    brights = np.array([r["bright"] for r in rows], dtype=np.float32)
    s_min, s_max = float(sharps.min()), float(sharps.max())
    b_min, b_max = float(brights.min()), float(brights.max())
    s_rng = max(s_max - s_min, 1e-6)
    b_rng = max(b_max - b_min, 1e-6)
    for r in rows:
        s_n = (r["sharp"] - s_min) / s_rng
        b_n = (r["bright"] - b_min) / b_rng
        r["score"] = float(0.7 * s_n + 0.3 * b_n)
    return rows

def select_topk_scored(paths: List[str], k: int, *, min_score: float = 0.0, strict_top: bool = False) -> tuple[List[dict], List[dict]]:
    """
    Returns (top_k_scored, all_scored).
    Each element is {'path','name','sharp','bright','score'}.
    """
    if k <= 0:
        return ([], [])
    rows = score_images(paths)
    if not rows:
        return ([], [])
    rows_sorted = sorted(rows, key=lambda r: r["score"], reverse=True)
    if strict_top:
        rows_sorted = [r for r in rows_sorted if r["score"] >= float(min_score)]
    return (rows_sorted[:k], rows)


def save_npy_for_student(h_code: str, vec: np.ndarray) -> str:
    out_dir = DIRS["EMBEDDINGS"]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{h_code}.npy")
    np.save(out_path, vec.astype(np.float32))
    return out_path


# ---- Enrollment (writes bytes to FaceEmbedding + npy file + metadata) ----
def enroll_student_from_folder(h_code: str, k: int = 3, force: bool = False, min_score: float = 0.0, strict_top: bool = False) -> dict:
    t0 = time.time()

    folder = student_gallery_dir(h_code)
    if not os.path.isdir(folder):
        return {"ok": False, "reason": f"Folder not found: {folder}"}

    # gather candidates
    cand = []
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

    embs, skipped = [], []
    sharp_list, bright_list, used_names = [], [], []
    used_detail = []  # list of dicts: name/score/sharp/bright

    for row in topk_rows:
        p = row["path"]
        bgr = load_bgr(p)
        if bgr is None:
            skipped.append(p)
            continue

        vec = compute_embedding_from_image(bgr)
        if vec is None:
            skipped.append(p)
            continue
        embs.append(vec)
        sharp_list.append(row["sharp"])
        bright_list.append(row["bright"])
        used_names.append(row["name"])
        used_detail.append({
            "name": row["name"],
            "score": round(float(row["score"]), 4),
            "sharp": round(float(row["sharp"]), 2),
            "bright": round(float(row["bright"]), 2),
        })

    if not embs:
        return {"ok": False, "reason": "No embeddings computed from top-k images"}

    # average then normalize
    mean_vec = np.mean(np.stack(embs, axis=0), axis=0)
    vec512 = l2_normalize(mean_vec)

    # write .npy (for PKL builder)
    abs_path = save_npy_for_student(h_code, vec512)
    rel_path = os.path.relpath(abs_path, settings.MEDIA_ROOT)

    # prepare FaceEmbedding payload
    vec_bytes = float32_to_bytes(vec512)
    emb_norm = l2_norm(vec512)
    sha = sha256_bytes(vec_bytes)
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

    payload = {
        "dim": 512,
        "vector": vec_bytes,
        "source_path": rel_path,  # representative path for reference
        "is_active": True,
        "last_enrolled_at": timezone.now(),
        "last_used_k": len(topk_rows),  # planned K after threshold
        "last_used_det_size": det,
        "images_considered": images_considered,
        "images_used": len(embs),  # actual used (may be < planned K)
        "avg_sharpness": float(np.mean(sharp_list)) if sharp_list else 0.0,
        "avg_brightness": float(np.mean(bright_list)) if bright_list else 0.0,
        "used_images": used_names,
        "used_images_detail": used_detail,
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


    return {
        "ok": True,
        "created": created,
        "h_code": h_code,
        "used_images": [r["path"] for r in topk_rows],
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
        },
    }

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