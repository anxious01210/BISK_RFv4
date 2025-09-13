from __future__ import annotations
import os, time
from pathlib import Path
from typing import List, Tuple
from django.conf import settings

# Reuse the same scorer the admin modal uses (keeps behavior consistent)
from .facescore import score_images  # returns rows with score/sharp/bright/name/path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def _list_images(dir_path: Path) -> List[Path]:
    if not dir_path.exists():
        return []
    out = []
    for p in dir_path.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            out.append(p)
    return sorted(out)

def curate_top_crops(dir_path: Path, keep_n: int) -> dict:
    """
    Keep the best 'keep_n' images in dir_path, remove the rest.
    Ranking = (score desc, sharp desc, mtime desc).
    Returns stats: {'total': int, 'kept': int, 'removed': int}
    """
    paths = _list_images(dir_path)
    total = len(paths)
    if total <= keep_n:
        return {"total": total, "kept": total, "removed": 0}

    # Score using the same pipeline as admin (ROI + cascade)
    rows = score_images([str(p) for p in paths])  # list of dicts with name/path/score/sharp/bright
    # Build lookup: path -> (score, sharp)
    meta = {}
    for r in rows or []:
        try:
            meta[Path(r["path"]).resolve()] = (float(r.get("score", 0.0)), float(r.get("sharp", 0.0)))
        except Exception:
            pass

    # Fallback: if some files missed scoring, give them low score to bias toward deletion
    def _rank_key(p: Path) -> Tuple[float, float, float]:
        s, sh = meta.get(p.resolve(), (0.0, 0.0))
        try:
            mt = float(p.stat().st_mtime)
        except Exception:
            mt = 0.0
        # Sort by: score desc, sharp desc, mtime desc
        return (s, sh, mt)

    ranked = sorted(paths, key=_rank_key, reverse=True)
    keep = set(ranked[:keep_n])
    remove = [p for p in ranked[keep_n:] if p.exists()]

    removed = 0
    for p in remove:
        try:
            p.unlink()
            removed += 1
        except Exception:
            # best-effort; ignore failures (permissions, races)
            pass

    return {"total": total, "kept": len(keep), "removed": removed}
