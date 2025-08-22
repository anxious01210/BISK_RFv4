# apps/attendance/utils/embedding_meta.py
from __future__ import annotations

import hashlib
import numpy as np
from typing import Iterable, Optional


def float32_to_bytes(vec: np.ndarray) -> bytes:
    """Ensure vec is 1D float32 and return raw bytes."""
    v = np.asarray(vec, dtype=np.float32)
    assert v.ndim == 1, "vector must be 1D"
    return v.tobytes()


def l2_norm(vec: np.ndarray) -> float:
    v = np.asarray(vec, dtype=np.float32)
    return float(np.linalg.norm(v))


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def build_enroll_notes(
    *,
    k_used: int,
    det_size: int,
    images_considered: int,
    images_used: int,
    provider: str | None = None,
    model: str | None = None,
    embedding_norm: Optional[float] = None,
    used_filenames: Optional[Iterable[str]] = None,
    extras: Optional[dict] = None,
) -> str:
    """
    Produce a short, human-friendly one-liner recording 'what happened' during enrollment.
    Keep it brief; it complements the structured fields.
    """
    parts = [
        f"k={k_used}",
        f"det={det_size}",
        f"considered={images_considered}",
        f"used={images_used}",
    ]
    if provider:
        parts.append(f"prov={provider}")
    if model:
        parts.append(f"model={model}")
    if embedding_norm is not None:
        parts.append(f"norm={embedding_norm:.4f}")
    if used_filenames:
        head = list(used_filenames)[:2]
        parts.append(f"files={','.join(head)}{'…' if images_used > len(head) else ''}")
    if extras:
        for k, v in extras.items():
            parts.append(f"{k}={v}")
    return " ".join(parts)
