# extras/runner/enroll_flow.py
import numpy as np
from typing import Tuple
from extras.runner.attendance_client import enroll  # you created this earlier

try:
    # optional (only if you added gallery_client.py earlier)
    from extras.runner.gallery_client import fetch_gallery
except Exception:
    fetch_gallery = None


def enroll_from_embedding(h_code: str,
                          embedding: "np.ndarray | list[float] | bytes",
                          dim: int | None = None,
                          camera_id: int | None = None,
                          source_path: str = "") -> dict:
    """
    Call this right after your runner computes an embedding for a student during enrollment.
    - h_code: the student code (must exist & be active)
    - embedding: float32 vector (shape (dim,)) or bytes of float32
    - dim: inferred if omitted
    - camera_id/source_path: optional metadata for tracking
    """
    res = enroll(h_code=h_code, embedding=embedding, dim=dim,
                 camera_id=camera_id, source_path=source_path)

    # Optional: immediately refresh local gallery so the new vector is used for matching
    if fetch_gallery is not None:
        try:
            M, codes, ids = fetch_gallery(
                dim=dim or (len(embedding) if not isinstance(embedding, bytes) else None) or 512,
                active=True)
            # TODO: hand M, codes, ids back to your matcher (store in a module/global)
            # e.g., set_global_gallery(M, codes, ids)
        except Exception as e:
            print("Gallery refresh failed:", e)

    return res


# --- CLI helper for quick manual testing (optional) ---
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print(
            "Usage: python extras/runner/enroll_flow.py <h_code> <vec.npy|comma_floats> [dim] [camera_id] [source_path]")
        sys.exit(2)

    h_code = sys.argv[1]
    vec_arg = sys.argv[2]
    if vec_arg.endswith(".npy"):
        v = np.load(vec_arg).astype(np.float32).ravel()
    else:
        v = np.array([float(x) for x in vec_arg.split(",")], dtype=np.float32)
    dim = int(sys.argv[3]) if len(sys.argv) >= 4 else int(v.size)
    cam = int(sys.argv[4]) if len(sys.argv) >= 5 else None
    src = sys.argv[5] if len(sys.argv) >= 6 else ""
    print(enroll_from_embedding(h_code, v, dim=dim, camera_id=cam, source_path=src))
