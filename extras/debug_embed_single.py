# extras/debug_embed_single.py
# Simple, hard-coded single-image test. No Django needed.
import os, cv2, numpy as np
from insightface.app import FaceAnalysis

# --- EDIT THESE THREE LINES ---
IMAGE_PATH = "/home/rio/PycharmProjects/BISK_RFv4/media/face_gallery/H150057/H150057.JPG"
H_CODE = "H150057"
DET_SET = 1024  # e.g. 640, 800, 1024, 1600, 2048
# --------------------------------

EMBED_DIR = os.path.join("media", "embeddings")
os.makedirs(EMBED_DIR, exist_ok=True)
OUT_NPY = os.path.join(EMBED_DIR, f"{H_CODE}.npy")


def load_bgr(path: str):
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0: return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def main():
    bgr = load_bgr(IMAGE_PATH)
    if bgr is None:
        raise SystemExit(f"Unreadable image: {IMAGE_PATH}")

    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(DET_SET, DET_SET))

    faces = app.get(bgr)
    if not faces:
        raise SystemExit("No face detected (try larger DET_SET).")

    # largest face
    f = max(faces, key=lambda ff: (ff.bbox[2] - ff.bbox[0]) * (ff.bbox[3] - ff.bbox[1]))
    vec = getattr(f, "normed_embedding", None) or getattr(f, "embedding", None)
    if vec is None:
        raise SystemExit("Face has no embedding vector.")

    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n == 0:
        raise SystemExit("Invalid embedding (norm=0).")
    if abs(n - 1.0) > 1e-3:
        v = v / n

    np.save(OUT_NPY, v.astype(np.float32))
    print(f"OK: saved {OUT_NPY} (dim={v.shape[0]}, det_set={DET_SET})")


if __name__ == "__main__":
    main()
