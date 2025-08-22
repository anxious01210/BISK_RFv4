# extras/runner/attendance_client.py
import os, base64, numpy as np, requests
import datetime as dt

SERVER = os.getenv("BISK_SERVER", "http://127.0.0.1:8000")
KEY = os.getenv("BISK_KEY", "dev-key-change-me")


def _iso(ts: dt.datetime | None) -> str | None:
    if ts is None:
        return None
    if ts.tzinfo is None:
        ts = ts.astimezone()  # make local-aware if naive
    return ts.isoformat()


def post_match(h_code: str,
               score: float,
               camera_id: int,
               ts: dt.datetime | None = None,
               crop_path: str = "") -> dict:
    """
    POST a recognition match to the Django server.

    Required: h_code (string), score (float), camera_id (int)
    Optional: ts (aware datetime) and crop_path (string)

    Env:
      BISK_SERVER (default http://127.0.0.1:8000)
      BISK_KEY    (default dev-key-change-me)
    """
    payload = {
        "h_code": h_code,
        "score": float(score),
        "camera_id": int(camera_id),
    }
    iso = _iso(ts)
    if iso:
        payload["ts"] = iso
    if crop_path:
        payload["crop_path"] = crop_path

    r = requests.post(
        f"{SERVER}/api/attendance/ingest/",
        json=payload,
        headers={"Content-Type": "application/json", "X-BISK-KEY": KEY},
        timeout=3,
    )
    r.raise_for_status()
    return r.json()


def fetch_gallery(dim: int = 512, active: bool = True):
    r = requests.get(
        f"{SERVER}/api/attendance/gallery/",
        params={"dim": dim, "active": 1 if active else 0},
        headers={"X-BISK-KEY": KEY},
        timeout=5,
    )
    r.raise_for_status()
    payload = r.json()
    embs = payload.get("embeddings", [])
    H = []  # vectors
    codes = []  # student codes
    ids = []  # embedding ids
    for item in embs:
        v = np.frombuffer(base64.b64decode(item["vec"]), dtype=np.float32)
        if v.size != dim:
            continue
        H.append(v)
        codes.append(item["h_code"])
        ids.append(item["id"])
    if not H:
        return np.zeros((0, dim), np.float32), [], []
    M = np.stack(H).astype(np.float32)
    # normalize rows for cosine
    M /= (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    return M, codes, ids


def cosine_match(vec: np.ndarray, M: np.ndarray, codes: list[str], min_score: float = 0.80):
    """
    vec: (dim,) float32 embedding (already normalized)
    M:   (N, dim) gallery matrix (normalized)
    """
    if M.shape[0] == 0:
        return None, 0.0
    sims = M @ vec  # cosine similarity
    idx = int(np.argmax(sims))
    score = float(sims[idx])
    return (codes[idx], score) if score >= min_score else (None, score)


def enroll(h_code: str,
           embedding: "np.ndarray | bytes | list[float]",
           dim: int | None = None,
           camera_id: int | None = None,
           source_path: str = "") -> dict:
    """
    Enroll a new embedding for a student.
    - embedding: a float32 vector (np.ndarray shape (dim,)), or raw bytes of float32, or a list of floats
    - dim: inferred from array length when omitted
    """
    # normalize inputs to float32 bytes
    if isinstance(embedding, bytes):
        raw = embedding
        if dim is None:
            if len(raw) % 4 != 0:
                raise ValueError("raw bytes length must be divisible by 4")
            dim = len(raw) // 4
    else:
        arr = np.asarray(embedding, dtype=np.float32).ravel()
        if dim is None:
            dim = int(arr.size)
        if arr.size != dim:
            raise ValueError(f"embedding size {arr.size} != dim {dim}")
        raw = arr.tobytes()

    vec_b64 = base64.b64encode(raw).decode("ascii")
    payload = {"h_code": h_code, "dim": int(dim), "vec": vec_b64}
    if camera_id is not None:
        payload["camera_id"] = int(camera_id)
    if source_path:
        payload["source_path"] = source_path

    r = requests.post(
        f"{SERVER}/api/attendance/enroll/",
        json=payload,
        headers={"Content-Type": "application/json", "X-BISK-KEY": KEY},
        timeout=5,
    )
    r.raise_for_status()
    return r.json()


if __name__ == "__main__":
    import sys, datetime as dt

    args = sys.argv[1:]
    if not args:
        print("Usage:\n"
              "  Match (old style): python extras/runner/attendance_client.py <h_code> <score> <camera_id> [<ISO-datetime>] [<crop_path>]\n"
              "  Match (explicit):  python extras/runner/attendance_client.py match <h_code> <score> <camera_id> [<ISO-datetime>] [<crop_path>]\n"
              "  Enroll:            python extras/runner/attendance_client.py enroll <h_code> <vec.npy|comma_floats> [<dim>] [<camera_id>] [<source_path>]\n")
        sys.exit(2)

    mode = "match"
    if args[0] in ("match", "enroll"):
        mode = args.pop(0)

    if mode == "match" and len(args) >= 3:
        h, s, cid = args[0], float(args[1]), int(args[2])
        ts = dt.datetime.fromisoformat(args[3]) if len(args) >= 4 else None
        crop = args[4] if len(args) >= 5 else ""
        print(post_match(h, s, cid, ts=ts, crop_path=crop))
        sys.exit(0)

    if mode == "enroll" and len(args) >= 2:
        h = args[0]
        vec_arg = args[1]
        # try loading a .npy file; else parse comma-separated floats
        if vec_arg.endswith(".npy"):
            v = np.load(vec_arg).astype(np.float32)
        else:
            v = np.array([float(x) for x in vec_arg.split(",")], dtype=np.float32)
        dim = int(args[2]) if len(args) >= 3 else int(v.size)
        cam = int(args[3]) if len(args) >= 4 else None
        src = args[4] if len(args) >= 5 else ""
        print(enroll(h, v, dim=dim, camera_id=cam, source_path=src))
        sys.exit(0)

    print("Bad args. See usage above.")
    sys.exit(2)
