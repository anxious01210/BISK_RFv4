# extras/runner/attendance_client.py
import os
import datetime as dt
import requests

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


if __name__ == "__main__":
    # quick CLI: python extras/runner/attendance_client.py H123456 0.87 1
    import sys

    if len(sys.argv) < 4:
        print(
            "Usage: python extras/runner/attendance_client.py <h_code> <score> <camera_id> [<YYYY-MM-DDTHH:MM:SS+ZZ:ZZ>] [<crop_path>]")
        sys.exit(2)
    h, s, cid = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
    ts = None
    if len(sys.argv) >= 5:
        ts = dt.datetime.fromisoformat(sys.argv[4])
    crop = sys.argv[5] if len(sys.argv) >= 6 else ""
    print(post_match(h, s, cid, ts=ts, crop_path=crop))
