# apps/cameras/utils.py
from pathlib import Path
from django.conf import settings


def snapshot_url_for(camera_id: int) -> str | None:
    """
    Return public URL for latest snapshot of a camera, or None if it doesn't exist.
    Works with local MEDIA_URL (no storage backends).
    """
    snap_path = Path(settings.SNAPSHOT_DIR) / f"{camera_id}.jpg"
    if not snap_path.exists():
        return None

    try:
        mtime = int(snap_path.stat().st_mtime)
    except Exception:
        mtime = 0

    base = settings.MEDIA_URL.rstrip("/")
    return f"{base}/snapshots/{camera_id}.jpg?v={mtime}"
