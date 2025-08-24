# apps/attendance/signals.py
def ensure_media_tree_signal(**kwargs):
    from .utils.media_paths import ensure_media_tree
    ensure_media_tree()
