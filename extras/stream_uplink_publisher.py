#!/usr/bin/env python3
# extras/stream_uplink_publisher.py
"""
Lightweight HTTP uplink publisher for BISK MJPEG sessions.

Use from any capture/recognition script to stream frames (JPEG/BGR) to:
  /attendance/stream/uplink/<session>/?key=...

Example:
    from extras.stream_uplink_publisher import StreamUplinkPublisher
    pub = StreamUplinkPublisher(server="http://127.0.0.1:8000",
                                session="cap_demo",
                                key="dev-stream-key-change-me",
                                max_fps=6)
    pub.publish_bgr(frame_bgr, quality=85)  # or pub.publish_jpeg(jpeg_bytes)
"""
from __future__ import annotations
import os, time, threading, queue
from typing import Optional
import requests

try:
    import cv2

    _CV2_OK = True
except Exception:
    _CV2_OK = False


class StreamUplinkPublisher:
    """
    Posts JPEG frames to /attendance/stream/uplink/<session>/ in a BACKGROUND THREAD.
    - Non-blocking: capture loop never sleeps for preview.
    - If the queue is full, we drop the oldest frame (keep it live).
    """

    def __init__(
            self,
            server: str,
            session: str,
            key: Optional[str] = None,
            timeout: float = 3.0,
            max_fps: int = 6,
    ):
        if not server or not session:
            raise ValueError("server and session are required")
        self.url = f"{server.rstrip('/')}/attendance/stream/uplink/{session}/"
        if key:
            sep = "&" if "?" in self.url else "?"
            self.url += f"{sep}key={key}"

        self._timeout = float(timeout)
        self._min_interval = 1.0 / float(max_fps) if max_fps and max_fps > 0 else 0.0
        self._sess = requests.Session()

        self._q: "queue.Queue[bytes]" = queue.Queue(maxsize=1)  # single-slot buffer
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._run, name="uplink-worker", daemon=True)
        self._worker.start()

    # ---------------------------- public API ---------------------------------

    def publish_jpeg(self, jpeg_bytes: bytes) -> None:
        """Enqueue a JPEG for posting (non-blocking). Drops oldest if busy."""
        if not jpeg_bytes:
            return
        try:
            self._q.put_nowait(jpeg_bytes)
        except queue.Full:
            try:
                _ = self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(jpeg_bytes)
            except queue.Full:
                pass

    def publish_bgr(self, frame_bgr, quality: int = 85) -> None:
        """Encode BGR -> JPEG then enqueue (non-blocking)."""
        if not _CV2_OK or frame_bgr is None:
            return
        q = max(2, min(int(quality), 100))
        ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            self.publish_jpeg(buf.tobytes())

    def close(self):
        """Optional: stop background worker (usually not needed; daemon thread)."""
        self._stop.set()
        try:
            self._worker.join(timeout=0.5)
        except Exception:
            pass

    # --------------------------- worker thread --------------------------------

    def _run(self):
        last_post = 0.0
        while not self._stop.is_set():
            try:
                frame = self._q.get(timeout=0.2)
            except queue.Empty:
                continue

            # rate limit here (off the capture thread)
            if self._min_interval:
                now = time.time()
                wait = self._min_interval - (now - last_post)
                if wait > 0:
                    time.sleep(wait)
                last_post = time.time()

            try:
                self._sess.post(
                    self.url,
                    data=frame,
                    headers={"Content-Type": "image/jpeg"},
                    timeout=self._timeout,
                )
            except requests.RequestException:
                # ignore transient posting errors
                pass


# --- Convenience factory from environment ------------------------------------

def from_env() -> StreamUplinkPublisher:
    """
    Build a publisher from environment variables:
      BISK_SERVER           (default "http://127.0.0.1:8000")
      BISK_PREVIEW_SESSION  (no default — REQUIRED)
      STREAM_UPLINK_KEY     (optional)  or BISK_UPLINK_KEY (fallback)
      BISK_UPLINK_MAXFPS    (default "6")
      BISK_UPLINK_TIMEOUT   (default "3.0")
    """
    server = os.getenv("BISK_SERVER", "http://127.0.0.1:8000")
    session = os.getenv("BISK_PREVIEW_SESSION")
    if not session:
        raise RuntimeError("BISK_PREVIEW_SESSION is required")
    key = os.getenv("STREAM_UPLINK_KEY") or os.getenv("BISK_UPLINK_KEY")
    max_fps = int(os.getenv("BISK_UPLINK_MAXFPS", "6"))
    timeout = float(os.getenv("BISK_UPLINK_TIMEOUT", "3.0"))
    return StreamUplinkPublisher(server=server, session=session, key=key, max_fps=max_fps, timeout=timeout)
