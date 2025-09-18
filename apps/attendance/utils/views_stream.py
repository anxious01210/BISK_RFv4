# apps/attendance/views_stream.py
from __future__ import annotations

import time
import queue
import threading
from typing import Dict
from django.http import StreamingHttpResponse, HttpResponseForbidden
from django.contrib.auth.decorators import login_required
from django.utils.crypto import get_random_string

# --- Simple in-memory session registry (per process) ---
# Each session has a Queue of JPEG bytes. Your capture code will publish into it.
_SESSIONS: Dict[str, "queue.Queue[bytes]"] = {}
_SESSIONS_LOCK = threading.Lock()


def get_or_create_session(session_id: str) -> "queue.Queue[bytes]":
    with _SESSIONS_LOCK:
        q = _SESSIONS.get(session_id)
        if q is None:
            q = queue.Queue(maxsize=3)  # small buffer to cap memory
            _SESSIONS[session_id] = q
        return q


def destroy_session(session_id: str) -> None:
    with _SESSIONS_LOCK:
        _SESSIONS.pop(session_id, None)


def publish_frame(session_id: str, jpeg_bytes: bytes) -> None:
    """
    Put a JPEG-encoded frame into the session queue.
    - Safe to call from your capture loop.
    - Drops the oldest frame if the queue is full (keeps stream 'live').
    """
    q = get_or_create_session(session_id)
    try:
        q.put_nowait(jpeg_bytes)
    except queue.Full:
        try:
            _ = q.get_nowait()
        except queue.Empty:
            pass
        finally:
            # best effort re-insert
            try:
                q.put_nowait(jpeg_bytes)
            except queue.Full:
                pass


# --- MJPEG stream view ---
BOUNDARY = "frame"


def _stream_generator(q: "queue.Queue[bytes]", keepalive_every: float = 5.0):
    """
    Yields multipart/x-mixed-replace chunks.
    If no frame arrives within `keepalive_every` seconds, emits a comment to keep the connection alive.
    """
    last_emit = time.time()
    while True:
        try:
            frame = q.get(timeout=0.5)
            yield (
                f"--{BOUNDARY}\r\n"
                "Content-Type: image/jpeg\r\n"
                f"Content-Length: {len(frame)}\r\n\r\n"
            ).encode("utf-8") + frame + b"\r\n"
            last_emit = time.time()
        # except queue.Empty:
        #     # Periodic keepalive to prevent proxies/clients from timing out
        #     if time.time() - last_emit >= keepalive_every:
        #         yield (f"--{BOUNDARY}\r\n"
        #                "Content-Type: text/plain\r\n\r\n"
        #                ": keepalive\r\n").encode("utf-8")
        #         last_emit = time.time()
        #     # continue
        except queue.Empty:
            # Emit a tiny JPEG keepalive so <img> doesn't show a broken icon
            if time.time() - last_emit >= keepalive_every:
                ka = _keepalive_jpeg()
                yield (
                    f"--{BOUNDARY}\r\n"
                    "Content-Type: image/jpeg\r\n"
                    f"Content-Length: {len(ka)}\r\n\r\n"
                ).encode("utf-8") + ka + b"\r\n"
                last_emit = time.time()


@login_required
def mjpeg_stream(request, session: str):
    """
    GET /stream/live/<session>.mjpg
    Requirements (for now):
      - Authenticated user (staff preferred; tighten later if needed).
    Returns a multipart MJPEG stream that emits whatever frames the session receives.
    """
    user = request.user
    if not (user.is_staff or user.is_superuser):
        return HttpResponseForbidden("Staff only")

    q = get_or_create_session(session)
    response = StreamingHttpResponse(
        streaming_content=_stream_generator(q),
        content_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
    )
    # Prevent buffering by proxies
    response["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response["Pragma"] = "no-cache"
    return response


# --- Helpers you can call from elsewhere (admin/capture code) ---

def new_session_id(prefix: str = "cap") -> str:
    """Generate a short random session id like 'cap_v7d9gq'."""
    return f"{prefix}_{get_random_string(6)}"


def session_exists(session_id: str) -> bool:
    with _SESSIONS_LOCK:
        return session_id in _SESSIONS


# --- Test publisher (synthetic frames) ---------------------------------------
import io
import threading
from datetime import datetime
from typing import Dict, Tuple

from django.http import JsonResponse
from django.views.decorators.http import require_GET

try:
    from PIL import Image, ImageDraw, ImageFont

    _PIL_OK = True
except Exception:
    _PIL_OK = False

# Track per-session synthetic publisher threads
_TEST_THREADS: Dict[str, Tuple[threading.Thread, threading.Event]] = {}
_TEST_THREADS_LOCK = threading.Lock()

# --- Keepalive as a tiny JPEG to avoid "broken image" icons -------------------
_KEEPALIVE_CACHE = None


def _keepalive_jpeg() -> bytes:
    """
    Returns a 1x1 black JPEG (cached). Uses Pillow if available, else a precomputed baseline JPEG.
    """
    global _KEEPALIVE_CACHE
    if _KEEPALIVE_CACHE is not None:
        return _KEEPALIVE_CACHE
    try:
        if _PIL_OK:
            import io
            from PIL import Image
            buff = io.BytesIO()
            Image.new("RGB", (1, 1), (0, 0, 0)).save(buff, format="JPEG", quality=70)
            _KEEPALIVE_CACHE = buff.getvalue()
            return _KEEPALIVE_CACHE
    except Exception:
        pass
    # Precomputed 1x1 black baseline JPEG
    _KEEPALIVE_CACHE = (
        b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00'
        b'\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\x09\x09\x08'
        b'\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d'
        b'\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342\xff\xc0\x00\x11'
        b'\x08\x00\x01\x00\x01\x03\x01"\x00\x02\x11\x01\x03\x11\x01\xff\xc4\x00'
        b' \x00\x00\x01\x05\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00'
        b'\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00\x1b\x01\x00\x03'
        b'\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03'
        b'\x04\x05\x06\x07\x08\t\n\x0b\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03'
        b'\x11\x00?\x00\xd2\xcf \xff\xd9'
    )
    return _KEEPALIVE_CACHE


def _run_synthetic_publisher(session_id: str, stop_event: threading.Event,
                             width: int = 1280, height: int = 720, fps: int = 8):
    """
    Generates simple JPEG frames with timestamp text and publishes them to the session.
    Runs until stop_event is set.
    """
    # Fallback font (Pillow will choose a default if truetype not found)
    font = None
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", 28)
    except Exception:
        pass

    frame_interval = max(1.0 / max(fps, 1), 0.02)
    counter = 0
    while not stop_event.is_set():
        img = Image.new("RGB", (width, height), (16, 16, 16))
        draw = ImageDraw.Draw(img)

        # Title
        title = f"BISK MJPEG TEST — session: {session_id}"
        draw.rectangle([10, 10, width - 10, 70], outline=(200, 200, 200), width=2)
        draw.text((20, 22), title, fill=(220, 220, 220), font=font)

        # Timestamp + counter
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        draw.text((20, 90), f"Now: {ts}", fill=(180, 255, 180), font=font)
        draw.text((20, 130), f"Frame: {counter}", fill=(180, 200, 255), font=font)

        # Moving box so you can see motion
        box_w, box_h = 160, 120
        x = 40 + (counter * 10) % max(1, (width - box_w - 80))
        y = 200 + (counter * 6) % max(1, (height - box_h - 240))
        draw.rectangle([x, y, x + box_w, y + box_h], outline=(255, 120, 80), width=6)

        # Encode to JPEG
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85, optimize=True)
        publish_frame(session_id, buf.getvalue())

        counter += 1
        time.sleep(frame_interval)


@login_required
@require_GET
def test_start(request, session: str):
    """
    GET /attendance/stream/test/start/<session>/?fps=8&width=1280&height=720
    Starts a synthetic publisher thread for <session> (staff only).
    """
    user = request.user
    if not (user.is_staff or user.is_superuser):
        return HttpResponseForbidden("Staff only")
    if not _PIL_OK:
        return JsonResponse({"ok": False, "error": "Pillow not available"}, status=500)

    try:
        fps = int(request.GET.get("fps", "8"))
    except Exception:
        fps = 8
    try:
        width = int(request.GET.get("width", "1280"))
        height = int(request.GET.get("height", "720"))
    except Exception:
        width, height = 1280, 720

    with _TEST_THREADS_LOCK:
        if session in _TEST_THREADS:
            # already running
            return JsonResponse({"ok": True, "running": True, "session": session, "note": "already running"})

        stop_evt = threading.Event()
        t = threading.Thread(
            target=_run_synthetic_publisher,
            kwargs={"session_id": session, "stop_event": stop_evt, "width": width, "height": height, "fps": fps},
            daemon=True,
        )
        _TEST_THREADS[session] = (t, stop_evt)
        t.start()

    return JsonResponse({"ok": True, "running": True, "session": session, "fps": fps, "size": [width, height]})


@login_required
@require_GET
def test_stop(request, session: str):
    """
    GET /attendance/stream/test/stop/<session>/
    Stops the synthetic publisher for <session>.
    """
    user = request.user
    if not (user.is_staff or user.is_superuser):
        return HttpResponseForbidden("Staff only")

    with _TEST_THREADS_LOCK:
        t_tuple = _TEST_THREADS.pop(session, None)
    if t_tuple:
        t, stop_evt = t_tuple
        stop_evt.set()
        return JsonResponse({"ok": True, "stopped": True, "session": session})

    return JsonResponse({"ok": True, "stopped": False, "session": session, "note": "not running"})


# --- HTTP Uplink: accept JPEG frames via POST and publish to a session --------
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST


def _uplink_authorized(request) -> bool:
    """
    Allow either:
      - logged-in staff/superuser, OR
      - a shared key passed as ?key=... matching settings.STREAM_UPLINK_KEY (optional).
    If STREAM_UPLINK_KEY is not set, only staff/superuser are allowed.
    """
    user = getattr(request, "user", None)
    if user and (user.is_staff or user.is_superuser):
        return True

    req_key = request.GET.get("key") or request.META.get("HTTP_X_STREAM_KEY")
    settings_key = getattr(settings, "STREAM_UPLINK_KEY", None)
    return bool(settings_key and req_key and req_key == settings_key)


@csrf_exempt
@require_POST
def uplink_frame(request, session: str):
    """
    POST /attendance/stream/uplink/<session>/?key=...   (or send header X-Stream-Key)
    Body options:
      - Content-Type: image/jpeg   (raw bytes in request.body)
      - multipart/form-data with file field 'frame'
    Publishes the JPEG frame into the in-memory session queue.
    """
    if not _uplink_authorized(request):
        return HttpResponseForbidden("Forbidden")

    data = None
    ct = (request.META.get("CONTENT_TYPE") or "").lower()

    if "multipart/form-data" in ct:
        f = request.FILES.get("frame")
        if f:
            data = f.read()
    else:
        # Accept raw body for image/jpeg
        if "image/jpeg" in ct or request.body:
            data = request.body

    if not data:
        return JsonResponse({"ok": False, "error": "no frame data"}, status=400)

    # publish to the session queue
    publish_frame(session, data)
    return JsonResponse({"ok": True, "published": True, "session": session, "bytes": len(data)})


# --- Spawn local uplink runner (FFmpeg→JPEG→HTTP POST) -----------------------
import os
import sys
import shlex
import subprocess
from pathlib import Path
from django.views.decorators.http import require_GET

# Track spawned runner processes per session
_RUNNERS: Dict[str, subprocess.Popen] = {}
_RUNNERS_LOCK = threading.Lock()


def _runner_script_path() -> str:
    """
    Resolve extras/mjpeg_uplink_runner.py from BASE_DIR.
    """
    base = Path(getattr(settings, "BASE_DIR", Path.cwd()))
    p = base / "extras" / "mjpeg_uplink_runner.py"
    return str(p)


def _default_server_from_request(request) -> str:
    """
    Build base server URL from the current request (e.g., "http://127.0.0.1:8000").
    """
    scheme = "https" if request.is_secure() else "http"
    return f"{scheme}://{request.get_host()}"


@login_required
@require_GET
def run_start(request, session: str):
    """
    GET /attendance/stream/run/start/<session>/?source=rtsp|webcam&rtsp=...&device=...&fps=6&size=1280x720&quality=3
         [&ffmpeg=/usr/bin/ffmpeg][&server=http://...][&key=...]
    Spawns extras/mjpeg_uplink_runner.py as a subprocess and records it under this session.
    Staff-only.
    """
    user = request.user
    if not (user.is_staff or user.is_superuser):
        return HttpResponseForbidden("Staff only")

    source = request.GET.get("source", "")
    if source not in ("rtsp", "webcam"):
        return JsonResponse({"ok": False, "error": "source must be 'rtsp' or 'webcam'"}, status=400)

    # Params
    rtsp = request.GET.get("rtsp")
    device = request.GET.get("device", "/dev/video0")
    fps = request.GET.get("fps", "6")
    size = request.GET.get("size", "1280x720")
    quality = request.GET.get("quality", "3")
    ffmpeg_bin = request.GET.get("ffmpeg", os.environ.get("FFMPEG", "ffmpeg"))
    server = request.GET.get("server", _default_server_from_request(request))
    key = request.GET.get("key", getattr(settings, "STREAM_UPLINK_KEY", ""))

    # Build command
    script = _runner_script_path()
    if not os.path.exists(script):
        return JsonResponse({"ok": False, "error": f"runner script not found: {script}"}, status=500)

    cmd = [sys.executable, script,
           "--ffmpeg", ffmpeg_bin,
           "--server", server,
           "--session", session,
           "--source", source,
           "--fps", str(fps),
           "--size", size,
           "--quality", str(quality)]
    if key:
        cmd += ["--key", key]
    if source == "rtsp":
        if not rtsp:
            return JsonResponse({"ok": False, "error": "--rtsp is required for source=rtsp"}, status=400)
        cmd += ["--rtsp", rtsp]
    else:
        cmd += ["--device", device]

    # If a runner exists for this session, don't start a duplicate
    with _RUNNERS_LOCK:
        proc = _RUNNERS.get(session)
        alive = proc and (proc.poll() is None)
        if alive:
            return JsonResponse(
                {"ok": True, "running": True, "session": session, "pid": proc.pid, "note": "already running"})

        # Start new
        try:
            # Use DEVNULL to avoid blocking on pipes; logs can be added later if needed
            devnull = subprocess.DEVNULL
            proc = subprocess.Popen(cmd, stdout=devnull, stderr=devnull)
            _RUNNERS[session] = proc
            running = True
        except Exception as e:
            return JsonResponse({"ok": False, "error": f"spawn failed: {e}"}, status=500)

    return JsonResponse({
        "ok": True,
        "running": running,
        "session": session,
        "pid": proc.pid,
        "cmd": " ".join(shlex.quote(c) for c in cmd),
    })


@login_required
@require_GET
def run_stop(request, session: str):
    """
    GET /attendance/stream/run/stop/<session>/
    Stops (terminates) the spawned runner for <session>.
    Staff-only.
    """
    user = request.user
    if not (user.is_staff or user.is_superuser):
        return HttpResponseForbidden("Staff only")

    with _RUNNERS_LOCK:
        proc = _RUNNERS.pop(session, None)

    if not proc:
        return JsonResponse({"ok": True, "stopped": False, "session": session, "note": "no runner for session"})

    try:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()
    except Exception:
        pass

    return JsonResponse({"ok": True, "stopped": True, "session": session})


# --- List saved cameras as JSON (for RTSP dropdown in modal) -----------------
from django.views.decorators.http import require_GET


# --- List saved cameras as JSON (for RTSP dropdown in modal) -----------------
from django.views.decorators.http import require_GET
from django.apps import apps as django_apps

@login_required
@require_GET
def cameras_json(request):
    """
    GET /attendance/stream/cameras.json
    Returns a lightweight list of active cameras:
      { ok: true, cameras: [{id, label, rtsp, transport}] }

    Uses the `cameras` app's Camera model:
      - name        -> label
      - rtsp_url    -> rtsp
      - rtsp_transport -> transport  ("auto" / "tcp" / "udp")
    """
    user = request.user
    if not (user.is_staff or user.is_superuser):
        return HttpResponseForbidden("Staff only")

    try:
        Camera = django_apps.get_model("cameras", "Camera")  # apps/cameras/models.py
    except Exception:
        return JsonResponse({"ok": True, "cameras": []})

    # all cameras (active)
    qs = Camera.objects.filter(is_active=True).order_by("name")
    # all cameras (active & non-active)
    # qs = Camera.objects.all().order_by("name")
    cameras = []
    for cam in qs:
        label = getattr(cam, "name", None) or f"Camera {cam.pk}"
        rtsp = getattr(cam, "rtsp_url", "") or ""
        transport = getattr(cam, "rtsp_transport", "auto") or "auto"
        # Optional: include location in the label if present
        loc = getattr(cam, "location", "") or ""
        if loc:
            label = f"{label} — {loc}"
        cameras.append({
            "id": cam.pk,
            "label": label,
            "rtsp": rtsp,
            "transport": transport,
        })

    return JsonResponse({"ok": True, "cameras": cameras})

