# BISK_RFv4 — Runner Telemetry & Heartbeats (Ops Guide)

_Last updated: 2025-08-16 • TZ: **Asia/Baghdad**_

This page explains how **runner telemetry (heartbeats)** relates to **running processes**, how status is derived, and how to tune, prune, and troubleshoot the system.

---

## 1) Concepts & Data Model

### RunningProcess (RP)
Represents a **spawned runner instance** (one per PID) for a `(camera, profile)` pair.
- Created by the **enforcer** when it launches `extras/recognize_ffmpeg.py`.
- Updated on every heartbeat: `last_heartbeat(_at)`, `last_error`.
- New PID ⇒ new RP row; old rows remain as history.

Key fields: `camera`, `profile`, `pid`, `status`, `started_at`, `effective_args`, `effective_env`, `last_heartbeat(_at)`, `last_error`.

### RunnerHeartbeat (HB)
Append-only **telemetry snapshots**; used for history/graphs/debugging.
- Inserted by heartbeat API at a **rate-limited** cadence.
- **Errors** (`last_error` non-empty) are always logged immediately.
- Typical fields: `camera`, `profile`, `pid`, `ts`, `fps`, `detected`, `matched`, `latency_ms`, `last_error`.
- Index: `(camera, profile, ts)` (or `-ts` on Postgres) for “latest per runner” queries.
- Admin: **latest-only** per runner by default; add `?all=1` for the firehose.

---

## 2) Telemetry Flow
1. Enforcer spawns runner → new **RP** row.
2. Runner starts FFmpeg, parses `stderr` (maps friendly errors).
3. Every `hb_interval` seconds, runner **POSTs** JSON to `RUNNER_HEARTBEAT_URL`.
4. API updates **RP** (`last_heartbeat`, `last_error`) and **optionally inserts** an **HB** row:
   - At most **one HB per `(camera,profile)`** every `HB_LOG_EVERY_SEC`.
   - **Always** insert on error.
   - **Seed** a first HB if none exist (after a prune).

---

## 3) Status Thresholds (based on RP heartbeat age)
```python
HEARTBEAT_INTERVAL_SEC = 10
HEARTBEAT_ONLINE_SEC   = 15
HEARTBEAT_STALE_SEC    = 45
HEARTBEAT_OFFLINE_SEC  = 120
```
- Online if age ≤ `HEARTBEAT_ONLINE_SEC`
- Stale if `HEARTBEAT_ONLINE_SEC < age ≤ HEARTBEAT_STALE_SEC`
- Offline if `age > HEARTBEAT_OFFLINE_SEC`

---

## 4) Keeping It Lightweight
```python
HB_LOG_EVERY_SEC = 60      # rate-limit HB inserts (0 = disable inserts entirely)
HEARTBEAT_SNAPSHOT_EVERY = 15  # snapshot cadence
```
- Use 60–120s in prod; ~10s for testing.
- Errors are logged immediately (bypass rate limit).
- Admin shows latest-only by default; avoid `?all=1` in day-to-day ops.

Optional FFmpeg input flags for low latency (revert if unstable):
```
-fflags nobuffer -flags low_delay -probesize 32 -analyzeduration 0
```

---

## 5) Pruning Heartbeats
Management command accepts **days/hours/minutes** (and short aliases):
```bash
# keep last 24h
python manage.py prune_heartbeats --d 0 --h 24
# keep last 2h 30m
python manage.py prune_heartbeats --h 2 --m 30
```

**systemd timer (example):**
```
[Unit]
Description=BISK: prune runner heartbeats
[Service]
Type=oneshot
WorkingDirectory=/path/to/BISK_RFv4
ExecStart=/path/to/venv/bin/python manage.py prune_heartbeats --d 0 --h 24
User=rio
Group=rio
```
```
[Unit]
Description=Run BISK heartbeat prune hourly
[Timer]
OnCalendar=hourly
Persistent=true
[Install]
WantedBy=timers.target
```
Enable:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now bisk-prune-heartbeats.timer
```

---

## 6) Settings Cheatsheet
```python
RUNNER_HEARTBEAT_URL = "http://127.0.0.1:8000/api/runner/heartbeat/"
RUNNER_HEARTBEAT_KEY = "dev-key-change-me"   # header: X-BISK-KEY

HB_LOG_EVERY_SEC = 60
HEARTBEAT_INTERVAL_SEC = 10
HEARTBEAT_SNAPSHOT_EVERY = 15

HEARTBEAT_ONLINE_SEC  = 15
HEARTBEAT_STALE_SEC   = 45
HEARTBEAT_OFFLINE_SEC = 120

ENFORCER_INTERVAL_SECONDS = 15
ENFORCER_LOCK_FILE = "/run/bisk/enforcer.lock"

CACHES = {
  "default": {
    "BACKEND": "django.core.cache.backends.redis.RedisCache",
    "LOCATION": "redis://127.0.0.1:6379/1",
    "KEY_PREFIX": "bisk",
  }
}
```
**Cache gate key:** `bisk:hb:rl:{camera_id}:{profile_id}`. If cache is down, inserts are allowed (fail-open).

---

## 7) Troubleshooting
- **405 on /api/runner/heartbeat/**: using GET; endpoint is **POST-only**.
- **403 bad key**: missing/wrong `X-BISK-KEY` header.
- **No HBs appear** but RP updates: probably insert path disabled/rate-limited; force one with non-empty `last_error`.
- **TypeError: unexpected keyword 'pid'**: add `pid` to `RunnerHeartbeat` or remove it from API insert (you added it).

**Quick end-to-end check**
```bash
KEY=$(python - <<'PY'
from django.conf import settings
print(getattr(settings,'RUNNER_HEARTBEAT_KEY',''))
PY
)

curl -sS -X POST http://127.0.0.1:8000/api/runner/heartbeat/   -H 'Content-Type: application/json' -H "X-BISK-KEY: $KEY"   -d '{"camera_id":1,"profile_id":1,"pid":9999,"fps":5,"detected":0,"matched":0,"latency_ms":0,"last_error":"debug"}'

python manage.py shell -c "from apps.scheduler.models import RunnerHeartbeat as H; print(H.objects.count()); print(list(H.objects.values('camera_id','profile_id','ts','last_error')[:3]))"
```

---

## 8) Optional Enhancements
- Add `rp = ForeignKey(RunningProcess, null=True, blank=True)` to HB and pass `rp_id` from runner for perfect join.
- On PostgreSQL, prefer `Index(fields=['camera','profile','-ts'])` for even faster “latest first” reads.
- Mask credentials in any RTSP strings shown in admin/UI.
