# BISK_RFv4 – StreamProfile & Enforcer Ops Rationale
**Last updated:** 2025-08-15

Project: **BISK_RFv4** — school attendance via face recognition  
Repo: https://github.com/anxious01210/BISK_RFv4  
Stack: Python 3.12, Django 5.2.5, DRF, ffmpeg/ffprobe, Redis (DB 1, key prefix `bisk:`), TZ=Asia/Baghdad.

---

## Overview (why these knobs exist)
- **Camera** = *what/where*: identity (name, URL, location, creds).
- **StreamProfile** = *how to run*: policy knobs (transport, fps, det_set, GPU, CPU affinity, priorities, telemetry cadence).
- **Enforcer** merges **StreamProfile → Camera (fallback)** at spawn time. This lets operators change behavior from Django Admin without code changes or restarts.

**Key ops benefit:** zero-deploy tuning per camera/profile with auditability (`effective_args`) and fast diagnosis (`last_error`).

---

## What each knob buys you

| Knob (where) | Controls | Why it matters | Real scenario |
|---|---|---|---|
| `rtsp_transport` (Profile) | RTSP over `tcp` vs `udp` | TCP is sturdier across NAT/switches; UDP is lower latency on clean LANs | Hallway cam behind a flaky switch drops frames → flip just that cam to `tcp` |
| `hwaccel`, `device`, `gpu_index` (Profile) | Decode/compute device (e.g., NVIDIA) | Avoid overloading the GTX 1060; choose who gets GPU vs CPU fallback | 08:00 peak → less-important cams on CPU, entrances on GPU |
| `cpu_affinity` (Profile) | Pin runner to CPU cores | Smoother frame pacing; stop ffmpeg threads from fighting Django/DB | Pin runners to cores `0,1`, leave `2,3` for OS/Postgres |
| `nice` (Profile) | Process priority | Prevent CPU spikes from hurting web/API responsiveness | Under encode spikes, Admin stays responsive |
| `fps` (Profile) | Input throttle | Predictable load; 3–6 FPS is often enough for attendance | Corridor cams at 4 FPS; lab cams at 8 FPS |
| `det_set` (Profile) | Detection resolution | Accuracy vs GPU mem/time tradeoff | Night profile at `640` to cut noise/OOM; daytime `1024` |
| `hb_interval` (Profile) | Heartbeat cadence | Faster failure detection vs less noise | Entrances at 3s (quick alerts), back-lot at 15s |
| `snapshot_every` (Profile) | Snapshot/telemetry rate | Control storage & I/O | Snap every 300s globally; front door at 60s |
| `extra_cli` (Profile) | Extra ffmpeg args | Try flags without migrations | Temporarily add `-stimeout 5000000` for a stubborn NVR |
| `pause_until` (Camera) | Temporary disable | Throttle without editing URLs/settings | Maintenance 14:00–16:00 → pause until 16:00 |
| `effective_args` (RunningProcess) | Exact spawn command | Forensics & reproducibility | Copy to shell, reproduce a DESCRIBE failure |
| `last_error` (RunningProcess) | Friendly fail reason | Operator clarity | Admin shows “auth failed”, “RTSP 404”, “timeout” within seconds |

> **Design intent:** keep Camera clean (identity + `pause_until`). Put *all knobs* into StreamProfile. Enforcer merges at spawn so the next run reflects your policy.

---

## Why DB (profiles) over code
- **Zero-deploy ops:** change Admin fields → behavior changes on next spawn (or via “Enforce now”).  
- **Per-camera policy:** one hallway can behave differently from an auditorium.  
- **Safety rails:** lower `fps` or set `nice` quickly during load spikes.  
- **Auditability:** `effective_args` + `last_error` = “what ran” and “why it failed.”  
- **Future-proof:** `extra_cli` lets you trial ffmpeg flags without schema changes.

---

## “Because of Y, later we can have X”
- Store **cpu_affinity/nice** → guarantee DB & embedding workers stay responsive during morning storms.  
- Store **hb_interval/last_error** → `/admin/system/` can show human labels and power alerting.  
- Store **det_set/fps** → implement **day vs night** profiles to cut GPU use at night.  
- Keep **rtsp_transport** → VLAN/router changes are one-field flips per cam.  
- Provide **extra_cli** → trial `-rw_timeout`, jitter buffer, etc., without touching code.

---

## Next steps (implementation plan)
1. **Wire StreamProfile knobs end-to-end (high impact)**  
   - Build `build_ffmpeg_args(camera, profile)` used by the enforcer `_start()` and save to `RunningProcess.effective_args`.
   - Apply `cpu_affinity` and `nice` after spawn.
   - Honor `hb_interval` and `snapshot_every` in the runner.
   - Confirm NVIDIA mapping (`-hwaccel cuda`, `-hwaccel_device N`) and fallbacks.
   - **Acceptance:** changing a knob changes next spawn; Admin shows exact command; snapshot/heartbeat intervals reflect config.

2. **Runner telemetry & clear errors**  
   - Map common ffmpeg/RTSP errors to friendly strings (401/403 auth, DESCRIBE/404, timeout, network).  
   - Include `last_error` in heartbeats; store and expose in Admin and `/dash/system/`.  
   - Clear on first stable frame/heartbeat.  
   - **Acceptance:** break a cam → Admin shows a human reason within seconds; fix it → clears.

3. **Admin quick pauses**  
   - Actions: Pause 30m / 2h / until tomorrow 08:00 (Asia/Baghdad).  
   - Badge on Camera & RunningProcess lists when paused.  
   - **Acceptance:** one click pauses; enforcer respects it; timestamp visible.

4. **More robust enforcer lock (optional)**  
   - Redis `SET NX EX` TTL lock (`bisk:lock:enforcer`) renewed each tick; file lock kept as fallback.  
   - **Acceptance:** single enforcer cluster‑wide; stale locks self‑heal.

5. **UI niceties**  
   - Auto-refresh toggle (10s default) on `/admin/system/`.  
   - “Enforce now” button (rate‑limited).

6. **Targeted tests**  
   - Args builder, pause actions, lock behavior, heartbeat error mapping.

---

## Ops snippets

**Check fields exist**
```bash
python manage.py shell -c "from django.apps import apps; R=apps.get_model('scheduler','RunningProcess'); print('effective_args' in [f.name for f in R._meta.concrete_fields])"
python manage.py shell -c "from django.apps import apps; C=apps.get_model('cameras','Camera'); print('pause_until' in [f.name for f in C._meta.concrete_fields])"
```

**See effective args of latest runner**
```bash
python manage.py shell -c "from apps.scheduler.models import RunningProcess as R; r=R.objects.latest('id'); print(r.effective_args)"
```

**Inspect nice/affinity of latest PID (Linux)**
```bash
python manage.py shell -c "from apps.scheduler.models import RunningProcess as R; import psutil; r=R.objects.latest('id'); p=psutil.Process(r.pid); print('nice=',p.nice(),'aff=',p.cpu_affinity())"
```

---

## Glossary
- **Enforcer**: background service that starts/stops camera runners according to schedules/profiles.  
- **Runner**: the process (ffmpeg + recognition) that pulls frames and produces heartbeats/snapshots/logs.  
- **Heartbeat**: periodic JSON from runner → server (`ok`, `pid`, `last_error`, etc.).

---

## Placement
Save this file under **`docs/ops/`** in the repo so operators and developers share the same mental model.