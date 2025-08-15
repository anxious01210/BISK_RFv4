# Redis Cache — Setup & Usage Guide (BISK_RFv4)

This guide documents **what Redis is**, **why we chose it** for BISK_RFv4, and **exact steps** to install, configure, and test it for Django caching. It also shows how our enforcer writes keys that the `/dash/system/` page reads — and includes a detailed **sanity test** using a `ping` key (what it proves and why we do it).

- Project: **BISK_RFv4**
- Purpose: **shared cache** between the Django web process and the **enforcer** (systemd service)
- Keys of interest: `enforcer:last_run`, `enforcer:last_ok`, `enforcer:last_error`, `enforcer:running_pid`
- Current cache config: `RedisCache` @ `redis://127.0.0.1:6379/1`, `KEY_PREFIX="bisk"`
- Target audience: developers/operators of this project

---

## 1) What is Redis and why we use it here?

**Redis** is an in-memory key-value store that’s extremely fast and supports persistence. In Django, Redis is often used as a **shared cache** between multiple processes. We need a shared cache so that:

- The **enforcer** (running under systemd) can write *breadcrumbs*:
  - `enforcer:last_run`, `enforcer:last_ok`, `enforcer:last_error`, `enforcer:running_pid`
- The **web app** (Django dev/prod server) can read them and display the live status at `/dash/system/`.

Alternatives like Django’s **LocMemCache** don’t share memory between processes, so the web server wouldn’t see keys written by the enforcer. Redis solves that cleanly and is production-ready.

**When Redis is a good fit**
- You need fast cross-process caching
- You want optional persistence and easy ops with `systemd`
- You might later use Redis for rate-limiting, locks, queues, etc.

**Downsides**
- Another service to install and monitor
- Needs basic ops know-how (systemd, persistence, security)

---

## 2) Installation (Ubuntu example)

> Do this on each host (dev / prod) where the app runs.

### 2.1 Install Redis server
```bash
sudo apt-get update
sudo apt-get install -y redis-server
sudo systemctl enable --now redis-server
systemctl is-active redis-server   # expect: active
redis-cli PING                     # expect: PONG
```

### 2.2 Install Python Redis client (in your project venv)
```bash
# from the project root, with .venv activated
pip install "redis>=5,<6"
```

Add to `requirements.txt`:
```text
redis>=5,<6
```

---

## 3) Django configuration

Edit **`bisk/settings.py`** and set the Django cache to Redis:

```python
CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.redis.RedisCache",
        "LOCATION": "redis://127.0.0.1:6379/1",
        "KEY_PREFIX": "bisk",  # helps namespace our keys
        # Optional: set default timeout; None = no expiry
        # "TIMEOUT": None,
    }
}
```

> Tip (production): you can use a Unix socket for slightly lower latency:
> ```python
> "LOCATION": "unix:///var/run/redis/redis-server.sock?db=1"
> ```
> Or add auth if Redis is on a different host:
> ```python
> "LOCATION": "redis://:PASSWORD@redis-host:6379/1"
> ```

After editing settings:
```bash
# restart Django web (dev server) and the enforcer service
# dev server: Ctrl+C then
python manage.py runserver

# enforcer (systemd):
sudo systemctl restart bisk-enforcer.service
```

---

## 4) Sanity Test: the `ping` key (what, why, and how)

We use a simple **`ping`** key to prove that:
1) Django is actually using **Redis** (not LocMem)  
2) The app **can write and read** from the configured Redis DB  
3) Our **KEY_PREFIX/versioning** work as expected  
4) (Optional) The key is visible in `redis-cli` with the expected name

### 4.1 Run the test from Django
```bash
python manage.py shell
>>> from django.core.cache import cache
>>> cache.set("ping", "pong", 60)   # set value with a 60s TTL
>>> cache.get("ping")
'pong'
```
- If this prints `'pong'`, Django can talk to Redis and basic reads/writes succeed.
- The TTL proves expiration works; after ~60 seconds `get('ping')` will return `None`.

### 4.2 Understand the key name in Redis
Django’s cache adds a **version** (default `1`). We also added **KEY_PREFIX="bisk"**.  
So your `ping` becomes **`bisk:1:ping`** in Redis.

Check it from Redis:
```bash
redis-cli -n 1 --scan --pattern 'bisk:1:ping'
redis-cli -n 1 --raw get 'bisk:1:ping'   # --raw prints plain strings nicely
redis-cli -n 1 ttl 'bisk:1:ping'         # shows remaining TTL in seconds
```
> If you used `ping` **before** setting `KEY_PREFIX`, your old key was `:1:ping`. That’s why `redis-cli -n 1 get ":1:ping"` may exist while `bisk:1:ping` doesn’t (or vice versa).

### 4.3 Common gotchas for `ping` showing `(nil)`
- **Wrong name**: you forgot the `KEY_PREFIX` and version → use `bisk:1:ping`  
- **Expired TTL**: you set `60`, it expired by the time you checked → set again or use a longer TTL  
- **Wrong DB index**: you’re scanning DB 0 while Django is on `/1` → use `-n 1`  
- **Different settings per process**: web and enforcer point to different caches → ensure both use the same `CACHES` block

This single `ping` test quickly isolates config vs. connectivity issues and confirms the exact key naming.

---

## 5) Verifying the cache works across processes

Our real goal is **cross-process sharing**:
- The **enforcer** writes:
  - `enforcer:last_run`, `enforcer:last_ok`, `enforcer:last_error`, `enforcer:running_pid`
- The **web app** reads them on `/dash/system/`.

You should see these with:
```bash
redis-cli -n 1 --scan --pattern "bisk:1:enforcer:*"
```

If `/dash/system/` shows live **PID / Last run / Last OK**, both processes are sharing Redis correctly.

---

## 6) Alternatives (recommended substitutions)

If you don’t want to run Redis on a particular host, two good Django backends still share keys across processes:

### 6.1 File-based cache (simple, no DB/service)
```python
CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.filebased.FileBasedCache",
        "LOCATION": "/tmp/bisk_cache",  # any writeable path
    }
}
```
Pros: very easy, no extra services.  
Cons: slower than Redis; local filesystem; cleanup needed sometimes.

### 6.2 Database cache (persists in your DB)
```python
CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.db.DatabaseCache",
        "LOCATION": "bisk_cache",  # table name
    }
}
```
Initialize once:
```bash
python manage.py createcachetable bisk_cache
```
Pros: persists, no extra service.  
Cons: more DB load vs Redis; slower than in-memory Redis.

> **Not recommended** for cross-process: `LocMemCache` (default) — fast but not shared across processes.

---

## 7) Troubleshooting

**A) `ModuleNotFoundError: No module named 'redis'`**  
Install the client in your venv: `pip install "redis>=5,<6"`

**B) `Connection refused` or `Cannot connect to redis`**  
Start the server: `sudo systemctl enable --now redis-server`  
Confirm: `systemctl is-active redis-server` should be `active`

**C) Keys don’t show in Redis**  
- Ensure your Django is actually using Redis:
  ```bash
  python manage.py shell -c "from django.conf import settings; print(settings.CACHES['default']['BACKEND'])"
  ```
- Remember the **prefix + version**: search for `bisk:1:*`
  ```bash
  redis-cli -n 1 --scan --pattern "bisk:1:*"
  ```

**D) `/dash/system/` still shows dashes for Enforcer panel**  
- Confirm both processes use the same cache settings (web & systemd enforcer)
- Restart both after changing settings:
  ```bash
  python manage.py runserver  # restart dev server
  sudo systemctl restart bisk-enforcer.service
  ```

**E) `--raw get` shows gibberish**  
Values may be pickled; that’s normal. Use `--raw` and store simple text if you need human-readability for ad-hoc debugging.

**F) Cleaning old keys (optional)**  
If you previously used no prefix, you might have `:1:enforcer:*` leftovers. Safe to delete:
```bash
for k in $(redis-cli -n 1 --scan --pattern ':1:enforcer:*'); do
  redis-cli -n 1 del "$k" >/dev/null
done
```

---

## 8) Security & production notes

- Keep Redis bound to **localhost** (`127.0.0.1`) unless you must expose it; then use a password and firewall rules.
- For higher durability, you can tune Redis persistence (RDB/AOF). Default Redis on Ubuntu enables RDB snapshots — OK for caching use.
- Monitor Redis memory usage (`redis-cli info memory`) and evictions; set sensible maxmemory if needed.

---

## 9) File locations & commits

- **Code:** `bisk/settings.py` (the `CACHES` block as shown above)
- **Dependency:** add `redis>=5,<6` to `requirements.txt`
- **Processes to restart after changes:**
  - dev server (or WSGI/ASGI server)
  - systemd enforcer (`sudo systemctl restart bisk-enforcer.service`)

Suggested commit message:
```
cache: switch to Redis backend with KEY_PREFIX=bisk; add sanity test; share enforcer breadcrumbs across processes
```

---

## 10) Where to store docs in this repo

Create a top-level **`docs/`** folder for future guides. For structure:

```
docs/
├─ ops/
│  ├─ REDIS_CACHE.md            # this file
│  ├─ ENFORCER_SYSTEMD_GUIDE.md # systemd service how-to
│  └─ DEPLOY_CHECKLIST.md
├─ backend/
│  ├─ SCHEDULER_OVERVIEW.md
│  └─ MODELS_GUIDE.md
└─ README.md
```

- Keep **operational** how-tos under `docs/ops/`
- Keep **code architecture** notes under `docs/backend/`

You can also keep deployment scripts under `deploy/` (already present) and reference them from `docs/ops/`.

---

## 11) Quick reference — commands used

```bash
# server
sudo apt-get install -y redis-server
sudo systemctl enable --now redis-server
systemctl is-active redis-server
redis-cli PING

# python client (venv)
pip install "redis>=5,<6"
echo 'redis>=5,<6' >> requirements.txt

# Django settings (bisk/settings.py)
# CACHES = { ... RedisCache ... KEY_PREFIX="bisk" ... }

# restarts
python manage.py runserver
sudo systemctl restart bisk-enforcer.service

# Django sanity test (what&why)
python manage.py shell
>>> from django.core.cache import cache
>>> cache.set("ping","pong",60)   # proves write+expiry
>>> cache.get("ping")             # proves read; expect 'pong'
>>> 
# Now confirm it in Redis with the exact namespacing:
redis-cli -n 1 --scan --pattern 'bisk:1:ping'
redis-cli -n 1 --raw get 'bisk:1:ping'
redis-cli -n 1 ttl 'bisk:1:ping'

# Redis inspection for enforcer breadcrumbs
redis-cli -n 1 --scan --pattern 'bisk:1:enforcer:*'
```

---

*Maintainer note:* If later we switch to Memcached or file/db cache for certain environments, update `bisk/settings.py`’s `CACHES` accordingly and restart both the web process and the enforcer. For prod, Redis remains the recommended default.
