# Live Code Mirror (Free) — Cloudflare Tunnel + Local Mirror (Ubuntu 24.04)

> **Short answer:** **Cloudflare Tunnel (Quick Tunnel) + a local mirror folder** is 100% **free** and keeps your code **auto‑synced from your laptop without any commit/push**. You develop normally; a tiny watcher mirrors your project to a **public copy** (with secrets excluded), and Cloudflare gives you a **public HTTPS URL** to browse it.
>
> If you later want a **stable URL** (not rotating) that’s still free, point a subdomain you control to Cloudflare with a **named tunnel**. Otherwise Quick Tunnel gives you a random URL each time.

---

## What you’ll get

- A **sanitized mirror** of your project at `~/public_mirror` (secrets/logs excluded).
- A **watcher** that auto‑syncs your edits into the mirror in seconds.
- A **local HTTP server** to view the mirror at `http://127.0.0.1:8000/`.
- A **public URL** (via Cloudflare Quick Tunnel) to share the mirror with others.
- (Optional) **Stable subdomain** and **systemd services** so everything runs on boot.

Tested on **Ubuntu 24.04**. Adjust paths as needed.

---

## Prerequisites

- You’re working on a local project, e.g. at `~/PycharmProjects/BISK_RFv4`.
- You have admin access to the machine.
- For a **stable URL**, you control a domain on Cloudflare (free plan is fine). For **Quick Tunnel**, a domain is **not** required.

> **Security note:** You’ll publish a sanitized copy of your code. Double‑check exclusions and masking before sharing the link.

---

## One‑time setup (Ubuntu 24.04)

### Step 1 — Install tools

```bash
sudo apt update
sudo apt install -y rsync inotify-tools
wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared-linux-amd64.deb
cloudflared --version
```

---

### Step 2 — Create a **sanitized mirror** of your project

> Goal: make a safe copy at `~/public_mirror` that excludes secrets and big folders.

1) Set paths (change if your project is elsewhere):

```bash
SRC="$HOME/PycharmProjects/BISK_RFv4"
DST="$HOME/public_mirror"
echo "SRC=$SRC"; echo "DST=$DST"
test -d "$SRC" && echo "OK: project exists" || echo "ERROR: wrong SRC path"
```

2) Create the mirror folder and exclude list:

```bash
mkdir -p "$DST"

cat > "$HOME/.mirror-exclude" <<'EOF'
.git/
.venv/
__pycache__/
*.pyc
*.log
media/
*.key
.env
EOF

cat "$HOME/.mirror-exclude"
```

3) First sync (copy → mirror, excluding secrets):

```bash
rsync -a --delete --exclude-from "$HOME/.mirror-exclude" "$SRC"/ "$DST"/
ls -la "$DST" | head -n 20
```

4) **Mask RTSP credentials** in the mirror (not your real code):

```bash
# Replaces rtsp://user:pass@host with rtsp://****:****@host in the MIRROR ONLY
find "$DST" \
  -type f \( -name "*.py" -o -name "*.json" -o -name "*.yml" -o -name "*.env" -o -name "*.txt" \) -print0 \
  | xargs -0 sed -i 's#\(rtsp://\)[^:@/]\+:[^@/]\+@#\1****:****@#g'

# Verify (should show ****:****@ if URLs exist)
grep -RIn "rtsp://" "$DST" | head -n 20 || true
```

> Add more excludes to `~/.mirror-exclude` as needed (e.g., `*.pem`, `node_modules/`, etc.). Re-run the sync when you update it.

---

### Step 3 — Auto‑sync on file changes (no git needed)

> Goal: a watcher keeps the mirror up to date as you edit files.

1) Create the watcher script:

```bash
mkdir -p "$HOME/bin"

cat <<'EOF' > "$HOME/bin/watch_mirror.sh"
#!/usr/bin/env bash
set -euo pipefail

SRC="$HOME/PycharmProjects/BISK_RFv4"
DST="$HOME/public_mirror"
EXC="$HOME/.mirror-exclude"

sync_once() {
  rsync -a --delete --exclude-from "$EXC" "$SRC"/ "$DST"/
  find "$DST" \
    -type f \( -name "*.py" -o -name "*.json" -o -name "*.yml" -o -name "*.env" -o -name "*.txt" \) -print0 \
    | xargs -0 sed -i 's#\(rtsp://\)[^:@/]\+:[^@/]\+@#\1****:****@#g'
  echo "$(date '+%F %T') synced"
}

sync_once

inotifywait -mr -e modify,create,delete,move "$SRC" \
| while read -r _; do
    sync_once
  done
EOF

chmod +x "$HOME/bin/watch_mirror.sh"
```

2) Run it in the background and verify:

```bash
nohup "$HOME/bin/watch_mirror.sh" >/tmp/mirror.log 2>&1 & echo $! > /tmp/watch_mirror.pid
sleep 1
tail -n 10 /tmp/mirror.log
```

3) Quick test the auto‑sync:

```bash
touch "$SRC/_mirror_test.txt"
sleep 2
ls -l "$DST/_mirror_test.txt"
rm -f "$SRC/_mirror_test.txt" "$DST/_mirror_test.txt"
```

**Stop / Restart the watcher:**

```bash
# Stop
kill "$(cat /tmp/watch_mirror.pid 2>/dev/null)" 2>/dev/null || true

# Start again
nohup "$HOME/bin/watch_mirror.sh" >/tmp/mirror.log 2>&1 & echo $! > /tmp/watch_mirror.pid
```

---

### Step 4 — Serve the mirror locally

> Goal: view the mirror at `http://127.0.0.1:8000/`.

```bash
cd "$DST"
nohup python3 -m http.server 8000 --bind 127.0.0.1 >/tmp/http.log 2>&1 &
sleep 1
curl -I http://127.0.0.1:8000 | head -n1
# Expected: HTTP/1.0 200 OK
```

> If 8000 is busy, use another port (e.g., 8010), and adjust the next step accordingly.

---

### Step 5 — Expose it publicly for free (Cloudflare **Quick Tunnel**)

> Goal: get a **public HTTPS URL** without router changes or cost.

```bash
cloudflared tunnel --url http://localhost:8000
```

You’ll see:
```
INF | All tunnels running. Visit: https://<random>.trycloudflare.com
```

That URL is your **public link** to the live, sanitized mirror. Keep the terminal running while it’s needed. Closing it stops the link.

> Later, use a **Named Tunnel** for a stable URL that stays the same across restarts (see below).

---

## Day‑to‑day usage

- Start/stop the **watcher** (Step 3) as needed — it auto‑syncs changes.
- Start the **HTTP server** (Step 4) when you want to serve the mirror.
- Run **Quick Tunnel** (Step 5) to get a public URL to share.
- Make changes in your project normally; they’ll show up at the public URL within seconds.

---

## Optional: Stable URL with a Named Cloudflare Tunnel (free)

> Requires a domain under Cloudflare DNS (free plan OK). End result: e.g., `https://code.yourdomain.com` points to your local mirror and persists across reboots.

1) **Login & create tunnel**:

```bash
cloudflared tunnel login            # opens browser to auth with Cloudflare
cloudflared tunnel create bisk-code # prints a TUNNEL ID
```

2) **Config file** at `/etc/cloudflared/config.yml` (use your Tunnel ID):

```yaml
tunnel: <YOUR_TUNNEL_ID>
credentials-file: /etc/cloudflared/<YOUR_TUNNEL_ID>.json

ingress:
  - hostname: code.yourdomain.com
    service: http://localhost:8000
  - service: http_status:404
```

3) **DNS record**: In Cloudflare dashboard, add a CNAME for `code.yourdomain.com` to the target shown by Cloudflare (usually `UUID.cfargotunnel.com`).

4) **Run as a service** (installs and enables systemd unit):

```bash
sudo cloudflared service install
sudo systemctl enable --now cloudflared
systemctl status cloudflared --no-pager
```

Now your mirror is at `https://code.yourdomain.com/` permanently.

---

## (Optional) Auto‑start everything on boot (systemd services)

> These are convenience units so you don’t have to start the watcher and HTTP server manually.

### A. Watcher service

Create `/etc/systemd/system/mirror-watcher.service`:

```ini
[Unit]
Description=Mirror watcher (rsync + mask)
After=network.target

[Service]
Type=simple
User=%i
ExecStart=/home/%i/bin/watch_mirror.sh
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

Enable for your user (replace `rio` with your username):

```bash
sudo systemctl enable --now mirror-watcher@rio.service
systemctl status mirror-watcher@rio --no-pager
```

### B. HTTP server service (simple Python server)

Create `/etc/systemd/system/mirror-http.service`:

```ini
[Unit]
Description=Mirror HTTP server (python -m http.server 8000)
After=network.target

[Service]
Type=simple
User=%i
WorkingDirectory=/home/%i/public_mirror
ExecStart=/usr/bin/python3 -m http.server 8000 --bind 127.0.0.1
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

Enable it:

```bash
sudo systemctl enable --now mirror-http@rio.service
systemctl status mirror-http@rio --no-pager
```

> The **cloudflared** named tunnel service was installed earlier via `cloudflared service install` and will auto‑start on boot.

---

## Security hygiene checklist

- ✅ Sensitive files **excluded** via `~/.mirror-exclude` (e.g., `.env`, keys, logs, `media/`).
- ✅ **Mask** any `rtsp://user:pass@host` occurrences in the **mirror**.
- ✅ Review the mirror at `http://127.0.0.1:8000/` before sharing the public link.
- ✅ If you add new sensitive files later, update `~/.mirror-exclude` and re‑sync.

---

## Troubleshooting

- **Command not found (`rsync`, `inotifywait`, `cloudflared`)**  
  Re‑install: `sudo apt install -y rsync inotify-tools` and re‑run the `.deb` for cloudflared.

- **Mirror not updating**  
  Check the watcher log: `tail -f /tmp/mirror.log`. Ensure `inotifywait` is running and `SRC` path is correct.

- **Port 8000 busy**  
  Pick another port (e.g., 8010) in Step 4 and update the tunnel config/command.

- **Quick Tunnel URL stops working**  
  The process must stay running. For a persistent URL, use a **named tunnel**.

- **Permission issues with systemd services**  
  Ensure paths are correct for your username. Check `journalctl -u mirror-watcher@rio -e`.

---

## Cheat‑sheet

```bash
# Start watcher (manual)
nohup "$HOME/bin/watch_mirror.sh" >/tmp/mirror.log 2>&1 & echo $! > /tmp/watch_mirror.pid

# Stop watcher (manual)
kill "$(cat /tmp/watch_mirror.pid 2>/dev/null)" 2>/dev/null || true

# Start local HTTP server (manual)
(cd "$HOME/public_mirror" && nohup python3 -m http.server 8000 --bind 127.0.0.1 >/tmp/http.log 2>&1 &)

# Quick Tunnel (ephemeral public URL)
cloudflared tunnel --url http://localhost:8000

# Named Tunnel (stable URL) — once configured
sudo systemctl restart cloudflared
```

---

## Cleanup / Uninstall

```bash
# Stop and disable services (if you created them)
sudo systemctl disable --now mirror-watcher@rio.service mirror-http@rio.service cloudflared

# Remove files
rm -rf "$HOME/public_mirror" "$HOME/bin/watch_mirror.sh" "$HOME/.mirror-exclude"
sudo rm -f /etc/systemd/system/mirror-watcher.service /etc/systemd/system/mirror-http.service
sudo systemctl daemon-reload
```

---

## Alternatives (also free)

- **ngrok**: easiest trial; URL rotates unless you have a paid plan for reserved domains.
- **Static bucket** (S3/R2) + `rclone` sync: no tunnel; updates every minute; great for read‑only code mirrors.

---

**You’re done.** Your live, sanitized mirror is a command away:
```bash
cloudflared tunnel --url http://localhost:8000
```
Share the printed `https://<random>.trycloudflare.com` URL when you want me to review your latest code.
