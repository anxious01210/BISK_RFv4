# BISK_RFv4 – Deployment & Restart Guide (Nginx Proxy Manager + Gunicorn)

This document summarizes all the steps needed to manage your Django/Flask projects
running behind **Nginx Proxy Manager (NPM)** with **Gunicorn** on your production server.

---

## ✅ 1. Architecture Summary

```
Browser → Cloudflare → FortiGate → NPM → 10.120.0.80:8001 → Gunicorn → Django
```

### Key settings:
- **Gunicorn bind:** `0.0.0.0:8001`
- **NPM Forward IP:** `10.120.0.80`
- **NPM Forward Port:** `8001`
- **Scheme:** `http`
- **Websockets:** Enabled

---

## ✅ 2. Restarting Your Django/Flask Project

Whenever you change your Python code (views, serializers, URLs, templates, etc.),
you must restart Gunicorn:

```bash
sudo systemctl restart bisk_rfv4-prod-gunicorn.service
```

Check status:

```bash
sudo systemctl status bisk_rfv4-prod-gunicorn.service
```

---

## ✅ 3. Running Django Migrations

When you change models or install apps:

```bash
source /srv/BISK_RFv4/.venv/bin/activate
cd /srv/BISK_RFv4
python manage.py migrate
```

Then restart:

```bash
sudo systemctl restart bisk_rfv4-prod-gunicorn.service
```

---

## ✅ 4. Updating Static Files

If you use Django static files:

```bash
source /srv/BISK_RFv4/.venv/bin/activate
cd /srv/BISK_RFv4
python manage.py collectstatic --noinput
sudo systemctl restart bisk_rfv4-prod-gunicorn.service
```

---

## ✅ 5. Useful Aliases (Optional)

Add these to `~/.bashrc` for faster workflows:

```bash
alias br="sudo systemctl restart bisk_rfv4-prod-gunicorn.service"
alias bs="sudo systemctl status bisk_rfv4-prod-gunicorn.service"
alias bm="python manage.py migrate"
alias bcs="python manage.py collectstatic --noinput"
```

Reload your shell:

```bash
source ~/.bashrc
```

---

## ✅ 6. Testing Backend Connectivity

Test Gunicorn directly:

```bash
curl -I http://10.120.0.80:8001
```

Expected: HTTP/1.1 200 or 302  
(400 is also normal if Django rejects the Host header.)

---

## ✅ 7. Testing From Inside Nginx Proxy Manager Container

```bash
sudo docker exec -it nginx-proxy-manager-app-1 curl -I http://10.120.0.80:8001
```

If this works, NPM → Gunicorn connectivity is correct.

---

## 🟢 Everything Is Now Production-Ready

You can add unlimited Django/Flask apps by:

1. Running each app’s Gunicorn on a different port.
2. Adding a Proxy Host in NPM.
3. Requesting a Let’s Encrypt certificate.
4. Updating your FortiGate DNAT rules (if needed).

---

Document generated automatically.
