# scripts/brand_drf.sh
#!/usr/bin/env bash
set -euo pipefail
BRAND_NAME="${1:-BISK AI Attendance API}"
BRAND_URL="${2:-/}"

python - "$BRAND_NAME" "$BRAND_URL" <<'PY'
import sys, pathlib, shutil, rest_framework, re
brand_name, brand_url = sys.argv[1], sys.argv[2]
src = pathlib.Path(rest_framework.__file__).parent / 'templates' / 'rest_framework' / 'base.html'
dst_dir = pathlib.Path('templates') / 'rest_framework'
dst = dst_dir / 'base.html'
dst_dir.mkdir(parents=True, exist_ok=True)
text = src.read_text(encoding='utf-8')
text = re.sub(
    r'{% block branding %}.*?{% endblock %}',
    '{% block branding %}\\n  <a class="navbar-brand" rel="nofollow" href="%s">%s</a>\\n{% endblock %}' % (brand_url, brand_name),
    text, flags=re.S)
dst.write_text(text, encoding='utf-8')
print(f'Wrote {dst}')
PY
