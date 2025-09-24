# From your project root (the one with manage.py)
python - <<'PY'
import pathlib, shutil, rest_framework
src = pathlib.Path(rest_framework.__file__).parent / 'templates' / 'rest_framework' / 'base.html'
dst = pathlib.Path('templates') / 'rest_framework' / 'base.html'
dst.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(src, dst)
print(f'Copied:\n  {src}\n-> {dst}')
PY
