"""Production static delivery without runserver, external services, or credentials."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from django.conf import settings
from django.test import SimpleTestCase


class ProductionStaticTests(SimpleTestCase):
    def test_production_assets_load_with_compression_and_safe_cache_headers(self):
        # A fresh interpreter exercises production middleware and manifest loading
        # without reusing the test runner's DEBUG/storage/middleware instances.
        probe = r'''
import gzip
import os
import re
from pathlib import Path
from urllib.parse import urljoin
import django
django.setup()
from django.conf import settings
from django.core.management import call_command
from django.contrib.staticfiles.storage import staticfiles_storage
from django.test import Client

assert settings.DEBUG is False
call_command("collectstatic", interactive=False, verbosity=0)
client = Client(HTTP_HOST="localhost")
assets = {
    "optimizer/studio-kit.css": "text/css",
    "optimizer/studio-theme.js": "javascript",
    "optimizer/ui-kit/keak-mark.svg": "image/svg+xml",
}
for name, content_type in assets.items():
    url = staticfiles_storage.url(name)
    assert url != settings.STATIC_URL + name, url
    response = client.get(url, HTTP_ACCEPT_ENCODING="gzip")
    assert response.status_code == 200, (url, response.status_code)
    assert content_type in response["Content-Type"], response["Content-Type"]
    assert "immutable" in response["Cache-Control"], response.headers
    assert "Set-Cookie" not in response.headers
    body = b"".join(response.streaming_content)
    if name.endswith(".js"):
        assert response.get("Content-Encoding") == "gzip", response.headers
        original = Path(settings.BASE_DIR) / "optimizer/static" / name
        assert gzip.decompress(body) == original.read_bytes()
    assert not (Path(settings.STATIC_ROOT) / name).exists(), name

# Follow the actual collected CSS dependencies, including nested stylesheet imports
# and the adapter's relative logo URLs. Successful top-level CSS alone isn't
# enough to prove the interface will render correctly in production.
pending = [staticfiles_storage.url(name) for name in (
    "optimizer/studio-kit.css", "optimizer/ui-kit/ci-tokens.css",
    "optimizer/ui-kit/ci-typography.css")]
seen = set()
while pending:
    url = pending.pop()
    if url in seen:
        continue
    seen.add(url)
    response = client.get(url)
    assert response.status_code == 200, (url, response.status_code)
    body = b"".join(response.streaming_content)
    if "text/css" in response["Content-Type"]:
        for reference in re.findall(r"url\([\"']?([^\"')]+)[\"']?\)", body.decode()):
            target = urljoin(url, reference)
            assert target.startswith(settings.STATIC_URL), target
            pending.append(target)
assert not any("Switzer" in url or "/fonts/" in url for url in seen), seen
assert any("keak-wordmark." in url for url in seen), seen

# Login HTML uses the collected manifest and no development static handler.
login = client.get("/login/")
assert login.status_code == 200
assert staticfiles_storage.url("optimizer/studio-kit.css").encode() in login.content
for path in ("/static/.env", "/static/db.sqlite3", "/static/../.env"):
    assert client.get(path).status_code == 404, path
print("Production static delivery, gzip, caching, login links, and exclusions passed")
'''
        with tempfile.TemporaryDirectory(prefix="studio-static-") as directory:
            env = dict(os.environ)
            env.update({
                "DJANGO_SETTINGS_MODULE": "optimizer_site.settings",
                "DJANGO_DEBUG": "0",
                "DJANGO_SECRET_KEY": settings.SECRET_KEY,
                "DJANGO_STATIC_ROOT": str(Path(directory) / "static"),
                "DB_PATH": str(Path(directory) / "app.sqlite3"),
                "SENTRY_DSN": "",
                "PYTHONPATH": os.pathsep.join((str(settings.BASE_DIR), str(settings.PROJECT_ROOT))),
            })
            result = subprocess.run([sys.executable, "-c", probe], env=env,
                                    cwd=settings.PROJECT_ROOT, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
