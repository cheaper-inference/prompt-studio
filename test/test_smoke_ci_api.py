"""End-to-end smoke test for the Cheaper Inference service API (/api/v1) — bearer variant.

Verifies the auth chain — bearer service token + trusted X-CI-User-Id — against a LOCAL
runserver, with no real CI key needed (it hits the read-only endpoints).

1) Start the server with matching TEST env:

   PROMPT_STUDIO_API_ENABLED=1 PROMPT_STUDIO_SERVICE_TOKEN=smoke-token \
   CHEAPERINFERENCE_API_KEY=dummy CHEAPERINFERENCE_BASE_URL=https://example.invalid/v1 \
   python manage.py runserver 127.0.0.1:8000

2) In another shell, run this with the SAME token:

   PROMPT_STUDIO_SERVICE_TOKEN=smoke-token python test/test_smoke_ci_api.py   # --base to override

No extra dependencies (stdlib only).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

TOKEN = os.environ.get("PROMPT_STUDIO_SERVICE_TOKEN", "smoke-token")
USER = os.environ.get("CI_SMOKE_USER_ID", "user-smoke-123")

_passed = _failed = 0


def _check(name: str, ok: bool, detail: str = "") -> None:
    global _passed, _failed
    if ok:
        _passed += 1
    else:
        _failed += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail and not ok else ""))


def call(base, method, path, body_obj=None, *, token=TOKEN, user_id=USER, omit_user=False):
    body = b"" if body_obj is None else json.dumps(body_obj, separators=(",", ":")).encode()
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    if user_id and not omit_user:
        headers["X-CI-User-Id"] = user_id
    req = urllib.request.Request(base + path, data=(body or None), headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--run-optimize", action="store_true",
                    help="also POST /api/v1/optimize (needs real CI creds on the server)")
    args = ap.parse_args()
    base = args.base.rstrip("/")
    print(f"\nSmoke — {base}  (bearer token from env)\n")

    # happy path: valid token + user id on read-only endpoints
    st, body = call(base, "GET", "/api/v1/runs")
    _check("GET /api/v1/runs → 200 with {runs}", st == 200 and '"runs"' in body, f"got {st}: {body[:120]}")

    st, _ = call(base, "GET", "/api/v1/standings")
    _check("GET /api/v1/standings → 200", st == 200, f"got {st}")

    # no service token → 401
    st, _ = call(base, "GET", "/api/v1/runs", token=None)
    _check("no bearer token → 401", st == 401, f"got {st}")

    # wrong service token → 401
    st, _ = call(base, "GET", "/api/v1/runs", token="not-the-token")
    _check("wrong bearer token → 401", st == 401, f"got {st}")

    # missing X-CI-User-Id → 401
    st, _ = call(base, "GET", "/api/v1/runs", omit_user=True)
    _check("missing X-CI-User-Id → 401", st == 401, f"got {st}")

    # unknown job id → 404 (auth passed, resource missing)
    st, _ = call(base, "GET", "/api/v1/jobs/does-not-exist")
    _check("GET /api/v1/jobs/<unknown> → 404", st == 404, f"got {st}")

    # non-CI callback_url → 400 (SSRF guard)
    st, _ = call(base, "POST", "/api/v1/optimize",
                 {"mode": "zero_shot", "task_description": "x",
                  "callback_url": "https://evil.example.com/hook"})
    _check("POST optimize w/ non-CI callback_url → 400", st == 400, f"got {st}")

    if args.run_optimize:
        st, body = call(base, "POST", "/api/v1/optimize",
                        {"mode": "zero_shot", "task_description": "keyword ideas",
                         "user_prompt": "Give three keywords for {{topic}}.",
                         "max_metric_calls": 4})
        _check("POST /api/v1/optimize → 200 {job_id}", st == 200 and '"job_id"' in body, f"got {st}: {body[:160]}")

    print(f"\n{_passed} passed, {_failed} failed\n")
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
