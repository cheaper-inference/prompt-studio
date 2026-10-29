"""Service-to-service gate for the Cheaper Inference (CI) integration — simple variant.

CI's backend is the ONLY trusted caller. Every /api/v1 request passes two checks:
  1. Service auth — a shared BEARER token only CI holds.
  2. User identity — CI asserts the signed-in user in the X-CI-User-Id header (trusted because
     the request already proved it is CI).
The scoped CI inference key is encrypted for the durable job and erased at termination.

All inert unless settings.PROMPT_STUDIO_API_ENABLED.
"""
from __future__ import annotations

import functools
import hmac
from urllib.parse import urlparse

from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt


def _err(msg: str, status: int = 401):
    return JsonResponse({"error": msg}, status=status)


def _client_ip(request) -> str:
    xff = request.META.get("HTTP_X_FORWARDED_FOR", "")
    return (xff.split(",")[0] if xff else request.META.get("REMOTE_ADDR", "")).strip()


def _ip_allowed(request) -> bool:
    allow = [x.strip() for x in (getattr(settings, "PROMPT_STUDIO_API_IP_ALLOWLIST", "") or "").split(",") if x.strip()]
    return True if not allow else _client_ip(request) in allow


def _token_ok(request) -> bool:
    expected = getattr(settings, "PROMPT_STUDIO_SERVICE_TOKEN", "")
    if not expected:
        return False
    auth = request.META.get("HTTP_AUTHORIZATION", "")
    token = request.META.get("HTTP_X_PROMPT_STUDIO_SERVICE_TOKEN", "") or (
        auth.split(" ", 1)[1] if auth.lower().startswith("bearer ") else "")
    return bool(token) and hmac.compare_digest(token, expected)


def callback_allowed(url: str) -> bool:
    """SSRF guard for webhook callback URLs: https + a CI host. Defaults to *.cheaperinference.com."""
    try:
        u = urlparse(url)
    except Exception:  # noqa: BLE001
        return False
    if u.scheme != "https" or not u.hostname or u.username or u.password or u.fragment:
        return False
    host = u.hostname.lower()
    hosts = [h.strip().lower() for h in (getattr(settings, "PROMPT_STUDIO_WEBHOOK_ALLOWED_HOSTS", "") or "").split(",") if h.strip()]
    if not hosts:
        return host == "cheaperinference.com" or host.endswith(".cheaperinference.com")
    return any(host == h or host.endswith("." + h) for h in hosts)


def ci_api(view):
    """Gate a view: enabled-flag -> IP -> bearer token -> X-CI-User-Id.
    On success attaches request.ci_user_id / request.ci_user_email. CSRF-exempt (no cookies)."""
    @csrf_exempt
    @functools.wraps(view)
    def wrapped(request, *a, **k):
        if not getattr(settings, "PROMPT_STUDIO_API_ENABLED", False):
            return _err("API is not enabled", 404)
        if not _ip_allowed(request):
            return _err("forbidden", 403)
        if not _token_ok(request):
            return _err("bad or missing service token", 401)
        uid = (request.META.get("HTTP_X_CI_USER_ID") or "").strip()
        if not uid:
            return _err("missing X-CI-User-Id", 401)
        request.ci_user_id = uid
        request.ci_user_email = (request.META.get("HTTP_X_CI_USER_EMAIL") or "").strip()
        return view(request, *a, **k)
    return wrapped
