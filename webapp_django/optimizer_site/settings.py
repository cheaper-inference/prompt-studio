"""Plain-Django settings for the prompt-optimizer reference app.

Small on purpose: a handful of contrib apps, a single SQLite database (users +
run History), and no admin. Optimization jobs use a durable leased queue and a separate worker process
sharing this service's persistent SQLite volume (see the Dockerfile). MLflow keeps
its own SQL store via the prompt_optimizer library.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from django.core.exceptions import ImproperlyConfigured

# webapp_django/ ; its parent is the GepaProject root that holds
# prompt_optimizer.py and llm_endpoint.py — make them importable.
BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Basic Sentry error monitoring for the web app and management commands (run_jobs).
from sentry_init import init_sentry  # noqa: E402

# Debug pages require an explicit local-development opt-in.
DEBUG = os.environ.get("DJANGO_DEBUG", "0") == "1"
SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "")
if (len(SECRET_KEY) < 50 or len(set(SECRET_KEY)) < 5
        or SECRET_KEY.startswith("django-insecure-")):
    raise ImproperlyConfigured(
        "Set DJANGO_SECRET_KEY to a generated secret of at least 50 characters. "
        "Use: python -c 'import secrets; print(secrets.token_urlsafe(64))'"
    )

# Self-service signup. OFF by default: issue accounts manually (manage.py adduser).
# Set DJANGO_ALLOW_SIGNUP=1 to reopen public sign-ups.
ALLOW_SIGNUP = os.environ.get("DJANGO_ALLOW_SIGNUP", "0") == "1"

# --- Cheaper Inference service API (simple variant). Inert unless PROMPT_STUDIO_API_ENABLED=1.
# CI is the only trusted caller: a shared bearer token (service auth) + a trusted X-CI-User-Id
# header (identity). The user's inference key rides in each request and is never stored.
PROMPT_STUDIO_API_ENABLED = os.environ.get("PROMPT_STUDIO_API_ENABLED", "0") == "1"
PROMPT_STUDIO_SERVICE_TOKEN = os.environ.get("PROMPT_STUDIO_SERVICE_TOKEN", "")
PROMPT_STUDIO_API_IP_ALLOWLIST = os.environ.get("PROMPT_STUDIO_API_IP_ALLOWLIST", "")        # comma-separated; empty = no IP gate
PROMPT_STUDIO_WEBHOOK_SECRET = os.environ.get("PROMPT_STUDIO_WEBHOOK_SECRET", "")  # sign our callbacks; falls back to PROMPT_STUDIO_SERVICE_TOKEN
PROMPT_STUDIO_WEBHOOK_ALLOWED_HOSTS = os.environ.get("PROMPT_STUDIO_WEBHOOK_ALLOWED_HOSTS", "")  # empty = *.cheaperinference.com


def env_list(name: str) -> list[str]:
    return [value.strip() for value in os.environ.get(name, "").split(",") if value.strip()]


ALLOWED_HOSTS = env_list("DJANGO_ALLOWED_HOSTS") or ["localhost", "127.0.0.1", "[::1]"]
railway_domain = os.environ.get("RAILWAY_PUBLIC_DOMAIN", "").strip()
if railway_domain and railway_domain not in ALLOWED_HOSTS:
    ALLOWED_HOSTS.append(railway_domain)
if "*" in ALLOWED_HOSTS:
    raise ImproperlyConfigured("DJANGO_ALLOWED_HOSTS must name hosts explicitly; '*' is unsafe.")

# Same-origin requests work without exceptions. Cross-origin browser clients must
# be explicitly trusted; do not trust every tenant of a hosting provider.
CSRF_TRUSTED_ORIGINS = env_list("DJANGO_CSRF_TRUSTED_ORIGINS")

# Enable only behind a controlled proxy that strips client-supplied forwarded
# headers and sets X-Forwarded-Proto itself (e.g. the Railway HTTPS proxy).
SECURE_PROXY_SSL_HEADER = (
    ("HTTP_X_FORWARDED_PROTO", "https")
    if os.environ.get("DJANGO_TRUST_PROXY", "0") == "1" else None
)
USE_X_FORWARDED_HOST = False
SESSION_COOKIE_SECURE = not DEBUG
CSRF_COOKIE_SECURE = not DEBUG

# Initialise optional telemetry only after validating configuration.
init_sentry()

INSTALLED_APPS = [
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.staticfiles",
    "optimizer",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
]

LOGIN_URL = "/login"
LOGIN_REDIRECT_URL = "/"
LOGOUT_REDIRECT_URL = "/login"

ROOT_URLCONF = "optimizer_site.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {"context_processors": []},
    },
]

WSGI_APPLICATION = "optimizer_site.wsgi.application"

# Run history (optimization + A/B) is persisted here so it survives restarts and
# is queryable. SQLite keeps it dependency-light; point DB_PATH at a persistent
# volume in production. Run `python manage.py migrate` before first use.
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        # WAL lets readers run while one worker writes; IMMEDIATE takes the write lock at
        # BEGIN, avoiding SQLite's lock-upgrade "database is locked" errors when several
        # queue workers (GP_JOB_WORKERS) write job progress at once.
        "OPTIONS": {
            "timeout": 30,
            "init_command": "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;",
            "transaction_mode": "IMMEDIATE",
        },
        # Point DB_PATH at a persistent volume in production (e.g. /app/data/db.sqlite3)
        # so run History and user accounts survive redeploys. Run migrate after deploy.
        "NAME": os.environ.get("DB_PATH") or (BASE_DIR / "db.sqlite3"),
    }
}

STATIC_URL = "static/"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# Send request errors to stdout so a 500 traceback shows up in the platform log
# stream (Railway). Without this, Django's default routes 500s only to the
# unconfigured 'mail_admins' handler with DEBUG=False, so the exception is silently
# swallowed — a generic "Server Error (500)" with nothing to read.
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {"console": {"class": "logging.StreamHandler"}},
    "root": {"handlers": ["console"], "level": "INFO"},
    "loggers": {
        "django.request": {"handlers": ["console"], "level": "ERROR", "propagate": False},
    },
}

# Separate Fernet key; persists encrypted delegated keys only until jobs finish.
PROMPT_STUDIO_CREDENTIAL_KEY = os.environ.get("PROMPT_STUDIO_CREDENTIAL_KEY", "")
PROMPT_STUDIO_INFERENCE_BASE_URL = os.environ.get("PROMPT_STUDIO_INFERENCE_BASE_URL", "")
