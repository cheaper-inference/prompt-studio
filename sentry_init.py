"""Basic Sentry error monitoring, shared by every GepaProject entry point.

Errors only: no tracing, profiling or logs. Disabled unless an operator supplies
SENTRY_DSN for their own project.

Reports from production only: Railway's "production" environment, or an
explicit SENTRY_ENVIRONMENT=production. Local runs, tests and other Railway
environments send nothing.
"""
from __future__ import annotations

import os

import sentry_sdk


def init_sentry() -> None:
    """Initialise Sentry once per process; later calls are no-ops."""
    dsn = os.environ.get("SENTRY_DSN", "").strip()
    if not dsn:
        return
    if sentry_sdk.is_initialized():
        return
    environment = os.environ.get(
        "SENTRY_ENVIRONMENT", os.environ.get("RAILWAY_ENVIRONMENT_NAME", "development")
    )
    if environment != "production":
        return
    sentry_sdk.init(
        dsn=dsn,
        environment=environment,
        traces_sample_rate=0,
        send_default_pii=False,
    )
