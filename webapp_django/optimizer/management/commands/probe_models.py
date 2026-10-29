"""Refresh Model Standings: probe every catalog model across the three roles.

This is the scheduled entry point. The webapp runs on your machine, so schedule
it with your OS scheduler (cron / launchd), NOT a cloud task — a cloud task can't
write to this local DB. Example (daily at 03:00, launchd/cron):

    0 3 * * *  cd /path/to/GepaProject/webapp_django && \
               /path/to/.venv/bin/python manage.py probe_models

The manual "Run diagnostics" button in the Standings tab does the same thing.
"""

from __future__ import annotations

from django.core.management.base import BaseCommand

from optimizer import services


class Command(BaseCommand):
    help = "Probe latency for every catalog model across task/judge/reflection roles."

    def add_arguments(self, parser):
        parser.add_argument("--timeout", type=int, default=60,
                            help="per-call timeout seconds (default 60)")
        parser.add_argument("--models", nargs="*", default=None,
                            help="specific model ids (default: whole catalog)")

    def handle(self, *args, **opts):
        res = services.run_diagnostics(models=opts.get("models"),
                                       timeout=opts["timeout"])
        self.stdout.write(self.style.SUCCESS(
            f"probed {res['probed']} (model, role) pairs across "
            f"{res['models']} model(s)"))
