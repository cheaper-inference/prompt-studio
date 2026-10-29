import time
from django.core.management.base import BaseCommand
from optimizer.durable import process_one

class Command(BaseCommand):
    help = "Process durable optimization jobs; run as a separate worker service."
    def add_arguments(self, parser):
        parser.add_argument("--once", action="store_true")
    def handle(self, *args, **options):
        import os
        import sys
        print(f"[durable] worker started · lane {os.environ.get('GP_JOB_LANE', 'any')}", file=sys.stderr, flush=True)
        while True:
            handled = process_one()
            if options["once"]:
                return
            if not handled:
                time.sleep(2)
