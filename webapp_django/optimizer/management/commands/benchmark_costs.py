import json
from decimal import Decimal
from django.core.management.base import BaseCommand
from optimizer.benchmark_timing import recent_costs


class Command(BaseCommand):
    help = "Read-only incurred cost of the last k benchmark runs (prints no prompts, answers or credentials)."

    def add_arguments(self, parser):
        parser.add_argument("limit", nargs="?", type=int, default=10)
        parser.add_argument("--json", action="store_true", help="Print the full per-run costs as JSON.")

    def handle(self, *args, limit, **options):
        runs = recent_costs(max(1, min(limit, 200)))
        if options["json"]:
            self.stdout.write(json.dumps(runs, indent=1))
            return
        usd = lambda v: "-" if v is None else f"${Decimal(v):.4f}"
        self.stdout.write(f"{'created':16} {'status':11} {'answers':>8} {'spend':>10} {'task':>10} {'judge':>10} "
                          f"{'per answer':>11} {'unpriced':>8}  models")
        # Spend comes from per-response receipts; calls without one are priced by CI's
        # ledger (by request ID), so a run with unpriced calls shows "n/a" (none priced)
        # or a lower bound "≥$x" (some priced), never a misleading $0.
        def shown(r, key):
            if not r["unpriced_calls"]:
                return usd(r[key])
            return "n/a" if not r["receipts"] else "≥" + usd(r[key])
        for r in runs:
            self.stdout.write(f"{r['created_at'][:16]:16} {r['status']:11} {r['answers']:>8} {shown(r, 'spend_usd'):>10} "
                              f"{shown(r, 'task_usd'):>10} {shown(r, 'judge_usd'):>10} {shown(r, 'per_answer_usd'):>11} "
                              f"{str(r['unpriced_calls'] if r['unpriced_calls'] is not None else '-'):>8}  {', '.join(r['models'] or [])}")
        priced = [r for r in runs if r["spend_usd"] is not None and not r["unpriced_calls"]]
        if not priced:
            self.stdout.write(f"total: n/a. None of the {len(runs)} runs is fully priced here; "
                              "CI prices unpriced calls from its ledger by request ID.")
            return
        total = sum((Decimal(r["spend_usd"]) for r in priced), Decimal("0"))
        coverage = f"{len(runs)} runs" if len(priced) == len(runs) else \
            f"{len(priced)} of {len(runs)} fully priced runs ({len(runs) - len(priced)} excluded; CI prices them)"
        self.stdout.write(f"total over {coverage}: ${total:.4f} (judge health checks are billed separately by CI)")
