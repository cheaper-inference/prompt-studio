import json
from django.core.management.base import BaseCommand
from optimizer.benchmark_timing import recent


class Command(BaseCommand):
    help = "Read-only timing summary of the latest benchmark runs (prints no prompts, answers or credentials)."

    def add_arguments(self, parser):
        parser.add_argument("limit", nargs="?", type=int, default=15)
        parser.add_argument("--json", action="store_true", help="Print the full summaries as JSON.")

    def handle(self, *args, limit, **options):
        runs = recent(max(1, min(limit, 200)))
        if options["json"]:
            self.stdout.write(json.dumps(runs, indent=1, default=str))
            return
        fmt = lambda v: "-" if v is None else str(v)
        self.stdout.write(f"{'created':16} {'status':11} {'cases':>5} {'answers':>8} {'judge chk':>9} {'1st call':>8} "
                          f"{'wall':>7} {'conc':>5} {'task med/p90':>13} {'judge med/p90':>14}  judge")
        for r in runs:
            t, j = r["task_call_s"], r["judge_call_s"]
            self.stdout.write(
                f"{r['created_at'][:16]:16} {r['status']:11} {r['cases']:>5} {r['answers']:>8} "
                f"{r['judge_check_s']:>8}s {fmt(r['to_first_call_s']):>8} {fmt(r['wall_s']):>7} {fmt(r['concurrency']):>5} "
                f"{fmt(t['median']) + '/' + fmt(t['p90']):>13} {fmt(j['median']) + '/' + fmt(j['p90']):>14}  {r['judge_model']}")
