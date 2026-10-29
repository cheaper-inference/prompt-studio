import time
from datetime import datetime, timezone
from django.core.management.base import BaseCommand, CommandError
from optimizer.benchmark_timing import parse, summarize
from optimizer.models import BenchmarkCall, JobState, Run


class Command(BaseCommand):
    help = "Live, read-only log of one benchmark's calls as they finish (default: latest running); prints no prompts or answers."

    def add_arguments(self, parser):
        parser.add_argument("job_id", nargs="?", help="Job id or prefix; default: the latest running benchmark.")
        parser.add_argument("--interval", type=float, default=3.0)

    def handle(self, *args, job_id=None, interval=3.0, **options):
        jobs = JobState.objects.filter(kind="benchmark")
        if job_id:
            matches = list(jobs.filter(pk__startswith=job_id).order_by("-created_at")[:6])
            if not matches:
                raise CommandError("No matching benchmark.")
            if len(matches) > 1:   # never guess which run an operator meant
                raise CommandError(f"Prefix {job_id!r} matches several benchmarks ({', '.join(m.id[:12] for m in matches[:5])}"
                                   f"{', …' if len(matches) > 5 else ''}); use a longer prefix.")
            job = matches[0]
        else:
            job = jobs.filter(status="running").order_by("-created_at").first()
            if job is None:
                raise CommandError("No running benchmark.")
        run = Run.objects.filter(job_id=job.id).first()
        scoring = ((run.config if run else None) or {}).get("scoring") or {}
        self.stdout.write(f"job {job.id}  judge={scoring.get('judge_model')}")
        self.stdout.write(f"{'seq':>4} {'role':5} {'model':24} {'var':9} {'case':>4} {'started':8} {'took':>7}")
        shown = set()
        while True:
            # Status first: once it reads terminal, the row read below is the final one.
            state = JobState.objects.filter(pk=job.id).values_list("status", flat=True).first()
            rows = BenchmarkCall.objects.filter(job_id=job.id).exclude(seq__in=shown).order_by("seq")
            in_flight = 0
            for r in rows:
                if r.in_flight:
                    in_flight += 1
                    continue
                shown.add(r.seq)
                started = parse(r.observed_at)
                took = "-" if r.duration_ms is None else f"{r.duration_ms / 1000:.1f}s"
                self.stdout.write(f"{r.seq:>4} {r.role:5} {str(r.model)[:24]:24} {str(r.variant):9} {str(r.case_index):>4} "
                                  f"{started.astimezone().strftime('%H:%M:%S') if started else '-':8} {took:>7}")
            if state not in ("queued", "running"):
                summary = summarize(run) if run else {}
                self.stdout.write(f"ended: {state} | wall {summary.get('wall_s')}s | calls {summary.get('calls')} | "
                                  f"concurrency {summary.get('concurrency')}")
                return
            time.sleep(interval)
