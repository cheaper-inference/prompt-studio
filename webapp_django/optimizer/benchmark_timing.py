"""Read-only timing and cost of benchmark runs from Run, JobState and BenchmarkCall rows.

Used by `manage.py benchmark_timings` (per-run timing), `manage.py benchmark_costs`
(per-run spend) and `manage.py watch_benchmark` (live per-call log). They load run
records (whose JSON holds prompts and answers) but output only metadata, timing and cost:
never prompts, answers or credentials.
"""
from datetime import datetime
from decimal import Decimal, InvalidOperation

from .models import BenchmarkCall, JobState, Run


def parse(value):
    try:
        return datetime.fromisoformat(value) if isinstance(value, str) else value
    except ValueError:
        return None


def seconds(start, end):
    return round((end - start).total_seconds(), 1) if start and end else None


def percentile(values, q):
    """Linear-interpolated percentile (q in 0..1): the true median for even counts,
    and p90 below the maximum once there are enough samples."""
    if not values:
        return None
    values = sorted(values)
    position = q * (len(values) - 1)
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return round(values[low] + (values[high] - values[low]) * (position - low), 1)


def summarize(run):
    """One run's timing. Durations come from BenchmarkCall.duration_ms (true per-call
    latency, valid when calls overlap); older runs without it report None."""
    manifest = run.config or {}
    scoring = manifest.get("scoring") or {}
    tried = (scoring.get("judge_selection") or {}).get("tried") or []
    job = JobState.objects.filter(pk=run.job_id).only("updated_at").first()
    rows = list(BenchmarkCall.objects.filter(job_id=run.job_id).order_by("seq"))
    starts = [parse(r.observed_at) for r in rows if parse(r.observed_at)]
    first, last_start = (min(starts), max(starts)) if starts else (None, None)
    durations = {"task": [], "judge": []}
    intervals = []   # (dispatch, arrival) per call, from migration 0011 on
    for r in rows:
        if r.duration_ms is not None and r.role in durations:
            durations[r.role].append(r.duration_ms / 1000)
        if r.duration_ms is not None and r.completed_at is not None:
            end = r.completed_at.timestamp()
            intervals.append((end - r.duration_ms / 1000, end))
    # Concurrency = summed call time / time with at least one call in flight (the union of the
    # intervals): idle gaps and bookkeeping before dispatch do not count, so sequential = 1.0.
    busy = sum(end - start for start, end in intervals)
    active, covered_to = 0.0, None
    for start, end in sorted(intervals):
        if covered_to is not None and start < covered_to:   # overlaps what is already counted
            start = covered_to
        if end > start:
            active += end - start
        covered_to = end if covered_to is None else max(covered_to, end)
    window = active or None
    billing = (run.result or {}).get("billing") if isinstance(run.result, dict) else None
    billing = billing if isinstance(billing, dict) else {}
    buckets = (billing.get("models") or {}).values()
    return {
        "run_id": str(run.id), "job_id": run.job_id, "status": run.status, "created_at": run.created_at.isoformat(),
        "models": manifest.get("models"), "cases": len(((manifest.get("dataset") or {}).get("rows")) or []),
        "variants": list((manifest.get("variants") or {}).keys()), "judge_model": scoring.get("judge_model"),
        "judge_check_s": round(sum(t.get("ms") or 0 for t in tried) / 1000, 1),
        # Submission to first model call: queue wait plus worker setup (MLflow, arms), not queue alone.
        "to_first_call_s": seconds(run.created_at, first),
        "wall_s": seconds(run.created_at, job.updated_at if job else None),
        "answers": f"{sum(b.get('completed_answers', 0) for b in buckets)}/{sum(b.get('total_answers', 0) for b in buckets)}",
        "calls": {"task": sum(r.role == "task" for r in rows), "judge": sum(r.role == "judge" for r in rows),
                  "in_flight": sum(r.in_flight for r in rows), "timed": sum(r.duration_ms is not None for r in rows)},
        "task_call_s": {"median": percentile(durations["task"], .5), "p90": percentile(durations["task"], .9)},
        "judge_call_s": {"median": percentile(durations["judge"], .5), "p90": percentile(durations["judge"], .9)},
        # Average calls in flight while the run was making calls (1.0 = strictly sequential).
        "concurrency": round(busy / window, 1) if window and window > 0 else None,
        "spend_usd": billing.get("reported_spend_usd"),
    }


def recent(limit):
    return [summarize(r) for r in Run.objects.filter(kind="benchmark").order_by("-created_at")[:limit]]


def _money(value):
    try:
        amount = Decimal(str(value))
        return amount if amount.is_finite() else None
    except (InvalidOperation, ValueError):
        return None


def cost(run):
    """One run's incurred spend: the run's reported total (CI-priced receipts), split by
    role from its per-call rows. Unpriced calls are counted, never treated as free.
    The judge health check is billed separately by CI and is not included."""
    manifest = run.config or {}
    billing = (run.result or {}).get("billing") if isinstance(run.result, dict) else None
    billing = billing if isinstance(billing, dict) else {}
    rows = list(BenchmarkCall.objects.filter(job_id=run.job_id).only("role", "billed_cost_usd"))
    # Per-call rows exist from migration 0009 on; older runs fall back to their stored receipts.
    priced = [(r.role, r.billed_cost_usd) for r in rows if r.billed_cost_usd] if rows else \
        [(x.get("role"), x.get("billed_cost_usd")) for x in billing.get("receipts") or [] if isinstance(x, dict)]
    by_role = None
    if rows or billing.get("receipts"):
        by_role = {"task": Decimal("0"), "judge": Decimal("0")}
        for role, value in priced:
            amount = _money(value)
            if amount is not None and role in by_role:
                by_role[role] += amount
    buckets = list((billing.get("models") or {}).values())
    answers = sum(b.get("completed_answers", 0) for b in buckets)
    total = _money(billing.get("reported_spend_usd"))
    return {
        "run_id": str(run.id), "job_id": run.job_id, "status": run.status, "created_at": run.created_at.isoformat(),
        "models": manifest.get("models"), "judge_model": (manifest.get("scoring") or {}).get("judge_model"),
        "answers": f"{answers}/{sum(b.get('total_answers', 0) for b in buckets)}",
        "spend_usd": str(total) if total is not None else None,
        "task_usd": str(by_role["task"]) if by_role else None, "judge_usd": str(by_role["judge"]) if by_role else None,
        "per_answer_usd": str((total / answers).quantize(Decimal("0.000001"))) if total is not None and answers else None,
        "receipts": billing.get("calls_with_receipts"), "unpriced_calls": billing.get("calls_missing_receipts"),
    }


def recent_costs(limit):
    return [cost(r) for r in Run.objects.filter(kind="benchmark").order_by("-created_at")[:limit]]
