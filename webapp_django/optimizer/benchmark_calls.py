"""Row storage for benchmark billing: one BenchmarkCall per task/judge call.

Live progress publishes only counters; the per-call lists are assembled from rows
exactly once, when the run ends or recovery interrupts it.
"""
from .models import BenchmarkCall

CALL_FIELDS = ('request_id', 'role', 'model', 'contender_model', 'variant', 'case_index',
               'completed', 'in_flight', 'observed_at')


def insert(job_id, seq, call):
    BenchmarkCall.objects.create(job_id=job_id, seq=seq, **{k: call.get(k) for k in CALL_FIELDS})


def update(job_id, seq, **fields):
    BenchmarkCall.objects.filter(job_id=job_id, seq=seq).update(**fields)


def complete_answer(job_id, contender_model, variant, case_index):
    BenchmarkCall.objects.filter(job_id=job_id, contender_model=contender_model, variant=variant,
                                 case_index=case_index).update(completed=True)


def assemble(job_id, billing):
    """Return terminal `billing` with `calls` and `receipts` rebuilt from rows.

    No rows means a complete zero-call ledger (the worker ended before its first call):
    both lists are returned empty, so clients can tell it from a live counters snapshot."""
    if not isinstance(billing, dict):
        return billing
    rows = list(BenchmarkCall.objects.filter(job_id=job_id).order_by('seq'))
    calls = [{k: getattr(row, k) for k in CALL_FIELDS} for row in rows]
    receipts = [{'request_id': row.request_id, 'status': row.receipt_status, 'billed_cost_usd': row.billed_cost_usd,
                 'currency': 'USD', 'role': row.role, 'model': row.receipt_model,
                 'contender_model': row.contender_model, 'variant': row.variant, 'case_index': row.case_index}
                for row in rows if row.receipt_status]
    return {**billing, 'calls': calls, 'receipts': receipts}
