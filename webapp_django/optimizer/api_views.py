"""Thin /api/v1 JSON views over services.py, gated by @ci_api (bearer + X-CI-User-Id).

Identity is the asserted CI user id (request.ci_user_id) and scopes persistence. The user's CI
inference key rides in the request (body api_key / header X-CI-Inference-Key), is encrypted for the durable job and erased when it reaches a terminal state. An optional `callback_url` (https, CI host)
enables the completion webhook; polling is always available.
"""
from __future__ import annotations

import json

from django.conf import settings
from django.http import HttpResponseNotAllowed, JsonResponse

from . import services
from .ci_api import callback_allowed, ci_api


def _body(request) -> dict:
    try:
        return json.loads(request.body or b"{}")
    except (ValueError, TypeError):
        return {}


def _prepare(request, required=()):
    """Build the job payload: trusted owner, frozen endpoint, key + optional webhook.
    Validates JSON and required fields up front. Returns (payload, error_message)."""
    try:
        payload = json.loads(request.body or b"{}")
    except (ValueError, TypeError):
        return None, "request body is not valid JSON"
    if not isinstance(payload, dict):
        return None, "request body must be a JSON object"
    missing = [k for k in required if k not in payload]
    if missing:
        return None, "missing required field(s): " + ", ".join(missing)
    payload["_owner"] = request.ci_user_id
    key = payload.pop("api_key", None) or request.META.get("HTTP_X_CI_INFERENCE_KEY")
    base = payload.pop("base_url", None)
    if base is None or base == "":
        base = settings.PROMPT_STUDIO_INFERENCE_BASE_URL
    if not isinstance(base, str) or not base:
        return None, "base_url must be a non-empty string"
    if key:
        payload["_api_key"] = key
    # Freeze the configured fallback before scored validation and durable enqueue;
    # a worker restart must not pick a different default from its environment.
    payload["_base_url"] = base.rstrip("/")
    cb = payload.pop("callback_url", None) or request.META.get("HTTP_X_CI_CALLBACK_URL")
    if cb:
        if not callback_allowed(cb):
            return None, "callback_url must be https and a Cheaper Inference host"
        payload["_callback_url"] = cb
    return payload, None


def _start(request, starter, required=()):
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])
    payload, err = _prepare(request, required)
    if err:
        return JsonResponse({"error": err}, status=400)
    payload["_idempotency_key"] = request.META.get("HTTP_IDEMPOTENCY_KEY")
    try:
        if payload.get("require_scored_evidence"):
            from .scored import validate_scored_payload
            validate_scored_payload(payload)
            if not payload["_idempotency_key"]:
                raise ValueError("Idempotency-Key is required for scored jobs")
        job = starter(payload)
    except ValueError as error:
        from .durable import IdempotencyConflict
        return JsonResponse({"error": str(error)}, status=409 if isinstance(error, IdempotencyConflict) else 400)
    return JsonResponse({"job_id": job.id, "status": "running"}, status=202)


@ci_api
def optimize(request):
    return _start(request, services.start_job, required=("mode",))


@ci_api
def experiment(request):
    return _start(request, services.start_experiment, required=("lever", "arms"))


@ci_api
def anchor(request):
    return _start(request, services.start_anchor, required=("mode",))


@ci_api
def diagnostics(request):
    return _start(request, services.start_diagnostics)


@ci_api
def job(request, job_id):
    from .models import Run
    benchmark = Run.objects.filter(job_id=job_id, kind="benchmark", owner=request.ci_user_id).first()
    st = services.get_job_state(job_id, request.ci_user_id)
    if not st or not st.get("status"):
        return JsonResponse({"error": "job not found"}, status=404)
    if benchmark is not None and benchmark.status == "cancelled":
        st = {**st, "status":"cancelled", "stage":"Cancelled. An in-flight call may still finish and be billed."}
    if benchmark is not None and benchmark.status == "cancelling":
        st = {**st, "status":"cancelling", "stage":"Cancelling after the current inference call finishes…"}
    if benchmark is not None and isinstance(st.get("result"), dict) and "billing" in st["result"]:
        # Include call IDs for CI ledger reconciliation; no credentials are retained.
        billing = st["result"]["billing"]
        if isinstance(billing, dict) and "calls" not in billing and st.get("status") not in ("queued", "running", "cancelling"):
            # Ended without its final snapshot (worker lost, or too large to store).
            from .benchmark_calls import assemble
            billing = assemble(job_id, billing)
        if isinstance(billing, dict):
            st = {**st, "billing": {key: value for key, value in billing.items() if key != "receipts"}}
            if st.get("status") != "done":
                st["result"] = None
    return JsonResponse({"job_id": job_id, **st})


@ci_api
def benchmark(request):
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])
    from . import benchmarks
    payload, error = _prepare(request, required=("baseline", "rows", "models"))
    if error:
        return JsonResponse({"error": error}, status=400)
    payload["_idempotency_key"] = request.META.get("HTTP_IDEMPOTENCY_KEY")
    try:
        return JsonResponse({"job_id": benchmarks.start(payload).id})
    except ValueError as exc:
        from .durable import IdempotencyConflict
        return JsonResponse({"error": str(exc)}, status=409 if isinstance(exc, IdempotencyConflict) else 400)


@ci_api
def draft_criteria(request):
    """Draft benchmark quality criteria from the baseline prompt-set (spec-draft-v2).
    Deterministic and instant: no model call, no job. The caller confirms or edits."""
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])
    body = _body(request)
    baseline = body.get("baseline") if isinstance(body, dict) else None
    if (not isinstance(baseline, list) or not baseline or len(baseline) > 20
            or not all(isinstance(m, dict) and m.get("role") in ("system", "user")
                       and isinstance(m.get("content"), str) and len(m["content"]) <= 100_000 for m in baseline)):
        return JsonResponse({"error": "baseline must be a list of 1-20 {role: system|user, content} messages"}, status=400)
    from ps_utils.spec_draft import VERSION, draft_criteria as draft
    return JsonResponse({"version": VERSION, "criteria": draft(baseline)})


@ci_api
def runs(request):
    from .models import Run
    qs = Run.objects.filter(owner=request.ci_user_id)[:200]
    return JsonResponse({"runs": [r.row() for r in qs]})


@ci_api
def run_detail(request, run_id):
    from .models import Run
    try:
        r = Run.objects.get(pk=run_id, owner=request.ci_user_id)
    except Exception:  # noqa: BLE001
        return JsonResponse({"error": "run not found"}, status=404)
    detail = r.detail()
    billing = (detail.get("result") or {}).get("billing") if isinstance(detail.get("result"), dict) else None
    if r.kind == "benchmark" and r.job_id and isinstance(billing, dict) and "calls" not in billing \
            and r.status not in ("queued", "running"):
        from .benchmark_calls import assemble
        detail["result"] = {**detail["result"], "billing": assemble(r.job_id, billing)}
    return JsonResponse(detail)


@ci_api
def standings(request):
    return JsonResponse(services.standings())


@ci_api
def cancel_job(request, job_id):
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])
    from .durable import cancel
    row = cancel(job_id, request.ci_user_id)
    if row is None:
        return JsonResponse({"error": "job not found"}, status=404)
    status = row.status
    if row.kind == "benchmark" and row.status == "running" and row.cancel_requested_at:
        status = "cancelled"
    return JsonResponse({"job_id": row.id, "status": status, "cancel_requested": True})


@ci_api
def capabilities(request):
    if request.method != "GET":
        return HttpResponseNotAllowed(["GET"])
    return JsonResponse({"durable_scored_jobs": 1, "idempotent_submission": True,
        "owner_scoped_jobs": True, "metered_inference": True,
        "scorer_revisions": ["gepa-holdout-v1"]})
