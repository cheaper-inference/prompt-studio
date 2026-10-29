"""Scored GEPA jobs with frozen training data and disjoint held-out evidence."""
from __future__ import annotations
import hashlib
import json
import re
from decimal import Decimal, InvalidOperation
from types import SimpleNamespace
from urllib.parse import urlparse
from django.conf import settings
from django.db import transaction
from .durable import assert_lease, checkpoint, finish_job, UncertainInference
from .models import InferenceCall, Run


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def validate_scored_payload(payload):
    if payload.get("mode") not in ("training_data", "few_shot"):
        raise ValueError("Scored jobs require training_data or few_shot mode")
    for field in ("_owner", "_api_key", "scorer_revision", "user_prompt", "task_model", "judge_model", "reflection_model"):
        if not isinstance(payload.get(field), str) or not payload[field].strip():
            raise ValueError(f"{field} is required")
    if payload.get("scorer_revision") != "gepa-holdout-v1":
        raise ValueError("Unsupported scorer revision")
    base = payload.get("_base_url", "").rstrip("/")
    if base != settings.PROMPT_STUDIO_INFERENCE_BASE_URL.rstrip("/") or urlparse(base).scheme != "https":
        raise ValueError("Inference base URL must match the configured HTTPS CI endpoint")
    calls = payload.get("max_metric_calls")
    if type(calls) is not int or not 1 <= calls <= 1000:
        raise ValueError("max_metric_calls must be between 1 and 1000")
    try:
        budget = Decimal(str(payload.get("max_cost_usd")))
        if not budget.is_finite() or not Decimal("0") < budget <= Decimal("10000"):
            raise ValueError("max_cost_usd must be positive and at most 10000")
    except InvalidOperation as error:
        raise ValueError("max_cost_usd is invalid") from error
    rows, holdout = payload.get("rows"), payload.get("holdout_rows")
    if not isinstance(rows, list) or not 2 <= len(rows) <= 1000 or not isinstance(holdout, list) or not 1 <= len(holdout) <= 1000:
        raise ValueError("Separate training and held-out rows are required")
    if not all(isinstance(row, dict) and isinstance(row.get("inputs"), dict) for row in rows + holdout):
        raise ValueError("Every case must have an inputs object")
    if any(not re.fullmatch(r"[0-9a-f]{64}", str(row.get("input_hash", ""))) for row in rows + holdout):
        raise ValueError("Every case requires the platform's frozen input_hash")
    trained = {_canonical(row["inputs"]) for row in rows}
    if any(_canonical(row["inputs"]) in trained for row in holdout):
        raise ValueError("Training and held-out inputs must be disjoint")
    if payload["mode"] == "training_data" and any(not isinstance(row.get("expected_output"), str) or not row["expected_output"] for row in rows + holdout):
        raise ValueError("Labeled mode requires an expected answer for every case")
    criteria = payload.get("quality_criteria")
    if payload["mode"] == "few_shot" and (not isinstance(criteria, list) or not criteria or any(not isinstance(value, str) or not value.strip() for value in criteria)):
        raise ValueError("Few-shot scoring requires explicit quality criteria")
    if payload.get("use_builtin_judge") or payload.get("custom_scorers") or payload.get("tracking_uri"):
        raise ValueError("Scored jobs cannot bypass the metered endpoint or use caller-selected stores")


def run_scored(job, payload):
    validate_scored_payload(payload)
    from prompt_optimizer import PromptOptimizer, check_output_constraints
    from .services import build_config, _tracking_uri
    from .metered_transport import MeteredTransport
    # Names isolate MLflow registry mutations and traces to this owner/job.
    scoped = {**payload, "name": f"scored_{job.id}"}
    config = build_config(scoped)
    config.validate()
    missing = [(row.get("id"), sorted(config.declared_vars() - set(row["inputs"])))
               for row in payload["rows"] + payload["holdout_rows"] if config.declared_vars() - set(row["inputs"])]
    if missing:
        raise ValueError("Every training and held-out case must fill every declared prompt variable")
    transport = MeteredTransport(job, payload)
    with transport.endpoint() as endpoint:
        opt = PromptOptimizer(endpoint=endpoint, tracking_uri=_tracking_uri({}))
        saved = assert_lease(job.id, job.lease_token).checkpoint
        if "optimized" not in saved:
            job.stage = "Optimizing training data"
            result = opt.optimize(config)
            if transport.failure:
                raise transport.failure
            checkpoint(job.id, job.lease_token, "optimized", {"result": result.to_dict(),
                "deployable": result.to_deployable(), "prompts": [{"role": p.role, "template": p.template} for p in result.prompts]})
        saved = assert_lease(job.id, job.lease_token).checkpoint
        optimized = saved["optimized"]
        prompts = [SimpleNamespace(**prompt) for prompt in optimized["prompts"]]
        held = []
        for index, row in enumerate(payload["holdout_rows"]):
            key = f"holdout_{index}"
            if key not in saved:
                job.stage = f"Scoring held-out case {index + 1}/{len(payload['holdout_rows'])}"
                transport.phase = f"baseline_{index}"
                baseline = opt.evaluate_prompts(config, config.prompts, [row])
                transport.phase = f"treatment_{index}"
                treatment = opt.evaluate_prompts(config, prompts, [row])
                if transport.failure:
                    raise transport.failure
                baseline_receipt = InferenceCall.objects.filter(job_id=job.id, call_key__startswith=f"baseline_{index}:").order_by("id").first()
                treatment_receipt = InferenceCall.objects.filter(job_id=job.id, call_key__startswith=f"treatment_{index}:").order_by("id").first()
                if not baseline_receipt or not baseline_receipt.request_id or not treatment_receipt or not treatment_receipt.request_id:
                    raise UncertainInference("Held-out task receipts are missing")
                item = {"id": str(row.get("id", index)),
                    "baseline_request_id": baseline_receipt.request_id, "treatment_request_id": treatment_receipt.request_id,
                    "constraint_passed": check_output_constraints(treatment["outputs"][0], config.output_constraints),
                    "baseline_constraint_passed": check_output_constraints(baseline["outputs"][0], config.output_constraints), "input_hash": row["input_hash"],
                    "baseline_score": baseline["scores"][0], "treatment_score": treatment["scores"][0],
                    "baseline_output": baseline["outputs"][0], "treatment_output": treatment["outputs"][0]}
                checkpoint(job.id, job.lease_token, key, item)
                saved[key] = item
            held.append(saved[key])
    calls = list(InferenceCall.objects.filter(job_id=job.id).order_by("id"))
    if not calls or any(call.status != "complete" for call in calls):
        raise UncertainInference("Not every inference call has a terminal receipt")
    ids = list(dict.fromkeys(call.request_id for call in calls if call.request_id))
    if not ids:
        raise UncertainInference("Inference request receipts are missing")
    result = {**optimized, "scored_evidence": {"scorer_revision": payload["scorer_revision"], "holdout": held,
        "inference_request_ids": ids, "inference_calls": [{"request_id": call.request_id, "model": call.model, "usage": call.usage, "http_status": call.http_status, "phase": call.phase, "purpose": call.purpose} for call in calls],
        "completion_known": True, "cost_authority": "ci_credential_ledger", "max_cost_usd": str(payload["max_cost_usd"]), "cost_complete": False}}
    _complete_scored(job, payload, result)


def _complete_scored(job, payload, result):
    """History and final state either commit under one live lease or roll back."""
    from .services import sanitize_payload
    from .storage import bounded_json_size
    # Refuse oversized history before allocating/writing its database copy.
    # finish_job rechecks the final result after its run_id has been attached.
    bounded_json_size(result, "result")
    config = sanitize_payload(payload)
    bounded_json_size(config, "payload")
    with transaction.atomic():
        assert_lease(job.id, job.lease_token)
        run, _ = Run.objects.update_or_create(owner=job.owner, job_id=job.id, defaults={"kind": "optimize", "status": "done",
            "label": str(payload.get("name", "Scored GEPA job"))[:200], "config": config, "result": result})
        result["run_id"] = str(run.id)
        finish_job(job.id, job.lease_token, "done", owner=job.owner, result=result, stage="Done", error="")
