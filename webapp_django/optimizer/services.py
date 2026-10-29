"""Backend logic for the prompt-optimizer app, framework-agnostic.

Everything here is plain Python that wraps the ``prompt_optimizer`` library:
token accounting, the model-downsizing cost model, config assembly from the
browser payload, and a small in-memory background job registry. Runs are always
live against the configured endpoint. The Django views are thin adapters over
these functions.
"""

from __future__ import annotations

import statistics
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from typing import Any

import sentry_sdk


# ---------------------------------------------------------------------------
# Token accounting + cost model
# ---------------------------------------------------------------------------
def count_tokens(text: str) -> int:
    text = text or ""
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return max(1, round(len(text) / 4))


def _dollars(tokens: float, price_per_million: float) -> float:
    return (tokens / 1_000_000.0) * price_per_million


def _cost_at(prompt_tok: float, out_tok: float, pin: float, pout: float, vol: float) -> float:
    """Serving cost for `vol` calls: (prompt_in + output) priced per model."""
    return (_dollars(prompt_tok, pin) + _dollars(out_tok, pout)) * vol


def _serving_from_usage(usage: Any, pin: float, pout: float) -> dict[str, Any] | None:
    """Actual per-call serving cost, measured from THIS run's task-model calls — average
    input/output tokens really observed, not a tiktoken guess or a fixed output count.
    Priced with the model's $/1M rates. None if no task calls were metered."""
    by = (usage or {}).get("by_role", {}) or {}
    t = by.get("task") or {}
    calls = t.get("calls") or 0
    if not calls:
        return None
    avg_in = t.get("input", 0) / calls
    avg_out = t.get("output", 0) / calls
    per = _dollars(avg_in, pin) + _dollars(avg_out, pout)
    return {"calls": calls, "avg_in": round(avg_in, 1), "avg_out": round(avg_out, 1),
            "per_call": per, "cost_1k": per * 1_000,
            "cost_10k": per * 10_000, "cost_100k": per * 100_000}


def compute_economics(baseline_prompt: str, optimized_prompt: str, econ: dict[str, Any]) -> dict[str, Any]:
    """Reference (frontier x baseline) vs optimized (target x optimized).

    The optimized prompt is usually LONGER; savings come from the per-token
    price gap between models, not from a shorter prompt.
    """
    out_tokens = float(econ.get("output_tokens", 200) or 200)
    volume = float(econ.get("monthly_calls", 100_000) or 100_000)
    # Prices come from the live catalog via the form; 0 if the endpoint was
    # unreachable and the user didn't type one (no hardcoded price seeds here).
    fin = float(econ.get("frontier_in") or 0.0)
    fout = float(econ.get("frontier_out") or 0.0)
    tin = float(econ.get("target_in") or 0.0)
    tout = float(econ.get("target_out") or 0.0)

    base_tok = count_tokens(baseline_prompt)
    opt_tok = count_tokens(optimized_prompt)
    ref_call = _dollars(base_tok, fin) + _dollars(out_tokens, fout)
    opt_call = _dollars(opt_tok, tin) + _dollars(out_tokens, tout)
    saving = ref_call - opt_call
    pct = (saving / ref_call * 100.0) if ref_call else 0.0

    return {
        "baseline_prompt_tokens": base_tok,
        "optimized_prompt_tokens": opt_tok,
        "prompt_token_delta": opt_tok - base_tok,
        "output_tokens": out_tokens,
        "reference": {
            "label": econ.get("frontier_name", "Frontier model"),
            "cost_per_call": ref_call, "cost_per_1k": ref_call * 1_000,
            "monthly_cost": ref_call * volume, "price_in": fin, "price_out": fout,
        },
        "optimized": {
            "label": econ.get("target_name", "Target model"),
            "cost_per_call": opt_call, "cost_per_1k": opt_call * 1_000,
            "monthly_cost": opt_call * volume, "price_in": tin, "price_out": tout,
        },
        "savings_per_call": saving, "savings_pct": pct,
        "monthly_savings": saving * volume, "monthly_calls": volume,
    }


# ---------------------------------------------------------------------------
# Config assembly + helpers (bridge to the prompt_optimizer library)
# ---------------------------------------------------------------------------
def variables_of(*templates: str) -> list[str]:
    from prompt_optimizer import p_variables

    seen: list[str] = []
    for t in templates:
        for v in sorted(p_variables(t or "")):
            if v not in seen:
                seen.append(v)
    return seen


def _tracking_uri(payload: dict[str, Any]) -> str:
    """MLflow store URI, resolved cwd-independently so we don't scatter mlflow.db
    files. Precedence: explicit payload value → MLFLOW_TRACKING_URI env (set on the
    deploy) → an absolute sqlite file at the repo root."""
    import os
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]   # optimizer/ -> webapp_django/ -> repo root
    return (payload.get("tracking_uri")
            or os.environ.get("MLFLOW_TRACKING_URI")
            or f"sqlite:///{root / 'mlflow.db'}")



# --------------------------------------------------------------------------- credentials
# The API path passes the user's CI key per request (transient); the standalone UI uses the
# deployment's env key. _endpoint_for builds an explicit-cred endpoint (no global env write)
# when a key is present, else falls back to the env-based configure().
_SECRET_KEYS = ("_api_key", "_base_url", "_callback_url")


def _endpoint_from(api_key=None, base_url=None):
    from llm_endpoint import configure
    if api_key and base_url:
        return configure(api_key=api_key, base_url=base_url, set_openai_env=False, env_file=None)
    return configure()


def _endpoint_for(payload):
    p = payload or {}
    return _endpoint_from(p.get("_api_key"), p.get("_base_url"))


def sanitize_payload(payload):
    """Strip transient credential / callback fields so nothing sensitive is persisted."""
    if not isinstance(payload, dict):
        return payload
    return {k: v for k, v in payload.items() if k not in _SECRET_KEYS}


def build_config(payload: dict[str, Any]):
    from prompt_optimizer import Mode, OptimizationConfig, Prompt

    mode = Mode(payload["mode"])
    if payload.get("_api_key") and (payload.get("use_builtin_judge") or payload.get("custom_scorers")):
        raise ValueError("Built-in / custom judges are unavailable on the API path "
                         "(they authenticate via the shared MLflow gateway env).")
    prompts: list[Prompt] = []
    if (payload.get("system_prompt") or "").strip():
        prompts.append(Prompt(role="system", template=payload["system_prompt"]))
    if (payload.get("user_prompt") or "").strip() or not prompts:
        prompts.append(Prompt(role="user", template=payload.get("user_prompt", "") or ""))

    common = dict(
        name=payload.get("name") or "dashboard_job",
        prompts=prompts,
        mode=mode,
        task_model=payload.get("task_model") or None,
        reflection_model=payload.get("reflection_model") or None,
        judge_model=payload.get("judge_model") or None,
        max_metric_calls=int(payload.get("max_metric_calls") or 30),
        constraint_gate=bool(payload.get("output_constraints")) and mode is Mode.FEW_SHOT,
        output_constraints=payload.get("output_constraints") or None,   # #2
        tighten_prompt=bool(payload.get("tighten_prompt")),             # #1
        reflection_max_chars=(int(payload["reflection_max_chars"])
                              if payload.get("reflection_max_chars") else None),
        task_reasoning_effort=payload.get("task_reasoning_effort") or "minimal",
    )
    if mode is Mode.TRAINING_DATA:
        return OptimizationConfig(
            training_data=payload.get("rows", []),
            target_key="expected_output",
            label_scorer=payload.get("label_scorer", "equivalence"),
            **common,
        )
    if mode is Mode.FEW_SHOT:
        criteria = [c for c in (payload.get("quality_criteria") or []) if c.strip()]
        return OptimizationConfig(
            examples=payload.get("rows", []),
            quality_criteria=criteria,
            **common,
        )
    return OptimizationConfig(task_description=payload.get("task_description") or "", **common)


# ---------------------------------------------------------------------------
# Background job registry — DB-backed so polling works across processes/workers
# and survives a server restart (the old in-memory dict did neither).
# ---------------------------------------------------------------------------
def _save_job_state(job: "Job") -> None:
    from .models import JobState
    from .durable import LeaseLost, TERMINAL_STATUSES, finish_job, lease_write
    from .storage import bounded_json_size
    bounded_json_size(job.result, "result")
    fields = {"owner": job.owner, "status": job.status, "stage": job.stage, "result": job.result, "error": job.error or ""}
    if job.lease_token:
        if job.status in TERMINAL_STATUSES:
            finish_job(job.id, job.lease_token, job.status, owner=job.owner, stage=job.stage, result=job.result, error=job.error or "")
        else:
            with lease_write(job.id, job.lease_token):
                if JobState.objects.filter(id=job.id, owner=job.owner, lease_token=job.lease_token, status="running").update(**fields) != 1:
                    raise LeaseLost("Job no longer owns its execution lease")
    else:
        if job.status in TERMINAL_STATUSES:
            fields["credential_ciphertext"] = ""
        JobState.objects.update_or_create(id=job.id, defaults=fields)


def get_job_state(job_id: str, owner: str) -> dict[str, Any] | None:
    from .models import JobState
    j = JobState.objects.filter(pk=job_id, owner=owner).first()
    if not j:
        return None
    return {"status": "running" if j.status == "queued" else j.status,
            "stage": j.stage, "result": j.result, "error": j.error}


def _fire_callback(job, notice_only: bool = False) -> None:
    """Best-effort signed POST of a job's terminal state to CI's callback URL (webhook, v1).
    Fired once when status becomes done/error. Never raises into the job thread."""
    url = getattr(job, "_callback_url", None)
    if not url:
        return
    import hashlib
    import hmac
    import json as _json
    import time as _time
    import urllib.request
    from django.conf import settings
    from .ci_api import callback_allowed
    if not callback_allowed(url):
        return
    payload = {"job_id": job.id, "user_id": job.owner, "status": job.status}
    if not notice_only:
        payload.update(result=job.result, error=job.error or "")
    body = _json.dumps(payload).encode()
    if notice_only and len(body) > 32_000:
        return
    ts = str(int(_time.time()))
    headers = {"Content-Type": "application/json", "X-PS-Timestamp": ts}
    secret = getattr(settings, "PROMPT_STUDIO_WEBHOOK_SECRET", "") or getattr(settings, "PROMPT_STUDIO_SERVICE_TOKEN", "")
    if not secret:
        return
    canonical = "\n".join([ts, "POST", url, hashlib.sha256(body).hexdigest()]).encode()
    headers["X-PS-Signature"] = hmac.new(secret.encode(), canonical, hashlib.sha256).hexdigest()
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, response_headers, newurl):
            return None
    try:
        with urllib.request.build_opener(NoRedirect()).open(urllib.request.Request(url, data=body, headers=headers, method="POST"), timeout=5):
            pass
    except Exception:  # noqa: BLE001 - webhook is best-effort; polling remains the source of truth
        traceback.print_exc()
        sentry_sdk.capture_exception()


def _capture_job_error(exc: BaseException) -> None:
    """Report a job failure the runner swallows to Sentry. Cancellation, lease loss,
    uncertain inference and an insufficient balance are expected outcomes, not bugs."""
    from .durable import JobCancelled, LeaseLost, UncertainInference
    if isinstance(exc, (JobCancelled, LeaseLost, UncertainInference)):
        return
    if getattr(exc, "code", None) == "insufficient_balance":
        return
    sentry_sdk.capture_exception(exc)


class Job:
    """A background job handle. Every status/stage/result/error change is persisted
    to the DB (via __setattr__), so the existing job bodies need no changes."""

    _PERSIST = {"status", "stage", "result", "error"}

    def __init__(self, id: str, status: str = "running", stage: str = "Preparing",
                 result: dict[str, Any] | None = None, error: str | None = None, owner: str = "", lease_token: str | None = None):
        object.__setattr__(self, "id", id)
        object.__setattr__(self, "owner", owner)
        object.__setattr__(self, "lease_token", lease_token)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "result", result)
        object.__setattr__(self, "error", error)
        object.__setattr__(self, "_callback_url", None)
        object.__setattr__(self, "_fired", False)
        _save_job_state(self)

    def __setattr__(self, k, v):
        object.__setattr__(self, k, v)
        if k in self._PERSIST:
            _save_job_state(self)
        if (k == "status" and v in ("done", "error")
                and getattr(self, "_callback_url", None) and not getattr(self, "_fired", False)):
            object.__setattr__(self, "_fired", True)
            import threading as _t
            _t.Thread(target=_fire_callback, args=(self,), daemon=True).start()


def endpoint_available() -> bool:
    return endpoint_status()["available"]


def endpoint_status() -> dict[str, Any]:
    """Whether the endpoint is configured, and — if not — a plain-language reason
    and the exact env vars still missing, so the UI can tell the user what to do.

    configure() re-reads .env and the environment on every call, so once the user
    creates a .env (or exports the vars) a re-check flips this to available without
    restarting the server.
    """
    try:
        from llm_endpoint import configure

        configure()
        return {"available": True, "detail": "", "missing": []}
    except Exception as exc:  # noqa: BLE001
        import os

        try:
            from llm_endpoint import DEFAULT_KEY_VAR, DEFAULT_URL_VAR
            required = (DEFAULT_KEY_VAR, DEFAULT_URL_VAR)
        except Exception:  # noqa: BLE001
            required = ("CHEAPERINFERENCE_API_KEY", "CHEAPERINFERENCE_BASE_URL")
        missing = [v for v in required if not os.environ.get(v)]
        return {"available": False, "detail": f"{type(exc).__name__}: {exc}", "missing": missing}


# ---------------------------------------------------------------------------
# Live model catalog (the single source for model names AND prices)
# ---------------------------------------------------------------------------
_MODELS_CACHE: dict[str, Any] = {"at": 0.0, "data": None}
_MODELS_TTL = 300.0   # seconds; the catalog changes rarely


def fetch_models() -> list[dict]:
    """The endpoint's live /v1/models catalog, normalized to
    ``[{id, provider, input, output}]`` where input/output are $/1M tokens (the
    listed billed rate; discount_percent is informational and not re-applied).

    This is the one source of truth for the model dropdowns and the cost-panel
    prices — no hardcoded list, no hardcoded prices. Cached briefly; returns the
    last good list (or []) if the endpoint is unreachable.
    """
    now = time.time()
    cached = _MODELS_CACHE["data"]
    if cached is not None and now - _MODELS_CACHE["at"] < _MODELS_TTL:
        return cached
    try:
        import json
        import urllib.request

        from llm_endpoint import configure

        ep = configure()
        req = urllib.request.Request(
            ep.base_url.rstrip("/") + "/models",
            headers={"Authorization": f"Bearer {ep.api_key}"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            payload = json.load(resp)

        def _num(v):
            try:
                return round(float(v), 6)
            except (TypeError, ValueError):
                return None

        out = []
        for m in payload.get("data", []):
            if not m.get("id"):
                continue
            p = m.get("pricing") or {}
            out.append({
                "id": m["id"],
                "provider": m.get("provider") or m.get("owned_by") or "",
                "input": _num(p.get("input_per_million")),
                "output": _num(p.get("output_per_million")),
                # raw capability signal (shape varies by endpoint) for text-model filtering
                "caps": (m.get("capabilities") or m.get("modalities")
                         or m.get("type") or m.get("mode")),
            })
        if not out:  # reached the endpoint but parsed nothing — log the shape to debug
            keys = list(payload.keys()) if isinstance(payload, dict) else type(payload).__name__
            raw = payload.get("data") if isinstance(payload, dict) else None
            print(f"[fetch_models] empty after parse. url={ep.base_url.rstrip('/')}/models "
                  f"top-level keys={keys} data_len={len(raw) if isinstance(raw, list) else 'n/a'} "
                  f"sample={ (raw[0] if isinstance(raw, list) and raw else None) }")
        _MODELS_CACHE.update(at=now, data=out)
        return out
    except Exception:  # noqa: BLE001 — endpoint down / unauthorized: degrade gracefully
        traceback.print_exc()
        sentry_sdk.capture_exception()
        return cached or []


def record_run(**fields) -> str:
    """A job cannot claim successful history persistence when the write failed."""
    from .models import Run
    if "config" in fields:
        fields["config"] = sanitize_payload(fields["config"])
    return str(Run.objects.create(**fields).id)


def _fmt_score(s: Any) -> str:
    return "—" if s is None else f"{float(s):.3f}"


def _tradeoff_from_target(opt, payload: dict[str, Any], target: dict[str, Any]) -> dict[str, Any] | None:
    """2nd pass: optimize a prompt for the benchmark anchor (frontier), then price BOTH
    it and the already-optimized target at scale — the honest, optimized-vs-optimized
    accuracy/lifetime-cost comparison. `target` carries the 1st pass result
    {model, score, prompt}. Returns None when no distinct anchor is set."""
    econ = payload.get("economics", {}) or {}
    anchor = (econ.get("frontier_name") or "").strip()
    tmodel = (target.get("model") or "").strip()
    if not anchor or anchor == tmodel:
        print(f"[anchor] no distinct anchor: frontier_name={anchor!r} target_model={tmodel!r} "
              f"econ_keys={list(econ.keys())} payload_keys={list(payload.keys())}")
        return None

    fin, fout = float(econ.get("frontier_in") or 0), float(econ.get("frontier_out") or 0)

    anchor_payload = dict(payload)
    anchor_payload["task_model"] = anchor
    ares = opt.optimize(build_config(anchor_payload))
    a_serv = _serving_from_usage(ares.usage, fin, fout) or {}
    t_serv = target.get("serving") or {}   # measured in pass 1

    def _row(model, score, chars, serv):
        return {"model": model, "score": score, "prompt_chars": chars,
                "avg_in": serv.get("avg_in"), "avg_out": serv.get("avg_out"),
                "cost_per_call": serv.get("per_call"),
                "cost_10k": serv.get("cost_10k"), "cost_100k": serv.get("cost_100k")}

    return {"basis": "measured",
            "anchor": _row(anchor, ares.final_score, ares.prompt_chars, a_serv),
            "target": _row(target.get("model"), target.get("score"), target.get("prompt_chars"), t_serv)}


def _run_anchor_job(job: Job, payload: dict[str, Any]) -> None:
    """The optional 2nd pass, fired separately after the 1st pass succeeds."""
    try:
        from prompt_optimizer import PromptOptimizer

        job.stage = ("Optimizing Benchmark Anchor (Frontier Model) for computing the "
                     "accuracy vs. lifetime cost trade-off")
        opt = PromptOptimizer(endpoint=_endpoint_for(payload), tracking_uri=_tracking_uri(payload))
        tradeoff = _tradeoff_from_target(opt, payload, payload.get("_target") or {})
        job.result = {"tradeoff": tradeoff}
        # When launched from History (a saved run), persist the trade-off back so it
        # shows on reopen instead of needing another paid anchor pass.
        rid = payload.get("_run_id")
        if rid and tradeoff:
            try:
                from django.db import connection

                from .models import Run
                try:
                    r = Run.objects.get(pk=rid, owner=payload.get("_owner", ""))
                    res = r.result or {}
                    res["tradeoff"] = tradeoff
                    r.result = res
                    r.save(update_fields=["result"])
                finally:
                    connection.close()
            except Exception:  # noqa: BLE001
                traceback.print_exc()
                sentry_sdk.capture_exception()
        job.stage = "Done"
        job.status = "done"
    except Exception as exc:  # noqa: BLE001
        _capture_job_error(exc)
        job.error = f"{type(exc).__name__}: {exc}"
        job.stage = "Error"
        job.status = "error"
        traceback.print_exc()
    finally:
        try:
            from django.db import connection
            connection.close()   # this worker thread's own DB connection
        except Exception:  # noqa: BLE001
            pass


def start_anchor(payload: dict[str, Any]) -> Job:
    from .durable import enqueue
    return enqueue("anchor", payload, payload.pop("_idempotency_key", None))


def _run_job(job: Job, payload: dict[str, Any]) -> None:
    t0 = time.time()
    label = f"{payload.get('mode', '?')} · {payload.get('name') or 'job'}"
    try:
        baseline = ((payload.get("system_prompt", "") or "") + "\n"
                    + (payload.get("user_prompt", "") or "")).strip()
        from prompt_optimizer import PromptOptimizer

        job.stage = "Configuring endpoint"
        opt = PromptOptimizer(endpoint=_endpoint_for(payload),
            tracking_uri=_tracking_uri(payload))
        job.stage = "Optimizing Task Model (GEPA Reflection Loop)"
        result = opt.optimize(build_config(payload))

        optimized = "\n".join(p.template or "" for p in result.prompts).strip()
        econ = compute_economics(baseline, optimized, payload.get("economics", {}))
        rd = result.to_dict()
        e = payload.get("economics", {}) or {}
        serving = _serving_from_usage(rd.get("usage"),
                                      float(e.get("target_in") or 0), float(e.get("target_out") or 0))
        job.result = {"result": rd, "deployable": result.to_deployable(),
                      "economics": econ, "serving": serving}
        imp = rd.get("improvement")
        summary = (f"{_fmt_score(rd.get('initial_score'))} → {_fmt_score(rd.get('final_score'))}"
                   + (f" ({'+' if (imp or 0) >= 0 else ''}{imp:.3f})" if imp is not None else ""))
        rid = record_run(kind="optimize", owner=payload.get("_owner", ""), job_id=job.id, label=label,
                         status="done", summary=summary,
                         opt_tokens=((rd.get("usage") or {}).get("total") or {}).get("total"),
                         duration_s=time.time() - t0, config=payload, result=job.result)
        if rid:
            job.result["run_id"] = rid   # so the live anchor pass can persist its trade-off back
        job.stage = "Done"
        job.status = "done"              # persists job.result (incl. run_id) for the poll
    except Exception as exc:  # noqa: BLE001
        _capture_job_error(exc)
        try:
            from prompt_optimizer import OptimizerFailed
        except Exception:  # noqa: BLE001 - the job may have failed importing it; still record that error
            OptimizerFailed = ()
        failed = exc if isinstance(exc, OptimizerFailed) else None
        # A rewriter failure carries a message written for the user (shown as-is in CI);
        # its code and raw cause go to the run record only.
        job.error = failed.message if failed else f"{type(exc).__name__}: {exc}"
        job.stage = "Error"
        job.status = "error"
        traceback.print_exc()
        record_run(kind="optimize", owner=payload.get("_owner", ""), job_id=job.id, label=label,
                   status="error", summary="failed",
                   error=f"{failed.code}: {failed.message} | {failed.detail[:500]}" if failed else job.error,
                   duration_s=time.time() - t0, config=payload)


def start_job(payload: dict[str, Any]) -> Job:
    from .durable import enqueue
    return enqueue("optimize", payload, payload.pop("_idempotency_key", None))


# ---------------------------------------------------------------------------
# A/B experiments: run the SAME job twice varying one lever, compare on held-out.
# First lever: reflection_max_chars (in-loop length regularization). Two arms.
# ---------------------------------------------------------------------------
def _ab_verdict(arms: list[dict]) -> str:
    """Lever-agnostic: compare arm B to arm A on held-out (primary) + prompt length."""
    if len(arms) < 2:
        return ""
    a, b = arms[0], arms[1]
    chars = f"prompt {a['prompt_chars']}→{b['prompt_chars']} chars"
    ha, hb = a["held_out"], b["held_out"]
    if ha is None or hb is None:
        return f"B vs A: {chars}. Held-out not scored — add held-out inputs to compare generalization."
    if hb > ha + 1e-9:
        return f"B beat A on held-out ({ha:.3f}→{hb:.3f}); {chars}."
    if hb < ha - 1e-9:
        return f"B lost to A on held-out ({ha:.3f}→{hb:.3f}); {chars}."
    return f"A and B tied on held-out ({ha:.3f}); {chars}."


def _run_experiment(job: Job, payload: dict[str, Any]) -> None:
    """Lever-agnostic A/B: run each arm's config (base payload + arm overrides),
    score on held-out, compare. Arms come from payload["arms"] as
    [{label, overrides:{field:value}}]; falls back to the reflection-cap lever."""
    t0 = time.time()
    lever = payload.get("lever") or "reflection_max_chars"
    arms_in = payload.get("arms")
    if not arms_in:  # back-compat default: the reflection-cap lever
        cap = int(payload["reflection_max_chars"]) if payload.get("reflection_max_chars") else None
        arms_in = [{"label": "A · no cap", "overrides": {"reflection_max_chars": None}},
                   {"label": f"B · cap {cap}", "overrides": {"reflection_max_chars": cap}}]
    label = f"{payload.get('mode', '?')} · {lever}"
    try:
        from prompt_optimizer import PromptOptimizer

        holdout = [r for r in (payload.get("holdout") or [])
                   if any(str(v).strip() for v in r.values())]
        opt = PromptOptimizer(endpoint=_endpoint_for(payload), tracking_uri=_tracking_uri(payload))
        arms: list[dict] = []
        n = len(arms_in)
        for i, spec in enumerate(arms_in):
            arm_label = spec.get("label") or f"Arm {i + 1}"
            p = dict(payload)
            p.update(spec.get("overrides") or {})
            cfg = build_config(p)
            job.stage = f"Arm {i + 1}/{n} — optimizing ({arm_label})"
            res = opt.optimize(cfg)
            job.stage = f"Arm {i + 1}/{n} — held-out eval ({arm_label})"
            held = opt.evaluate_prompts(cfg, res.prompts, holdout) if holdout else {"mean": None, "n": 0}
            usage_total = (res.usage or {}).get("total", {})
            arms.append({
                "label": arm_label,
                "overrides": spec.get("overrides") or {},
                "in_sample": res.final_score,
                "held_out": held["mean"],
                "held_out_n": held["n"],
                "prompt_chars": res.prompt_chars,
                "opt_tokens": usage_total.get("total"),
                "optimized_prompt": "\n".join(pp.template or "" for pp in res.prompts).strip(),
            })
        verdict = _ab_verdict(arms)
        job.result = {"experiment": lever, "arms": arms, "verdict": verdict}
        job.stage = "Done"
        job.status = "done"
        record_run(kind="ab", owner=payload.get("_owner", ""), job_id=job.id, label=label,
                   status="done", summary=verdict[:300],
                   opt_tokens=sum(a["opt_tokens"] or 0 for a in arms) or None,
                   duration_s=time.time() - t0, config=payload,
                   result={"arms": arms, "verdict": verdict, "holdout": holdout, "experiment": lever})
    except Exception as exc:  # noqa: BLE001
        _capture_job_error(exc)
        job.error = f"{type(exc).__name__}: {exc}"
        job.stage = "Error"
        job.status = "error"
        traceback.print_exc()
        record_run(kind="ab", owner=payload.get("_owner", ""), job_id=job.id, label=label,
                   status="error", summary="failed",
                   error=job.error, duration_s=time.time() - t0, config=payload)


def start_experiment(payload: dict[str, Any]) -> Job:
    from .durable import enqueue
    return enqueue("experiment", payload, payload.pop("_idempotency_key", None))


# ---------------------------------------------------------------------------
# Latency diagnostics -> "Model Standings"
# Probe every catalog model across the three roles; publish measured latency so
# model choice is evidence-based. The timing engine is diagnose._time_calls (one
# source of truth); we only supply the three role-shaped prompts and store rows.
# ---------------------------------------------------------------------------
_ROLE_PROMPTS = {
    "task": [{"role": "user", "content": "Reply with the single word: ok"}],
    "judge": [{"role": "user", "content":
               "You are grading an answer. Question: name a color. Answer: blue. "
               "Is the answer acceptable? Reply yes or no."}],
    "reflection": [{"role": "user", "content":
                    "Improve this instruction to be clearer, in one sentence: "
                    "'Classify the sentence.' Output only the rewritten instruction."}],
}
_ROLE_N = {"task": 3, "judge": 3, "reflection": 2}

# Only chat/text models can answer these probes; embedding, image, audio, rerank,
# moderation, etc. reject a chat.completions call and would show as spurious "err".
_NONTEXT_HINTS = ("embed", "rerank", "moderation", "whisper", "tts", "audio",
                  "speech", "transcrib", "image", "dall", "diffusion", "sdxl",
                  "flux", "video", "veo", "sora", "clip", "ocr", "guard", "rank")
_TEXT_CAPS = {"chat", "completions", "completion", "text", "text-generation",
              "tools", "reasoning", "instruct"}
_NONTEXT_CAPS = {"embeddings", "embedding", "image", "images", "audio", "speech",
                 "tts", "stt", "rerank", "reranker", "moderation"}


def _capset(m: dict) -> set:
    c = m.get("caps")
    if isinstance(c, dict):
        return {str(k).lower() for k, v in c.items() if v}
    if isinstance(c, (list, tuple, set)):
        return {str(x).lower() for x in c}
    if isinstance(c, str):
        return {c.lower()}
    return set()


def is_text_model(m: dict) -> bool:
    """Whether a catalog model can answer a chat.completions probe. Name hints are a
    hard exclude; declared capabilities refine the rest; default is include (so a
    genuine text model is never dropped for lacking metadata)."""
    ident = f"{m.get('id', '')} {m.get('provider', '')}".lower()
    if any(h in ident for h in _NONTEXT_HINTS):
        return False
    caps = _capset(m)
    if caps and not (caps & _TEXT_CAPS) and (caps & _NONTEXT_CAPS):
        return False
    return True


def _probe_stats(lats: list[float]) -> dict[str, float]:
    if not lats:
        return {"mean_s": None, "min_s": None, "max_s": None, "p90_s": None}
    s = sorted(lats)
    return {"mean_s": statistics.mean(lats), "min_s": min(lats), "max_s": max(lats),
            "p90_s": s[max(0, int(len(s) * 0.9) - 1)]}


def run_diagnostics(models: list[str] | None = None, roles: list[str] | None = None,
                    timeout: int = 60, api_key: str | None = None,
                    base_url: str | None = None) -> dict[str, Any]:
    """Time each (model, role) once and write a Probe row per pair. Reuses the same
    per-call timing engine as diagnose.py. Returns a small summary."""
    from django.db import connection

    from llm_endpoint import configure, time_calls as _time_calls

    from .models import Probe

    ep = _endpoint_from(api_key, base_url)
    client = ep.client
    roles = roles or ["task", "judge", "reflection"]

    skipped: list[str] = []
    if models is None:                              # full catalog sweep: text-only
        catalog = [m for m in fetch_models() if m.get("id")]
        models = [m["id"] for m in catalog if is_text_model(m)]
        skipped = [m["id"] for m in catalog if not is_text_model(m)]
        if not models:
            models = [ep.task_model]
        if skipped:                                 # prune stale non-text err rows
            try:
                Probe.objects.filter(model__in=skipped).delete()
            except Exception:  # noqa: BLE001
                pass

    probed = 0
    try:
        for model in models:
            for role in roles:
                lats, err = _time_calls(client, model, _ROLE_PROMPTS[role],
                                        n=_ROLE_N[role], timeout=timeout)
                Probe.objects.create(model=model, role=role, n=len(lats),
                                     ok=(err is None), error=err or "",
                                     **_probe_stats(lats))
                probed += 1
    finally:
        connection.close()
    return {"probed": probed, "models": len(models), "roles": roles,
            "skipped": len(skipped)}


def _run_diagnostics_job(job: Job, payload: dict[str, Any]) -> None:
    try:
        job.stage = "Probing models (task · judge · reflection)"
        job.result = {"diagnostics": run_diagnostics(
            models=payload.get("models") or None,
            roles=payload.get("roles") or None,
            timeout=int(payload.get("timeout") or 60),
            api_key=payload.get("_api_key"), base_url=payload.get("_base_url"))}
        job.stage = "Done"
        job.status = "done"
    except Exception as exc:  # noqa: BLE001
        _capture_job_error(exc)
        job.error = f"{type(exc).__name__}: {exc}"
        job.stage = "Error"
        job.status = "error"
        traceback.print_exc()


def start_diagnostics(payload: dict[str, Any]) -> Job:
    from .durable import enqueue
    return enqueue("diagnostics", payload, payload.pop("_idempotency_key", None))


def standings() -> dict[str, Any]:
    """Latest measurement per (model, role), pivoted to one row per model. Sorted
    by reflection latency (the role slow models punish hardest), fastest first."""
    from .models import Probe

    latest: dict[tuple, Any] = {}
    newest = None
    errored: set = set()
    for p in Probe.objects.all()[:3000]:      # -created_at ordering: first seen wins
        if p.created_at and (newest is None or p.created_at > newest):
            newest = p.created_at
        if not p.ok:                          # errored probes never load into the table
            errored.add(p.model)
            continue
        latest.setdefault((p.model, p.role), p)

    rows: dict[str, dict] = {}
    for (model, role), p in latest.items():
        row = rows.setdefault(model, {"model": model, "roles": {}})
        row["roles"][role] = {
            "mean_s": p.mean_s, "p90_s": p.p90_s, "min_s": p.min_s, "max_s": p.max_s,
            "ok": p.ok, "n": p.n,
            "at": p.created_at.isoformat() if p.created_at else None,
        }

    def _refl(r):
        return (r["roles"].get("reflection") or {}).get("mean_s") or 1e9

    hidden = len([m for m in errored if m not in rows])   # models with only failures
    return {"rows": sorted(rows.values(), key=_refl),
            "last_run": newest.isoformat() if newest else None, "hidden": hidden}
