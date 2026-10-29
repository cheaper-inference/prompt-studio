"""Durable single-job leases. No provider call is dispatched from a web request."""
from __future__ import annotations
import hashlib
import json
import sys
import uuid
from contextlib import contextmanager
from datetime import timedelta
from types import SimpleNamespace
import sentry_sdk
from django.conf import settings
from django.db import IntegrityError, transaction
from django.db.models import Q
from django.db.models.functions import Now
from django.utils import timezone
from .models import JobState
from .storage import bounded_json_size

TERMINAL_STATUSES = ("done", "error", "interrupted", "cancelled")
# MiniMe jobs whose worker stopped mid-way (a restart) are not resumed on their own: the task is
# paused, and the user's "Start optimizing" resumes it. Nothing is lost: a learning run keeps its
# GEPA checkpoint and saved rewrite, and judge calls are cached.
PAUSE_ON_INTERRUPT = ("minime_learn", "minime_round")

# Lanes: a worker in the "short" lane only takes short MiniMe jobs that a person or the chart is
# waiting for (new outputs, re-judging, checking and scoring versions), so they never queue behind
# a learning run; an "any" worker takes everything, oldest first. GP_JOB_LANE sets a worker's lane.
LANES = ("any", "short")


def is_short(row) -> bool:
    payload = row.payload or {}
    return row.kind == "minime_round" or (row.kind == "minime_learn"
                                          and bool(payload.get("rejudge_only") or payload.get("score_only")))

class LeaseLost(RuntimeError):
    pass

class JobCancelled(RuntimeError):
    pass

class UncertainInference(RuntimeError):
    pass

class IdempotencyConflict(ValueError):
    pass

def encrypt_secret(value: dict) -> str:
    if not value:
        return ""
    from cryptography.fernet import Fernet
    key = getattr(settings, "PROMPT_STUDIO_CREDENTIAL_KEY", "")
    if not key:
        raise ValueError("PROMPT_STUDIO_CREDENTIAL_KEY is required for durable credentials")
    return Fernet(key.encode()).encrypt(json.dumps(value).encode()).decode()

def decrypt_secret(value: str) -> dict:
    if not value:
        return {}
    from cryptography.fernet import Fernet
    return json.loads(Fernet(settings.PROMPT_STUDIO_CREDENTIAL_KEY.encode()).decrypt(value.encode()))

def enqueue(kind: str, payload: dict, idempotency_key: str | None = None):
    owner = payload.get("_owner")
    if not isinstance(owner, str) or not owner or len(owner) > 254:
        raise ValueError("An authenticated job owner is required")
    if idempotency_key and (len(idempotency_key) > 128 or not all(c.isalnum() or c in "_:-" for c in idempotency_key)):
        raise ValueError("Invalid idempotency key")
    secret = {k: payload[k] for k in ("_api_key", "_callback_url") if payload.get(k)}
    snapshot = {k: v for k, v in payload.items() if k not in secret}
    bounded_json_size(snapshot, "payload")
    digest = hashlib.sha256(json.dumps({"kind": kind, "payload": snapshot}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    def existing():
        row = JobState.objects.filter(owner=owner, idempotency_key=idempotency_key).first()
        if row and row.payload_hash != digest:
            raise IdempotencyConflict("Idempotency key already names a different job")
        return row
    if idempotency_key:
        found = existing()
        if found:
            return found
    try:
        with transaction.atomic():
            return JobState.objects.create(id=uuid.uuid4().hex, owner=owner, kind=kind,
                status="queued", stage="Queued", idempotency_key=idempotency_key,
                payload=snapshot, payload_hash=digest, credential_ciphertext=encrypt_secret(secret))
    except IntegrityError:
        if not idempotency_key:
            raise
        return existing()

def _sync_benchmark_history(job_id, status, *, result=None, error=""):
    # Existing Run JSON/history only; benchmark jobs never replay paid calls.
    from .models import Run
    fields = {"status": status, "summary": "Benchmark complete" if status == "done" else status.title(), "error": error or ""}
    if result is not None:
        fields["result"] = result
    Run.objects.filter(job_id=job_id, kind="benchmark").update(**fields)


def _serial_key(job):
    """MiniMe jobs of one task and kind/variant run one at a time: (kind, task, variant), else None.
    A rejudge or scoring request may queue behind a running one (it is not absorbed by it), but it
    must not start until that one ends, or both judge the same labels and the older batch can
    overwrite the newer one."""
    if job.kind not in ("minime_round", "minime_learn"):
        return None
    p = job.payload or {}
    return (job.kind, p.get("task_id"),
            "score" if p.get("score_only") else "rejudge" if p.get("rejudge_only") else "main")


def _serial_busy(candidate, key, now) -> bool:
    """Another job with this key is running on a live lease."""
    return any(_serial_key(j) == key for j in JobState.objects.filter(
        kind=candidate.kind, status="running", lease_until__gte=now).exclude(pk=candidate.pk))


def claim(lease_seconds=120, lane="any"):
    # Repair historical terminal rows left behind by a process dying before its
    # old finally cleanup. New terminal transitions erase credentials atomically.
    JobState.objects.filter(status__in=TERMINAL_STATUSES).exclude(credential_ciphertext="").update(credential_ciphertext="")
    now = timezone.now()
    eligible = Q(status="queued") | Q(status="running", lease_until__lt=now)
    rows = JobState.objects.filter(eligible).order_by("created_at")
    if lane == "short":
        rows = rows.filter(kind__in=["minime_round", "minime_learn"])
    # Serial keys running on a live lease, read once per poll: queued jobs waiting on them, and jobs this
    # lane does not take, are passed over without counting toward the window, so a backlog of blocked
    # jobs never hides runnable work behind it. The claim below still re-checks atomically.
    busy = {_serial_key(j) for j in JobState.objects.filter(
        kind__in=("minime_round", "minime_learn"), status="running", lease_until__gte=now)} - {None}
    window, considered = (200 if lane == "short" else 20), 0

    def pages(size=100):
        """The eligible rows in queue order, a materialized page at a time (no open cursor while claim()
        writes), paged by position (created_at, id) so rows leaving the set mid-poll shift nothing."""
        ordered, after = rows.order_by("created_at", "id"), None
        while True:
            page = list(ordered.filter(Q(created_at__gt=after[0]) | Q(created_at=after[0], id__gt=after[1]))[:size]
                        if after else ordered[:size])
            yield from page
            if len(page) < size:
                return
            after = (page[-1].created_at, page[-1].id)

    for candidate in pages():
        if lane == "short" and candidate.status == "queued" and not is_short(candidate):
            continue                    # a learning run: left for an "any" worker
        if candidate.status == "queued" and _serial_key(candidate) in busy:
            continue                    # waits for the running one with its key
        considered += 1
        if considered > window:
            break
        if candidate.status == "running" and candidate.kind in PAUSE_ON_INTERRUPT:
            changed = JobState.objects.filter(pk=candidate.pk, status="running", lease_until__lt=now).update(
                status="cancelled", stage="Paused", credential_ciphertext="", lease_token=None, lease_until=None)
            if changed:
                print(f"[durable] {candidate.kind} {candidate.id}: its worker stopped; the task is paused until "
                      f"Start optimizing", file=sys.stderr, flush=True)
                from .minime import pause_task
                pause_task((candidate.payload or {}).get("task_id"))
            continue
        if candidate.status == "running" and not candidate.payload.get("require_scored_evidence"):
            status = "cancelled" if candidate.cancel_requested_at else "interrupted"   # stopped by the user: cancelled
            error = "" if status == "cancelled" else "Job cannot be safely replayed after worker interruption"
            with transaction.atomic():
                changed = JobState.objects.filter(pk=candidate.pk, status="running", lease_until__lt=now).update(
                    status=status, stage=status.title(), error=error, credential_ciphertext="", lease_token=None, lease_until=None)
                if changed and candidate.kind == "benchmark":
                    _sync_benchmark_history(candidate.id, status, error=error)
            continue
        token = uuid.uuid4().hex
        key = _serial_key(candidate)
        # The serial check and the compare-and-set are one transaction (SQLite IMMEDIATE serializes
        # writers), so two workers cannot both start jobs with the same key. The CAS alone works on
        # the deployed SQLite database too.
        with transaction.atomic():
            if key and _serial_busy(candidate, key, now):
                continue                    # waits for the running one (claimed on a later poll)
            changed = JobState.objects.filter(pk=candidate.pk).filter(eligible).update(
                status="running", stage="Resuming" if candidate.status == "running" else "Starting",
                lease_token=token, lease_until=now + timedelta(seconds=lease_seconds), updated_at=now)
        if changed:
            return JobState.objects.get(pk=candidate.pk, lease_token=token)
    return None

def assert_lease(job_id: str, token: str):
    if not token:
        raise LeaseLost("An execution lease is required")
    row = JobState.objects.filter(pk=job_id, lease_token=token, status="running", lease_until__gt=Now()).first()
    if not row:
        raise LeaseLost("Job lease expired or changed")
    if row.cancel_requested_at:
        raise JobCancelled("Cancellation requested")
    return row


def _live_lease(job_id, token, *, allow_cancelled=False, owner=None):
    if not token:
        raise LeaseLost("An execution lease is required")
    query = JobState.objects.filter(pk=job_id, lease_token=token, status="running", lease_until__gt=Now())
    if owner is not None:
        query = query.filter(owner=owner)
    return query if allow_cancelled else query.filter(cancel_requested_at__isnull=True)


def _raise_lease_loss(job_id, token, *, allow_cancelled=False, owner=None):
    if not allow_cancelled and _live_lease(job_id, token, allow_cancelled=True, owner=owner).filter(cancel_requested_at__isnull=False).exists():
        raise JobCancelled("Cancellation requested")
    raise LeaseLost("Job lease expired or changed")


@contextmanager
def lease_write(job_id: str, token: str, *, allow_cancelled=False):
    """Fence related writes with a live lease at both transaction boundaries.

    The first CAS takes a write lock on SQLite as well as PostgreSQL. The final
    CAS uses database time, so expiry while writing rolls back every enclosed
    checkpoint/receipt write. Terminal transitions use finish_job instead.
    """
    with transaction.atomic():
        if _live_lease(job_id, token, allow_cancelled=allow_cancelled).update(updated_at=Now()) != 1:
            _raise_lease_loss(job_id, token, allow_cancelled=allow_cancelled)
        row = JobState.objects.get(pk=job_id, lease_token=token)
        yield row
        if _live_lease(job_id, token, allow_cancelled=allow_cancelled).update(updated_at=Now()) != 1:
            _raise_lease_loss(job_id, token, allow_cancelled=allow_cancelled)


def finish_job(job_id: str, token: str, status: str, *, owner=None, **fields):
    """Commit terminal state and credential erasure in the same fenced write.

    Retain the token until process_one's callback/cleanup: it identifies the
    worker that committed completion, without retaining its delegated secret.
    """
    if status not in TERMINAL_STATUSES:
        raise ValueError("A terminal job status is required")
    if set(fields) - {"stage", "result", "error"}:
        raise ValueError("Unsupported terminal job field")
    if "result" in fields:
        bounded_json_size(fields["result"], "result")
    allow_cancelled = status != "done"
    with transaction.atomic():
        changed = _live_lease(job_id, token, allow_cancelled=allow_cancelled, owner=owner).update(
            **fields, status=status, credential_ciphertext="", updated_at=Now())
        if changed != 1:
            _raise_lease_loss(job_id, token, allow_cancelled=allow_cancelled, owner=owner)
        _sync_benchmark_history(job_id, status, result=fields.get("result"), error=fields.get("error", ""))

def heartbeat(job_id: str, token: str, seconds=120):
    changed = _live_lease(job_id, token, allow_cancelled=True).update(
        lease_until=Now() + timedelta(seconds=seconds), updated_at=Now())
    if not changed:
        raise LeaseLost("Job lease expired or changed")

def cancel(job_id: str, owner: str):
    with transaction.atomic():
        active = JobState.objects.filter(pk=job_id, owner=owner, status__in=["queued", "running"])
        active.update(cancel_requested_at=timezone.now(), updated_at=timezone.now())
        active.filter(status="queued").update(status="cancelled", stage="Cancelled", credential_ciphertext="")
        row = JobState.objects.filter(pk=job_id, owner=owner).first()
        if row and row.kind == "benchmark" and row.cancel_requested_at and row.status in ("running", "cancelled"):
            # Keep the live worker lease for its already-dispatched receipt, but
            # make cancellation terminal in benchmark history/UI immediately.
            _sync_benchmark_history(job_id, "cancelled")
        return row


def checkpoint(job_id: str, token: str, key: str, value):
    with lease_write(job_id, token) as row:
        values = dict(row.checkpoint or {})
        values[key] = value
        bounded_json_size(values, "checkpoint")
        if _live_lease(job_id, token).update(checkpoint=values, updated_at=Now()) != 1:
            raise LeaseLost("Checkpoint lease changed")

def process_one():
    """Run one claimed job in a dedicated worker process. Completed model calls
    are replayable; an ambiguous dispatch stops the resumed job for reconciliation."""
    import threading
    from django.db import close_old_connections
    from . import services
    import os
    lane = os.environ.get("GP_JOB_LANE", "any")
    row = claim(lane=lane if lane in LANES else "any")
    if row is None:
        return False
    stop = threading.Event()
    def renew():
        while not stop.wait(20):
            close_old_connections()
            try:
                heartbeat(row.id, row.lease_token)
            except LeaseLost:
                print(f"[durable] {row.kind} {row.id} lost its lease; heartbeat stops", file=sys.stderr, flush=True)
                return
            except Exception as exc:  # noqa: BLE001
                # A transient error (SQLite "database is locked" while another job writes) must not
                # end the heartbeat: the lease would lapse and a live job be reaped as interrupted.
                print(f"[durable] heartbeat for {row.kind} {row.id} failed, retrying: {type(exc).__name__}: {exc}",
                      file=sys.stderr, flush=True)
            finally:
                close_old_connections()
    keeper = threading.Thread(target=renew, name="job-lease-heartbeat")
    keeper.start()
    payload = {}
    try:
        payload = {**row.payload, **decrypt_secret(row.credential_ciphertext)}
        job = services.Job(id=row.id, owner=row.owner, lease_token=row.lease_token)
        assert_lease(row.id, row.lease_token)
        if row.kind == "benchmark":
            from .benchmarks import run_queued
            run_queued(job, payload)
        elif payload.get("require_scored_evidence"):
            from .scored import run_scored
            run_scored(job, payload)
        else:
            from . import minime
            runner = {"optimize": services._run_job, "experiment": services._run_experiment,
                      "anchor": services._run_anchor_job, "diagnostics": services._run_diagnostics_job,
                      "minime_learn": minime.run_learn, "minime_round": minime.run_round}[row.kind]
            runner(job, payload)
    except LeaseLost:
        # A stale worker cannot overwrite its successor's state.
        pass
    except Exception as error:
        status = "cancelled" if isinstance(error, JobCancelled) else "interrupted" if isinstance(error, UncertainInference) else "error"
        if status == "error":
            sentry_sdk.capture_exception(error)
        try:
            finish_job(row.id, row.lease_token, status, stage=status.title(), error=type(error).__name__)
        except LeaseLost:
            pass
    finally:
        stop.set()
        keeper.join(timeout=5)
        terminal = JobState.objects.filter(pk=row.id, lease_token=row.lease_token, status__in=TERMINAL_STATUSES).first()
        JobState.objects.filter(pk=row.id, lease_token=row.lease_token, status__in=TERMINAL_STATUSES).update(
            credential_ciphertext="", lease_token=None, lease_until=None)
        # The scored runner commits directly to JobState, so Job.__setattr__
        # cannot deliver this notice. Only the worker that owned the terminal
        # commit may notify; callbacks carry no output or billing evidence.
        if terminal and payload.get("_callback_url"):
            services._fire_callback(SimpleNamespace(id=terminal.id, owner=terminal.owner,
                status=terminal.status, result=terminal.result, error=terminal.error,
                _callback_url=payload["_callback_url"]), notice_only=bool(payload.get("require_scored_evidence")))
        close_old_connections()
    return True
