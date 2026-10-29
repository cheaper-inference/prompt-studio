"""MiniMe's background jobs: starting, stopping and pausing them, Start optimizing, and what the page shows of them."""
from __future__ import annotations
import traceback
from django.utils import timezone
from ..models import JobState, MiniMeTask
from . import config as _config
from . import judging as _judging
from . import learning as _learning
from . import scoring as _scoring
from . import store as _store


def active_job_rows(task):
    """[(kind, JobState)] of this task's queued or running MiniMe jobs."""
    return [(j.kind, j) for j in JobState.objects.filter(owner=task.owner, kind__in=_config.JOB_KINDS, status__in=("queued", "running"))
            if (j.payload or {}).get("task_id") == str(task.id)]


def optimizing(task) -> bool:
    """Is MiniMe working on this task? Set by Start optimizing, cleared by Stop or a worker restart.
    While off, reviews are kept but no job starts on its own."""
    return bool((task.settings or {}).get("optimizing"))


def start_optimizing(task, owner) -> list:
    """Start optimizing: resume what is pending (a learning run with enough feedback, or one cut
    off mid-way, which picks up its GEPA checkpoint or saved rewrite), new outputs when few wait,
    and the blind-check scoring of every version."""
    _store._update_settings(task, lambda s: {**s, "optimizing": True})
    task.refresh_from_db()
    s = task.settings or {}
    started, running = [], active_jobs(task)
    if "minime_learn" not in running and (_store.feedback_since_learn(task) >= _learning.learn_every(task)
                                           or s.get("gepa_run") or s.get("pending_rewrite")):
        started.append(start("minime_learn", owner, {"task_id": str(task.id)}).id)
    stale = bool(_judging.stale_waiting(task, _store.latest(task).version))   # other rules, or the previous judge
    if stale and "minime_learn" not in running and not started:       # rules changed while paused
        started.append(start("minime_learn", owner, {"task_id": str(task.id), "rejudge_only": True}).id)
    started += autostart(task, owner)
    scoring = ensure_scored(task, owner)
    return started + ([scoring.id] if scoring else [])


def pause_if_ready(task) -> bool:
    """Reaching Ready pauses the task: all its jobs stop and it waits for the user (Start
    optimizing keeps learning). Only on the way into Ready: a task the user starts again while
    ready is not paused again until it has left Ready and come back. Returns True if it paused."""
    task.refresh_from_db()
    ready = _scoring.readiness(task)["ready"]
    was = bool((task.settings or {}).get("ready_paused"))
    if ready == was:
        return False
    _store._update_settings(task, lambda s: {**s, "ready_paused": ready})
    if not ready or not optimizing(task):
        return False
    stop_jobs(task, quiet=True)
    _store.log(task, "Ready to use: MiniMe paused optimizing. Start optimizing to keep learning.")
    return True


def pause_task(task_id):
    """A worker stopped mid-job (a restart): pause the task until the user starts it again."""
    try:
        task = MiniMeTask.objects.get(pk=task_id)
    except (MiniMeTask.DoesNotExist, ValueError, TypeError):
        return
    stop_jobs(task, quiet=True)      # like Stop: the task's queued jobs end too (no paid calls while paused)
    _store.log(task, "Paused: the worker restarted. Start optimizing to resume.", kind="routine")


def _stopped(exc):
    """A job the user stopped ends as cancelled (the worker records it), not as an error."""
    from ..durable import JobCancelled
    if isinstance(exc, JobCancelled):
        raise exc


def stop_jobs(task, quiet=False) -> int:
    """Stop every queued or running job of this task. Queued ones end at once; a running one stops
    at its next progress write (a GEPA search at its next iteration, keeping its checkpoint, so a
    later run resumes it instead of starting over). Returns how many were stopped."""
    from ..durable import cancel
    _store._update_settings(task, lambda s: {**s, "optimizing": False})
    rows = active_job_rows(task)
    for kind, job in rows:
        cancel(job.id, task.owner)
    if not quiet:
        _store.log(task, "You stopped optimizing.", kind="routine")
    return len(rows)


def _cancel_requested(job) -> bool:
    jid = getattr(job, "id", None)
    return bool(jid) and JobState.objects.filter(pk=jid).exclude(cancel_requested_at=None).exists()


def active_jobs(task) -> dict:
    """{kind: {stage, since}} of this task's queued or running MiniMe jobs (since: when it was queued)."""
    out = {}
    for j in JobState.objects.filter(owner=task.owner, kind__in=_config.JOB_KINDS, status__in=("queued", "running")):
        if (j.payload or {}).get("task_id") == str(task.id):
            out[j.kind] = {"stage": "Stopping" if j.cancel_requested_at else j.stage, "since": j.created_at.isoformat(),
                           "rejudge": bool((j.payload or {}).get("rejudge_only") or (j.payload or {}).get("score_only")),
                           "stopping": bool(j.cancel_requested_at)}
    return out


def _plain_error(error) -> str:
    """A failed job's error in a few words the user can act on."""
    e = str(error or "")
    if " 503" in e or "No compatible route" in e or "ServiceUnavailable" in e:
        return "endpoint unavailable (503)"
    if " 429" in e or "RateLimit" in e:
        return "rate-limited (429)"
    if " 401" in e or "Authentication" in e:
        return "API key rejected (401)"
    if "Timeout" in e or "timed out" in e:
        return "timeout"
    return e[:200] or "unknown error"


def failed_jobs(task) -> dict:
    """{kind: {error, detail, at}} when this task's latest finished job of a kind failed and none
    of that kind is running: so the page can say why nothing new arrived, instead of 'nothing to review'."""
    out, seen = {}, set()
    for j in (JobState.objects.filter(owner=task.owner, kind__in=_config.JOB_KINDS, status__in=("done", "error", "interrupted"))
              .order_by("-created_at")[:50]):
        if ((j.payload or {}).get("task_id") != str(task.id) or j.kind in seen or (j.payload or {}).get("rejudge_only")
                or (j.payload or {}).get("score_only")):
            continue
        seen.add(j.kind)
        if j.status == "error":
            out[j.kind] = {"error": _plain_error(j.error), "detail": str(j.error or "")[:500], "at": j.updated_at.isoformat()}
        elif j.status == "interrupted":
            out[j.kind] = {"error": "stopped (restart or worker cut off)",
                           "detail": str(j.error or "")[:500], "at": j.updated_at.isoformat()}
    running = active_jobs(task)
    return {k: v for k, v in out.items() if k not in running}


def autostart(task, owner) -> list:
    """Start what the loop needs next: a round when few answers wait, learning after enough feedback.
    Only while optimizing (Start optimizing)."""
    task.refresh_from_db()
    if not optimizing(task):
        return []
    started, running = [], active_jobs(task)
    if _store.feedback_since_learn(task) >= _learning.learn_every(task) and "minime_learn" not in running:
        started.append(start("minime_learn", owner, {"task_id": str(task.id)}).id)
    if task.labels.filter(passed=None).count() < _config.LOW_QUEUE and "minime_round" not in running:
        started.append(start("minime_round", owner, {"task_id": str(task.id)}).id)
    return started


# --------------------------------------------------------------------------- jobs
def _variant(payload) -> str:
    p = payload or {}
    return "score" if p.get("score_only") else "rejudge" if p.get("rejudge_only") else "main"


def start(kind, owner, payload):
    """Enqueue a job for a task, at most one per (task, kind, variant) queued or running: the check
    and the enqueue are one transaction (SQLite IMMEDIATE serializes writers across workers and
    requests), so two quick reviews or clicks cannot both start a round or a learning run. A job
    being stopped does not count. A scoring job that tests a new version (test) always starts.
    Rejudge and scoring jobs merge only into a queued one: a running one may be past its last check
    for new work, so the request queues behind it instead of being absorbed and lost."""
    from django.db import transaction
    from ..durable import enqueue
    statuses = ("queued", "running") if _variant(payload) == "main" else ("queued",)
    with transaction.atomic():
        if not payload.get("test"):
            for j in JobState.objects.filter(owner=owner, kind=kind, status__in=statuses,
                                             cancel_requested_at=None):
                if (j.payload or {}).get("task_id") == payload.get("task_id") and _variant(j.payload) == _variant(payload):
                    return j
        return enqueue(kind, {**payload, "_owner": owner})


def _stage(job, verb):
    """progress callback: 'Writing 40%' (before each batch)."""
    def report(done, total):
        job.stage = f"{verb} {round(100 * done / total) if total else 0}%" + _judging._judge_speed()
    return report


def _finish(job, result=None, error=None):
    if error:
        job.error, job.stage, job.status = error, "Error", "error"
    else:
        job.result, job.stage, job.status = result or {}, "Done", "done"


def _after_job(payload) -> bool:
    """After a job: pause the task if it has just reached Ready (pause_if_ready)."""
    try:
        task = MiniMeTask.objects.get(pk=payload["task_id"])
        return pause_if_ready(task)
    except MiniMeTask.DoesNotExist:
        return False
    except Exception:  # noqa: BLE001 - best effort
        traceback.print_exc()
        return False


def ensure_scored(task, owner, force=False, test=None):
    """When versions still lack verdicts on some blind checks and nothing runs for this task, start
    a scoring-only job, so the chart catches up on its own (not only after the next learning run).
    Asks at most once per SCORE_RETRY for the same backlog, so a failing scorer is not restarted
    in a loop."""
    from datetime import datetime
    if not optimizing(task) or (active_job_rows(task) and not test):
        return None
    to_go = sum(not e["scored"] for e in _scoring.blind_scores(task))
    if test:                                       # a run applied a rewrite: always check it
        return start("minime_learn", owner, {"task_id": str(task.id), "score_only": True, "test": test})
    if not to_go:
        return None
    tried = (task.settings or {}).get("score_tried") or {}
    now = timezone.now()
    if not force and tried.get("to_go") == to_go and tried.get("at") and \
            (now - datetime.fromisoformat(tried["at"])).total_seconds() < _config.SCORE_RETRY:
        return None
    _store._update_settings(task, lambda s: {**s, "score_tried": {"at": now.isoformat(), "to_go": to_go}})
    return start("minime_learn", owner, {"task_id": str(task.id), "score_only": True})


def run_rejudge(task, owner):
    """After a hand edit: re-judge waiting answers in the background (a learn job that only re-judges).
    Only while optimizing; otherwise Start optimizing does it."""
    if not optimizing(task):
        return None
    return start("minime_learn", owner, {"task_id": str(task.id), "rejudge_only": True})


__minime_names__ = ('_after_job', '_cancel_requested', '_finish', '_plain_error', '_stage', '_stopped', '_variant', 'active_job_rows', 'active_jobs', 'autostart', 'ensure_scored', 'failed_jobs', 'optimizing', 'pause_if_ready', 'pause_task', 'run_rejudge', 'start', 'start_optimizing', 'stop_jobs',)
