"""Reading and writing a task's stored state: versions, reviews, settings and the history log."""
from __future__ import annotations
from django.db import transaction
from django.db.models import Max, Q
from django.utils import timezone
from ps_utils.minime.candidates import blind_only, placeholders
from ps_utils.minime.qcr import QCR, TASK_FIELDS
from ..models import MiniMeTask, MiniMeVersion
from . import config as _config


# --------------------------------------------------------------------------- reads
def get_task(task_id, owner) -> MiniMeTask:
    task = MiniMeTask.objects.filter(pk=task_id, owner=owner).first()
    if task is None:
        raise _config.MiniMeError("task not found")
    return task


def latest(task) -> MiniMeVersion:
    return task.versions.order_by("-version").first()


def current_qcr(task) -> QCR:
    v = latest(task)
    qcr = QCR.from_dict(v.qcr)
    qcr.version = v.version
    return qcr


def confirmed(task, split="learn"):
    """Reviewed rows: the user's verdict and comment (optional, on a kept or a flipped verdict)."""
    return [{"id": str(l.pk), "input": l.input, "output": l.output, "passed": l.passed, "reason": l.reason,
             # a comment on a kept verdict, or on an uploaded one (a labelled sheet: no keep/flip action)
             "note": bool(l.reason.strip()) and (l.action == "keep" or (l.action == "" and l.source == "upload")),
             "flip": l.action == "flip",
             "judged_pass": l.judged_pass, "judged_failed": l.judged_failed or [], "judged_version": l.judged_version,
             "at": l.created_at, "judged_at": judged_at(l)}
            for l in task.labels.filter(split=split).exclude(passed=None)]


def feedback_since_learn(task) -> int:
    """Feedback not yet seen by a learn run (accepted or not), so a rejected run does not repeat:
    every No, and every comment on a Yes."""
    qs = task.labels.filter(split="learn").filter(
        Q(action="flip") | (Q(action="keep") & ~Q(reason=""))
        # uploaded rows (a labelled sheet) carry no action: every FAIL, and every PASS with a reason
        | (Q(source="upload") & Q(action="") & ~Q(passed=None) & (Q(passed=False) | ~Q(reason=""))))
    since = (task.settings or {}).get("learned_through")
    return (qs.filter(labelled_at__gt=since) if since else qs).count()


def _update_settings(task, change) -> dict:
    """Read-modify-write of task.settings on the stored row, so a job and a request (or two jobs)
    never overwrite each other's keys from a stale copy."""
    with transaction.atomic():
        fresh = MiniMeTask.objects.select_for_update().get(pk=task.pk)
        fresh.settings = change(dict(fresh.settings or {}))
        fresh.save(update_fields=["settings", "updated_at"])
    task.settings = fresh.settings
    return fresh.settings


def log(task, text, **event) -> None:
    """One entry in "Prompt Rewriting History". A learn run also carries its structured result (event)."""
    entry = {"at": timezone.now().isoformat(), "text": text, **event}
    _update_settings(task, lambda s: {**s, "log": ([entry] + list(s.get("log") or []))[:_config.LOG_SIZE]})


def effective_models(task) -> dict:
    """The models MiniMe's jobs will actually use: the task's setting, else MiniMe's defaults
    (TASK_MODEL for answers, JUDGE_MODEL for judging, REWRITER_MODEL for GEPA's rewriter)."""
    s = task.settings or {}
    return {"task_model": s.get("task_model") or _config.TASK_MODEL, "judge_model": s.get("judge_model") or _config.JUDGE_MODEL,
            "rewriter_model": s.get("rewriter_model") or _config.REWRITER_MODEL}


def input_pools(task) -> dict:
    """{'learn': [...], 'heldout': [...]}. Learning asks use the user's seed inputs plus generated
    ones; blind (held-out) asks use a reserved slice of generated inputs that learning never sees.
    Until that slice exists, blind asks fall back to the seeds."""
    pool = list((task.settings or {}).get("input_pool") or [])
    blind = [x for x in pool if blind_only(x)]
    learn = list(task.inputs) + [x for x in pool if not blind_only(x)]
    return {"learn": learn, "heldout": blind or list(task.inputs)}


# --------------------------------------------------------------------------- writes
def _definition(raw) -> dict:
    d = {k: str((raw or {}).get(k) or "").strip() for k in TASK_FIELDS}
    if not d["objective"]:
        raise _config.MiniMeError("describe the task")
    return d


def _inputs(inputs) -> list:
    inputs = [str(i).strip() for i in inputs or [] if str(i).strip()]
    if not inputs:
        raise _config.MiniMeError("at least one real task input is required")
    return inputs


def _settings(raw, base=None) -> dict:
    out = dict(base or {})
    for k in ("task_model", "judge_model", "rewriter_model"):
        if raw and k in raw:
            out[k] = str(raw.get(k) or "").strip()
    if raw and raw.get("round_size") not in (None, ""):
        try:
            out["round_size"] = max(2, min(int(raw["round_size"]), 50))
        except (TypeError, ValueError):
            raise _config.MiniMeError("round size must be a number")
    return out


def _prompt(text) -> str:
    """The deployed prompt: at most one placeholder, since MiniMe fills in one input."""
    text = str(text or "").strip()
    names = placeholders(text)
    if len(names) > 1:
        raise _config.MiniMeError("your prompt has " + ", ".join("{{%s}}" % n for n in names)
                          + ": use one placeholder for the input")
    return text


def judge_since(task):
    """When the judge last changed: verdicts written before it are stale (another judge made them)."""
    from datetime import datetime
    at = (task.settings or {}).get("judge_since")
    return datetime.fromisoformat(at) if at else None


def judged_at(label):
    """When a label's verdict was written (rows from before judged_at existed: when created)."""
    return label.judged_at or label.created_at


_ANY = object()


def _new_version(task, qcr, source, report=None, base=None, judge_since=_ANY):
    """The next version of the rules. With base (the version a rewrite started from), it is created
    only if that is still the latest, and with judge_since (the judge epoch it was chosen in) only
    if the judge has not changed since; both in the same transaction (SQLite IMMEDIATE: writers are
    serialized), so a hand edit or a judge change made meanwhile always wins; otherwise None."""
    with transaction.atomic():
        latest = task.versions.aggregate(m=Max("version"))["m"] or 0
        if base is not None and latest != base:
            return None
        if judge_since is not _ANY and (MiniMeTask.objects.get(pk=task.pk).settings or {}).get("judge_since") != judge_since:
            return None
        number = latest + 1
        qcr.version = number
        return MiniMeVersion.objects.create(task=task, version=number, source=source, qcr=qcr.to_dict(),
                                            report=report or {})


__minime_names__ = ('_definition', '_inputs', '_new_version', '_prompt', '_settings', '_update_settings', 'confirmed', 'current_qcr', 'effective_models', 'feedback_since_learn', 'get_task', 'input_pools', 'judge_since', 'judged_at', 'latest', 'log',)
