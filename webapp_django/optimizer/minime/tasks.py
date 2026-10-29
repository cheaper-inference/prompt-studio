"""Task-level operations behind the API: create, edit, review, undo, and the page's summary."""
from __future__ import annotations
from django.db import transaction
from django.utils import timezone
from ps_utils.minime.candidates import blind_only
from ps_utils.minime.labels import LabelError, parse_labels, parse_verdict
from ps_utils.minime.qcr import QCR, STATUSES, TASK_FIELDS, Statement, judge_prompt, seed_qcr
from ..models import MiniMeLabel, MiniMeTask, MiniMeVersion
from . import config as _config
from . import jobs as _jobs
from . import judging as _judging
from . import learning as _learning
from . import scoring as _scoring
from . import store as _store


def summary(task) -> dict:
    v = _store.latest(task)
    qcr = QCR.from_dict(v.qcr)
    settings = task.settings or {}
    q = _scoring.quality(task)
    return {"id": str(task.id), "name": task.name, "definition": task.definition, "inputs": task.inputs,
            "deployed_prompt": task.deployed_prompt,
            "settings": {k: settings.get(k) for k in ("task_model", "judge_model", "rewriter_model", "round_size")},
            "input_pool": [{"text": x, "blind": blind_only(x)} for x in settings.get("input_pool") or []],
            "rewriter_default": _config.REWRITER_MODEL,
            "models": _store.effective_models(task),
            "version": v.version, "statements": [{"id": s.id, "text": s.text, "origin": s.origin}
                                                 for s in qcr.active()],
            "judge_prompt": judge_prompt(qcr),
            "log": settings.get("log") or [], "jobs": _jobs.active_jobs(task), "failed": _jobs.failed_jobs(task),
            "waiting": task.labels.filter(passed=None).count(),
            "reviewed": task.labels.exclude(passed=None).count(),
            "feedback_since_learn": _store.feedback_since_learn(task), "learn_every": _learning.learn_every(task), "quality": q,
            "ready": _scoring.readiness(task, q), "series": _scoring.quality_series(task), "optimizing": _jobs.optimizing(task)}


def next_candidate(task) -> dict | None:
    """The next answer to review. On a blind row the judge's verdict is not sent at all."""
    l = task.labels.filter(passed=None).first()
    if l is None:
        return None
    blind = l.split == "heldout" or l.judged_pass is None
    out = {"id": l.pk, "input": l.input, "output": l.output, "blind": blind}
    if not blind:
        out.update(judged_pass=l.judged_pass, failed=[f["text"] for f in l.judged_failed or []],
                   judged_version=l.judged_version)        # older than the rules: the page says so
    return out


def create_task(owner, name, definition, inputs, deployed_prompt="", settings=None) -> MiniMeTask:
    d, inputs, deployed_prompt = _store._definition(definition), _store._inputs(inputs), _store._prompt(deployed_prompt)
    qcr = seed_qcr(d)
    with transaction.atomic():
        task = MiniMeTask.objects.create(owner=owner, name=(name or "").strip()[:200] or "MiniMe task",
                                         definition=d, inputs=inputs, deployed_prompt=deployed_prompt,
                                         settings=_store._settings(settings, {"round_size": _config.ROUND_SIZE}))
        MiniMeVersion.objects.create(task=task, version=0, source="seed", qcr=qcr.to_dict())
    _store.log(task, f"Started with {len(qcr.active())} statement(s) from your description.")
    return task


def delete_task(task) -> None:
    """Remove the task with all its labels and versions (cascade). Its queued jobs are cancelled;
    a running one notices on its next write and ends quietly."""
    from ..durable import cancel
    for kind, job in _jobs.active_job_rows(task):
        cancel(job.id, task.owner)
    task.delete()


def update_task(task, deployed_prompt=None, inputs=None, settings=None, input_pool=None, name=None) -> MiniMeTask:
    """The name, the deployed prompt, the real inputs and the models can change. The description
    changes through reword_task (minor) or clone_task (substantial)."""
    if name is not None and str(name).strip():
        task.name = str(name).strip()[:200]
        task.save(update_fields=["name", "updated_at"])
    if deployed_prompt is not None:
        task.deployed_prompt = _store._prompt(deployed_prompt)
    if inputs is not None:
        task.inputs = _store._inputs(inputs)
    task.save(update_fields=["deployed_prompt", "inputs", "updated_at"])
    if settings is not None:
        before = _store.effective_models(task)["judge_model"]
        _store._update_settings(task, lambda s: _store._settings(settings, s))
        task.refresh_from_db()
        after = _store.effective_models(task)["judge_model"]
        if after != before:            # another instrument: re-score every version at the new setting
            _scoring._judge_epoch(task)
            _store.log(task, f"Judge changed to {after}. The versions are scored again on the blind checks with this judge.")
            _jobs.run_rejudge(task, task.owner)    # the waiting answers too (only while optimizing; else Start does it)
    if input_pool is not None:          # the user can only prune the generated inputs, not add hidden ones
        keep = {str(x).strip() for x in input_pool}
        _store._update_settings(task, lambda s: {**s, "input_pool": [x for x in s.get("input_pool") or [] if x in keep]})
    return task


def reword_task(task, definition) -> MiniMeVersion | None:
    """A minor change to the description (rewording, touch-ups): the task keeps its reviews and history.
    A new rubric version carries the new wording: answer writing and the MiniMe prompt use it. The
    statements are unchanged, so verdicts are too (the judge checks statements). None when nothing changed."""
    current = _store.current_qcr(task)
    before = {k: str(current.task.get(k) or "").strip() for k in TASK_FIELDS}
    d = _store._definition({**before, **{k: (definition or {})[k] for k in _config.DESCRIPTION_FIELDS if k in (definition or {})}})
    changes = [{"field": k, "before": before[k], "after": d[k]} for k in _config.DESCRIPTION_FIELDS if d[k] != before[k]]
    if not changes:
        return None
    v = _store._new_version(task, QCR(task=d, statements=current.statements, evaluator_model=current.evaluator_model),
                     "edit", {"description": changes})
    task.definition = d
    task.save(update_fields=["definition", "updated_at"])
    _store.log(task, "You reworded the task. Reviews and history are kept; new outputs use the new wording.",
        kind="describe", changes=changes, version=v.version)
    return v


def clone_task(task, owner, definition, name=None) -> MiniMeTask:
    """A substantial change to the description: a new task, since the reviews were given against the
    old one. It starts from this task's current rubric (the statements learned so far, which can then
    be edited), real inputs, prompt and models; reviews and history start fresh."""
    current = _store.current_qcr(task)
    base = {k: str(current.task.get(k) or "").strip() for k in TASK_FIELDS}
    base["requirements"] = "\n".join(s.text for s in current.active())
    d = {**base, **{k: (definition or {})[k] for k in TASK_FIELDS if k in (definition or {})}}
    settings = {k: (task.settings or {}).get(k) for k in ("task_model", "judge_model", "rewriter_model", "round_size")}
    new = create_task(owner, (name or "").strip() or f"{task.name} (new)", d, task.inputs, task.deployed_prompt,
                      {k: v for k, v in settings.items() if v not in (None, "")})
    _store.log(new, f"Cloned from \u201c{task.name}\u201d with a new description: the rubric carried over, reviews start fresh.")
    _store.log(task, f"Cloned into \u201c{new.name}\u201d with a new description.")
    return new


def edit_qcr(task, statements) -> MiniMeVersion:
    """The user's own edits: a new version; waiting answers are re-judged with it."""
    current = _store.current_qcr(task)
    old = {s.id: s for s in current.statements}
    n = 1 + max([int(s.id[1:]) for s in current.statements if s.id[1:].isdigit()] or [0])
    out, seen = [], set()
    for item in statements or []:
        text, status = str(item.get("text") or "").strip(), item.get("status") or "active"
        if status not in STATUSES:
            raise _config.MiniMeError("invalid status")
        base = old.get(item.get("id"))
        if not text and base is None:
            continue
        if not text:                      # an emptied statement is retired
            text, status = base.text, "retired"
        if base:
            seen.add(base.id)
            out.append(Statement(**{**base.__dict__, "text": text, "status": status,
                                    "flip_rate": base.flip_rate if text == base.text else None}))
        else:
            out.append(Statement(id=f"s{n}", text=text, status=status, origin="user"))
            n += 1
    out += [s for s in current.statements if s.id not in seen and s.status == "retired"]
    key = lambda ss: [(s.id, s.text, s.status) for s in ss if s.status != "retired"]
    if key(out) == key(current.statements):
        raise _config.MiniMeError("Nothing changed.")
    qcr = QCR(task=current.task, statements=out, evaluator_model=current.evaluator_model)
    v = _store._new_version(task, qcr, "edit")
    _store.log(task, "You edited the rubric. Waiting outputs will be re-judged.")
    return v


def apply_proposal(task, at) -> MiniMeVersion:
    """The user overrides the safety check: apply a learning proposal that was not applied
    automatically (it got some reviewed answers wrong). Only while the rubric it was built on is
    still the current one; otherwise its edits would silently undo later changes."""
    entries = list((task.settings or {}).get("log") or [])
    entry = next((e for e in entries if e.get("at") == at and e.get("kind") == "learn"), None)
    if entry is None or not entry.get("proposal"):
        raise _config.MiniMeError("that proposal is not available any more")
    if entry.get("applied"):
        raise _config.MiniMeError("already applied")
    if entry.get("based_on") != _store.latest(task).version:
        raise _config.MiniMeError("the rubric has changed since this was proposed, so it no longer applies")
    qcr = QCR.from_dict(entry["proposal"])
    v = _store._new_version(task, qcr, "learn", {"edits": entry.get("edits") or [], "applied_by_user": True,
                                          "broke": [x["id"] for x in entry.get("broke") or []]})

    def mark(settings):
        for e in settings.get("log") or []:
            if e.get("at") == at:
                e["applied"], e["version"] = True, v.version
        return settings
    _store._update_settings(task, mark)
    broke = len(entry.get("broke") or [])
    _store.log(task, f"You applied the proposed change (it disagrees with you on {broke} reviewed output(s)). "
              "Waiting outputs will be re-judged.")
    return v


def undo_change(task, at) -> MiniMeVersion:
    """Undo an applied learning change: the rubric from before it becomes the current one again
    (a new version, so history is kept). Only while that change is still the current rubric."""
    entries = list((task.settings or {}).get("log") or [])
    entry = next((e for e in entries if e.get("at") == at and e.get("kind") == "learn"), None)
    version = entry and (entry.get("version") if (entry.get("accepted") or entry.get("applied")) else None)
    if version is None:
        raise _config.MiniMeError("that change is not applied")
    if entry.get("undone"):
        raise _config.MiniMeError("already undone")
    if _store.latest(task).version != version:
        raise _config.MiniMeError("the rubric has changed since; edit the rules instead")
    prior = task.versions.filter(version__lt=version).order_by("-version").first()
    v = _store._new_version(task, QCR.from_dict(prior.qcr), "edit", {"undo_of": version})

    def mark(settings):
        for e in settings.get("log") or []:
            if e.get("at") == at:
                e["undone"] = v.version
        return settings
    _store._update_settings(task, mark)
    _store.log(task, f"You undid the change of v{version}: back to the rules of v{prior.version}. Waiting outputs will be re-judged.")
    return v


def upload_labels(task, text, input_col="input", output_col="output", split="learn") -> int:
    """Rows the user already judged (verdict + reason), e.g. from a sheet."""
    if split not in MiniMeLabel.SPLITS:
        raise _config.MiniMeError("split must be learn or heldout")
    try:
        parsed = parse_labels(text, input_col, output_col, source="upload")
    except LabelError as error:
        raise _config.MiniMeError(str(error)) from error
    now = timezone.now()
    MiniMeLabel.objects.bulk_create([MiniMeLabel(task=task, source="upload", split=split, input=r["input"],
                                                 output=r["output"], passed=r["passed"], reason=r["reason"],
                                                 labelled_at=now) for r in parsed])
    task.save(update_fields=["updated_at"])
    return len(parsed)


def interpret_feedback(task, label, seen, comment) -> bool:
    """True when free-text feedback accepts the verdict the user saw (it may still criticise the
    reasons: that is a comment on a kept verdict). One call to the task model, which is fast."""
    complete = _judging.llm(_judging._payload(task, {"task_id": str(task.id), "_owner": task.owner}))[0]
    verdict = "PASS" if seen else "FAIL"
    reply = complete(_config.INTERPRET.format(output=label.output[:2000], said=_judging.assessment(seen, label.judged_failed),
                                      comment=comment, verdict=verdict))
    return not str(reply or "").strip().upper().startswith("CHANGE")


def review(task, label_id, action, verdict=None, reason="", shown_pass=None) -> MiniMeLabel:
    """keep / flip the judge's verdict, or (blind rows) give one. reason: the user's comment, optional
    on any answer; it is what learning reads (on a kept verdict too: right verdict, wrong reason).
    shown_pass: the verdict the user saw. Keep and flip refer to it, not to the stored one, which a
    re-judge (after a rubric edit) may have changed while the answer was on screen."""
    label = task.labels.filter(pk=label_id).first()
    if label is None:
        raise _config.MiniMeError("output not found")
    reason = str(reason or "").strip()
    blind = label.split == "heldout" or label.judged_pass is None
    if action == "comment" and not blind:
        # Only words: whether they agree with MiniMe's verdict is for the model to read, not the user to tick.
        if not reason:
            raise _config.MiniMeError("write your feedback first")
        seen = shown_pass if isinstance(shown_pass, bool) else label.judged_pass
        try:
            agrees = interpret_feedback(task, label, seen, reason)
        except Exception as exc:  # noqa: BLE001
            raise _config.MiniMeError(f"could not read your feedback right now ({type(exc).__name__}); try again")
        label.passed, label.action = (seen if agrees else not seen), ("keep" if agrees else "flip")
    elif action in ("keep", "flip") and not blind:
        seen = shown_pass if isinstance(shown_pass, bool) else label.judged_pass
        label.passed = seen if action == "keep" else not seen
        label.action = action
    elif action == "label":
        passed = parse_verdict(verdict)
        if passed is None:
            raise _config.MiniMeError("choose Pass or Fail")
        label.passed, label.action = passed, ""
    else:
        raise _config.MiniMeError("unknown action")
    label.reason, label.labelled_at = reason, timezone.now()
    label.save(update_fields=["passed", "action", "reason", "labelled_at"])
    task.save(update_fields=["updated_at"])
    return label


def dismiss_tip(task, at) -> None:
    def mark(settings):
        for e in settings.get("log") or []:
            if e.get("at") == at and e.get("kind") == "tip":
                e["dismissed"] = True
        return settings
    _store._update_settings(task, mark)


__minime_names__ = ('apply_proposal', 'clone_task', 'create_task', 'delete_task', 'dismiss_tip', 'edit_qcr', 'interpret_feedback', 'next_candidate', 'review', 'reword_task', 'summary', 'undo_change', 'update_task', 'upload_labels',)
