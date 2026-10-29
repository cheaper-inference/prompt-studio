"""Rounds of new outputs to review, and the first draft of the rules."""
from __future__ import annotations
import traceback
from django.db.models import Max
from django.utils import timezone
from ps_utils.minime.candidates import expand_inputs, generate_round, plan_round
from ps_utils.minime.qcr import render_task, task_context
from ps_utils.minime.rubric_text import DRAFT_TEMPLATE, from_text, parse
from ..models import MiniMeLabel, MiniMeTask
from . import config as _config
from . import jobs as _jobs
from . import judging as _judging
from . import scoring as _scoring
from . import store as _store


def draft_rubric(payload, task_text, requirements) -> str:
    """The rewriter model drafts the first rubric from the task description: one checkable
    requirement per statement, counts and lengths phrased so they compile to code."""
    from ..services import _endpoint_for
    ep = _endpoint_for(payload)
    reply = ep.client.chat.completions.create(
        model=payload.get("rewriter_model") or _config.REWRITER_MODEL, max_tokens=_config.REWRITER_MAX_TOKENS, timeout=_config.REWRITER_TIMEOUT,
        messages=[{"role": "user", "content": DRAFT_TEMPLATE.format(task=task_text, requirements=requirements or "(not given)")}])
    return reply.choices[0].message.content or ""


def _draft(task, payload, job) -> None:
    """Start small: the user's own lines are v0, exactly as written. Only when the description
    gives no rules at all does the rewriter draft a few (at most DRAFT_MAX) from the objective,
    once, before the first round; learning grows the rubric from there. Best effort."""
    if (task.settings or {}).get("drafted") or _store.latest(task).version != 0:
        return
    _store._update_settings(task, lambda s: {**s, "drafted": True})
    current = _store.current_qcr(task)
    if current.active():
        return
    job.stage = "Drafting rubric"
    try:
        text = draft_rubric(payload, task_context(current.task), current.task.get("requirements") or "")
        lines = parse(text)[:_config.DRAFT_MAX]
        candidate, edits = from_text("\n".join(f"{n}. {t}" for n, t in enumerate(lines, 1)), current, None)
    except Exception as exc:  # noqa: BLE001
        _store.log(task, f"Could not draft the rubric ({type(exc).__name__}); starting from your description as written.",
            kind="routine")
        return
    if edits:
        candidate.evaluator_model = current.evaluator_model
        v = _store._new_version(task, candidate, "learn", {"edits": edits, "draft": True})
        _store.log(task, f"Drafted the rubric from your description: {len(candidate.active())} statement(s).",
            kind="draft", edits=edits, version=v.version, rewriter=payload.get("rewriter_model"))


def run_round(job, payload):
    _scoring._ensure_judge_epoch(payload)
    try:
        task = MiniMeTask.objects.get(pk=payload["task_id"], owner=payload["_owner"])
        payload = _judging._payload(task, payload)
        _draft(task, payload, job)
        qcr = _store.current_qcr(task)
        size = int((task.settings or {}).get("round_size") or _config.ROUND_SIZE)
        number = (task.labels.aggregate(m=Max("round"))["m"] or 0) + 1
        plan = plan_round(qcr, size=size, heldout_share=_config.BLIND_SHARE, random_share=_config.RANDOM_SHARE,
                          comply_share=_config.COMPLY_SHARE, blind_cycle=_config.BLIND_CYCLE, offset=number, tested=_scoring.rule_tests(task))
        complete, make_judge, make_item_judge, _ = _judging.llm(payload)
        configured = _judging.configured_at(payload)
        pool = list((task.settings or {}).get("input_pool") or [])
        if len(pool) < _config.POOL_MAX:
            job.stage = "Finding inputs"
            new = expand_inputs(render_task(qcr.task), task.inputs, pool, complete, n=_config.POOL_GROWTH)
            if new:
                _store._update_settings(task, lambda s: {**s, "input_pool": (list(s.get("input_pool") or []) + new)[:_config.POOL_MAX]})
                _store.log(task, f"Found {len(new)} new inputs similar to yours: " + ", ".join(new[:4]) + ("…" if len(new) > 4 else "") + ".",
                    kind="routine")
        candidates = generate_round(qcr, plan, _store.input_pools(task), complete, seed=number,
                                    deployed_prompt=task.deployed_prompt or None, progress=_jobs._stage(job, "Writing"),
                                    parallel=_config.PARALLEL)
        # Judged PARALLEL at a time, and each batch saved at once: the first outputs can be reviewed
        # while the rest are still being judged.
        for start in range(0, len(candidates), _config.PARALLEL):
            job.stage = f"Judging {round(100 * start / len(candidates))}%" + _judging._judge_speed()
            batch = list(enumerate(candidates))[start:start + _config.PARALLEL]
            verdicts = _judging._verdicts(qcr, [{"id": c["id"], "input": c["input"], "output": c["output"]} for _, c in batch],
                                 make_judge, make_item_judge)
            MiniMeLabel.objects.bulk_create([
                MiniMeLabel(task=task, round=number, position=n, source="generated", split=c["split"],
                            input=c["input"], output=c["output"], kind=c["kind"], target=c["target"] or "",
                            expected_pass=c["expected_pass"], judged_pass=verdicts[c["id"]][0],
                            judged_failed=verdicts[c["id"]][1], judged_version=qcr.version,
                            judged_at=configured)   # when the judge was read: a later change makes it stale
                for n, c in batch])
        _store.log(task, f"{len(candidates)} new outputs to review.", kind="routine")
        _jobs._finish(job, {"round": number, "candidates": len(candidates)})
    except MiniMeTask.DoesNotExist:
        _jobs._finish(job, {"skipped": "the task was deleted"})     # deleted while this job ran: nothing to do
    except Exception as exc:  # noqa: BLE001
        _jobs._stopped(exc)
        traceback.print_exc()
        _jobs._finish(job, error=f"{type(exc).__name__}: {exc}")


__minime_names__ = ('_draft', 'draft_rubric', 'run_round',)
