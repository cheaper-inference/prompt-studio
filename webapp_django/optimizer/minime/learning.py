"""Learning new rules: the cold-start rewrite, GEPA (called directly), the follow-up check, tips and chaining."""
from __future__ import annotations
import re
import shutil
import time
import traceback
import uuid
import zlib
from pathlib import Path
from django.utils import timezone
from ps_utils.minime.qcr import QCR, task_context
from ps_utils.minime.rubric_text import (MAX_EDITS, REFLECTION_TEMPLATE, REFLECTION_TEMPLATE_BOLD,
    RubricRejected, from_text, to_text)
from ..models import MiniMeTask
from . import config as _config
from . import jobs as _jobs
from . import judging as _judging
from . import scoring as _scoring
from . import store as _store


def _records(rows):
    return [{"inputs": {"input": r["input"], "output": r["output"]},
             "expectations": {"expected_response": "PASS" if r["passed"] else "FAIL", "reason": r["reason"],
                              "note": r["note"]}}
            for r in rows]


def split_train(rows, n_pareto=_config.PARETO_SIZE, share=_config.PARETO_SHARE):
    """(D_feedback, D_pareto): GEPA's split of D_train (Algorithm 1, line 1), disjoint.

    D_train is every reviewed shown answer. D_pareto (n_pareto = min(PARETO_SIZE, share x |D_train|))
    scores every candidate admitted to the pool, builds the Pareto front and picks the result;
    D_feedback is the rest, the only source of reflection minibatches. Stratified by your verdict
    and by whether you gave feedback (a No or a comment), so both sets hold passes, fails and
    corrections in the same proportions. Deterministic (a stable hash of each row's id).
    The test set (blind checks) is outside D_train altogether."""
    if len(rows) < 2:
        return list(rows), list(rows)          # nothing to split yet: GEPA needs both sets non-empty
    size = max(1, min(n_pareto, round(len(rows) * share)))
    groups = {}
    for r in sorted(rows, key=lambda r: zlib.crc32(r["id"].encode())):
        groups.setdefault((bool(r["passed"]), r.get("flip") or r.get("note")), []).append(r)
    pareto = []
    for group in groups.values():          # proportional share of each stratum, at least one where possible
        pareto += group[:max(1, round(len(group) * size / len(rows)))] if len(group) > 1 else []
    pareto = pareto[:size] or [rows[0]]
    chosen = {r["id"] for r in pareto}
    feedback = [r for r in rows if r["id"] not in chosen]
    return feedback, pareto


def _grade(row, said, check):
    """GEPA's metric and its feedback (the ASI) for one reviewed output: 1 when the rubric's verdict
    is the user's and, under a comment on a kept verdict, the assessment meets the comment."""
    gold = "PASS" if row["passed"] else "FAIL"
    reason = row.get("reason") or ""
    if said.split(" ")[0] == gold and row.get("note"):
        if check(row["output"], said, reason):
            return 1.0, f"Correct: the expert also says {gold}, and the comment is met."
        return 0.0, (f"Right verdict but not for the expert's reason: the rubric said {said}; "
                     f"the expert agrees it is {gold} and comments: {reason}.")
    if said.split(" ")[0] == gold:
        return 1.0, f"Correct: the expert also says {gold}."
    return 0.0, f"Wrong: the rubric said {said}; the expert says {gold}" + (f", because: {reason}" if reason else "") + "."


def run_gepa(current, rows, payload, make_judge, make_item_judge, judge_model, check, max_edits=MAX_EDITS, before=None,
             progress=None, split=None, run_dir=None, stop=None, timing=None):
    """GEPA (the gepa library, called directly) on the QCR text: one component, the numbered
    statements. D_train (all reviewed shown answers) is split into D_feedback (trainset: reflection
    minibatches, admission to the pool) and D_pareto (valset: scores of every pool member, the
    Pareto front, the result); the test set (blind checks) stays out. The metric is agreement with
    the user's confirmed verdict and, under a comment on a kept verdict, an assessment the comment
    is satisfied with; the feedback is the user's comment (the ASI). Our adapter gives the rewriter
    exactly that per output: input, output, the rubric's verdict, the expert's, and the feedback.
    Returns the best rewrite's text."""
    import gepa
    from concurrent.futures import ThreadPoolExecutor
    from ..durable import JobCancelled, LeaseLost
    from ..services import _endpoint_for

    d_feedback, d_pareto = split or split_train(rows)       # D_train = rows
    logger = _GepaLogger(payload["task_id"], run_dir)
    if timing is not None:
        timing["logger"] = logger
    before = before or {}
    comp = f"minime-{payload['task_id'][:8]}"               # the component's name (kept, so older checkpoints resume)
    # Algorithm 1: the seed is scored on D_pareto, then each admitted proposal costs b + |D_pareto|.
    budget = min(_config.LEARN_BUDGET, len(d_pareto) + _config.PROPOSALS * (_config.MINIBATCH + len(d_pareto)))

    def assess(text, row):
        """What a candidate rubric says about one reviewed output."""
        if progress:
            progress(budget)
        try:
            candidate, _ = from_text(text, current, max_edits)
        except RubricRejected:
            return "UNUSABLE RUBRIC"
        if row["id"] in before:                              # judge only the statements that changed
            (verdict, failed), = _judging._reuse_verdicts(current, candidate, [row], before, make_judge, make_item_judge).values()
        else:
            (verdict, failed), = _judging._verdicts(candidate, [{"id": "x", "input": row["input"], "output": row["output"]}],
                                           make_judge, make_item_judge).values()
        return _judging.assessment(verdict, failed)

    def one(text, row):
        from django.db import connections
        try:
            said = assess(text, row)
        except (JobCancelled, LeaseLost):
            raise
        except Exception as exc:  # noqa: BLE001 - one output the judge could not score counts as wrong, the search goes on
            said = f"JUDGE ERROR ({type(exc).__name__})"
        finally:
            connections.close_all()                          # this thread's connections only
        score, feedback = _grade(row, said, check)
        return said, score, feedback

    class Adapter:
        propose_new_texts = None

        def evaluate(self, batch, candidate, capture_traces=False):
            text = candidate[comp]
            with ThreadPoolExecutor(max_workers=max(1, min(_config.GEPA_EVAL_PARALLEL, len(batch)))) as pool:
                got = list(pool.map(lambda row: one(text, row), batch))
            return gepa.EvaluationBatch(
                outputs=[said for said, _, _ in got], scores=[score for _, score, _ in got],
                trajectories=[{"row": row, "said": said, "feedback": fb} for row, (said, _, fb) in zip(batch, got)]
                if capture_traces else None)

        def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
            records = [{"Input": t["row"]["input"], "Output": t["row"]["output"], "Rubric's verdict": t["said"],
                        "Expert's verdict": "PASS" if t["row"]["passed"] else "FAIL", "Feedback": t["feedback"]}
                       for t in eval_batch.trajectories or []]
            return {name: records for name in components_to_update}

    ep = _endpoint_for(payload)
    rewriter = payload.get("rewriter_model") or _config.REWRITER_MODEL

    def reflection_lm(prompt):          # same guarded call: a cut-off proposal never becomes a candidate
        return rewriter_call(ep, rewriter, prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}])

    template = (REFLECTION_TEMPLATE_BOLD if max_edits is None else
                REFLECTION_TEMPLATE.replace(f"At most {MAX_EDITS} edits", f"At most {max_edits} edits"))
    kwargs = dict(seed_candidate={comp: to_text(current)}, trainset=d_feedback, valset=d_pareto, adapter=Adapter(),
                  reflection_lm=reflection_lm, reflection_prompt_template=template, max_metric_calls=budget,
                  reflection_minibatch_size=_config.MINIBATCH, candidate_selection_strategy="pareto", logger=logger,
                  display_progress_bar=False, use_mlflow=False)
    if run_dir:                     # checkpoint after every iteration; resume from it when present
        kwargs["run_dir"] = run_dir
    if stop:                        # the user pressed Stop: end after this iteration, checkpoint kept
        kwargs["stop_callbacks"] = [lambda state: stop()]
    result = gepa.optimize(**kwargs)
    logger.close_round()
    best = result.best_candidate
    return best[comp] if isinstance(best, dict) else best


def _example(row, limit=400):
    """A reviewed answer as shown under a learning result: what it was and what the user said."""
    clip = lambda t: t if len(t) <= limit else t[:limit].rstrip() + "…"
    return {"id": row["id"], "input": clip(row["input"]), "output": clip(row["output"]),
            "verdict": "PASS" if row["passed"] else "FAIL", "reason": clip(row["reason"] or "")}


def _describe(edit):
    return {"add": lambda e: f"added “{e['after']}”",
            "sharpen": lambda e: f"reworded “{e['before']}” → “{e['after']}”",
            "retire": lambda e: f"dropped “{e['before']}”"}[edit["op"]](edit)


def gepa_log(task_id, message):
    """One GEPA line for the terminal, marked so it can be grepped: [gepa <task id, 8 chars>] ..."""
    import sys
    tag = str(task_id).replace("-", "")[:8]
    for line in str(message).splitlines() or [""]:
        print(f"[gepa {tag}] {line}", file=sys.stderr, flush=True)


class _GepaLogger:
    """GEPA's own progress lines (Iteration n: ...), marked like gepa_log, and kept in the run's
    run_log.txt as before. Replaces GEPA's default logger, which redirects the whole process's
    stdout while it runs."""

    def __init__(self, task_id, run_dir=None):
        self.task_id, self.path = task_id, (str(Path(run_dir) / "run_log.txt") if run_dir else None)
        self.round, self.since = None, time.monotonic()
        self.rounds = []          # [{round, wall, judge_busy, rewriter, prompt_kb}] for the run summary

    def close_round(self):
        """One timing line for the round that just ended: its judge time, its rewriter call (timed
        exactly, with the prompt's size) and the rest."""
        if self.round is None:
            return
        w = _judging.CLOCK.window(self.since, time.monotonic())
        self.rounds.append({"round": self.round, "wall": w["wall"], "judge_busy": w["busy"], "rewriter": w["rewriter"],
                            "prompt_kb": round(w["prompt_chars"] / 1000, 1)})
        self.round = None
        gepa_log(self.task_id, f"⏱ round {self.rounds[-1]['round']}: {_judging._mins(w['wall'])} · {_judging._judge_line(w)}"
                               + (f" · rewriter {_judging._mins(w['rewriter'])} ({w['prompt_chars'] / 1000:.1f} KB prompt)"
                                  if w["rewrites"] else "")
                               + f" · other {_judging._mins(max(0.0, w['wall'] - w['busy'] - w['rewriter']))}")

    def log(self, message):
        import re
        m = re.match(r"Iteration (\d+):", str(message))
        if m and int(m.group(1)) != self.round:
            since = self.since if self.round is None and not self.rounds else time.monotonic()
            self.close_round()
            self.round, self.since = int(m.group(1)), since
        gepa_log(self.task_id, message)
        if self.path:
            try:
                with open(self.path, "a", encoding="utf-8") as out:
                    out.write(f"{message}\n")
            except OSError:
                pass


def _gepa_dir(run_id):
    return str(_judging.data_dir() / "minime_gepa" / run_id)


def _gepa_run(task, current, rows) -> dict:
    """The task's GEPA run: resumed when one is under way on the current rubric, else a new one.

    A run keeps its own D_pareto (frozen ids, in order: GEPA's saved scores are per D_pareto
    instance) and a checkpoint directory GEPA saves to after every iteration. Restarts, crashes
    and newly reviewed outputs therefore never send it back to 0%: on resume it continues from
    its last iteration, with D_feedback grown by everything reviewed meanwhile. Only a change of
    the rules (your edit or undo) starts a fresh run."""
    run = (task.settings or {}).get("gepa_run")
    if run and run.get("based_on") == current.version and _same_epoch(task, run) and Path(run.get("run_dir", "")).exists():
        return {**run, "resumed": True}
    if run:
        shutil.rmtree(run.get("run_dir", ""), ignore_errors=True)
    _, d_pareto = split_train(rows)
    run_id = f"{str(task.id)[:8]}-v{current.version}-{uuid.uuid4().hex[:8]}"
    run = {"id": run_id, "based_on": current.version, "pareto_ids": [r["id"] for r in d_pareto],
           "run_dir": _gepa_dir(run_id), "started": timezone.now().isoformat(), "judge_since": _judge_epoch_of(task)}
    _store._update_settings(task, lambda s: {**s, "gepa_run": run})
    return run


def _judge_epoch_of(task):
    """The judge epoch (settings['judge_since']) a GEPA run or saved rewrite was made in. Its scores
    come from that judge, so they are reused only in the same epoch. Records from before this key
    existed carry none and count as the current epoch (a judge change now drops them: _judge_epoch)."""
    return (task.settings or {}).get("judge_since")


def _epoch_now(payload):
    """The task's judge epoch as stored now (None if the task is gone)."""
    try:
        return _judge_epoch_of(MiniMeTask.objects.get(pk=payload["task_id"]))
    except MiniMeTask.DoesNotExist:
        return None


def _same_epoch(task, record) -> bool:
    return record.get("judge_since", _judge_epoch_of(task)) == _judge_epoch_of(task)


def _end_gepa_run(task, mine=None):
    """GEPA returned (its result is saved as the pending rewrite): drop the checkpoint. mine: the run
    this job searched in; its folder goes too, even if the task no longer names it (a judge change
    retired it while the search ran)."""
    run = (task.settings or {}).get("gepa_run") or {}
    for d in {run.get("run_dir"), (mine or {}).get("run_dir")} - {None, ""}:
        shutil.rmtree(d, ignore_errors=True)
    _store._update_settings(task, lambda s: {k: v for k, v in s.items() if k != "gepa_run"})


def _retired(task, run) -> bool:
    """The task no longer names this GEPA run as its checkpoint (a judge change dropped it)."""
    task.refresh_from_db()
    return ((task.settings or {}).get("gepa_run") or {}).get("id") != run.get("id")


def learn_every(task) -> int:
    """New outputs with feedback that start the next run: one per comment during the cold start
    (direct rewrites are a single quick call), GEPA_EVERY once GEPA runs, so rewrites and feedback
    co-evolve instead of every comment spawning a rewrite."""
    if learning_phase(_store.confirmed(task)) == "gepa":
        return _config.GEPA_EVERY
    return _scoring.learning_mode(task)["every"]


def learning_phase(rows) -> str:
    """'gepa' once D_train can be split usefully: enough answers for D_pareto to rank candidates
    (both verdicts present) and for D_feedback to hold minibatches with feedback; 'direct' before."""
    passes = sum(1 for r in rows if r["passed"])
    feedback = sum(1 for r in rows if r.get("flip") or r.get("note"))
    if (len(rows) >= _config.GEPA_MIN_REVIEWS and passes >= _config.GEPA_MIN_EACH and len(rows) - passes >= _config.GEPA_MIN_EACH
            and feedback >= _config.GEPA_MIN_FEEDBACK):
        return "gepa"
    return "direct"


def rewriter_call(ep, model, messages) -> str:
    """One rewriter reply, complete or nothing. A reply cut off at the output cap (finish_reason
    "length"), or empty, is asked for once more with twice the cap (a model that reasons first can
    spend the cap before it writes); if that one is cut off or empty too, "" is returned: truncated
    text must never become rules (a cut-off "2. Do not" would install a half requirement, and the
    statements after the cut would be retired)."""
    import sys
    chars = sum(len(str(m.get("content") or "")) for m in messages)
    for cap in (_config.REWRITER_MAX_TOKENS, 2 * _config.REWRITER_MAX_TOKENS):
        started = time.monotonic()
        try:
            reply = ep.client.chat.completions.create(model=model, max_tokens=cap, timeout=_config.REWRITER_TIMEOUT,
                                                      messages=messages)
        finally:
            _judging.CLOCK.rewrite(started, time.monotonic(), chars)
        choice = reply.choices[0]
        content, finish = choice.message.content or "", getattr(choice, "finish_reason", None)
        if content.strip() and finish != "length":
            return content
        retry = cap == _config.REWRITER_MAX_TOKENS
        print(f"[minime] rewriter reply {'cut off' if finish == 'length' else 'empty'} ({model}, max_tokens {cap}, "
              f"{len(content)} chars, finish: {finish}); " + ("asking again with a larger cap" if retry else "giving up: no rewrite"),
              file=sys.stderr, flush=True)
    return ""


def reflect_once(payload, prompt) -> str:
    """One call to the rewriter (the direct reflection of the cold start, and the coach)."""
    from ..services import _endpoint_for
    return rewriter_call(_endpoint_for(payload), payload.get("rewriter_model") or _config.REWRITER_MODEL,
                         [{"role": "user", "content": prompt}])


def direct_rewrite(current, rows, before, payload, max_edits) -> str:
    """Cold start: GEPA's reflection step once, on the answers that matter: those with your
    feedback, and those the current rubric gets wrong. Same template as GEPA's, so the rewrite
    reads the same signal; the full check on all reviewed answers still decides."""
    clip = lambda t, n=600: t if len(t) <= n else t[:n].rstrip() + "…"
    wrong = [r for r in rows if before.get(r["id"], (None,))[0] != r["passed"]]
    said = [r for r in rows if (r.get("flip") or r.get("note")) and r not in wrong]
    shown = (wrong + said)[:_config.DIRECT_EXAMPLES]
    side = "\n\n".join(
        f"# Example {n}\nAnswer:\n{clip(r['output'])}\nThe rubric said: {_judging.assessment(*before.get(r['id'], (None, [])))}\n"
        f"The expert says: {'PASS' if r['passed'] else 'FAIL'}" + (f", because: {clip(r['reason'], 400)}" if r["reason"] else "")
        for n, r in enumerate(shown, 1))
    template = (REFLECTION_TEMPLATE_BOLD if max_edits is None else
                REFLECTION_TEMPLATE.replace(f"At most {MAX_EDITS} edits", f"At most {max_edits} edits"))
    return reflect_once(payload, template.replace("<curr_param>", to_text(current)).replace("<side_info>", side))


def _clear_pending(task):
    _store._update_settings(task, lambda s: {k: v for k, v in s.items() if k != "pending_rewrite"})


def _gepa_progress(job):
    """Shown while GEPA searches: how many candidate checks so far, and how long a judge call takes."""
    seen = {"n": 0}

    def tick(total=None):
        seen["n"] += 1
        if seen["n"] % 5 == 1:
            done = f"{min(99, round(100 * seen['n'] / total))}%" if total else f"{seen['n']} checks"
            job.stage = f"GEPA {done}" + _judging._judge_speed()
    return tick


def test_pick(task, test, make_judge, make_item_judge, payload, job=None) -> dict:
    """After a run applied a rewrite: which reviewed outputs it fixes and which it now gets wrong,
    against the rules it replaced. Shown on its learning card (the run no longer waits for this)."""
    versions = {v.version: v for v in task.versions.filter(version__in=[test["version"], test["based_on"]])}
    if len(versions) < 2:
        return {}
    old, new = (QCR.from_dict(versions[k].qcr) for k in (test["based_on"], test["version"]))
    old.version, new.version = test["based_on"], test["version"]
    rows = _store.confirmed(task)
    check = _judging.note_checker(payload)
    progress = _jobs._stage(job, "Checking the new rules") if job else None
    was_v = _judging._current_verdicts(task, old, rows, make_judge, make_item_judge, progress)
    now_v = _judging._current_verdicts(task, new, rows, make_judge, make_item_judge, progress)
    by_id = {r["id"]: r for r in rows}
    was = {i: _judging._right(by_id[i], *was_v[i], check) for i in by_id}
    now = {i: _judging._right(by_id[i], *now_v[i], check) for i in by_id}
    fixed = sorted((i for i in by_id if not was[i] and now[i]), key=int)
    broke = sorted((i for i in by_id if was[i] and not now[i]), key=int)
    examples = lambda ids: [_example(by_id[i]) for i in ids]

    def fill(settings):
        for e in settings.get("log") or []:
            if e.get("kind") == "learn" and e.get("version") == test["version"] and e.get("checking"):
                e.update(checking=False, fixed=examples(fixed), broke=examples(broke), reviewed=len(rows),
                         text=e.get("text", "").rstrip(".") + f" (fixes {len(fixed)}" + (f", gets {len(broke)} wrong" if broke else "") + ").")
        return settings
    _store._update_settings(task, fill)
    from django.db import transaction
    with transaction.atomic():
        v = task.versions.select_for_update().get(version=test["version"])
        v.report = {**(v.report or {}), "fixed": fixed, "broke": broke}
        v.save(update_fields=["report"])
    gepa_log(task.id, f"v{test['version']} on your reviews: fixes {len(fixed)}, gets {len(broke)} wrong")
    return {"fixed": fixed, "broke": broke}


def run_learn(job, payload):
    _scoring._ensure_judge_epoch(payload)
    if payload.get("score_only"):                 # after a run: score the versions (the chart), then check its pick
        _scoring._score_versions(job, payload)
        if payload.get("test"):
            try:
                task = MiniMeTask.objects.get(pk=payload["task_id"], owner=payload["_owner"])
                full = _judging._payload(task, payload)
                _, make_judge, make_item_judge, _ = _judging.llm(full)
                got = test_pick(task, payload["test"], make_judge, make_item_judge, full, job)
                if got:
                    from types import SimpleNamespace
                    _coach(SimpleNamespace(result={"accepted": True, "edits": [1], **got}), payload)
            except MiniMeTask.DoesNotExist:
                return _jobs._finish(job, {"skipped": "the task was deleted"})
            except Exception as exc:  # noqa: BLE001 - best effort: the card stays "checking"
                _jobs._stopped(exc)
                traceback.print_exc()
        _jobs._finish(job, {"scored": True})
        return _jobs._after_job(payload)
    job.learn_mark = timezone.now().isoformat()   # feedback this run sees: marked as used only once it finished,
    epoch = _epoch_now(payload)                   # so a run cut short (restart, error) is simply redone; and not
    _learn(job, payload)                          # when the judge changed meanwhile (it learned with the old one)
    mark = job.learn_mark
    if job.status == "done" and not payload.get("rejudge_only") and _epoch_now(payload) == epoch:
        try:
            # Checked again under the settings lock: a judge change in between keeps the feedback unconsumed.
            _store._update_settings(MiniMeTask.objects.get(pk=payload["task_id"]),
                                    lambda s: {**s, "learned_through": mark} if s.get("judge_since") == epoch else s)
        except MiniMeTask.DoesNotExist:
            pass
        if not (job.result or {}).get("test"):     # an applied run gets its tips after the check
            _coach(job, payload)
    if job.status == "done" and _jobs._after_job(payload):
        return                      # Ready reached: paused, nothing more starts
    if job.status == "done":       # score the versions and check the pick in a job of its own, queued
        try:                       # before the next run so the chart does not wait for that run
            _jobs.ensure_scored(MiniMeTask.objects.get(pk=payload["task_id"]), payload["_owner"], force=True,
                          test=(job.result or {}).get("test"))
        except MiniMeTask.DoesNotExist:
            pass
        except Exception:  # noqa: BLE001 - best effort; the page asks again
            traceback.print_exc()
    _chain(job, payload)


def _coach_facts(task, rows, result):
    """What the coach reads: the run's outcome, agreement by kind, the rules disputed most, comment habits."""
    q = _scoring.quality(task)["detail"]
    facts = [f"{len(rows)} outputs reviewed with MiniMe's verdict shown."]
    if result.get("accepted"):
        facts.append(f"The last rewrite was applied: it fixed {len(result.get('fixed') or [])} and broke "
                     f"{len(result.get('broke') or [])} reviewed outputs.")
    elif result.get("edits"):
        facts.append(f"The last rewrite was not applied: it fixed {len(result.get('fixed') or [])} and broke "
                     f"{len(result.get('broke') or [])} reviewed outputs.")
    for name, r in (("everyday outputs", q["representative"]), ("outputs built to break a rule", q["adversarial"])):
        if r["of"]:
            facts.append(f"Hidden-verdict checks on {name}: the judge agreed with you on {r['hit']} of {r['of']}.")
    disputed = {}
    for r in rows:
        if r.get("flip"):
            for f in r.get("judged_failed") or []:
                disputed[f["text"]] = disputed.get(f["text"], 0) + 1
    if disputed:
        top = sorted(disputed.items(), key=lambda kv: -kv[1])[:3]
        facts.append("Rules the person most often overruled: " + "; ".join(f"\"{t}\" ({n}x)" for t, n in top) + ".")
    flips = [r for r in rows if r.get("flip")]
    bare = sum(1 for r in flips if not (r.get("reason") or "").strip())
    if flips:
        facts.append(f"{bare} of {len(flips)} disagreements came without a comment.")
    return "\n".join(f"- {f}" for f in facts)


def coach_reply(reply):
    """(tips, rules) from the coach's reply: at most 2 tips ("- ..."), and for each the rule it suggests
    adding ("RULE: ..." on the line after it), or None. ([], []) for NONE."""
    text = str(reply or "").strip()
    if text.upper().startswith("NONE"):
        return [], []
    tips, rules = [], []
    for line in (l.strip() for l in text.splitlines()):
        rule = re.match(r"(?:[-*]\s*)?\**rule\s*:\s*\**\s*(.+)$", line, re.I)
        if rule and tips and rules[-1] is None:
            text = rule.group(1).strip()
            whole = re.fullmatch(r'"([^"]*)"|“([^“”]*)”', text)     # quotes around the whole rule, not a quoted term in it
            rules[-1] = ((next(g for g in whole.groups() if g is not None) if whole else text).strip()[:300]) or None
        elif line.startswith("- "):
            tips.append(line[2:].strip())
            rules.append(None)
    return tips[:2], rules[:2]


def _coach(job, payload):
    """After a learning run: up to 2 concrete tips for the person, from the rewriter, logged and shown.
    Best effort: a failure here never affects the run."""
    try:
        task = MiniMeTask.objects.get(pk=payload["task_id"], owner=payload["_owner"])
        result = job.result or {}
        rows = _store.confirmed(task)
        traded = result.get("edits") and len(result.get("broke") or []) >= len(result.get("fixed") or [])
        if len(rows) < _config.COACH_MIN_REVIEWS and not traded:
            return
        given = [t for e in (task.settings or {}).get("log") or [] if e.get("kind") == "tip" for t in e.get("tips") or []][:6]
        recent = sorted((r for r in rows if r.get("flip") or r.get("note") or r.get("reason")), key=lambda r: int(r["id"]))[-12:]
        comments = "\n".join(f"- {'PASS' if r['passed'] else 'FAIL'}" + (f": {r['reason'][:200]}" if r.get("reason") else " (no comment)")
                              for r in recent) or "- (none)"
        current = _store.current_qcr(task)
        reply = reflect_once(_judging._payload(task, payload), _config.COACH_PROMPT.format(
            task=task_context(current.task), rules=to_text(current) or "(none yet)", facts=_coach_facts(task, rows, result),
            comments=comments, given="\n".join(f"- {t}" for t in given) or "- (none)"))
        tips, rules = coach_reply(reply)
        if tips:
            _store.log(task, "MiniMe suggests: " + " ".join(tips), kind="tip", tips=tips,
                       **({"rules": rules} if any(rules) else {}))
    except Exception:  # noqa: BLE001
        traceback.print_exc()


def _learn(job, payload):
    try:
        task = MiniMeTask.objects.get(pk=payload["task_id"], owner=payload["_owner"])
        raw = payload
        payload = _judging._payload(task, payload)
        complete, make_judge, make_item_judge, judge_model = _judging.llm(payload)
        if payload.get("rejudge_only"):
            # Catch up before finishing: judge again until the waiting answers are current, each pass with
            # the judge as the settings name it now (a judge change mid-pass stops that pass).
            done = 0
            for _ in range(5):
                task.refresh_from_db()
                qcr = _store.current_qcr(task)
                if not _judging.stale_waiting(task, qcr.version):
                    break
                fresh = _judging._payload(task, raw)
                _, judge, item_judge, _ = _judging.llm(fresh)
                failures = []
                done += _judging.rejudge_waiting(task, qcr, judge, item_judge, job, _judging.configured_at(fresh), failures)
                if failures:      # those outputs stay waiting to be judged; another pass would only fail again
                    _store._update_settings(task, lambda s: {**s, "unjudged_since": timezone.now().isoformat()})
                    _store.log(task, f"MiniMe could not judge {len(failures)} waiting outputs "
                               f"({_judging.failure_text(failures)}); they are judged again later.", kind="routine")
                    break
            return _jobs._finish(job, {"rejudged": done})
        rows = _store.confirmed(task)
        flips = _store.feedback_since_learn(task)
        if flips == 0:
            raise _config.MiniMeError("nothing to learn: no feedback since the last run")
        check = _judging.note_checker(payload)
        mode = _scoring.learning_mode(task)
        current = _store.current_qcr(task)
        epoch = _judge_epoch_of(task)          # the judge epoch this run learns in: its results are that judge's
        _judging.CLOCK.reset()
        marks, timing = [("start", time.monotonic())], {}
        before = _judging._current_verdicts(task, current, rows, make_judge, make_item_judge,
                                   _jobs._stage(job, "Reading reviews"))
        marks.append(("reading reviews", time.monotonic()))
        phase = learning_phase(rows)
        pending = (task.settings or {}).get("pending_rewrite")
        if pending and pending.get("based_on") == current.version and _same_epoch(task, pending):
            # GEPA already ran on this rubric (the run that found this was cut off before applying it):
            # go straight to the check, and count only the feedback that GEPA saw.
            text = pending["text"]
            job.learn_mark = pending["mark"]
            job.stage = "Reusing saved rewrite"
        elif phase == "direct":
            job.stage = "Direct rewrite"
            text = direct_rewrite(current, rows, before, payload, mode["max_edits"])
            marks.append(("direct rewrite", time.monotonic()))
            mark = getattr(job, "learn_mark", None) or timezone.now().isoformat()
            _store._update_settings(task, lambda s: {**s, "pending_rewrite": {"text": text, "based_on": current.version,
                                                                       "mark": mark, "at": timezone.now().isoformat(),
                                                                       "judge_since": epoch}})
        else:
            run = _gepa_run(task, current, rows)
            by_id = {r["id"]: r for r in rows}
            d_pareto = [by_id[i] for i in run["pareto_ids"] if i in by_id]
            frozen = set(run["pareto_ids"])
            d_feedback = [r for r in rows if r["id"] not in frozen] or d_pareto
            job.stage = "GEPA resuming" if run.get("resumed") else "GEPA 0%"
            gepa_log(task.id, f"{'resume' if run.get('resumed') else 'start'} on v{current.version}: "
                              f"D_feedback {len(d_feedback)}, D_pareto {len(d_pareto)}, "
                              f"{_store.feedback_since_learn(task)} new feedback, judge {judge_model}, "
                              f"rewriter {payload.get('rewriter_model') or _config.REWRITER_MODEL}")
            from ..durable import JobCancelled
            searcher = str(getattr(job, "id", "") or "")
            if searcher:   # a judge change stops this job only (not the one that starts the new epoch)
                _store._update_settings(task, lambda s: {**s, "gepa_run": {**s["gepa_run"], "job": searcher}}
                                        if (s.get("gepa_run") or {}).get("id") == run["id"] else s)
            try:
                text = run_gepa(current, rows, payload, make_judge, make_item_judge, judge_model, check, mode["max_edits"],
                                before, _gepa_progress(job), split=(d_feedback, d_pareto), run_dir=run["run_dir"],
                                stop=lambda: _jobs._cancel_requested(job), timing=timing)
            except JobCancelled:            # stopped mid-call (a progress write): a retired checkpoint goes too
                if _retired(task, run):
                    shutil.rmtree(run["run_dir"], ignore_errors=True)
                    gepa_log(task.id, "stopped: the judge changed, so its checkpoint was dropped; the next run starts fresh")
                raise
            marks.append(("GEPA search", time.monotonic()))
            if _jobs._cancel_requested(job):     # stopped mid-search: keep the checkpoint for the next run
                if _retired(task, run):         # unless a judge change retired it: its scores are the old judge's
                    shutil.rmtree(run["run_dir"], ignore_errors=True)
                    gepa_log(task.id, "stopped: the judge changed, so its checkpoint was dropped; the next run starts fresh")
                    raise JobCancelled("Stopped: the judge changed")
                gepa_log(task.id, "stopped by the user: checkpoint kept, Start optimizing resumes it")
                raise JobCancelled("Stopped by the user")
            gepa_log(task.id, "search done: " + ("the rules are unchanged" if text.strip() == to_text(current).strip()
                                                 else "a rewrite was picked (best on D_pareto)"))
            mark = getattr(job, "learn_mark", None) or timezone.now().isoformat()
            # Saved first, then the checkpoint is retired: cut off between the two, either the result
            # or the resumable search survives (never neither).
            _store._update_settings(task, lambda s: {**s, "pending_rewrite": {"text": text, "based_on": current.version,
                                                                       "mark": mark, "at": timezone.now().isoformat(),
                                                                       "judge_since": epoch}})
            _end_gepa_run(task, run)
        # Every outcome records its timing (⏱ lines and the job result), not only an applied rewrite.
        timed = lambda: _run_timing(task.id, marks, timing.get("logger"))
        try:
            candidate, edits = from_text(text, current, mode["max_edits"])
        except RubricRejected as why:
            _clear_pending(task)
            _store.log(task, f"Learning found no usable change ({why}).")
            # Keep diagnostics useful without copying private prompt/model text
            # into process logs or diagnostic job-result fields.
            gepa_log(task.id, f"rewrite unusable ({why}); {len(text)} chars")
            return _jobs._finish(job, {"accepted": False, "why": str(why), "timing": timed()})
        if not edits:
            _clear_pending(task)
            _store.log(task, "Learning found no change that helps yet.")
            return _jobs._finish(job, {"accepted": False, "why": "no edits", "timing": timed()})
        # Applied as GEPA picked it (best on D_pareto, as in the paper): no second gate. fixed/broke over
        # all reviewed outputs is shown with it; the blind checks score it next, and Undo takes it back.
        # Created only if the rules are still the ones it started from, checked in the same transaction:
        # you edited them while this ran, yours win.
        candidate.evaluator_model = judge_model
        v = _store._new_version(task, candidate, "learn", {"edits": edits, "rows": len(rows)}, base=current.version,
                                judge_since=epoch)
        if v is None:
            _clear_pending(task)
            judge_changed = _epoch_now(payload) != epoch
            _store.log(task, "The judge changed while learning ran; its rewrite was set aside and is searched again."
                       if judge_changed else "A rewrite finished after you changed the rules; it was set aside.", kind="routine")
            return _jobs._finish(job, {"accepted": False, "timing": timed(),
                                       "why": "the judge changed meanwhile" if judge_changed else "the rules changed meanwhile"})
        _clear_pending(task)                  # only once the new version exists: a crash before keeps the rewrite
        if phase != "direct":
            gepa_log(task.id, f"applied as v{v.version}: {len(edits)} edit(s); checking it on your reviews next")
        _store.log(task, f"Learned: {'; '.join(_describe(e) for e in edits)}.",
            kind="learn", phase=phase, accepted=True, version=v.version, edits=edits, checking=True,
            reviewed=len(rows), bold=not mode["settled"],
            rewriter=payload["rewriter_model"], judge=judge_model, based_on=current.version)
        _judging.rejudge_waiting(task, _store.current_qcr(task), make_judge, make_item_judge, job,
                                 _judging.configured_at(payload))
        marks.append(("re-judge waiting", time.monotonic()))
        times = _run_timing(task.id, marks, timing.get("logger"))
        _jobs._finish(job, {"accepted": True, "version": v.version, "edits": edits, "timing": times,
                      "test": {"version": v.version, "based_on": current.version}})
    except MiniMeTask.DoesNotExist:
        _jobs._finish(job, {"skipped": "the task was deleted"})     # deleted while this job ran: nothing to do
    except Exception as exc:  # noqa: BLE001
        _jobs._stopped(exc)
        traceback.print_exc()
        _jobs._finish(job, error=f"{type(exc).__name__}: {exc}")


def _run_timing(task_id, marks, logger=None) -> dict:
    """The run's timing summary: one [gepa] line per phase and one for the run, and the same as a
    dict (kept in the job's result, so runs can be compared later)."""
    phases = [(name, b - a) for (_, a), (name, b) in zip(marks, marks[1:])]
    t0, t1 = marks[0][1], marks[-1][1]
    whole = _judging.CLOCK.window(t0, t1)
    rounds = list(getattr(logger, "rounds", []) or [])
    rewriter = whole["rewriter"]
    for name, (a, b) in ((n, (x, y)) for (_, x), (n, y) in zip(marks, marks[1:])):
        gepa_log(task_id, f"⏱ {name}: {_judging._mins(b - a)} · {_judging._judge_line(_judging.CLOCK.window(a, b))}")
    wall = t1 - t0 or 1e-9
    share = lambda x: f"{100 * x / wall:.0f}%"
    other = max(0.0, wall - whole["busy"] - rewriter)
    gepa_log(task_id, f"⏱ run: {_judging._mins(wall)} · judge busy {share(whole['busy'])} ({whole['calls']} calls, "
                      f"avg {whole['avg']:.0f} s, {whole['hits']} cached) · rewriter {share(rewriter)} "
                      f"({whole['rewrites']} calls, {whole['prompt_chars'] / 1000:.0f} KB prompts) · other {share(other)}"
                      + (f" · {len(rounds)} GEPA rounds" if rounds else ""))
    return {"wall": round(wall, 1), "phases": {n: round(d, 1) for n, d in phases},
            "judge": {"busy": round(whole["busy"], 1), "calls": whole["calls"], "cached": whole["hits"],
                      "avg": round(whole["avg"], 1), "longest": round(whole["longest"], 1)},
            "rewriter": round(rewriter, 1),
            "rounds": [{k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()} for r in rounds]}


def _chain(job, payload):
    """Always reflecting: feedback that arrived while this run worked starts the next run at once,
    without waiting for the user's next click. Not after an error (the page offers a retry)."""
    if job.status != "done" or payload.get("rejudge_only") or payload.get("score_only"):
        return
    try:
        task = MiniMeTask.objects.get(pk=payload["task_id"], owner=payload["_owner"])
        if _jobs.optimizing(task) and _store.feedback_since_learn(task) >= learn_every(task):
            _jobs.start("minime_learn", payload["_owner"], {"task_id": str(task.id)})
    except MiniMeTask.DoesNotExist:
        pass
    except Exception:  # noqa: BLE001
        traceback.print_exc()


__minime_names__ = ('_GepaLogger', '_chain', '_clear_pending', '_coach', '_coach_facts', '_describe', '_end_gepa_run', '_epoch_now', '_example', '_gepa_dir', '_gepa_progress', '_gepa_run', '_grade', '_judge_epoch_of', '_learn', '_records', '_retired', '_run_timing', '_same_epoch', 'coach_reply', 'direct_rewrite', 'gepa_log', 'learn_every', 'learning_phase', 'reflect_once', 'rewriter_call', 'run_gepa', 'run_learn', 'split_train', 'test_pick',)
