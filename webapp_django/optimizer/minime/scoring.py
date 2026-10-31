"""How good the judge is: blind-check scores per version, the chart's data, and the Ready checks."""
from __future__ import annotations
import traceback
from django.utils import timezone
from ps_utils.minime.evaluate import metrics
from ps_utils.minime.qcr import QCR
from ps_utils.minime.rubric_text import MAX_EDITS
from ..models import MiniMeTask
from . import config as _config
from . import jobs as _jobs
from . import judging as _judging
from . import store as _store


def quality(task) -> dict:
    """How often the judge agrees with the user on blind checks (its verdict was hidden, so the user
    was not anchored by it): the CURRENT rules on every blind check, as the GEPA paper scores a
    prompt on the whole test set. Two kinds:

      representative  outputs like the deployed ones, or written to pass: a proxy for field
                      performance (the headline); the good outputs test that it does not over-fail
      adversarial     outputs built to break a rule: tougher, a measure of robustness

    Each is {pct, hit, of}; pct is None below QUALITY_MIN checks. action: one next step."""
    now = blind_scores(task)[-1]
    field, robust = now["all_rep"], now["all_adv"]
    if field["pct"] is None or robust["pct"] is None:
        action = "Not enough hidden-verdict checks of both kinds yet. Keep reviewing."
    elif field["pct"] < _config.READY_AGREE["representative"]:
        action = "It disagrees with you on everyday outputs. When it is wrong, disagree and say exactly why."
    elif robust["pct"] < _config.READY_AGREE["adversarial"]:
        action = "It slips on tougher outputs. When it misses one, disagree and name the rule it missed."
    else:
        action = "On track on both everyday and tougher outputs."
    return {"pct": field["pct"], "agreed": field["hit"], "n": field["of"], "min": _config.QUALITY_MIN, "version": now["v"],
            "detail": {"representative": field, "adversarial": robust, "action": action}}


def _blind_labels(task):
    # Every reviewed blind check, judged when it was shown or not (an uploaded sheet, or a check whose
    # judging failed): score_blind gives the unjudged ones their verdicts in full.
    return list(task.labels.filter(split="heldout").exclude(passed=None)
                .exclude(labelled_at=None).order_by("labelled_at"))


def _epochs(task) -> list:
    """The versions grouped by rules: one list per rule change, in order (rewording the task
    description keeps the rules, so that version joins the group before it)."""
    epochs = []
    for v in task.versions.order_by("version"):
        if epochs and "description" in (v.report or {}):
            epochs[-1].append(v)
        else:
            epochs.append([v])
    return epochs


def _epoch_verdicts(members, labels, since=None) -> dict:
    """{label id: verdict} this group's rules gave each blind check: scored for it (report["blind"],
    see score_blind), or the verdict shown if the check was reviewed while it was in force (and
    judged since the judge last changed: `since`)."""
    ids, scored = {v.version for v in members}, {}
    for v in members:
        scored.update((v.report or {}).get("blind") or {})
    out = {}
    for l in labels:
        key = str(l.pk)
        shown = l.judged_pass if l.judged_version in ids and (since is None or _store.judged_at(l) >= since) else None
        verdict = scored[key] if key in scored else shown
        if verdict is not None:
            out[key] = verdict
    return out


def _judge_since(task):
    """When the judge last changed (its verdicts from before are not reused for scoring)."""
    return _store.judge_since(task)


def _judge_epoch(task):
    """Start a new judge epoch: the versions' blind scores were made with another judge, so they are
    dropped and re-scored (the chart then compares every version with the same judge)."""
    import shutil
    now = timezone.now()
    # A GEPA checkpoint and a saved rewrite hold the old judge's scores: dropped, so the next run
    # searches again with this judge instead of mixing the two. A search still running in that
    # checkpoint (the job the run names as searching in it) is asked to stop instead: deleting its
    # folder under it fails its next checkpoint write, and it removes the folder itself when it ends.
    # Any other job, such as a learn job starting up and opening this epoch, is left alone.
    # The checkpoint is retired and the search cancelled in one transaction: a worker that sees the
    # cancellation also sees its run retired, so it drops the folder instead of keeping it as paused.
    from django.db import transaction
    run = (task.settings or {}).get("gepa_run") or {}
    searching = [j for kind, j in _jobs.active_job_rows(task)
                 if kind == "minime_learn" and j.status == "running" and str(j.id) == str(run.get("job") or "")
                 ] if run.get("run_dir") else []
    judge = _store.effective_models(task)["judge_model"]     # the judge this epoch is for (see _ensure_judge_epoch)
    with transaction.atomic():
        _store._update_settings(task, lambda s: {k: v for k, v in {**s, "judge_since": now.isoformat(), "judge_epoch_model": judge}.items()
                                                 if k not in ("judge_reasoning", "judge_reasoning_since", "no_reasoning_models",
                                                              "gepa_run", "pending_rewrite")})
        if searching:
            from ..durable import cancel
            for j in searching:
                cancel(j.id, task.owner)
    if searching:
        _store.log(task, "The rewrite in progress was stopped: it was scoring rules with the previous judge.", kind="routine")
    elif run.get("run_dir"):
        shutil.rmtree(run["run_dir"], ignore_errors=True)
    for v in task.versions.all():
        if (v.report or {}).get("blind"):
            v.report = {k: x for k, x in v.report.items() if k != "blind"}
            v.save(update_fields=["report"])


def blind_scores(task) -> list:
    """Each version of the rules scored on the blind checks, one entry per rule change, in order:
    [{v, at, versions, type, changes, rep, adv, all_rep, all_adv}].

      rep / adv          on the SAME blind checks for every version (those every version has a
                         verdict for), so versions compare like for like, as in the GEPA paper's
                         test set; score_blind fills older versions in as new checks arrive
      all_rep / all_adv  on every blind check this version has a verdict for (the current one:
                         the headline, including checks reviewed since the last scoring)

    Each is {pct, hit, of}; pct is None below QUALITY_MIN."""
    labels = _blind_labels(task)
    epochs = _epochs(task)
    since = _judge_since(task)
    verdicts = [_epoch_verdicts(e, labels, since) for e in epochs]
    common = set.intersection(*(set(v) for v in verdicts)) if verdicts else set()

    def rate(hits):
        return {"pct": round(100 * sum(hits) / len(hits)) if len(hits) >= _config.QUALITY_MIN else None, "hit": sum(hits), "of": len(hits)}

    def score(got, keep):
        hits = {"rep": [], "adv": []}
        for l in labels:
            if str(l.pk) in got and str(l.pk) in keep:
                hits["adv" if l.kind == "violate" else "rep"].append(got[str(l.pk)] == l.passed)
        return rate(hits["rep"]), rate(hits["adv"])
    out, prev = [], None
    for e, got in zip(epochs, verdicts):
        rep, adv = score(got, common)
        all_rep, all_adv = score(got, got)
        out.append({"v": e[0].version, "at": e[0].created_at, "versions": [v.version for v in e],
                    "rep": rep, "adv": adv, "all_rep": all_rep, "all_adv": all_adv,
                    "scored": len(got) == len(labels),          # a verdict on every blind check so far
                    "type": _change_type(e[0]), "changes": _rule_diff(prev, e[0].qcr)})
        prev = e[0].qcr
    if not out:
        empty = rate([])
        out = [{"v": 0, "at": None, "versions": [], "rep": empty, "adv": empty, "all_rep": empty, "all_adv": empty,
                "type": "seed", "changes": []}]
    return out


def _change_type(v):
    r = v.report or {}
    return "seed" if v.version == 0 else "undo" if r.get("undo_of") else "edit" if v.source == "edit" else "learned"


def _rule_diff(before, after) -> list:
    """[{op: add | sharpen | retire, before, after}] between two versions' active statements, by id."""
    if before is None:
        return []
    act = lambda q: {s["id"]: s["text"] for s in (q or {}).get("statements") or [] if s.get("status", "active") == "active"}
    old, new = act(before), act(after)
    return ([{"op": "add", "before": "", "after": t} for i, t in new.items() if i not in old]
            + [{"op": "sharpen", "before": old[i], "after": t} for i, t in new.items() if i in old and old[i] != t]
            + [{"op": "retire", "before": t, "after": ""} for i, t in old.items() if i not in new])


def score_blind(task, make_judge, make_item_judge, job=None) -> int:
    """Give every version of the rules a verdict on every blind check reviewed so far, newest
    version first, and keep them on the version (report["blind"]), so all versions are compared on
    the same test set. A check judged with a version already has its verdict; statements a version
    shares with the one a check was shown with keep their result, so only the statements that
    differ are judged. Saved per version, so an interrupted run keeps what it did. Returns the
    number of verdicts added."""
    from django.db import transaction
    labels = _blind_labels(task)
    since = _judge_since(task)
    ids = {str(l.pk) for l in labels}
    # Verdicts written before the judge last changed are not reused (_current_verdicts judges those
    # rows again in full), and neither are missing ones.
    rows = [r for r in _store.confirmed(task, split="heldout") if r["id"] in ids]
    added = 0
    for members in reversed(_epochs(task)):
        head = members[0]
        known = _epoch_verdicts(members, labels, since)
        todo = [r for r in rows if r["id"] not in known]
        if not todo:
            continue
        qcr = QCR.from_dict(head.qcr)
        qcr.version = head.version
        verdicts = _judging._current_verdicts(task, qcr, todo, make_judge, make_item_judge,
                                     _jobs._stage(job, f"Scoring blind checks v{head.version}") if job else None)
        with transaction.atomic():
            # The judge changed while these were judged (its epoch reset dropped the scores): do not
            # write the old judge's verdicts back. Checked in the same transaction as the write.
            if _judge_since(MiniMeTask.objects.get(pk=task.pk)) != since:
                return added
            v = task.versions.select_for_update().get(pk=head.pk)
            v.report = {**(v.report or {}), "blind": {**((v.report or {}).get("blind") or {}),
                                                      **{i: verdict for i, (verdict, _) in verdicts.items()}}}
            v.save(update_fields=["report"])
        added += len(todo)
    return added


def _rule_changes(task):
    """[(version, created_at, type)] of the versions that changed the rules (rewording the task
    description does not): type is learned | edit | undo."""
    out = []
    for v in task.versions.exclude(version=0).order_by("version"):
        r = v.report or {}
        if "description" in r:
            continue
        out.append((v.version, v.created_at, "undo" if r.get("undo_of") else "edit" if v.source == "edit" else "learned"))
    return out


def quality_series(task) -> dict:
    """Blind-check agreement over time, for the chart: after each blind check, the rolling
    agreement on the last QUALITY_WINDOW of each kind (rep = representative, adv = adversarial),
    placed at n = outputs reviewed so far (shown and blind). marks: each rule change, at the
    number of outputs reviewed before it."""
    from bisect import bisect_left, bisect_right
    times = sorted(task.labels.exclude(passed=None).exclude(labelled_at=None).values_list("labelled_at", flat=True))
    blind = (task.labels.filter(split="heldout").exclude(passed=None).exclude(judged_pass=None)
             .exclude(labelled_at=None).order_by("labelled_at"))
    seen, points = {"rep": [], "adv": []}, []
    for l in blind:
        kind = "adv" if l.kind == "violate" else "rep"
        seen[kind].append(l.passed == l.judged_pass)
        pt = {"n": bisect_right(times, l.labelled_at), "kind": kind}
        for k, rows in seen.items():
            w = rows[-_config.QUALITY_WINDOW:]
            pt[k] = round(100 * sum(w) / len(w)) if len(w) >= _config.QUALITY_MIN else None
            pt[k + "_hit"], pt[k + "_of"] = sum(w), len(w)
        points.append(pt)
    marks = [{"v": v, "n": bisect_left(times, at), "type": t} for v, at, t in _rule_changes(task)]
    scores = blind_scores(task)
    # Same test set for every version once all are scored; until then (scoring catches up in the
    # background) each version on the checks it has, flagged so the chart says so.
    to_go = sum(not e["scored"] for e in scores)
    pick = (lambda e, k: e[k]) if not to_go else (lambda e, k: e["all_" + k])
    versions = [{"v": e["v"], "n": bisect_left(times, e["at"]) if e["at"] and e["v"] else 0,
                 "rep": pick(e, "rep"), "adv": pick(e, "adv"), "type": e["type"], "changes": e["changes"]} for e in scores]
    return {"points": points, "marks": marks, "versions": versions, "to_score": to_go, "reviewed": len(times), "window": _config.QUALITY_WINDOW,
            "targets": {"rep": _config.READY_AGREE["representative"], "adv": _config.READY_AGREE["adversarial"]}}


def _stability(task):
    """(settled, flips, shown, rules_changed): settled = at most READY_FLIPS Nos in the last
    READY_STABLE shown verdicts and no rule change meanwhile. Until then MiniMe learns boldly."""
    shown = list(task.labels.filter(split="learn").exclude(passed=None).exclude(action="")
                 .order_by("-labelled_at")[:_config.READY_STABLE])
    flips = sum(l.action == "flip" for l in shown)
    changed = [at for _, at, _ in _rule_changes(task)]
    since = shown[-1].labelled_at if len(shown) == _config.READY_STABLE else None
    rules_changed = since is not None and any(c >= since for c in changed)
    return since is not None and flips <= _config.READY_FLIPS and not rules_changed, flips, shown, rules_changed


def learning_mode(task) -> dict:
    """Bold while the rules are new, careful once settled: how often to learn and how big a change
    may be. GEPA's pick is applied either way (no gate after the run)."""
    settled = _stability(task)[0]
    return {"settled": settled, "every": _config.LEARN_EVERY if settled else _config.LEARN_EVERY_EARLY,
            "max_edits": MAX_EDITS if settled else _config.MAX_EDITS_EARLY}


def readiness(task, q=None) -> dict:
    """Is the judge good enough to copy and use? Advice, not a gate: the user decides when to stop.

      targets  performance: rolling blind agreement per kind vs its target (the chart's lines)
      checks   consistency, tri-state: status met (✓) | not (✗, decided: not yet) | open (○, too
               early to tell); short is the one-line reading, detail the hover text

    ready = every check met and both lines at or above target (on READY_BLIND+ checks each)."""
    q = q or quality(task)
    d = q["detail"]
    targets = {}
    for k in ("representative", "adversarial"):
        r = d[k]
        ok = r["of"] >= _config.READY_BLIND and r["pct"] is not None and r["pct"] >= _config.READY_AGREE[k]
        status = "met" if ok else "open" if r["of"] < _config.READY_BLIND or r["pct"] is None else "not"
        pct = "—" if r["pct"] is None else f"{r['pct']}%"
        short = f"{pct} · {r['of']}/{_config.READY_BLIND}" if status == "open" else f"{pct} · {r['hit']}/{r['of']}"
        targets[k] = {"pct": r["pct"], "hit": r["hit"], "of": r["of"], "target": _config.READY_AGREE[k], "need": _config.READY_BLIND, "ok": ok,
                      "status": status, "short": short, "label": f"{k.capitalize()} ≥{_config.READY_AGREE[k]}%",
                      "detail": f"Agreement with you on {k} blind checks, current rules: {r['hit']} of {r['of']}. "
                                f"Needs {_config.READY_AGREE[k]}% or more on at least {_config.READY_BLIND}."}
    blind = task.labels.filter(split="heldout").exclude(passed=None)
    passed, failed = blind.filter(passed=True).count(), blind.filter(passed=False).count()
    stable, flips, shown, rules_changed = _stability(task)
    changes = _rule_changes(task)
    last = changes[-1] if changes else None
    learn = task.labels.filter(split="learn").exclude(passed=None).exclude(action="")
    since = learn.filter(labelled_at__gte=last[1]).count() if last else learn.count()
    reviewed = list(task.labels.exclude(passed=None).exclude(judged_pass=None))
    caught = {f.get("id") for l in reviewed if l.passed is False for f in (l.judged_failed or [])}
    rules = _store.current_qcr(task).active()
    untested = [s.id for s in rules if s.id not in caught]
    good_through = any(l.passed is True and l.judged_pass is True for l in reviewed)
    plural = lambda n, w: f"{n} {w}{'' if n == 1 else 's'}"

    both = "met" if passed >= _config.READY_EACH and failed >= _config.READY_EACH else "open"
    if len(shown) < _config.READY_STABLE or rules_changed:
        st = "open"
        st_short = f"{min(since, _config.READY_STABLE)}/{_config.READY_STABLE} since v{last[0]}" if last else f"{len(shown)}/{_config.READY_STABLE} reviewed"
    else:
        st = "met" if stable else "not"
        st_short = "" if stable else "still moving"
    rl = "met" if rules and not untested and good_through else "open"
    rl_short = ("no rules yet" if not rules else "" if rl == "met" else
                plural(len(untested), "rule") + " untested" if untested else "no full pass yet")
    checks = [
        {"key": "both", "label": "Tested on good & bad outputs", "status": both,
         "short": f"{passed} · {failed}",
         "detail": f"Blind checks you passed: {passed}, failed: {failed}. Needs {_config.READY_EACH} of each."},
        {"key": "stable", "label": "Rules settled", "status": st, "short": st_short,
         "detail": (f"{plural(flips, 'disagreement')} in your last {len(shown)} shown verdicts. "
                    f"Settled: at most {_config.READY_FLIPS} in {_config.READY_STABLE} in a row, with no rule change meanwhile."
                    + (f" The rules changed at v{last[0]}; {since} shown since." if last and st == "open" else ""))},
        {"key": "rules", "label": "Each rule tested", "status": rl, "short": rl_short,
         "detail": ("No rules yet." if not rules else
                    ("Each rule caught a bad output." if not untested else "Not yet caught a bad output: " + ", ".join(untested) + ".")
                    + (" A good output passed all rules." if good_through else " No good output has passed all rules yet."))},
    ]
    for c in checks:
        c["ok"] = c["status"] == "met"
    items = [c["ok"] for c in checks] + [t["ok"] for t in targets.values()]
    # What is left, in a few words: "1 more blind check of each kind", "rules have not settled", ...
    more = {k: _config.READY_BLIND - t["of"] for k, t in targets.items() if t["of"] < _config.READY_BLIND}
    todo = []
    if len(more) == 2 and len(set(more.values())) == 1:
        n = next(iter(more.values()))
        todo.append(f"{n} more blind check{'' if n == 1 else 's'} of each kind")
    else:
        todo += [f"{n} more {k} blind check{'' if n == 1 else 's'}" for k, n in more.items()]
    todo += [f"{k} below {t['target']}%" for k, t in targets.items() if t["status"] == "not"]
    todo += [c["label"].lower() for c in checks if not c["ok"]]
    return {"ready": all(items), "done": sum(items), "of": len(items), "checks": checks, "targets": targets, "todo": todo}


def metrics_report(task) -> dict:
    """Agreement on the blind slice (held-out). Not shown in the UI yet; kept for assurance."""
    blind = _store.confirmed(task, "heldout")
    results = [{"id": r["id"], "predicted": r["judged_pass"]} for r in blind]
    return metrics(blind, results)


def rule_tests(task) -> dict:
    """{statement id: reviewed bad outputs it caught} (as "Each rule tested" counts them), so a round's
    rule-breaking outputs go to untested rules first."""
    out = {}
    for l in task.labels.exclude(passed=None).exclude(judged_pass=None).filter(passed=False):
        for f in l.judged_failed or []:
            out[f.get("id")] = out.get(f.get("id"), 0) + 1
    return out


def _score_versions(job, payload):
    """After a run or a re-judge: every version on every blind check so far (score_blind), so the
    chart compares versions on the same test set. Best effort: a failure here never fails the job."""
    try:
        task = MiniMeTask.objects.get(pk=payload["task_id"], owner=payload["_owner"])
        _, make_judge, make_item_judge, _ = _judging.llm(_judging._payload(task, payload))
        score_blind(task, make_judge, make_item_judge, job)
    except MiniMeTask.DoesNotExist:
        pass
    except Exception:  # noqa: BLE001
        traceback.print_exc()


def _ensure_judge_epoch(payload):
    """Before a job judges: the task's epoch must be for the judge it will use now.

    Tasks scored before judge epochs existed (or at the since-removed reasoning setting) start one
    (once). A task that uses MiniMe's default judge changes judge when the default does, without
    any setting changing: its epoch records the judge it was for (judge_epoch_model), so the change
    is seen here and handled like one made in Settings (new epoch, waiting answers re-judged).
    Epochs from before that key used the explicit setting, else the default of that time."""
    try:
        task = MiniMeTask.objects.get(pk=payload["task_id"])
    except (MiniMeTask.DoesNotExist, KeyError, ValueError):
        return
    s = task.settings or {}
    if not s.get("judge_since"):
        _judge_epoch(task)
        return
    judge = _store.effective_models(task)["judge_model"]
    was = s.get("judge_epoch_model") or s.get("judge_model") or _config.JUDGE_MODEL_BEFORE
    if was != judge:
        _judge_epoch(task)
        _store.log(task, f"Judge changed to {judge} (MiniMe's default). The versions are scored again on the "
                         f"blind checks with this judge.")
        _jobs.run_rejudge(task, task.owner)
    elif not s.get("judge_epoch_model"):
        _store._update_settings(task, lambda x: {**x, "judge_epoch_model": judge})


__minime_names__ = ('_blind_labels', '_change_type', '_ensure_judge_epoch', '_epoch_verdicts', '_epochs', '_judge_epoch', '_judge_since', '_rule_changes', '_rule_diff', '_score_versions', '_stability', 'blind_scores', 'learning_mode', 'metrics_report', 'quality', 'quality_series', 'readiness', 'rule_tests', 'score_blind',)
