"""The judge: model calls, the persistent judge cache, retries and timing, and verdicts on reviewed outputs."""
from __future__ import annotations
import json
import time
from pathlib import Path
from django.utils import timezone
from ps_utils.minime.evaluate import judge_rows
from ps_utils.minime.qcr import QCR
from ..models import MiniMeLabel
from . import config as _config
from . import store as _store


JUDGE_TIMES = []     # seconds per (uncached) judge call, the last 50 in this worker process


class _Clock:
    """Where a learning run's time goes, for the [gepa] timing lines. Judge calls run on many
    threads at once, so their busy time is the union of the call intervals (time with at least
    one call in flight), not the sum. One job runs per worker process, so one clock per process."""

    def __init__(self):
        import threading
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        with self.lock:
            self.calls, self.hits, self.rewrites = [], [], []

    def rewrite(self, start, end, chars):
        with self.lock:
            self.rewrites.append((start, end, chars))

    def call(self, start, end):
        with self.lock:
            self.calls.append((start, end))

    def hit(self):
        with self.lock:
            self.hits.append(time.monotonic())

    def window(self, t0, t1=None) -> dict:
        """{calls, hits, busy, total, avg, longest, wall} for judge calls that started in [t0, t1)."""
        t1 = time.monotonic() if t1 is None else t1
        with self.lock:
            spans = sorted((a, min(b, t1)) for a, b in self.calls if t0 <= a < t1)
            hits = sum(1 for h in self.hits if t0 <= h < t1)
            rw = [(a, b, c) for a, b, c in self.rewrites if t0 <= a < t1]
        busy, end = 0.0, None
        for a, b in spans:
            if end is None or a > end:
                busy += b - a
                end = b
            elif b > end:
                busy += b - end
                end = b
        total = sum(b - a for a, b in spans)
        return {"calls": len(spans), "hits": hits, "busy": busy, "total": total, "wall": t1 - t0,
                "rewriter": sum(min(b, t1) - a for a, b, _ in rw), "rewrites": len(rw), "prompt_chars": sum(c for *_, c in rw),
                "avg": total / len(spans) if spans else 0.0, "longest": max((b - a for a, b in spans), default=0.0)}


CLOCK = _Clock()


def _mins(seconds) -> str:
    return f"{seconds:.0f} s" if seconds < 90 else f"{seconds / 60:.1f} min"


def _judge_line(w) -> str:
    """'judge 74 s busy (31 calls, avg 18 s, longest 44 s, 12 cached)'"""
    if not w["calls"]:
        return f"judge 0 s ({w['hits']} cached)"
    return (f"judge {_mins(w['busy'])} busy ({w['calls']} calls, avg {w['avg']:.0f} s, longest {w['longest']:.0f} s"
            + (f", {w['hits']} cached" if w["hits"] else "") + ")")


def _judge_retry_line(exc, attempt):
    """One line per retried judge call instead of a traceback: what failed and, when the judge
    answered, how its reply started and why it ended (length = cut off)."""
    import sys
    reply = getattr(exc, "reply", None)
    finish = getattr(exc, "finish", None)
    print(f"[minime] judge call failed (try {attempt + 1}/{_config.JUDGE_RETRIES + 1}), retrying in "
          f"{_config.JUDGE_RETRY_PAUSE * (attempt + 1)} s: {type(exc).__name__}: {exc}"
          + (f" · reply: {reply!r}" if reply is not None else "")
          + (f" · finish: {finish}" if finish else ""), file=sys.stderr, flush=True)


def _judge_speed() -> str:
    """' · judge ~12s per call' from the recent judge calls (empty before the first one)."""
    if not JUDGE_TIMES:
        return ""
    recent = sorted(JUDGE_TIMES[-20:])
    return f" · ~{recent[len(recent) // 2]:.0f}s/call"


def data_dir() -> Path:
    """Where MiniMe keeps its judge cache and GEPA checkpoints: MINIME_DATA_DIR when set (production
    points it at the persistent volume, so a deploy keeps both), else webapp_django/mlruns."""
    import os
    from django.conf import settings as dj
    return Path(os.environ.get("MINIME_DATA_DIR") or Path(dj.BASE_DIR) / "mlruns")


def _judge_cache_path():
    return str(data_dir() / "minime_judge_cache.sqlite")


class _JudgeStore:
    """Judge results kept across jobs and restarts (a small SQLite file of its own, apart from the
    app database): a run that is resumed, or retried, or a later run re-judging the same output
    on the same rule with the same judge model, reuses the result instead of paying for it again.
    Evaluation errors are never stored.

    Judge calls run on several threads at once, and they share this store: every use of the one
    SQLite connection holds the lock. Without it, two threads' queries interleave and one reads
    the other's row (an item verdict list where a rule verdict was expected, or nothing)."""

    def __init__(self, model):
        import threading
        self.model, self.mem, self.db, self.lock = model, {}, None, threading.RLock()

    def _conn(self):
        if self.db is None:
            import sqlite3
            Path(_judge_cache_path()).parent.mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(_judge_cache_path(), timeout=30, check_same_thread=False)
            self.db.execute("pragma journal_mode=wal")
            self.db.execute("create table if not exists judged (key text primary key, result text)")
        return self.db

    def _key(self, key):
        import hashlib
        return hashlib.sha256(json.dumps([self.model, *key], default=str).encode()).hexdigest()

    def __contains__(self, key):
        with self.lock:
            if key in self.mem:
                return True
            try:
                row = self._conn().execute("select result from judged where key=?", (self._key(key),)).fetchone()
                if not row or not isinstance(row[0], str):
                    return False
                self.mem[key] = json.loads(row[0])
            except Exception:  # noqa: BLE001 - the cache is an optimisation, never a failure
                return False
            return True

    def __getitem__(self, key):
        with self.lock:
            return self.mem[key]

    def discard(self, key):
        with self.lock:
            self.mem.pop(key, None)
            try:
                with self._conn():
                    self._conn().execute("delete from judged where key=?", (self._key(key),))
            except Exception:  # noqa: BLE001
                pass

    def __setitem__(self, key, result):
        text = json.dumps(result, default=str)
        with self.lock:
            self.mem[key] = result
            if "evaluation_error" in text:
                return
            try:
                with self._conn():
                    self._conn().execute("insert or replace into judged values (?, ?)", (self._key(key), text))
            except Exception:  # noqa: BLE001
                pass


def _cached(make_judge, make_item_judge, model=""):
    """Judge calls memoized, across jobs and restarts (_JudgeStore). GEPA re-judges the same answers
    with many candidate rubrics that share most statements, so most calls repeat exactly (temperature 0)."""
    store = _JudgeStore(model)

    def timed(fn):
        """One judge call, timed; a transient failure (an empty or garbled reply from the endpoint, a
        timeout, a 5xx) is retried twice with a short pause before it fails the job."""
        for attempt in range(_config.JUDGE_RETRIES + 1):
            started = time.monotonic()
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001
                if attempt == _config.JUDGE_RETRIES:
                    raise                               # the job reports it, with the full traceback
                _judge_retry_line(exc, attempt)
                time.sleep(_config.JUDGE_RETRY_PAUSE * (attempt + 1))
            finally:
                ended = time.monotonic()
                JUDGE_TIMES.append(ended - started)
                del JUDGE_TIMES[:-50]
                CLOCK.call(started, ended)

    def fits(key, result):
        """A result of the right shape for its call: a batch judge gives {criteria: [one per
        statement]}, an item judge [one verdict per item]. Anything else is not reused."""
        if key[0] == "batch":
            return isinstance(result, dict) and isinstance(result.get("criteria"), list) and len(result["criteria"]) == len(key[1])
        return isinstance(result, list) and len(result) == len(key[2])

    def cached(key, run):
        with store.lock:                        # look up and read as one step
            hit = store[key] if key in store else None
        if hit is not None:
            if fits(key, hit):
                CLOCK.hit()
                return hit
            print(f"[minime] judge cache: dropping a {type(hit).__name__} result kept for a {key[0]} call; judging again")
            store.discard(key)
        box = {}
        timed(lambda: run(lambda result: box.update(result=result)))
        if not fits(key, box.get("result")):
            raise RuntimeError(f"judge gave a {type(box.get('result')).__name__} result for a {key[0]} call")
        store[key] = box["result"]
        return box["result"]

    def judge(texts, receive):
        def call(inputs=None, outputs=None):
            key = ("batch", tuple(texts), str(inputs), str(outputs))
            receive(cached(key, lambda got: make_judge(texts, got)(inputs=inputs, outputs=outputs)))
        return call

    def item_judge(requirement, items, receive):
        def call(inputs=None, outputs=None):
            key = ("item", requirement, tuple(items), str(inputs), str(outputs))
            receive(cached(key, lambda got: make_item_judge(requirement, items, got)(inputs=inputs, outputs=outputs)))
        return call
    return judge, (item_judge if make_item_judge else None)


def _judge_client(ep):
    """The endpoint's client, bounded for judge calls."""
    return ep.client.with_options(timeout=_config.JUDGE_TIMEOUT, max_retries=0)


def llm(payload):
    """(complete, make_judge, make_item_judge, judge_model, endpoint, optimizer) for one job."""
    from prompt_optimizer import Mode, OptimizationConfig, Prompt, PromptOptimizer
    from ..services import _endpoint_for
    ep = _endpoint_for(payload)
    # The judge gets its own bounded client: JUDGE_TIMEOUT per attempt and no SDK retries (MiniMe retries
    # itself, JUDGE_RETRIES times). With the endpoint's 300 s and the SDK's 2 retries, one hung call could
    # hold a batch for 45 min.
    import dataclasses
    judge_ep = dataclasses.replace(ep)
    object.__setattr__(judge_ep, "_client", _judge_client(ep))
    opt = PromptOptimizer(endpoint=judge_ep)                 # no tracking: MiniMe does not use MLflow
    judge_model = payload.get("judge_model") or ep.judge_model
    config = OptimizationConfig(name="minime", prompts=[Prompt(role="user", template="x")], mode=Mode.FEW_SHOT,
                                examples=[{"inputs": {}}], quality_criteria=["x"], judge_model=judge_model)
    # No reasoning setting is sent: the judge runs at its model's default, as in Prompt Studio's optimize
    # and benchmark flows (judge_reasoning_effort = None). Speed comes from choosing a fast judge model.
    model = payload.get("task_model") or ep.task_model

    def complete(prompt):
        reply = ep.client.chat.completions.create(model=model, max_tokens=_config.GEN_MAX_TOKENS,
                                                  messages=[{"role": "user", "content": prompt}])
        return reply.choices[0].message.content or ""
    make_judge, make_item_judge = _cached(
        lambda texts, receive: opt._build_guidelines_scorer(config, texts, on_judgment=receive, plain=True),
        lambda requirement, items, receive: opt._build_item_judge(config, requirement, items, receive),
        judge_model)
    return complete, make_judge, make_item_judge, judge_model


def _payload(task, payload):
    """Job payload with the task's saved models filled in, else MiniMe's defaults. _configured_at is
    the judge epoch of the same settings snapshot the judge model is read from (judge_since; verdict-
    writing jobs set it up first): the verdicts of a judge built from this payload are stamped with it
    (judged_at), so any judge change after that snapshot, even one before this call, makes them stale."""
    s = task.settings or {}
    return {**payload, "_configured_at": s.get("judge_since") or timezone.now().isoformat(), "task_model": payload.get("task_model") or s.get("task_model") or _config.TASK_MODEL,
            "judge_model": payload.get("judge_model") or s.get("judge_model") or _config.JUDGE_MODEL,
            "rewriter_model": payload.get("rewriter_model") or s.get("rewriter_model") or _config.REWRITER_MODEL}


def _verdicts(qcr, rows, make_judge, make_item_judge, progress=None, failures=None):
    """{row id: (predicted, [{id, text} failed])}, judging PARALLEL rows at once.
    progress(done, total) before each batch.

    failures: None, and the first row the judge could not answer (after its retries) raises. A list,
    and such a row gets no verdict (None, []) and its (row id, exception) is appended: the others
    are still judged. A row judged without a usable verdict (predicted None: an evaluation error
    judge_rows absorbed) is appended too."""
    from ps_utils.rubric_checks import _run_concurrently
    texts = {s.id: s.text for s in qcr.statements}
    results = {}
    for start in range(0, len(rows), _config.PARALLEL):
        batch = rows[start:start + _config.PARALLEL]
        if progress:
            progress(start, len(rows))
        def judge(row):
            return lambda: results.__setitem__(row["id"], judge_rows(qcr, [row], make_judge, make_item_judge)[0])
        errors = _run_concurrently([judge(row) for row in batch])
        for row, error in zip(batch, errors):
            if error is None:
                continue
            if failures is None:
                raise error
            failures.append((row["id"], error))
            results[row["id"]] = {"predicted": None, "statements": {}}
        if failures is not None:
            failures += [(row["id"], RuntimeError("the judge gave no usable verdict")) for row, error in zip(batch, errors)
                         if error is None and results[row["id"]]["predicted"] is None]
    out = {}
    for row in rows:
        r = results[row["id"]]
        failed = [{"id": sid, "text": texts[sid]} for sid, ok in r["statements"].items() if ok is False]
        out[row["id"]] = (r["predicted"], failed)
    return out


def _reuse_verdicts(current, candidate, rows, before, make_judge, make_item_judge, progress=None):
    """The candidate rubric's verdicts, judging only what changed. A rewrite keeps most statements
    word for word, and their results on each answer are already known (`before`: the current
    rubric's {row id: (verdict, failed)}), so only new or reworded statements are judged.
    Rows with no known verdict are judged in full, and so is a FAIL none of whose failed statements
    the candidate keeps: only failed statements are stored, so the others' results are not known
    (one may have been an evaluation error), and defaulting them to PASS could promote it."""
    old = {s.text for s in current.active()}
    kept = {s.text for s in candidate.active() if s.text in old}
    fresh = [s for s in candidate.active() if s.text not in old]

    def reusable(r):
        verdict, failed = before.get(r["id"], (None, []))
        return verdict is True or (verdict is False and any(f["text"] in kept for f in failed or []))
    known = [r for r in rows if reusable(r)]
    unknown = [r for r in rows if not reusable(r)]
    out = _verdicts(candidate, unknown, make_judge, make_item_judge) if unknown else {}
    new = (_verdicts(QCR(task=candidate.task, statements=fresh, evaluator_model=candidate.evaluator_model),
                     known, make_judge, make_item_judge, progress) if fresh and known else {})
    for r in known:
        failed = [f for f in before[r["id"]][1] if f["text"] in kept]
        verdict, more = new.get(r["id"], (True, []))
        out[r["id"]] = (False if failed or verdict is False else verdict, failed + more)
    return out


def _current_verdicts(task, current, rows, make_judge, make_item_judge, progress=None):
    """The current rubric's verdicts on reviewed rows. Rows were judged with the version of their
    day; statements unchanged since then keep their stored result, so only statements added or
    reworded since are judged (rows without a stored verdict are judged in full)."""
    versions = {v.version: QCR.from_dict(v.qcr) for v in task.versions.all()}
    since = _store.judge_since(task)
    if since:                    # verdicts written before the judge last changed: another judge's
        rows = [r if r.get("judged_at") is None or r["judged_at"] >= since else {**r, "judged_pass": None} for r in rows]
    out, groups = {}, {}
    for r in rows:
        if r["judged_version"] == current.version and r["judged_pass"] is not None:
            out[r["id"]] = (r["judged_pass"], r["judged_failed"])
        else:
            groups.setdefault(r["judged_version"] if r["judged_pass"] is not None else None, []).append(r)
    for version, group in groups.items():
        then = versions.get(version)
        if then is None:
            out.update(_verdicts(current, group, make_judge, make_item_judge, progress))
        else:
            known = {r["id"]: (r["judged_pass"], r["judged_failed"]) for r in group}
            out.update(_reuse_verdicts(then, current, group, known, make_judge, make_item_judge, progress))
    return out


def stale_waiting(task, version) -> list:
    """Waiting outputs whose verdict is not current: judged with other rules, not judged, or judged
    before the judge last changed (another judge's)."""
    since = _store.judge_since(task)
    return [l for l in task.labels.filter(passed=None) if l.judged_version != version or l.judged_pass is None
            or (since and _store.judged_at(l) < since)]


def configured_at(payload):
    """The judge epoch the payload's judge was read in (see _payload); now if it was not."""
    from datetime import datetime
    at = (payload or {}).get("_configured_at")
    return datetime.fromisoformat(at) if at else timezone.now()


def judge_down(batch, failures) -> bool:
    """No row of this batch could be judged: the judge is unavailable, so calling it for the next
    batches would only wait out the same timeouts."""
    return bool(batch) and len({row_id for row_id, _ in failures} & {r["id"] for r in batch}) == len(batch)


def failure_text(failures) -> str:
    """'APITimeoutError: Request timed out.' for the first failure, short enough for a log line."""
    error = failures[0][1]
    return f"{type(error).__name__}: {error}"[:200]


def rejudge_waiting(task, qcr, make_judge, make_item_judge, job=None, configured=None, failures=None) -> int:
    """Waiting outputs judged again with these rules, in review order (the one on screen first), and
    saved batch by batch, so the page shows the new verdict as soon as it exists.

    configured: when this judge was read from the settings (configured_at). Every verdict is stamped
    with it, and the run stops at the first batch after a judge change: a later timestamp would make
    the old judge's verdicts look current. Returns how many were judged.

    An output the judge cannot answer keeps its old state (still waiting to be judged) and is added to
    failures, if given; the rest are judged. A batch with no answer at all means the judge is down:
    the run stops there."""
    configured = configured or timezone.now()
    failures = [] if failures is None else failures
    waiting = stale_waiting(task, qcr.version)
    judged = 0
    for start in range(0, len(waiting), _config.PARALLEL):
        if job:
            job.stage = f"Re-judging {round(100 * start / len(waiting))}%" + _judge_speed()
        since = _store.judge_since(type(task).objects.get(pk=task.pk))
        if since and since > configured:      # the judge changed: this one is obsolete
            break
        batch = waiting[start:start + _config.PARALLEL]
        rows = [{"id": str(l.pk), "input": l.input, "output": l.output} for l in batch]
        before = len(failures)
        verdicts = _verdicts(qcr, rows, make_judge, make_item_judge, failures=failures)
        missed = {row_id for row_id, _ in failures[before:]}
        answered = [l for l in batch if str(l.pk) not in missed]
        for l in answered:
            l.judged_pass, l.judged_failed = verdicts[str(l.pk)]
            l.judged_version, l.judged_at = qcr.version, configured
        MiniMeLabel.objects.bulk_update(answered, ["judged_pass", "judged_failed", "judged_version", "judged_at"])
        judged += len(answered)
        if judge_down(rows, failures[before:]):
            break
    return judged


def assessment(verdict, failed) -> str:
    """The judge's verdict as text: what GEPA sees, and what a comment is checked against."""
    if verdict is None:
        return "NO VERDICT"
    return "PASS" if verdict else "FAIL (breaks: " + "; ".join(f["text"] for f in failed or []) + ")"


def note_checker(payload):
    """check(output, said, comment) -> bool: does an assessment address a reviewer's comment?
    One call to the judge model per distinct question, cached for the job."""
    cache, endpoint = {}, []

    def check(output, said, comment):
        key = (output, said, comment)
        if key not in cache:
            if not endpoint:                    # only when there is a comment to check
                from ..services import _endpoint_for
                endpoint.append(_endpoint_for(payload))
            ep = endpoint[0]
            model = payload.get("judge_model") or ep.judge_model
            reply = ep.client.chat.completions.create(
                model=model, max_tokens=_config.GEN_MAX_TOKENS,
                messages=[{"role": "user", "content": _config.NOTE_CHECK.format(output=output, said=said, comment=comment)}])
            cache[key] = (reply.choices[0].message.content or "").strip().upper().startswith("YES")
        return cache[key]
    return check


def _right(row, verdict, failed, check) -> bool:
    """A reviewed answer handled right: the user's verdict, and, under a comment on a kept verdict,
    an assessment the comment is satisfied with."""
    if verdict != row["passed"]:
        return False
    return not row["note"] or check(row["output"], assessment(verdict, failed), row["reason"])


__minime_names__ = ('CLOCK', 'JUDGE_TIMES', '_Clock', '_JudgeStore', '_cached', '_current_verdicts', '_judge_cache_path', '_judge_client', '_judge_line', '_judge_retry_line', '_judge_speed', '_mins', '_payload', '_reuse_verdicts', '_right', '_verdicts', 'assessment', 'configured_at', 'data_dir', 'failure_text', 'judge_down', 'llm', 'note_checker', 'rejudge_waiting', 'stale_waiting',)
