"""Benchmark rubric scoring: mandatory code checks first, then the LLM judge."""

class RubricEvaluationError(ValueError):
    """Only these allowlisted diagnostics may be exposed in benchmark reports."""
    messages = {
        'invalid_json': 'Judge did not return a JSON object.',
        'invalid_shape': 'Judge must return one boolean and one detail per criterion.',
        'invalid_detail': 'Judge criterion details must be objects.',
        'missing_explanation': 'Judge omitted a criterion explanation.',
        'invalid_evidence': 'Judge evidence must be an exact excerpt of the answer, or null.',
        'invalid_label': 'Judge did not return a boolean match verdict.',
        'invalid_items': 'Judge must return one numbered verdict per item, in item order.',
    }

    def __init__(self, code):
        self.code = code
        super().__init__(self.messages[code])


MAX_CONCURRENT_JUDGE_CALLS = 4


def _run_concurrently(calls):
    """Run independent judge calls at once; return each call's exception (or None), in order.

    The last call runs on the caller's thread; the others on short-lived threads, each with a
    copy of the caller's context (MLflow tracing state) and its own database connection, which
    it closes when done. Every call finishes before this returns, so no call outlives the answer.
    """
    import contextvars
    from concurrent.futures import ThreadPoolExecutor

    def guarded(call):
        try:
            call()
            return None
        except Exception as error:  # noqa: BLE001 - reported per call to the caller
            return error

    if len(calls) < 2:
        return [guarded(call) for call in calls]

    def in_thread(call, context):
        try:
            return context.run(guarded, call)
        finally:
            try:
                from django.db import connections
                connections.close_all()   # this thread's connections only
            except Exception:  # noqa: BLE001 - Django absent or not configured
                pass

    head, last = calls[:-1], calls[-1]
    # The caller runs `last` itself, so the pool gets one slot fewer: at most MAX calls at once.
    with ThreadPoolExecutor(max_workers=max(1, min(len(head), MAX_CONCURRENT_JUDGE_CALLS - 1))) as pool:
        futures = [pool.submit(in_thread, call, contextvars.copy_context()) for call in head]
        last_error = guarded(last)
        return [future.result() for future in futures] + [last_error]


def combined_scorer(compiled, make_judge, on_judgment, make_item_judge=None):
    """Score one answer against a compiled rubric (see rubric_parser.compile_rubric).

    Code checks are mandatory and always run first. A criterion whose code checks
    fail is failed without a judge call. Item-scoped criteria (judge_scope 'item')
    get their own judge call on the keywords code extracted from the items; the
    judge decides each item alone and code applies "each". The judge_text of the
    remaining criteria goes to the judge in one call, in criterion order. These calls
    are independent (each decides different criteria), so they run at once. An
    unusable judge response on one call does not discard the other calls' verdicts;
    any other error (cancellation, lost lease, transport) is raised after all calls end.
    """
    from .rubric_parser import run_check, extract_items, extract_judged_text

    def judge(inputs=None, outputs=None):
        answer = str(outputs or '')
        details, pending = [], []
        for entry in compiled:
            results = [dict(check=check, **dict(zip(('passed', 'explanation', 'observed'), run_check(answer, check))))
                       for check in entry['checks']]
            code_ok = all(r['passed'] for r in results)
            code_note = ' '.join(r['explanation'] for r in results)
            observed = {}
            for r in results:
                observed.update(r['observed'])
            detail = {'criterion': entry['criterion'], 'evidence': None, 'checks': results,
                      'method': 'deterministic' if entry['judge_text'] is None else 'llm_judge',
                      'compiled_method': entry['method']}
            if results:
                detail['observed'] = observed
            if not code_ok or entry['judge_text'] is None:
                detail.update(passed=code_ok, status='pass' if code_ok else 'fail',
                              explanation=code_note, method='deterministic')
            else:
                detail.update(passed=None, status='evaluation_error',
                              explanation='No usable judgment was produced.', code_explanation=code_note)
                pending.append(len(details))
            details.append(detail)

        def publish():
            errors = sum(d['passed'] is None for d in details)
            passed = sum(d['passed'] is True for d in details)
            # All criteria must pass: one failed criterion decides the case even if
            # another verdict is missing. Unknown only when nothing failed yet.
            failed = any(d['passed'] is False for d in details)
            overall = False if failed else None if errors else True
            on_judgment({'criteria': [dict(d) for d in details], 'passed': passed, 'total': len(details),
                         'aggregation': 'all_criteria', 'overall_pass': overall,
                         'evaluation_errors': errors})

        def fold(index, passed, explanation, evidence):
            note = details[index].pop('code_explanation', '')
            status = 'evaluation_error' if passed is None else 'pass' if passed else 'fail'
            details[index].update(passed=passed, status=status, evidence=evidence,
                                  explanation=f"Code: {note} Judge: {explanation}" if note else explanation)

        def item_call(index, items):
            keywords = [extract_judged_text(item, compiled[index].get('item_unit')) for item in items]
            def receive(verdicts):
                rows = [{'item': n + 1, 'text': text, 'rule': rule, 'passed': v['passed'], 'explanation': v['explanation']}
                        for n, ((text, rule), v) in enumerate(zip(keywords, verdicts))]
                details[index]['items'] = rows
                failing = [r for r in rows if not r['passed']]
                listed = '; '.join(f'"{r["text"]}"' for r in rows)
                summary = (f'All {len(rows)} items pass: {listed}.' if not failing else
                           'Failing items: ' + '; '.join(f"{r['item']} (\"{r['text']}\"): {r['explanation']}" for r in failing))
                fold(index, not failing, summary, failing[0]['text'] if failing else None)
            return lambda: make_item_judge(compiled[index]['judge_text'], [t for t, _ in keywords], receive)(
                inputs=inputs, outputs=outputs)

        calls, batch = [], []
        for index in pending:
            items = extract_items(answer)[0] if compiled[index].get('judge_scope') == 'item' else None
            if items and make_item_judge is not None:
                details[index]['judge_scope'] = 'item'
                calls.append(([index], item_call(index, items)))
            else:
                batch.append(index)   # answer scope, or no items to judge one by one
        if batch:
            def receive(result):
                for index, verdict in zip(batch, result['criteria']):
                    fold(index, verdict['passed'], verdict['explanation'], verdict['evidence'])
            calls.insert(0, (batch, lambda: make_judge([compiled[i]['judge_text'] for i in batch], receive)(
                inputs=inputs, outputs=outputs)))

        publish()  # Keep code evidence even if a provider call fails later.
        unusable = None
        try:
            errors = _run_concurrently([call for _, call in calls])
            for (indexes, _), error in zip(calls, errors):
                if isinstance(error, RubricEvaluationError):   # this call's verdicts only
                    for index in indexes:
                        details[index].pop('code_explanation', None)
                        details[index].update(explanation=str(error), error_code=error.code)
                    unusable = unusable or error
            fatal = next((e for e in errors if e is not None and not isinstance(e, RubricEvaluationError)), None)
            if fatal is not None:
                raise fatal
        finally:
            publish()
        if unusable is not None:
            raise unusable
        return sum(d['passed'] is True for d in details) / len(details)
    return judge
