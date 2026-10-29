"""Run a QCR over labelled rows and measure it against the user's verdicts.

FAIL is the positive class: fail catch = recall on true fails, power = recall on
true passes, FDP = share of predicted fails that are true passes. Rows the judge
could not score (no usable verdict) are counted and left out of the metrics.
"""
from ..rubric_checks import RubricEvaluationError, combined_scorer


def judge_rows(qcr, rows, make_judge, make_item_judge=None):
    """[{id, predicted, statements: {id: bool|None}, criteria}] in row order.

    predicted is the AND of the active statements (None when nothing failed but a
    verdict is missing). With no active statement, nothing can be violated: PASS."""
    entries, owners = qcr.compile()
    results = []
    for row in rows:
        if not entries:
            results.append({'id': row['id'], 'predicted': True, 'statements': {}, 'criteria': []})
            continue
        evidence = []
        try:
            combined_scorer(entries, make_judge, evidence.append, make_item_judge)(inputs=row['input'], outputs=row['output'])
        except RubricEvaluationError:
            pass   # the evidence already marks the affected criteria as evaluation errors
        last = evidence[-1]
        verdicts = {}
        for owner, detail in zip(owners, last['criteria']):
            prior, value = verdicts.get(owner, True), detail['passed']
            verdicts[owner] = False if False in (prior, value) else None if None in (prior, value) else True
        results.append({'id': row['id'], 'predicted': last['overall_pass'], 'statements': verdicts,
                        'criteria': last['criteria']})
    return results


def _ratio(num, den):
    return num / den if den else None


def metrics(rows, results):
    truth = {r['id']: r['passed'] for r in rows}
    scored = [(truth[r['id']], r['predicted']) for r in results if r['predicted'] is not None]
    tp = sum(t and p for t, p in scored)              # pass kept
    tf = sum(not t and not p for t, p in scored)      # fail caught
    fp = sum(not t and p for t, p in scored)          # fail missed
    ff = sum(t and not p for t, p in scored)          # pass wrongly failed
    n = len(scored)
    agree = _ratio(tp + tf, n)
    kappa = None
    if n:
        chance = ((tp + ff) * (tp + fp) + (tf + fp) * (tf + ff)) / (n * n)
        kappa = (agree - chance) / (1 - chance) if chance < 1 else None
    return {'n': n, 'unscored': len(results) - n, 'agreement': agree,
            'power': _ratio(tp, tp + ff), 'fail_catch': _ratio(tf, tf + fp),
            'fdp': _ratio(ff, ff + tf), 'kappa': kappa,
            'confusion': {'pass_kept': tp, 'fail_caught': tf, 'fail_missed': fp, 'pass_failed': ff}}


def flip_rates(runs):
    """{statement id: share of rows whose verdict differs across repeated runs}.
    Missing verdicts are ignored; a row with fewer than two verdicts is not counted."""
    rates = {}
    ids = {sid for run in runs for r in run for sid in r['statements']}
    for sid in sorted(ids):
        counted = flipped = 0
        for per_row in zip(*runs):
            seen = {r['statements'].get(sid) for r in per_row} - {None}
            if sum(r['statements'].get(sid) is not None for r in per_row) >= 2:
                counted += 1
                flipped += len(seen) > 1
        rates[sid] = _ratio(flipped, counted)
    return rates


def predictions(results):
    """{row id: predicted verdict}: the compact per-row record a later version is compared against."""
    return {r['id']: r['predicted'] for r in results}


def regressions(rows, previous, results):
    """Row ids the previous version got right and this one gets wrong, against today's labels
    (a row relabelled since is judged by its new label for both). Unscored rows are not counted."""
    truth = {r['id']: r['passed'] for r in rows}
    now = predictions(results)
    return sorted((i for i, before in (previous or {}).items()
                   if i in truth and before is not None and now.get(i) is not None
                   and before == truth[i] and now[i] != truth[i]), key=str)


def evaluate(qcr, rows, make_judge, make_item_judge=None, repeats=1):
    """{metrics, flip_rates, results}. Metrics come from the first run; repeats >= 2
    re-runs every row to measure each statement's flip rate (near-determinism)."""
    runs = [judge_rows(qcr, rows, make_judge, make_item_judge) for _ in range(max(1, repeats))]
    return {'metrics': metrics(rows, runs[0]), 'flip_rates': flip_rates(runs) if len(runs) > 1 else {},
            'results': runs[0]}
