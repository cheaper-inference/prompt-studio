"""MiniMe labels: (input, output, verdict, reason) rows, all free text.

Verdict and reason are mandatory on every row; a file with any row missing either
is rejected as a whole, listing every bad line, so no partial label set is learned from.
"""
import csv
import io

_VERDICTS = {'pass': True, 'true': True, 'fail': False, 'false': False}


class LabelError(ValueError):
    pass


def parse_verdict(value):
    """'Pass'/'PASS'/'true' -> True, 'Fail'/'false' -> False, anything else -> None."""
    return _VERDICTS.get(str(value or '').strip().lower())


def _delimiter(header):
    counts = {d: header.count(d) for d in (',', ';', '\t')}
    return max(counts, key=counts.get)


def load_labels(path, input_col='input', output_col='output', id_col='id'):
    with open(path, newline='', encoding='utf-8-sig') as f:
        return parse_labels(f.read(), input_col, output_col, id_col, source=path)


def parse_labels(text, input_col='input', output_col='output', id_col='id', source='labels'):
    """[{id, input, output, passed, reason}] in file order.

    Column names are configurable (the keyword sheets use 'service' and 'answer').
    The delimiter (',', ';' or tab) is taken from the header line. A missing id
    column numbers rows from 1."""
    text = text.lstrip('\ufeff')
    reader = csv.DictReader(io.StringIO(text), delimiter=_delimiter(text.split('\n', 1)[0]))
    missing = [c for c in (input_col, output_col, 'verdict', 'reason') if c not in (reader.fieldnames or [])]
    if missing:
        raise LabelError(f'{source}: missing column(s): {", ".join(missing)}')
    rows, problems = [], []
    for n, raw in enumerate(reader, start=1):
        passed = parse_verdict(raw['verdict'])
        reason = (raw['reason'] or '').strip()
        issues = ([] if passed is not None else [f'verdict {raw["verdict"]!r} is not Pass/Fail']) + \
                 ([] if reason else ['reason is empty'])
        row_id = (raw.get(id_col) or '').strip() or str(n)
        if issues:
            problems.append(f'row {row_id}: ' + '; '.join(issues))
            continue
        rows.append({'id': row_id, 'input': raw[input_col] or '', 'output': raw[output_col] or '',
                     'passed': passed, 'reason': reason})
    if problems:
        raise LabelError(f'{source}: {len(problems)} invalid row(s):\n' + '\n'.join(problems))
    return rows
