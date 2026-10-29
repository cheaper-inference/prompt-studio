"""Draft benchmark quality criteria (QCR) from the baseline prompt-set: v2, deterministic.

The baseline prompt is the user's intent; the benchmark spec should follow from it.
Each sentence of the baseline is compiled with the rubric parser (the same code that
grades answers). Every objective requirement it finds (counts, lengths, quoted terms,
JSON, uniqueness, list format) becomes one canonical criterion, rendered so it compiles
back to the same check, and cites the exact prompt sentence it came from. A stated semantic
requirement ("Each keyword must be relevant to the healthcare domain") is drafted
verbatim, so it compiles to the same judge criterion; context and goal sentences
("Your goal is to ...") are not requirements and are skipped, as are sentences naming a
{{variable}}, which a judge cannot resolve per case. Drafts are proposals: the user
confirms or edits them before benchmarking. Only the baseline is read, never an
optimized prompt, so an optimizer cannot redefine success.
"""
import re

from .rubric_parser import compile_rubric, group_criteria

VERSION = 'spec-draft-v2'

# A judged sentence becomes a criterion only when it states a requirement. Task verbs
# ("Write 3 taglines", "Suggest 3 keywords") are not: they contribute only their code checks.
_DIRECTIVE = re.compile(
    r"\b(?:must|should|shall|needs? to|ha(?:s|ve) to|ensure|make sure|avoid|never|always|do not|don't|"
    r"cannot|can't|may not|required?)\b|^(?:[-*•]\s*)?(?:keep|use|focus|prefer|prioriti[sz]e|target|include|"
    r"exclude|stay|limit)\b", re.I)
_TEMPLATED = re.compile(r'\{\{[^{}]+\}\}')

_SINGULAR = {'entries': 'entry', 'bullet points': 'bullet point', 'key phrases': 'key phrase'}


def _singular(unit):
    unit = unit.lower()
    return _SINGULAR.get(unit, unit[:-1] if unit.endswith('s') and not unit.endswith('ss') else unit)


def _plural(unit):
    unit = _singular(unit)
    return {'entry': 'entries'}.get(unit, unit + 's')


def _sentences(text):
    """[(sentence, span)]: instruction units with the verbatim prompt text each came from.
    Nested lists are rejoined ("Rules:" + "- At most 5 words") so sub-lines keep their
    scope; such a unit cites its own line, not the lead-in."""
    lines = [line.strip() for line in (text or '').splitlines() if line.strip()]
    units = []
    for grouped, source in group_criteria(lines):
        own = lines[source[-1]]
        for sentence in re.split(r'(?<=[.!?])\s+(?=[A-Z0-9*_"“\'(-])', grouped):
            sentence = sentence.strip()
            if sentence:
                units.append((sentence, sentence if sentence in text else own))
    return units


def _resolve(sentence, span):
    """(text, span, compiled): a nested rule ("Follow these rules:" + "- Keep it short")
    is stated by its own line when that line alone grades the same, so the lead-in
    neither clutters the criterion nor turns a code-only rule into a judged one."""
    entry = compile_rubric([sentence])[0]
    own = re.sub(r'^(?:[-*•]|\d+[.)])\s+', '', span).strip()
    if own != sentence:
        alone = compile_rubric([own])[0]
        same_checks = len(alone['checks']) == len(entry['checks']) and all(
            _same(a, b) for a, b in zip(alone['checks'], entry['checks']))
        # A judged line must keep the lead-in's scope ("Each keyword must:" + "- be relevant").
        if same_checks and (alone['method'] == 'deterministic' or alone['judge_scope'] == entry['judge_scope']):
            return own, span, alone
    return sentence, span, entry


def _bound(check):
    low, high = check['min'], check['max']
    if low is not None and low == high:
        return f'exactly {low}'
    if low is not None and high is not None:
        return f'between {low} and {high}'
    return f'at most {high}' if high is not None else f'at least {low}'


def render(check, unit):
    """Canonical criterion text for one check; compile_rubric([text]) yields the same check."""
    unit = _singular(unit or 'item')
    kind = check['kind']
    if kind == 'item_count':
        if check['min'] == check['max']:
            return f"{check['min']} {_plural(unit) if check['min'] != 1 else unit} in total."
        return f"{_bound(check).capitalize()} {_plural(unit)}."
    if kind == 'length':
        size = f"{_bound(check)} {check['unit']}"
        return (f'Each {unit} must contain {size}.' if check['scope'] == 'item'
                else f'The answer must contain {size}.')
    if kind == 'contains':
        verb = 'must not include' if check['negate'] else 'must include'
        who = f'Each {unit}' if check['scope'] == 'item' else 'The answer'
        return f'{who} {verb} "{check["term"]}".'
    if kind == 'json':
        return 'Return valid JSON.' if check['shape'] == 'any' else f"Return a JSON {check['shape']}."
    if kind == 'unique_items':
        return f'No duplicate {_plural(unit)}.'
    if kind == 'only_items':
        style = {'numbered': 'numbered ', 'bulleted': 'bulleted '}.get(check['markers'], '')
        return f'Output only the {style}{_plural(unit)}, nothing else.' if style else \
            f'Output only the {_plural(unit)}, one per line.'
    if kind == 'list_style':
        return f"Present the {_plural(unit)} as a {check['markers']} list."
    raise ValueError(f'Unknown check kind {kind!r}')


def _same(a, b):
    keys = set(a) | set(b)
    return all(a.get(k) == b.get(k) for k in keys if k != 'unit')


def draft_criteria(messages):
    """[{criterion, source_span, source_role, kind, check, origin}] in prompt order.

    kind: 'code' (one rendered check), 'judge' (a stated semantic requirement, verbatim)
    or 'hybrid' (verbatim; its code checks plus a judged remainder).

    messages: [{role, content}] of the baseline prompt-set (system first, then user).
    One criterion per distinct requirement; when several sentences state the same check,
    the first is cited. Every source_span is verbatim prompt text."""
    units = [(message.get('role'), *_resolve(sentence, span))
             for message in messages or [] for sentence, span in _sentences(message.get('content') or '')]
    # A rule without its own item noun ("Do not include any explanations") speaks about
    # the prompt's items: the first noun the prompt names ("3 keywords").
    default_unit = next((entry['item_unit'] for *_, entry in units if entry['item_unit']), None)
    # A stated semantic requirement is drafted verbatim. A hybrid one ("at most 5 words and
    # relevant to healthcare") compiles to its code checks plus the judged remainder, so
    # those checks are not drafted again on their own. A sentence that is not a
    # requirement ("Suggest 3 keywords for {{service}}") contributes only its code checks.
    verbatim = [unit for unit in units if unit[3]['method'] in ('llm_judge', 'hybrid')
                and _DIRECTIVE.search(unit[1]) and not _TEMPLATED.search(unit[1])]
    covered = [check for *_, entry in verbatim for check in entry['checks']]
    drafts, seen = [], set()
    for unit in units:
        role, text, span, entry = unit
        if unit in verbatim:
            if text.lower() not in seen:
                seen.add(text.lower())
                drafts.append({'criterion': text, 'source_span': span, 'source_role': role,
                               'kind': 'judge' if entry['method'] == 'llm_judge' else 'hybrid', 'check': None,
                               'origin': 'stated', '_unit': None})
            continue
        for check in entry['checks']:
            if any(_same(check, c) for c in covered) or any(d['check'] and _same(check, d['check']) for d in drafts):
                continue
            rendered = render(check, entry['item_unit'] or default_unit)
            if not any(_same(check, c) for c in compile_rubric([rendered])[0]['checks']):
                continue   # never propose a criterion that would grade differently
            drafts.append({'criterion': rendered, 'source_span': span, 'source_role': role,
                           'kind': 'code', 'check': check, 'origin': 'stated', '_unit': entry['item_unit'] or default_unit})
    # "Do not include any explanations" + "Present them as a numbered list": the exclusivity
    # rule must not demand bare lines when the prompt asks for a list style, or the two
    # drafted criteria would contradict each other on every compliant answer.
    style = next((d['check']['markers'] for d in drafts if d['kind'] == 'code' and d['check']['kind'] == 'list_style'), None)
    for d in drafts:
        if (style and d['kind'] == 'code' and d['check']['kind'] == 'only_items' and d['check']['markers'] == 'none'
                and not re.search(r'\b(?:no|without) (?:numbering|numbers|bullets|markers)\b|\bplain\b|\bone per line\b', d['source_span'], re.I)):
            d['check'] = {**d['check'], 'markers': style}
            d['criterion'] = render(d['check'], d['_unit'])
    for d in drafts:
        d.pop('_unit')
    return drafts
