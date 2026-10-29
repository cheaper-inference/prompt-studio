"""The QCR as GEPA's single text component, and back: a numbered list of statements.

GEPA rewrites the list as text. from_text parses the rewrite and diffs it against the
current QCR, so every change reaches the user as a structured edit:
  add      a new statement
  sharpen  an existing statement reworded (it keeps its id and history)
  retire   a statement the rewrite dropped
A rewrite that parses to nothing, or changes more than MAX_EDITS statements, is rejected:
small, incremental, readable changes only.
"""
import difflib
import re
from dataclasses import replace

from .qcr import QCR, Statement

MAX_EDITS = 3
SHARPEN_RATIO = 0.5    # a rewritten line this similar to a dropped one is the same statement, reworded
EMPTY = 'No requirements yet.'
# 1. / 1) / 1: / - / * / •, also with Markdown bold around the number (**1.** or __1.__), as models often write.
_LINE = re.compile(r'^\s*(?:\*\*|__)?(?:\d+[.):]|[-*•])(?:\*\*|__)?\s+(.+?)\s*$')

# GEPA's reflection prompt (gepa 0.1.4 placeholders: <curr_param>, <side_info>).
REFLECTION_TEMPLATE = """I judge answers to a task with a quality rubric. An answer FAILS if it breaks any one statement, otherwise it PASSES. The rubric:
```
<curr_param>
```

Below are answers, the rubric's verdicts, and an expert's feedback. Where the rubric was wrong, or right for the wrong reason, the feedback carries the expert's comment: that is the signal to act on.
```
<side_info>
```

Make the smallest edits that bring the rubric's verdicts, and its reasons, in line with the expert: add a statement, sharpen one (reword it more precisely), or remove one. At most 3 edits. Every statement must be one requirement that can be checked just by reading the answer. Keep the statements that work, unchanged and in order. Generalize from the reasons; do not copy details that fit only one example.

Return the whole rubric, one statement per numbered line, within ``` blocks."""


# While the rules are new: large strides. No edit cap; restructuring is welcome.
REFLECTION_TEMPLATE_BOLD = """I judge answers to a task with a quality rubric. An answer FAILS if it breaks any one statement, otherwise it PASSES. The rubric:
```
<curr_param>
```

Below are answers, the rubric's verdicts, and an expert's feedback. Where the rubric was wrong, or right for the wrong reason, the feedback carries the expert's comment: that is the signal to act on.
```
<side_info>
```

Rewrite the rubric so its verdicts, and its reasons, match the expert's. Be bold: add, sharpen, merge, split or remove statements, and restructure the rubric if that serves the expert better. Every statement must be one requirement that can be checked just by reading the answer. State counts, lengths and required or forbidden words plainly (e.g. "At most 200 characters."), so they can be checked in code. Generalize from the comments; do not copy details that fit only one example.

Return the whole rubric, one statement per numbered line, within ``` blocks."""

DRAFT_TEMPLATE = """You write the quality rubric an LLM judge will use to grade answers to this task:
{task}

How the author describes a good answer:
{requirements}

Write a small starting rubric: at most 3 numbered statements, only the most essential requirements the task itself states or clearly implies. It will grow later from the author's feedback. An answer passes only if it meets every statement. Each statement is one requirement that can be checked just by reading the answer. Do not invent requirements the author did not state or imply. State counts, lengths and required or forbidden words plainly (e.g. "At most 200 characters.", "Mention @handle."), so they can be checked in code.

Return only the numbered list."""


class RubricRejected(ValueError):
    pass


def to_text(qcr):
    active = qcr.active()
    return '\n'.join(f'{n}. {s.text}' for n, s in enumerate(active, start=1)) or EMPTY


_FENCE = re.compile(r'```[^\n]*\n(.*?)```', re.S)


def parse(text):
    """Statement texts from a numbered (or bulleted) list, in order, without duplicates. When the
    reply has ``` blocks, only the last one is the rubric: the rewriter may reason in numbered
    lists before it, and those notes must never become statements. If that block holds no list,
    the reply is unusable (nothing is returned): an earlier list is never taken in its place."""
    blocks = _FENCE.findall(text or '')
    body = blocks[-1] if blocks else (text or '')
    lines = [m.group(1) for line in body.splitlines() if (m := _LINE.match(line))]
    return list(dict.fromkeys(l for l in lines if l))


def from_text(text, current, max_edits=MAX_EDITS):
    """(new QCR, edits) from a rewritten list. Retired statements stay in the QCR (status
    'retired') so their history is kept; new ones get the next free id, origin 'learned'."""
    if not (text or '').strip():
        raise RubricRejected('the rewriter returned no usable reply (empty or cut off)')
    texts = parse(text)
    if not texts:
        raise RubricRejected('the rewrite has no statements')
    active = current.active()
    by_text = {s.text: s for s in active}
    kept, added = [], []
    for t in texts:
        (kept if t in by_text else added).append(t)
    dropped = [s for s in active if s.text not in texts]
    # Pair each added line with the most similar dropped statement: that is a sharpen.
    sharpened = {}
    for t in list(added):
        best = max(dropped, key=lambda s: difflib.SequenceMatcher(None, s.text, t).ratio(), default=None)
        if best is not None and best.id not in sharpened and \
                difflib.SequenceMatcher(None, best.text, t).ratio() >= SHARPEN_RATIO:
            sharpened[best.id] = t
            added.remove(t)
            dropped.remove(best)
    edits = ([{'op': 'sharpen', 'id': sid, 'before': next(s.text for s in active if s.id == sid), 'after': t}
              for sid, t in sharpened.items()]
             + [{'op': 'retire', 'id': s.id, 'before': s.text} for s in dropped])
    if max_edits is not None and len(edits) + len(added) > max_edits:
        raise RubricRejected(f'{len(edits) + len(added)} edits; at most {max_edits} are allowed')
    n = 1 + max([int(m.group(1)) for s in current.statements if (m := re.fullmatch(r's(\d+)', s.id))] or [0])
    order = {t: i for i, t in enumerate(texts)}
    statements = []
    for s in current.statements:
        if s.status != 'active':
            statements.append(s)
        elif s.id in sharpened:
            statements.append(replace(s, text=sharpened[s.id], flip_rate=None))
        elif s in dropped:
            statements.append(replace(s, status='retired'))
        else:
            statements.append(s)
    for t in added:
        statements.append(Statement(id=f's{n}', text=t, origin='learned'))
        edits.append({'op': 'add', 'id': f's{n}', 'after': t})
        n += 1
    # Active statements follow the rewrite's order; retired ones keep their place at the end.
    rank = lambda s: order.get(s.text, len(order)) if s.status == 'active' else len(order) + 1
    statements.sort(key=rank)
    return QCR(task=current.task, statements=statements, version=current.version,
               evaluator_model=current.evaluator_model), edits
