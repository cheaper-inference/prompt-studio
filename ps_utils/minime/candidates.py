"""Active learning: choose what to ask the user to label next, and generate those answers.

  plan_round      deterministic: which asks give the most learning per label
  generate_round  one LLM call per ask -> blind, shuffled candidates to label

Each round has two streams. Held-out asks ('representative': the plain task on a random input) estimate
deployed performance and are never used for revision. Learning asks target where the QCR is weakest, in
priority order:
  gap        more answers with the flaw of a fail no statement catches (builds support for a new statement)
  violate    an answer breaking exactly one statement, for candidates under MIN_SUPPORT (and, with
             slots left, active statements with the least support)
  (missed fails, whose stated statement did not fire, get violate asks on that statement)
  near_miss  an answer that comes close to a statement without breaking it: sharpens its boundary,
             for weak candidates and for statements whose verdict flips across runs
Every candidate records what the engine expects (expected_pass) for later scoring; it is never shown
to the person labelling.
"""
import random
import re
import zlib

from ..rubric_checks import _run_concurrently
from .qcr import render_task, task_context

MIN_SUPPORT = 3          # a statement needs this many supporting labels to be learned reliably
EPSILON = 0.01           # flip rate above this: the statement is not near-deterministic
HELDOUT_SHARE = 0.2
ROUND_SIZE = 25
MAX_GAPS_SHARE = 0.5     # gaps may take at most half of the learning slots

_TASK = 'You are performing this task:\n{task}\n\nInput:\n{input}\n\n'
_RULES = 'Requirements:\n{rules}\n\n'
_ONLY = 'Reply with the answer only, exactly as it would be delivered: no preface, no commentary.'
PROMPTS = {
    'representative': _TASK + _ONLY,
    'violate': _TASK + _RULES + ('Write a realistic answer that meets every requirement except this one, which it '
                                 'must clearly break:\n{target}\n\n' + _ONLY),
    'near_miss': _TASK + _RULES + ('Write a realistic answer that meets every requirement, and comes as close as '
                                   'possible to breaking this one without breaking it:\n{target}\n\n' + _ONLY),
    'comply': _TASK + _RULES + ('Write a realistic, strong answer that meets every requirement.\n\n' + _ONLY),
    'gap': _TASK + _RULES + ('A reviewer failed an earlier answer for this reason:\n"{reason}"\n\nWrite a new '
                             'realistic answer that meets every requirement but has the same flaw.\n\n' + _ONLY),
}
EXPECTED = {'representative': None, 'violate': False, 'near_miss': True, 'gap': False, 'comply': True}

INPUT_PROMPT = ('You are helping test this task:\n{task}\n\nHere are example inputs:\n{examples}\n\n'
                'Write {n} new inputs of the same kind: realistic, varied, and clearly different from the examples '
                'and from each other. One per line, the input only: no numbering, no commentary.')
_BULLET = re.compile(r'^\s*(?:\d+[.)]|[-*•])\s*')


def _key(text):
    return ' '.join(str(text).lower().split())


def expand_inputs(task_text, seeds, existing, generate, n=8, max_len=300):
    """New inputs similar to the user's seed examples, so MiniMe does not learn only from those few.
    One LLM call; returns up to n new inputs, none duplicating a seed or an existing one."""
    known = {_key(x) for x in list(seeds) + list(existing)}
    examples = '\n'.join(f'- {x}' for x in (list(seeds) + list(existing))[:20])
    reply = generate(INPUT_PROMPT.format(task=task_text, examples=examples, n=n))
    out = []
    for line in str(reply or '').splitlines():
        text = _BULLET.sub('', line).strip().strip('"').strip()
        if text and len(text) <= max_len and _key(text) not in known:
            known.add(_key(text))
            out.append(text)
    return out[:n]


def blind_only(text, share=4):
    """A stable 1-in-`share` slice of generated inputs reserved for blind checks: MiniMe never
    learns from answers to these inputs, so judge quality measures unseen inputs."""
    return zlib.crc32(_key(text).encode()) % share == 0


def plan_round(qcr, report=None, flip_rates=None, size=ROUND_SIZE, heldout_share=HELDOUT_SHARE, epsilon=EPSILON,
               random_share=0.0, adversarial_share=0.0, comply_share=0.0, blind_cycle=None, offset=0, tested=None):
    """[{kind, split, target, reason}] for one round: learning asks first, then held-out ones.

    report holds findings (uncovered_fails, missed_fails), flip_rates evaluate()'s. random_share of the
    learning slots go to untargeted (representative) answers, so blind spots still surface.
    adversarial_share of the held-out asks break one active statement (round-robin) instead of
    being representative, so the blind checks contain enough bad answers to measure catching them.
    comply_share of the learning slots go to answers written to meet every active statement, so good
    answers are seen too (an over-strict statement then gets flipped, and the judge is tested on PASS).
    blind_cycle, when given, sets the held-out kinds instead of adversarial_share: they follow the cycle,
    continued from round `offset`, so small rounds still mix all kinds over time. tested, when given,
    is {statement id: bad outputs it has caught}: the rule-breaking asks go to untested rules first,
    then the least tested, rotating by round so every rule gets its turn. Deterministic."""
    report, flip_rates = report or {}, flip_rates or {}
    # A positive share always keeps at least one blind check (round(2 * 0.2) would be 0: never Ready).
    heldout = max(1, round(size * heldout_share)) if size > 1 and heldout_share > 0 else 0
    randoms = round((size - heldout) * random_share)
    complies = round((size - heldout) * comply_share) if qcr.active() else 0
    slots = max(0, size - heldout - randoms - complies)
    asks = []

    def add(kind, target=None, reason=None):
        if len(asks) < slots:
            asks.append({'kind': kind, 'split': 'learn', 'target': target, 'reason': reason})

    for gap in report.get('uncovered_fails', [])[:int(slots * MAX_GAPS_SHARE)]:
        add('gap', reason=gap['reason'])
    for s in qcr.statements:
        if s.status == 'candidate':
            for _ in range(max(0, MIN_SUPPORT - len(s.support))):
                add('violate', s.id)
            add('near_miss', s.id)
    for miss in report.get('missed_fails', []):     # a statement whose check misses a fail it should catch
        for sid in miss['statements']:
            add('violate', sid)
    for s in qcr.active():
        if (flip_rates.get(s.id) or 0) > epsilon:
            add('near_miss', s.id)
            add('near_miss', s.id)
    active = list(qcr.active())
    if tested is not None and active:     # untested first, then least tested; ties rotate by round
        n = len(active)
        order = {s.id: i for i, s in enumerate(active)}
        active.sort(key=lambda s: (tested.get(s.id, 0), (order[s.id] - offset * max(1, slots)) % n))
    else:
        active.sort(key=lambda s: len(s.support))
    while active and len(asks) < slots:
        for s in active:
            add('violate', s.id)
    while len(asks) < slots:          # nothing to target yet: representative answers to learn from
        add('representative')
    asks += [{'kind': 'representative', 'split': 'learn', 'target': None, 'reason': None}] * randoms
    asks += [{'kind': 'comply', 'split': 'learn', 'target': None, 'reason': None}] * complies
    targets = [s.id for s in qcr.active()]
    if blind_cycle:
        out = []
        for n in range(offset * heldout, offset * heldout + heldout):
            kind = blind_cycle[n % len(blind_cycle)] if targets else 'representative'
            # The rule a violation targets advances with the violations asked so far, not with the
            # cycle position (violations sit at fixed positions, which would always pick the same rules).
            asked = sum(blind_cycle[k % len(blind_cycle)] == 'violate' for k in range(n))
            out.append({'kind': kind, 'split': 'heldout', 'target': targets[asked % len(targets)] if kind == 'violate' else None,
                        'reason': None})
        return asks + out
    adversarial = round(heldout * adversarial_share) if targets else 0
    blind = [{'kind': 'violate', 'split': 'heldout', 'target': targets[n % len(targets)], 'reason': None}
             for n in range(adversarial)]
    return asks + blind + [{'kind': 'representative', 'split': 'heldout', 'target': None, 'reason': None}] * (heldout - adversarial)


PLACEHOLDER = re.compile(r'\{\{\s*([A-Za-z_][\w-]*)\s*\}\}')


def placeholders(prompt):
    """The distinct {{name}} placeholders in a prompt, in order of appearance."""
    return list(dict.fromkeys(m.group(1) for m in PLACEHOLDER.finditer(prompt or '')))


def deployed(prompt, value):
    """The deployed prompt on one input. Its one placeholder ({{input}}, {{theme}}, ... whatever the
    name) is replaced everywhere it appears; with none, the input is appended. A prompt with several
    different placeholders is refused when saved; here it falls back to appending."""
    names = placeholders(prompt)
    if len(names) == 1:
        return PLACEHOLDER.sub(lambda m: value, prompt)
    return f'{prompt.rstrip()}\n\nInput:\n{value}'


def generate_round(qcr, plan, inputs, generate, seed=0, deployed_prompt=None, progress=None, parallel=1):
    """Shuffled candidates [{id, input, output, kind, split, target, expected_pass}].

    inputs: task inputs, a list, or {'learn': [...], 'heldout': [...]} to keep blind asks on inputs
    the learning asks never use; each ask draws one at random from its split's list (seeded).
    generate(prompt) -> answer text. Learning asks state only active statements as requirements
    (plus the target), so candidates are judged against what the QCR currently holds.
    deployed_prompt: the prompt actually in production; representative answers come from it,
    so the held-out slice measures what is deployed (without one, from the task definition).
    progress(done, total) is called before each batch; parallel answers are written at once."""
    pools = inputs if isinstance(inputs, dict) else {'learn': inputs, 'heldout': inputs}
    if not pools.get('learn') or not pools.get('heldout'):
        raise ValueError('at least one task input is required')
    rng = random.Random(seed)
    task = render_task(qcr.task)
    # Targeted asks (violate, comply, near miss, gap) state the rules as they are now, listed
    # separately: the description's original requirements would contradict an edited rule.
    context = task_context(qcr.task)
    texts = {s.id: s.text for s in qcr.statements}
    rules = [s.text for s in qcr.active()]
    asks = []
    for n, ask in enumerate(plan, start=1):     # prompts first, in order: the seeded draws stay reproducible
        target = texts.get(ask['target'])
        listed = rules + ([target] if target and target not in rules else [])
        value = rng.choice(pools[ask['split']])
        if ask['kind'] == 'representative' and deployed_prompt:
            prompt = deployed(deployed_prompt, value)
        else:
            prompt = PROMPTS[ask['kind']].format(task=task if ask['kind'] == 'representative' else context,
                                                 input=value, target=target or '',
                                                 reason=ask.get('reason') or '',
                                                 rules='\n'.join(f'- {r}' for r in listed) or '- (none yet)')
        asks.append((n, ask, value, prompt))
    answers = {}
    for start in range(0, len(asks), max(1, parallel)):    # then answers, `parallel` at a time
        batch = asks[start:start + max(1, parallel)]
        if progress:
            progress(start, len(asks))
        def write(n, prompt):
            return lambda: answers.__setitem__(n, generate(prompt).strip())
        errors = _run_concurrently([write(n, prompt) for n, _, _, prompt in batch])
        failed = next((e for e in errors if e is not None), None)
        if failed is not None:
            raise failed
    out = [{'id': f'c{n}', 'input': value, 'output': answers[n], 'kind': ask['kind'],
            'split': ask['split'], 'target': ask['target'], 'expected_pass': EXPECTED[ask['kind']]}
           for n, ask, value, _ in asks]
    rng.shuffle(out)
    return out
