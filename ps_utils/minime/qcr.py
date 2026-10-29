"""The QCR: a versioned set of judgement statements (the user's lessons learned for one task).

Each statement is atomic and binary. It compiles with the benchmark rubric parser, so the
same code checks and judge prompts grade answers here and in benchmarks. The verdict is
the AND of all active statements.
"""
import json
import re
from dataclasses import asdict, dataclass, field

from ..rubric_parser import compile_rubric

STATUSES = ('candidate', 'active', 'retired')
TASK_FIELDS = ('objective', 'input', 'output', 'requirements', 'goal')

# The task definition rendered as one text: context for answer generation.
SEED_TEMPLATE = ('Objective: {objective}\n\nInput: {input}\n\nOutput: {output}\n\n'
                 '{requirements}\n\nGoal: {goal}')


@dataclass
class Statement:
    id: str
    text: str
    status: str = 'active'
    origin: str = 'seed'          # seed | reason | user
    support: list = field(default_factory=list)          # label ids whose reason states it
    counterexamples: list = field(default_factory=list)  # label ids that contradict it
    flip_rate: float | None = None

    def __post_init__(self):
        if self.status not in STATUSES:
            raise ValueError(f'status must be one of {STATUSES}, got {self.status!r}')


@dataclass
class QCR:
    task: dict
    statements: list = field(default_factory=list)
    version: int = 0
    evaluator_model: str | None = None

    def active(self):
        return [s for s in self.statements if s.status == 'active']

    def compile(self):
        """(entries, owners): compiled rubric entries for the active statements, and the
        statement id each entry belongs to. Statements compile one by one, so one never
        merges with its neighbour (the parser regroups lead-in lines)."""
        entries, owners = [], []
        for s in self.active():
            for entry in compile_rubric([s.text]):
                entries.append(dict(entry, criterion_index=len(entries)))
                owners.append(s.id)
        return entries, owners

    def to_dict(self):
        return {'task': self.task, 'version': self.version, 'evaluator_model': self.evaluator_model,
                'statements': [asdict(s) for s in self.statements]}

    @classmethod
    def from_dict(cls, data):
        return cls(task=data['task'], version=data.get('version', 0), evaluator_model=data.get('evaluator_model'),
                   statements=[Statement(**s) for s in data.get('statements', [])])

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, encoding='utf-8') as f:
            return cls.from_dict(json.load(f))


def render_task(task):
    unknown = set(task) - set(TASK_FIELDS)
    if unknown:
        raise ValueError(f'unknown task field(s): {", ".join(sorted(unknown))}')
    if not str(task.get('objective') or '').strip():
        raise ValueError('task objective is required')
    return SEED_TEMPLATE.format(**{k: str(task.get(k) or '').strip() for k in TASK_FIELDS}).strip()


def seed_qcr(task):
    """QCR v0 from the task definition: each line of the quality description ('requirements',
    plain language) becomes one statement, verbatim. Deterministic, no LLM. Statements compile
    like benchmark criteria: code checks where the wording allows, the LLM judge otherwise.
    May be empty: GEPA then learns every statement from the user's corrections."""
    render_task(task)   # validates the fields
    lines = [re.sub(r'^\s*(?:\d+[.)]|[-*•])\s+', '', line).strip()
             for line in str(task.get('requirements') or '').splitlines()]
    texts = list(dict.fromkeys(line for line in lines if line))
    return QCR(task=dict(task), statements=[Statement(id=f's{n}', text=t) for n, t in enumerate(texts, start=1)])


JUDGE_TEMPLATE = """You judge answers to this task:
{task}

An answer PASSES only if it meets every requirement below; if it breaks any one, it FAILS.
{rules}

Read the input and the answer, check each requirement, then reply with PASS or FAIL. On FAIL, name the requirement it breaks.

Input:
{{{{input}}}}

Answer:
{{{{output}}}}"""


def task_context(task):
    """The description without the requirements (the numbered rules state them, as learned so far)
    and without the fields left empty."""
    render_task(task)   # validates the fields
    labels = {'objective': 'Objective', 'input': 'Input', 'output': 'Output', 'goal': 'Goal'}
    return '\n'.join(f'{label}: {str(task.get(k) or "").strip()}' for k, label in labels.items()
                     if str(task.get(k) or '').strip())


def judge_prompt(qcr):
    """The QCR as one standalone judge prompt (the MiniMe prompt), portable to any LLM:
    the task, the active statements as numbered requirements, and {{input}} / {{output}} slots.
    MiniMe itself evaluates statements one by one (code checks where possible); this is the
    same rubric as a single, copyable prompt."""
    rules = '\n'.join(f'{n}. {s.text}' for n, s in enumerate(qcr.active(), start=1)) or '(no requirements yet)'
    return JUDGE_TEMPLATE.format(task=task_context(qcr.task), rules=rules)
