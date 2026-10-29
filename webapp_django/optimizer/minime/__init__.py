"""MiniMe on the optimizer site: the judge works, the user corrects, GEPA learns.

  round   write outputs and judge each with the current rules (about 1 in 5 is a blind check:
          its verdict is hidden, so it measures the judge; it is never learned from)
  review  the user agrees or not with each verdict, with an optional comment (the ASI)
  learn   after enough Nos or comments, GEPA rewrites the rules on the reviewed outputs; the
          rewrite becomes structured edits (add / sharpen / retire) and is applied as GEPA picked it
  score   every version is scored on the same blind checks; Ready = agreement at target plus
          consistency checks, and reaching it pauses the task

Nothing runs until the user presses Start optimizing; Stop (or a worker restart) pauses the task.
Labels are only ever written by the user.

The code lives in submodules; every name stays importable from here (``from optimizer import minime``):
  config    MiniMe's settings: sizes, budgets, models, thresholds and the prompts it sends.
  store     Reading and writing a task's stored state: versions, reviews, settings and the history log.
  jobs      MiniMe's background jobs: starting, stopping and pausing them, Start optimizing, and what the page shows of them.
  judging   The judge: model calls, the persistent judge cache, retries and timing, and verdicts on reviewed outputs.
  scoring   How good the judge is: blind-check scores per version, the chart's data, and the Ready checks.
  rounds    Rounds of new outputs to review, and the first draft of the rules.
  learning  Learning new rules: the cold-start rewrite, GEPA (called directly), the follow-up check, tips and chaining.
  tasks     Task-level operations behind the API: create, edit, review, undo, and the page's summary.

Tests patch a function on the submodule that defines it, e.g. ``patch.object(mm.judging, "llm", ...)``:
calls between submodules go through the module (``_judging.llm``), so the patch takes effect everywhere.
"""
from . import config, store, jobs, judging, scoring, rounds, learning, tasks
from ps_utils.minime.candidates import blind_only, expand_inputs, generate_round, placeholders, plan_round
from ps_utils.minime.evaluate import judge_rows, metrics
from ps_utils.minime.labels import LabelError, parse_labels, parse_verdict
from ps_utils.minime.qcr import QCR, STATUSES, TASK_FIELDS, Statement, judge_prompt, render_task, seed_qcr, task_context
from ps_utils.minime.rubric_text import (DRAFT_TEMPLATE, MAX_EDITS, REFLECTION_TEMPLATE, REFLECTION_TEMPLATE_BOLD,
                                        RubricRejected, from_text, parse, to_text)
from ..models import JobState, MiniMeLabel, MiniMeTask, MiniMeVersion  # noqa: F401  (as the old module offered them)

for _module in (config, store, jobs, judging, scoring, rounds, learning, tasks):
    for _name in _module.__minime_names__:
        globals()[_name] = getattr(_module, _name)
del _module, _name
