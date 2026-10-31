"""MiniMe's settings: sizes, budgets, models, thresholds and the prompts it sends."""
from __future__ import annotations
from ps_utils.minime.qcr import TASK_FIELDS


ROUND_SIZE = 10            # answers per round


BLIND_CYCLE = ("violate", "representative", "violate", "comply")   # blind checks, rotating over rounds: half built


                           # to break one rule (catching bad answers), the rest like deployed ones or written to pass
COMPLY_SHARE = 0.3         # learning asks written to meet every rule, so good answers are seen (and over-strict rules flipped)


BLIND_SHARE = 0.2          # verdict hidden: the held-out slice


RANDOM_SHARE = 0.3         # learning asks drawn at random (untargeted), so blind spots surface


LOW_QUEUE = 6              # fewer answers waiting than this: the next round starts on its own (a round takes minutes)


LEARN_EVERY = 3            # Nos and comments since the last learn run that start the next one, once settled


LEARN_EVERY_EARLY = 1      # ... while the rules are new: reflect on every one, change boldly


MAX_EDITS_EARLY = None     # no cap on edits per learning run while new (MAX_EDITS, 3, once settled)


LEARN_BUDGET = 250         # GEPA metric calls per learn run, at most


GEPA_EVERY = 5             # once GEPA runs: new outputs with feedback before the next run (not one per comment)


GEPA_MIN_REVIEWS = 20      # GEPA (split D_train into D_feedback / D_pareto) from this many reviewed answers,


GEPA_MIN_EACH = 4          # ... with at least this many PASS and this many FAIL,


GEPA_MIN_FEEDBACK = 3      # ... and this many Nos or comments; before that: one direct reflection (cold start)


DIRECT_EXAMPLES = 20       # answers shown to the rewriter in a direct reflection


PROPOSALS = 6              # GEPA budget, in rewriter proposals: seed on D_pareto + this many x (minibatch + D_pareto)


MINIBATCH = 3              # b: answers per reflection minibatch (GEPA's default)


PARETO_SIZE = 40           # n_pareto: |D_pareto| at most (GEPA, Algorithm 1 line 1) ...


PARETO_SHARE = 1 / 3       # ... and this share of D_train while D_train is small


GEN_MAX_TOKENS = 8192      # reasoning models spend tokens thinking before the answer


REWRITER_MODEL = "claude-opus-4.8"   # GEPA's reflective rewriter: reads the Nos and comments, proposes the edits


JUDGE_MODEL_BEFORE = "deepseek-v4.1-flash"   # the default until 2026-10-29: what judged tasks that use the default
TASK_MODEL = "gpt-4o-mini-fast"      # writes the answers to review (default; Settings can change it)


DRAFT_MAX = 3              # statements drafted when the description gives no rules


JUDGE_MODEL = "deepseek-v4-flash-0731-fast"  # MiniMe's judge: fast, since GEPA makes hundreds of judge calls






REWRITER_MAX_TOKENS = 4096           # output cap: CI reserves a call's worst-case cost before answering


REWRITER_TIMEOUT = 180               # seconds; a rewrite of the whole rubric can take a while


PARALLEL = 8               # answers written / judged at once (each judged answer may itself make several calls)


LOG_SIZE = 100            # "Prompt Rewriting History" entries kept (routine ones included; the page shows a few at a time)


POOL_MAX = 60              # generated inputs kept per task (similar to the user's seed inputs)


POOL_GROWTH = 8            # new inputs asked for per round, until POOL_MAX


JOB_KINDS = ("minime_round", "minime_learn")


class MiniMeError(ValueError):
    pass


QUALITY_WINDOW = 10        # judge quality = agreement on the most recent blind checks of each kind


QUALITY_MIN = 3


READY_AGREE = {"representative": 90, "adversarial": 85}   # % agreement on blind checks


READY_BLIND = 10           # blind checks of each kind before agreement counts


READY_EACH = 5             # blind outputs you passed, and failed


READY_STABLE = 20          # this many shown verdicts in a row with at most READY_FLIPS flips and no rubric change


READY_FLIPS = 1


DESCRIPTION_FIELDS = tuple(k for k in TASK_FIELDS if k != "requirements")   # the statements own the requirements


JUDGE_RETRIES = 2


JUDGE_TIMEOUT = 120         # seconds per judge attempt (normal calls: 5-15 s, slowest seen 99 s)
JUDGE_RETRY_PAUSE = 3       # seconds, growing per attempt
UNJUDGED_RETRY = 300      # seconds before outputs the judge could not answer are tried again
WAITING_RECHECK = 30      # seconds between catch-up re-judges of waiting outputs whose verdict is not current


NOTE_CHECK = """A reviewer read how a judge assessed an answer, agreed with the verdict, and left a comment.

Answer:
{output}

The judge said: {said}

The reviewer's comment: {comment}

Does the judge's assessment fit the comment: the same verdict, for the reason the reviewer gives, and
nothing the reviewer objects to? Reply with YES or NO only."""


GEPA_EVAL_PARALLEL = 20    # outputs judged at once while GEPA scores a candidate (D_pareto, at most 40, in 1-2 waves)


SCORE_RETRY = 600          # seconds before the page asks again to score versions that are still behind


COACH_MIN_REVIEWS = 15      # tips only with this much evidence, or after a rewrite that broke as much as it fixed


COACH_PROMPT = """You help a person teach an automatic judge their taste for this task:
{task}

The judge's current rules (an output passes only if it meets every rule):
{rules}

What happened:
{facts}

Recent feedback from the person (their verdict, and their comment if any):
{comments}

Tips already given (do not repeat them):
{given}

Give the person at most 2 short, concrete tips on how to give feedback that teaches the judge faster,
grounded in the facts above: e.g. comments that only give a verdict, vague taste words that should become
checkable rules, or rules the judge keeps getting wrong. One line each, starting with "- ", addressed to
the person, plain language, no preamble. When a tip is that something should become a rule (a taste word,
a complaint that keeps coming back) and the judge has no such rule yet, put the rule itself on the next line
as "RULE: <one sentence, phrased like the judge's rules above>", ready to add as is.
If nothing useful stands out, reply with NONE."""


__minime_names__ = ('BLIND_CYCLE', 'BLIND_SHARE', 'COACH_MIN_REVIEWS', 'COACH_PROMPT', 'COMPLY_SHARE', 'DESCRIPTION_FIELDS', 'DIRECT_EXAMPLES', 'DRAFT_MAX', 'GEN_MAX_TOKENS', 'GEPA_EVAL_PARALLEL', 'GEPA_EVERY', 'GEPA_MIN_EACH', 'GEPA_MIN_FEEDBACK', 'GEPA_MIN_REVIEWS', 'JOB_KINDS', 'JUDGE_MODEL', 'JUDGE_MODEL_BEFORE', 'JUDGE_RETRIES', 'JUDGE_RETRY_PAUSE', 'JUDGE_TIMEOUT', 'LEARN_BUDGET', 'LEARN_EVERY', 'LEARN_EVERY_EARLY', 'LOG_SIZE', 'LOW_QUEUE', 'MAX_EDITS_EARLY', 'MINIBATCH', 'MiniMeError', 'NOTE_CHECK', 'PARALLEL', 'PARETO_SHARE', 'PARETO_SIZE', 'POOL_GROWTH', 'POOL_MAX', 'PROPOSALS', 'QUALITY_MIN', 'QUALITY_WINDOW', 'RANDOM_SHARE', 'READY_AGREE', 'READY_BLIND', 'READY_EACH', 'READY_FLIPS', 'READY_STABLE', 'REWRITER_MAX_TOKENS', 'REWRITER_MODEL', 'REWRITER_TIMEOUT', 'ROUND_SIZE', 'SCORE_RETRY', 'TASK_MODEL', 'UNJUDGED_RETRY', 'WAITING_RECHECK',)
