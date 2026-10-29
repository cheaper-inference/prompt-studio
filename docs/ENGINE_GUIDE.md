# Prompt Studio engine guide

The Python library lives in `prompt_optimizer.py`; `llm_endpoint.py` configures
the task, reflection, and judge roles. Start with the runnable example in the
[README](../README.md#use-it-from-python).

## Three optimization modes

| Mode | Inputs | Behavior |
| --- | --- | --- |
| `Mode.ZERO_SHOT` | `prompts`, optional `task_description` | Uses MLflow's description-guided `MetaPromptOptimizer`. No examples are scored; assess the rewritten prompt separately. |
| `Mode.FEW_SHOT` | `prompts`, `examples`, criteria and/or reference answers | Uses GEPA and a structured judge to score outputs against the available quality signal. |
| `Mode.TRAINING_DATA` | `prompts`, `training_data`, target answers | Uses GEPA and a label scorer, with correctness or equivalence as the comparison method. |

For data-backed optimization, a reflection model proposes candidate prompts from
execution feedback, a task model produces outputs, and a judge scores them.
Candidate selection retains promising alternatives. Better evaluation criteria
make the search more useful; optimization can also exploit weaknesses in a scorer.

## Endpoint configuration

Export `CHEAPERINFERENCE_API_KEY` and `CHEAPERINFERENCE_BASE_URL`, or pass
`api_key` and `base_url` directly to `llm_endpoint.configure()`. The names are
historical; the endpoint can be another OpenAI-compatible provider.

Choose model IDs that your endpoint supports. `configure()` accepts `task_model`,
`reflection_model`, and `judge_model`; the latter two default to the task model.
Provider support for JSON mode and reasoning settings can vary.

By default, `configure()` also sets `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and
`OPENAI_API_BASE` for integrations that read process environment variables.
The explicit client paths use the returned `Endpoint` object.

`optimizer.preflight()` makes a request for each model role and returns a list of
failures. It is an optional live check and can incur provider charges.

## Configuration reference

`OptimizationConfig.validate()` checks the selected mode's input contract before
optimization. Only fields relevant to that mode are needed.

| Setting | Purpose |
| --- | --- |
| `name`, `mode`, `prompts` | Name the run, choose a mode, and supply prompt text, `Prompt` objects, or role/template dictionaries. |
| `training_data`, `target_key` | Labeled examples and the answer column; `target_key` defaults to `expected_output`. |
| `input_keys` | Columns used to fill `{{variables}}`; inferred when possible. |
| `label_scorer` | `correctness` or `equivalence` for labeled examples. |
| `examples`, `quality_criteria` | Few-shot examples and rules for good outputs; criteria can be a string or a list. |
| `per_row_criteria`, `criteria_key` | Use example-specific criteria; the column defaults to `guidelines`. |
| `task_description` | Explain the task for a zero-shot rewrite. |
| `task_model`, `reflection_model`, `judge_model` | Override the endpoint's per-role defaults. |
| `max_metric_calls` | Evaluation-call budget used by the optimizer; default `50`, not a monetary limit or a ceiling on every role's requests. |
| `reflection_minibatch_size` | Examples considered during reflection; default `3`. |
| `candidate_selection_strategy` | Candidate selection strategy; default `pareto`. |
| `request_timeout` | Request timeout in seconds; default `90`. |
| `enable_tracking` | Opt into MLflow run/metric/trace tracking; default `False`. The prompt registry is still used. |

Keep template variables in double braces, for example `{{question}}`. Flat labeled
rows such as `{"question": "...", "expected_output": "..."}` are normalized to
the inputs/expectations format. Input placeholders must be supplied by the data.

## Output constraints

`output_constraints` can describe `format` (`json`, `lines`, or `csv`),
`json_fields`, `line_count`, and `allowed_values`. These requirements are folded
into the prompt and checked deterministically during scoring.

For few-shot runs, `constraint_gate=True` makes constraint failure a zero score
before the quality judge runs. Otherwise constraint checks and quality scoring
are combined. `tighten_prompt=True` attempts a final compression pass while
preserving template variables.

## Results and storage

`result.summary()` provides a readable overview. `result.to_deployable()` returns
prompt messages, variables, model metadata, inference parameters, and available
scores. `result.to_json()` serializes the result. Zero-shot runs have no scored
evaluation dataset, so do not treat them as a measured improvement.

The MLflow prompt registry needs a SQL-backed tracking URI. For local use,
`sqlite:///mlflow.db` creates a store in the working directory. The Django app
also creates its own database for accounts, run history, and durable jobs.
No database is shipped with the project.

## Evaluate before deploying

- Define criteria that match the behavior you want. A high judge score is only
  useful if the scorer captures the task correctly.
- Keep held-out inputs separate from optimization data. Repeatedly selecting
  prompts on a test set turns it into validation data.
- Compare baseline and candidate prompts using the same scoring method.
- Inspect failed examples and coverage alongside aggregate scores.
- Measure your actual token usage, provider charges, and latency. A longer prompt
  on a cheaper model can cost less, but neither quality nor savings is guaranteed.

For local accounts, background workers, and production configuration, see the
[web app guide](../webapp_django/README.md).
