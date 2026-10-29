# Prompt Optimizer — Django web app ("Prompt Studio")

A guided studio over the `prompt_optimizer` library (MLflow `optimize_prompts` /
GEPA). You learn an instruction-rich prompt so a **cheaper target model** matches a
**frontier benchmark**, and see the **accuracy vs. lifetime-cost trade-off — measured
from real runs, not guessed.** Same library and cost story as before; the framework,
data model, and UX have grown up considerably.

## What it does

- **Optimize** — a 4-step flow:
  1. **Approach** — pick a use case: **Training data** (labelled), **Few-shot**
     (examples + criteria), or **Zero-shot** (one-line description).
  2. **Prompt** — system/user prompt with `{{variables}}`.
  3. **Examples** — labelled rows / few-shot examples.
  4. **Let's Optimize** — task/reflection/judge models, task reasoning effort,
     optimization budget (with a GEPA `~15–20×` budget hint), and collapsible
     **Output constraints** and **Prompt-length controls**.
  - **Pass 1** optimizes your **task model** and shows a **Score Card / Playbook**:
    quality, **measured** per-call serving cost, prompt, deployable payload, and the
    one-time optimization token cost.
  - **Pass 2 (opt-in)** optimizes the **benchmark anchor (frontier)** too and renders
    the **Accuracy vs. Lifetime Cost Trade-off** — score + `$ /call`, `$ @10k`,
    `$ @100k` for both models, priced from each run's *actual* task-call tokens ×
    live catalog prices (hover the ⓘ for the method).

- **A/B Experiments** — pick a completed run as the **baseline** (no re-entry), vary
  exactly one **lever** and compare arms on **held-out** inputs:
  - Levers: `reflection_max_chars`, `reflection_model`, `task_reasoning_effort`.
  - Held-out scoring: few-shot → quality-criteria judge; **training-data → gold
    `expected_output` via the run's label scorer** (equivalence/correctness);
    zero-shot is unsupported (no reflection step, nothing to score against).

- **Model Standings** — measured latency per model × role (task/judge/reflection),
  **text models only**; refresh on demand or via `python manage.py probe_models`.

- **History** — durable, per-user log of every optimize + A/B run, showing the
  background **job id** (matches `/api/job/<id>` logs). Resume the anchor pass on a
  past optimize run; the computed trade-off is saved back to that run.

- **Auth** — Django username/password self-signup; all views and History are scoped
  per user.

## Stack

- **Backend:** plain Django 5 + **SQLite**. Logic lives in `optimizer/services.py`;
  views are thin `JsonResponse` adapters. Optimization / A-B / anchor / diagnostics
  run in **background threads**, and **job state is persisted to the DB**
  (`JobState`) — so progress polling works across processes/workers and survives a
  restart (orphaned `running` jobs are marked interrupted on boot). Model list and
  prices come **live from the endpoint's `/v1/models`** — the single source of truth,
  no hardcoded models or prices.
- **Frontend:** one server-rendered template (`optimizer/templates/optimizer/index.html`),
  vanilla JS, no build step, wrapped in `{% verbatim %}` so `{{variables}}` survive.

## Layout

```
webapp_django/
  manage.py
  optimizer_site/        # project: settings, urls, wsgi
  optimizer/
    services.py          # economics/cost model, config building, job engine (DB-backed),
                         #   A/B lever engine, held-out eval, latency diagnostics, catalog
    views.py             # thin JSON views behind auth (+ public /healthz)
    models.py            # Run (history), JobState (job status), Probe (latency)
    apps.py              # startup: mark orphaned running jobs interrupted
    management/commands/probe_models.py   # scheduled Model Standings refresh
    migrations/          # 0001 Run · 0002 Probe · 0003 JobState · 0004 job_id
    templates/optimizer/{index.html, auth.html}
```

`settings.py` adds the repo root (holding `prompt_optimizer.py` / `llm_endpoint.py`)
to `sys.path`, so the app imports the library directly.

## Run (local)

```bash
pip install -r ../requirements.txt          # django, gunicorn, mlflow, gepa, openai, litellm, tiktoken
cd GepaProject/webapp_django
export DJANGO_SECRET_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(64))')"
export DJANGO_DEBUG=1                       # local HTTP only; never enable on a public deploy
python manage.py migrate                    # creates db.sqlite3 (users, History, job state)

export CHEAPERINFERENCE_API_KEY=...  CHEAPERINFERENCE_BASE_URL=...
python manage.py runserver 127.0.0.1:8000   # single process is fine
```

Create an account with `python manage.py adduser demo` (prompts for a password).
Open http://127.0.0.1:8000 and sign in. Self-signup is disabled by default.
For optimization jobs, also run `python manage.py run_jobs` in another terminal
with the same exported environment, or use `python ../run_service.py` to
supervise the web process and worker together. The badge shows whether the
endpoint is reachable; with no creds the app still runs (dropdowns just stay empty).

> **One replica.** The deploy runs one replica: one Gunicorn process plus
> `GP_JOB_WORKERS` queue workers (default 1, max 8), all on the same host and the
> same SQLite file in WAL mode. Jobs are claimed atomically, so each runs once.
> Multiple replicas sharing the SQLite file are not supported — move to Postgres
> to scale out.

## Configuration (env)

| Var | Purpose |
|---|---|
| `CHEAPERINFERENCE_API_KEY` / `CHEAPERINFERENCE_BASE_URL` | endpoint creds (task/judge via OpenAI SDK, reflection via litellm) |
| `DB_PATH` | SQLite file path — **point at a persistent volume in prod** |
| `GP_JOB_WORKERS` | queue workers `run_service.py` starts on this host (default `1`, range 1–8) |
| `GP_BENCHMARK_CONCURRENCY` | benchmark cases run at once, across all models and variants, queued case by case (default `1`, range 1–8); each case may have up to 4 judge calls in flight |
| `MLFLOW_TRACKING_URI` | MLflow store — else an absolute `mlflow.db` at the repo root (cwd-independent) |
| `DJANGO_DEBUG` | `1` for local HTTP and debug pages; default `0` enables secure session/CSRF cookies |
| `DJANGO_SECRET_KEY` | required for all Django commands; generated secret of at least 50 characters, stable across processes/restarts |
| `DJANGO_ALLOWED_HOSTS` | comma-separated hostnames, no scheme/port; defaults to localhost, 127.0.0.1 and [::1]; `*` is rejected |
| `RAILWAY_PUBLIC_DOMAIN` | exact deployment hostname, automatically added to allowed hosts |
| `DJANGO_CSRF_TRUSTED_ORIGINS` | optional comma-separated cross-origin browser origins including scheme; empty by default, same-origin needs no entry |
| `DJANGO_TRUST_PROXY` | `1` only behind a controlled proxy that sets/strips X-Forwarded-Proto; default `0`, set `1` on Railway |
| `SENTRY_DSN` | optional, your own Sentry project; unset/blank disables error reporting |
| `SENTRY_ENVIRONMENT` | reports only for `production`; defaults to RAILWAY_ENVIRONMENT_NAME, then development |
| `PROMPT_STUDIO_INFERENCE_BASE_URL` | optional scored service jobs: your explicitly configured HTTPS inference base URL; no default |

Django reads these settings from exported environment variables, not `.env`.
Keep the generated signing key in a secret store or ignored local environment
file and export the same value for subsequent commands. Generating a new key
invalidates existing sessions. Tests also require a generated key; CI creates
an ephemeral one. Missing, short, low-diversity, and `django-insecure-` keys fail
startup without logging the supplied value.

## Deploy (Railway / Docker)

- The `Dockerfile` builds the image; its `CMD` runs `migrate --noinput` then
  **gunicorn `--workers 1 --threads 8`** bound to `$PORT`.
- **Healthcheck path is `/healthz`** — a public, auth-free, DB-free liveness endpoint
  (`railway.toml` / `railway.json`). `/api/health` is auth-gated for the UI.
- **Mount a Volume at `/app/data`** and set `DB_PATH=/app/data/db.sqlite3` +
  `MLFLOW_TRACKING_URI=sqlite:////app/data/mlflow.db` so the app DB (users, History,
  job state) and the MLflow store persist across redeploys — the container filesystem
  is ephemeral.
- Set the endpoint creds as service env vars.
- Set a stable generated `DJANGO_SECRET_KEY`, keep `DJANGO_DEBUG=0`, and serve
  browser traffic over HTTPS. Set `DJANGO_TRUST_PROXY=1` on Railway. Configure
  `DJANGO_ALLOWED_HOSTS` for custom domains; Railway's public domain is added
  automatically. Only add CSRF trusted origins for intentional cross-origin
  browser clients. Supply your own `SENTRY_DSN` to opt in to production errors.

## Benchmark timing and cost (read-only)

These commands print only run metadata, per-call timing and cost, never prompts, answers or
credentials (they do load run records, whose JSON holds those). Run them locally or on the
deploy (`railway ssh`, then `python manage.py …`).

```bash
python manage.py benchmark_timings 20          # per run: judge check, time to first call, wall time,
                                               # task/judge call latency (median/p90), concurrency
python manage.py benchmark_timings 20 --json   # full summaries
python manage.py watch_benchmark [job_id]      # live: one line per finished call with its latency
python manage.py benchmark_costs 10            # per run: spend, task/judge split, cost per answer,
                                               # unpriced calls; total over the runs listed
```

Per-call latency comes from `BenchmarkCall.duration_ms` (recorded from runs on this version on);
"concurrency" is the average number of calls in flight while the run was making calls.

## API

Browser routes use Django session authentication and CSRF. A separate API is
available for trusted services. Set `GEPA_SERVICE_API_KEY` to a long random
secret and send it as `Authorization: Bearer <secret>`.

| Method + path | Purpose |
|---|---|
| `GET /` | the studio UI (auth required) |
| `GET /healthz` | public liveness (200) — platform healthcheck |
| `GET /api/health` | endpoint-config status (UI badge) |
| `GET /api/me` | current username |
| `GET /api/models` | live `/v1/models` catalog (ids + prices) |
| `POST /api/variables` | prompts → `{{variables}}` |
| `POST /api/optimize` | 1st-pass job → `{job_id}` |
| `POST /api/anchor` | 2nd (benchmark-anchor) pass → `{job_id}` |
| `POST /api/experiment` | A/B run → `{job_id}` |
| `GET /api/runs` · `GET /api/runs/<uuid>` | History list / detail (per user) |
| `POST /api/diagnostics` · `GET /api/standings` | latency probe / Model Standings |
| `GET /api/job/<id>` | job status; on `done` includes `result`/`serving`/`tradeoff`/`run_id` |
| `login/ · signup/ · logout/` | auth pages |

The `POST /api/optimize` body maps 1:1 to `prompt_optimizer.OptimizationConfig`.

## Notes

- The MLflow prompt registry needs a SQL store; the app resolves it via
  `MLFLOW_TRACKING_URI` → absolute repo-root `mlflow.db` (see `_tracking_uri` in
  `services.py`).
- `__pycache__/*.pyc` should not be committed; `.dockerignore` already keeps them out
  of the image.
