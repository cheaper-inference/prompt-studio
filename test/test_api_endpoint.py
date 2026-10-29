"""Representative /api/v1 calls — one per optimization approach (bearer variant).

Self-contained: standard library only (no requests). Submits one job per approach,
waits for each to finish, and prints the final score.

Start a local server first (PROMPT_STUDIO_API_ENABLED=1, matching PROMPT_STUDIO_SERVICE_TOKEN, real endpoint creds):

    PROMPT_STUDIO_API_ENABLED=1 PROMPT_STUDIO_SERVICE_TOKEN=smoke-token \
    CHEAPERINFERENCE_API_KEY=… CHEAPERINFERENCE_BASE_URL=… \
    python manage.py runserver 127.0.0.1:8000

Then, with the SAME token (the inference key defaults to CHEAPERINFERENCE_API_KEY):

    PROMPT_STUDIO_SERVICE_TOKEN=smoke-token python test/test_api_endpoint.py
"""
import json
import os
import time
import urllib.error
import urllib.request

API_BASE    = os.environ.get("API_BASE", "http://127.0.0.1:8000")
SERVICE_TOK = os.environ.get("PROMPT_STUDIO_SERVICE_TOKEN", "smoke-token")
USER_ID     = os.environ.get("USER_ID", "user-smoke-123")
# Inference key: use whatever's already in the environment for testing.
INFER_KEY   = os.environ.get("USER_CI_KEY") or os.environ.get("CHEAPERINFERENCE_API_KEY")
# Kept small so a full run finishes quickly; raise for a real optimization.
MAX_CALLS   = int(os.environ.get("MAX_METRIC_CALLS", "12"))
POLL_TIMEOUT = int(os.environ.get("POLL_TIMEOUT", "1800"))   # seconds
POLL_EVERY   = int(os.environ.get("POLL_EVERY", "5"))        # seconds


def call(method, path, body=None, *, user_id=None, inference_key=None):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {SERVICE_TOK}"}
    if user_id:       headers["X-CI-User-Id"] = user_id
    if inference_key: headers["X-CI-Inference-Key"] = inference_key
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(API_BASE + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode() or "{}")
        except Exception:  # noqa: BLE001
            return e.code, {}
    except Exception as e:  # noqa: BLE001
        return None, {"error": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------------------
# One representative request body per approach.
# ---------------------------------------------------------------------------

# 1) FEW-SHOT — Part A (rules) + Part B (a few example answers: input + good output).
FEW_SHOT = {
    "mode": "few_shot",
    "name": "keywords_explorer",
    "system_prompt": "You are a helpful search marketing assistant that identifies prospective marketing keywords for local service businesses.",
    "user_prompt": "Recommend 5 high-click search keywords for a {{service}} clinic.",
    "quality_criteria": [
        "Return exactly 5 keywords, one per line, no numbering, bullets, or quotes.",
        "Each keyword starts with a 2-word generic term (e.g. urgent care, primary care).",
        "Extend each with a proximity/convenience modifier (near me, same day, open now).",
        "Keywords are unique, under 5 words, ordered by expected clicks (high to low).",
    ],
    "rows": [
        {"service": "urgent care",
         "expected_output": "urgent care near me\nurgent care open now\nwalk in clinic near me\nsame day urgent care\n24 hour urgent care"},
        {"service": "pediatric clinic",
         "expected_output": "pediatric clinic near me\nkids clinic open now\nchild care clinic nearby\nsame day pediatric care\nwalk in pediatric clinic"},
        {"service": "dental office",
         "expected_output": "dental office near me\nemergency dentist near me\nwalk in dental office\nsame day dental care\ndentist open saturday"},
    ],
    "task_model": "gemini-2.5-flash",
    "reflection_model": "gemini-2.5-flash",
    "judge_model": "gemini-2.5-flash",
    "task_reasoning_effort": "minimal",
    "max_metric_calls": MAX_CALLS,
}

# 2) ZERO-SHOT — no data, just a one-line task description (MetaPromptOptimizer).
ZERO_SHOT = {
    "mode": "zero_shot",
    "name": "keywords_explorer_zs",
    "task_description": "Search engine marketing keyword exploration for healthcare clinics.",
    "user_prompt": "Suggest 5 high-CTR search keywords for a {{service}} clinic.",
    "task_model": "gemini-2.5-flash",
    "reflection_model": "gemini-2.5-flash",
    "judge_model": "gemini-2.5-flash",
    "task_reasoning_effort": "minimal",
    "max_metric_calls": MAX_CALLS,
}

# 3) TRAINING-DATA (labelled) — every row has a gold `expected_output`, scored by a label judge.
TRAINING_DATA = {
    "mode": "training_data",
    "name": "context_qa",
    "user_prompt": "Answer the question based on the context provided.\n\nContext: {{context}}\nQuestion: {{question}}\n\nAnswer:",
    "label_scorer": "correctness",   # or "equivalence"
    "rows": [
        {"context": "The clinic is open Monday to Friday, 8am to 5pm.",
         "question": "Is the clinic open on Sunday?", "expected_output": "No"},
        {"context": "New patients should arrive 15 minutes early to complete intake forms.",
         "question": "How early should new patients arrive?", "expected_output": "15 minutes early"},
        {"context": "We accept most major insurers but not out-of-network HMOs.",
         "question": "Are out-of-network HMOs accepted?", "expected_output": "No"},
    ],
    "task_model": "gemini-2.5-flash",
    "reflection_model": "gemini-2.5-flash",
    "judge_model": "gemini-2.5-flash",
    "max_metric_calls": MAX_CALLS,
}


def _fmt(x):
    return "—" if x is None else f"{float(x):.3f}"


def submit(label, body):
    st, j = call("POST", "/api/v1/optimize", body, user_id=USER_ID, inference_key=INFER_KEY)
    if st != 200 or "job_id" not in j:
        print(f"[{label}] submit FAILED ({st}): {j}")
        return None
    print(f"[{label}] job_id={j['job_id']}")
    return j["job_id"]


def wait_for(label, job_id):
    """Poll until the job reaches a terminal state (or timeout). Prints stage transitions."""
    t0, last_stage = time.time(), None
    while True:
        st, j = call("GET", f"/api/v1/jobs/{job_id}", user_id=USER_ID)
        status, stage = j.get("status"), j.get("stage")
        if stage and stage != last_stage:
            print(f"[{label}]   … {stage}")
            last_stage = stage
        if status in ("done", "error", "interrupted"):
            return j
        if status is None:
            print(f"[{label}]   job not found: {j}")
            return j
        if time.time() - t0 > POLL_TIMEOUT:
            print(f"[{label}]   TIMEOUT after {POLL_TIMEOUT}s (last status={status})")
            return j
        time.sleep(POLL_EVERY)


def run(label, body):
    job_id = submit(label, body)
    if not job_id:
        return
    j = wait_for(label, job_id)
    if j.get("status") == "done":
        rd = (j.get("result") or {}).get("result") or {}
        print(f"[{label}] DONE  initial={_fmt(rd.get('initial_score'))}  "
              f"final={_fmt(rd.get('final_score'))}  Δ={_fmt(rd.get('improvement'))}  "
              f"run_id={(j.get('result') or {}).get('run_id')}")
    else:
        print(f"[{label}] {j.get('status')}: {j.get('error') or ''}")


if __name__ == "__main__":
    for label, body in (("few_shot", FEW_SHOT), ("zero_shot", ZERO_SHOT), ("training_data", TRAINING_DATA)):
        run(label, body)
