"""Django views — thin JSON adapters over optimizer.services, behind auth."""

from __future__ import annotations

import hmac
import json
from functools import wraps

import sentry_sdk
from django.conf import settings
from django.contrib.auth import authenticate, login as auth_login, logout as auth_logout
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.http import HttpRequest, HttpResponseNotAllowed, JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.csrf import csrf_exempt, ensure_csrf_cookie

from . import services


def _json_body(request: HttpRequest) -> dict:
    try:
        value = json.loads(request.body or b"{}")
        return value if isinstance(value, dict) else {}
    except (ValueError, TypeError):
        return {}


def api_login_required(view):
    """API guard: 401 JSON instead of a login redirect for unauthenticated calls."""
    @wraps(view)
    def wrapped(request, *a, **k):
        if not request.user.is_authenticated:
            return JsonResponse({"error": "authentication required"}, status=401)
        return view(request, *a, **k)
    return wrapped


def service_api_required(view):
    """Bearer-only guard for trusted server-to-server calls.

    It intentionally does not accept browser sessions: the service routes are
    CSRF-exempt, so accepting cookies here would weaken the web UI's boundary.
    """
    @wraps(view)
    def wrapped(request, *a, **k):
        expected = settings.GEPA_SERVICE_API_KEY
        if not expected:
            sentry_sdk.capture_message("GEPA_SERVICE_API_KEY is not configured", level="error")
            return JsonResponse({"error": "service API is not configured"}, status=503)
        authorization = request.headers.get("Authorization", "")
        scheme, _, supplied = authorization.partition(" ")
        valid = scheme.lower() == "bearer" and hmac.compare_digest(
            supplied.encode("utf-8"), expected.encode("utf-8")
        )
        if not valid:
            return JsonResponse({"error": "invalid service credential"}, status=401)
        return view(request, *a, **k)
    return wrapped


def _job_response(job) -> JsonResponse:
    body = {"status": job.status, "stage": job.stage}
    if job.status == "done":
        body.update(job.result or {})
    elif job.status == "error":
        body["error"] = job.error
    return JsonResponse(body)


# --------------------------------------------------------------------------- auth
def signup(request: HttpRequest):
    if request.user.is_authenticated:
        return redirect("index")
    if not settings.ALLOW_SIGNUP:            # self-signup disabled — accounts are issued manually
        return redirect("login")
    error = ""
    if request.method == "POST":
        u = (request.POST.get("username") or "").strip()
        p = request.POST.get("password") or ""
        if not u or not p:
            error = "Username and password are required."
        elif User.objects.filter(username=u).exists():
            error = "That username is taken."
        else:
            auth_login(request, User.objects.create_user(username=u, password=p))
            return redirect("index")
    return render(request, "optimizer/auth.html", {"mode": "signup", "error": error, "allow_signup": True})


def login_view(request: HttpRequest):
    if request.user.is_authenticated:
        return redirect("index")
    error = ""
    if request.method == "POST":
        u = (request.POST.get("username") or "").strip()
        user = authenticate(request, username=u, password=request.POST.get("password") or "")
        if user is None:
            error = "Invalid username or password."
        else:
            auth_login(request, user)
            return redirect(request.GET.get("next") or "index")
    return render(request, "optimizer/auth.html", {"mode": "login", "error": error, "allow_signup": settings.ALLOW_SIGNUP})


def logout_view(request: HttpRequest):
    auth_logout(request)
    return redirect("login")


# --------------------------------------------------------------------------- app
def healthz(request: HttpRequest):
    """Public liveness probe for the platform healthcheck (Railway et al.). No auth,
    no DB, no endpoint call — a 200 means the process is up and serving. Point the
    Railway 'Healthcheck Path' here (/healthz). App + endpoint status, which the UI
    needs, stays behind auth at /api/health."""
    return JsonResponse({"status": "ok"})


@login_required
@ensure_csrf_cookie
def index(request: HttpRequest):
    return render(request, "optimizer/index.html")


@api_login_required
def me(request: HttpRequest):
    return JsonResponse({"username": request.user.username})


@api_login_required
def health(request: HttpRequest):
    status = services.endpoint_status()
    return JsonResponse({
        "endpoint_available": status["available"],
        "detail": status["detail"],
        "missing": status["missing"],
    })


@api_login_required
def models(request: HttpRequest):
    return JsonResponse({"models": services.fetch_models()})


@api_login_required
def variables(request: HttpRequest):
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])
    p = _json_body(request)
    return JsonResponse({"variables": services.variables_of(
        p.get("system_prompt", ""), p.get("user_prompt", ""))})


@api_login_required
def optimize(request: HttpRequest):
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])
    payload = _json_body(request)
    payload["_owner"] = request.user.username
    return JsonResponse({"job_id": services.start_job(payload).id})


@api_login_required
def anchor(request: HttpRequest):
    """Optional 2nd pass: optimize for the benchmark anchor and compute the trade-off.
    Client posts the same payload plus _target (1st pass model/score/prompt)."""
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])
    payload = _json_body(request)
    payload["_owner"] = request.user.username
    return JsonResponse({"job_id": services.start_anchor(payload).id})


@api_login_required
def experiment(request: HttpRequest):
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])
    payload = _json_body(request)
    payload["_owner"] = request.user.username
    return JsonResponse({"job_id": services.start_experiment(payload).id})


@api_login_required
def runs(request: HttpRequest):
    from .models import Run
    qs = Run.objects.filter(owner=request.user.username)[:200]
    return JsonResponse({"runs": [r.row() for r in qs]})


@api_login_required
def run_detail(request: HttpRequest, run_id: str):
    from .models import Run
    try:
        r = Run.objects.get(pk=run_id, owner=request.user.username)
    except Exception:  # noqa: BLE001
        return JsonResponse({"error": "run not found"}, status=404)
    return JsonResponse(r.detail())


@api_login_required
def diagnostics(request: HttpRequest):
    """Kick off a latency sweep (all catalog models × 3 roles) as a background job."""
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])
    payload = _json_body(request)
    payload["_owner"] = request.user.username
    return JsonResponse({"job_id": services.start_diagnostics(payload).id})


@api_login_required
def standings(request: HttpRequest):
    return JsonResponse(services.standings())


@api_login_required
def job_status(request: HttpRequest, job_id: str):
    job = services.get_job_state(job_id, request.user.username)   # DB-backed: any worker/process can serve this
    if not job:
        return JsonResponse({"error": "unknown job"}, status=404)
    status = "error" if job["status"] in ("interrupted", "cancelled") else job["status"]
    body = {"status": status, "stage": job["stage"]}
    if status == "done":
        body.update(job["result"] or {})
    elif status == "error":
        body["error"] = job["error"]
    return JsonResponse(body)


# --------------------------------------------------------------------------- MiniMe
def _minime(handler):
    """Session-auth JSON API for one owner's MiniMe tasks; MiniMeError -> 400."""
    @api_login_required
    @wraps(handler)
    def wrapped(request, *a, **k):
        from .minime import MiniMeError
        try:
            return handler(request, *a, **k)
        except MiniMeError as error:
            return JsonResponse({"error": str(error)}, status=400)
    return wrapped


@_minime
def minime_tasks(request: HttpRequest):
    from . import minime as mm
    from .models import MiniMeTask
    owner = request.user.username
    if request.method == "POST":
        p = _json_body(request)
        task = mm.create_task(owner, p.get("name"), p.get("definition"), p.get("inputs"),
                              p.get("deployed_prompt"), p.get("settings"))
        return JsonResponse(mm.summary(task))
    return JsonResponse({"tasks": [{"id": str(t.id), "name": t.name, "updated_at": t.updated_at.isoformat()}
                                   for t in MiniMeTask.objects.filter(owner=owner)[:200]]})


@_minime
def minime_task(request: HttpRequest, task_id):
    from . import minime as mm
    task = mm.get_task(task_id, request.user.username)
    if request.method == "DELETE":
        mm.delete_task(task)
        return JsonResponse({"deleted": str(task_id)})
    if request.method == "POST":
        p = _json_body(request)
        mm.update_task(task, p.get("deployed_prompt"), p.get("inputs"), p.get("settings"), p.get("input_pool"),
                       p.get("name"))
    return JsonResponse(mm.summary(task))


@_minime
def minime_describe(request: HttpRequest, task_id):
    """Change the task description: mode "reword" keeps this task, "clone" starts a new one."""
    from . import minime as mm
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])
    owner = request.user.username
    task = mm.get_task(task_id, owner)
    p = _json_body(request)
    if p.get("mode") == "clone":
        new = mm.clone_task(task, owner, p.get("definition"), p.get("name"))
        return JsonResponse(mm.summary(new))
    if p.get("mode") != "reword":
        raise mm.MiniMeError("mode must be reword or clone")
    mm.reword_task(task, p.get("definition"))
    return JsonResponse(mm.summary(task))


@_minime
def minime_qcr(request: HttpRequest, task_id):
    from . import minime as mm
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])
    task = mm.get_task(task_id, request.user.username)
    mm.edit_qcr(task, _json_body(request).get("statements"))
    mm.run_rejudge(task, request.user.username)
    return JsonResponse(mm.summary(task))


@_minime
def minime_apply(request: HttpRequest, task_id):
    from . import minime as mm
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])
    task = mm.get_task(task_id, request.user.username)
    mm.apply_proposal(task, _json_body(request).get("at"))
    mm.run_rejudge(task, request.user.username)
    return JsonResponse(mm.summary(task))


@_minime
def minime_undo(request: HttpRequest, task_id):
    from . import minime as mm
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])
    task = mm.get_task(task_id, request.user.username)
    mm.undo_change(task, _json_body(request).get("at"))
    mm.run_rejudge(task, request.user.username)
    return JsonResponse(mm.summary(task))


@_minime
def minime_stop(request: HttpRequest, task_id):
    from . import minime as mm
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])
    task = mm.get_task(task_id, request.user.username)
    mm.stop_jobs(task)
    return JsonResponse(mm.summary(task))


@_minime
def minime_dismiss_tip(request: HttpRequest, task_id):
    from . import minime as mm
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])
    task = mm.get_task(task_id, request.user.username)
    mm.dismiss_tip(task, _json_body(request).get("at"))
    return JsonResponse(mm.summary(task))


@_minime
def minime_labels(request: HttpRequest, task_id):
    from . import minime as mm
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])
    task = mm.get_task(task_id, request.user.username)
    p = _json_body(request)
    n = mm.upload_labels(task, p.get("csv") or "", p.get("input_col") or "input",
                         p.get("output_col") or "output", p.get("split") or "learn")
    return JsonResponse({"added": n})


@_minime
def minime_label(request: HttpRequest, task_id, label_id: int):
    from . import minime as mm
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])
    task = mm.get_task(task_id, request.user.username)
    p = _json_body(request)
    mm.review(task, label_id, p.get("action"), p.get("verdict"), p.get("reason"), p.get("shown_pass"))
    mm.pause_if_ready(task)              # a blind check can complete Ready: then pause, start nothing
    mm.autostart(task, request.user.username)
    return JsonResponse({"next": mm.next_candidate(task), "task": mm.summary(task)})


@_minime
def minime_queue(request: HttpRequest, task_id):
    from . import minime as mm
    task = mm.get_task(task_id, request.user.username)
    if request.method == "POST":          # nothing waiting and nothing running: start the loop again
        mm.autostart(task, request.user.username)
    mm.ensure_scored(task, request.user.username)   # versions behind on the blind checks: catch up
    return JsonResponse({"next": mm.next_candidate(task), "task": mm.summary(task)})


@_minime
def minime_job(request: HttpRequest, task_id, action: str):
    from . import minime as mm
    if action not in ("round", "learn", "start"):
        return JsonResponse({"error": "unknown action"}, status=404)
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])
    task = mm.get_task(task_id, request.user.username)
    if action == "start":                 # Start optimizing: resume everything pending
        mm.start_optimizing(task, request.user.username)
        return JsonResponse(mm.summary(task))
    mm._update_settings(task, lambda s: {**s, "optimizing": True})   # Retry / Get more outputs: an explicit start
    job = mm.start(f"minime_{action}", request.user.username, {"task_id": str(task.id)})
    return JsonResponse({"job_id": job.id})
