from django.urls import path

from optimizer import api_views, views

urlpatterns = [
    path("healthz", views.healthz, name="healthz"),   # public liveness for platform healthcheck
    path("", views.index, name="index"),
    path("login/", views.login_view, name="login"),
    path("signup/", views.signup, name="signup"),
    path("logout/", views.logout_view, name="logout"),
    path("api/me", views.me, name="me"),
    path("api/health", views.health, name="health"),
    path("api/models", views.models, name="models"),
    path("api/variables", views.variables, name="variables"),
    path("api/optimize", views.optimize, name="optimize"),
    path("api/anchor", views.anchor, name="anchor"),
    path("api/experiment", views.experiment, name="experiment"),
    path("api/runs", views.runs, name="runs"),
    path("api/runs/<uuid:run_id>", views.run_detail, name="run_detail"),
    path("api/diagnostics", views.diagnostics, name="diagnostics"),
    path("api/standings", views.standings, name="standings"),
    path("api/job/<str:job_id>", views.job_status, name="job_status"),
    path("api/minime/tasks", views.minime_tasks),
    path("api/minime/tasks/<uuid:task_id>", views.minime_task),
    path("api/minime/tasks/<uuid:task_id>/qcr", views.minime_qcr),
    path("api/minime/tasks/<uuid:task_id>/labels", views.minime_labels),
    path("api/minime/tasks/<uuid:task_id>/labels/<int:label_id>", views.minime_label),
    path("api/minime/tasks/<uuid:task_id>/queue", views.minime_queue),
    path("api/minime/tasks/<uuid:task_id>/apply", views.minime_apply),
    path("api/minime/tasks/<uuid:task_id>/describe", views.minime_describe),
    path("api/minime/tasks/<uuid:task_id>/undo", views.minime_undo),
    path("api/minime/tasks/<uuid:task_id>/tip", views.minime_dismiss_tip),
    path("api/minime/tasks/<uuid:task_id>/stop", views.minime_stop),
    path("api/minime/tasks/<uuid:task_id>/<str:action>", views.minime_job),

    # --- Cheaper Inference service API (v1) — bearer token + X-CI-User-Id gated ---
    path("api/v1/optimize", api_views.optimize),
    path("api/v1/capabilities", api_views.capabilities),
    path("api/v1/benchmarks", api_views.benchmark),
    path("api/v1/benchmarks/draft-criteria", api_views.draft_criteria),
    path("api/v1/experiments", api_views.experiment),
    path("api/v1/anchor", api_views.anchor),
    path("api/v1/diagnostics", api_views.diagnostics),
    path("api/v1/standings", api_views.standings),
    path("api/v1/jobs/<str:job_id>", api_views.job),
    path("api/v1/jobs/<str:job_id>/cancel", api_views.cancel_job),
    path("api/v1/runs", api_views.runs),
    path("api/v1/runs/<uuid:run_id>", api_views.run_detail),
]
