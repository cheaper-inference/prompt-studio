"""Durable run history — the log of learnings.

One table, ``Run``, holds BOTH kinds of run:
  * ``kind="optimize"`` — a single optimization job.
  * ``kind="ab"``       — an A/B experiment (control + variant arms).

Shared, queryable fields are columns; the kind-specific detail (the full config
snapshot and the outcome/artifacts) lives in JSON. ``owner`` is the stub for the
future auth layer — a nullable string now, to become a User FK once sign-in lands
(then set it from ``request.user`` and filter history per user).
"""

from __future__ import annotations

import uuid

from django.db import models


class Run(models.Model):
    KIND_OPTIMIZE = "optimize"
    KIND_AB = "ab"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)
    owner = models.CharField(max_length=254, blank=True, default="")   # future: User FK
    job_id = models.CharField(max_length=32, blank=True, default="")   # the background job id (matches /api/job logs)
    kind = models.CharField(max_length=16)                             # "optimize" | "ab"
    label = models.CharField(max_length=200, blank=True, default="")
    status = models.CharField(max_length=16, default="done")           # "done" | "error"
    summary = models.CharField(max_length=300, blank=True, default="") # one-line outcome
    opt_tokens = models.IntegerField(null=True, blank=True)
    duration_s = models.FloatField(null=True, blank=True)
    error = models.TextField(blank=True, default="")
    config = models.JSONField(default=dict)   # the job / control snapshot
    result = models.JSONField(default=dict)   # optimize: result dict; ab: {arms, verdict, holdout}

    class Meta:
        ordering = ["-created_at"]

    def row(self) -> dict:
        """Compact record for the history list."""
        return {
            "id": str(self.id),
            "job_id": self.job_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "kind": self.kind,
            "label": self.label,
            "status": self.status,
            "summary": self.summary,
            "opt_tokens": self.opt_tokens,
            "duration_s": self.duration_s,
        }

    def detail(self) -> dict:
        """Full record for revisiting one run."""
        d = self.row()
        d.update({"owner": self.owner, "error": self.error,
                  "config": self.config, "result": self.result})
        return d


class JobState(models.Model):
    """Durable status of a background job (optimize / A/B / anchor), so polling works
    across processes and survives a server restart — the in-memory registry didn't.
    A leased worker process updates this row; polling is scoped to its owner."""

    id = models.CharField(primary_key=True, max_length=32)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    status = models.CharField(max_length=16, default="running")   # running | done | error
    owner = models.CharField(max_length=254, default="", db_index=True)
    kind = models.CharField(max_length=32, default="optimize")
    idempotency_key = models.CharField(max_length=128, null=True, blank=True)
    payload = models.JSONField(default=dict)
    payload_hash = models.CharField(max_length=64, default="")
    credential_ciphertext = models.TextField(default="", blank=True)
    lease_token = models.CharField(max_length=32, null=True, blank=True)
    lease_until = models.DateTimeField(null=True, blank=True)
    cancel_requested_at = models.DateTimeField(null=True, blank=True)
    checkpoint = models.JSONField(default=dict)
    receipt_storage_bytes = models.PositiveBigIntegerField(default=0)
    stage = models.CharField(max_length=200, default="Preparing")
    result = models.JSONField(null=True, blank=True)
    error = models.TextField(blank=True, default="")

    class Meta:
        ordering = ["-created_at"]
        constraints = [models.UniqueConstraint(fields=["owner", "idempotency_key"], name="unique_owned_job_submission")]


class InferenceCall(models.Model):
    """A durable dispatch barrier and receipt for every task/judge/reflection call."""
    job = models.ForeignKey(JobState, on_delete=models.CASCADE, related_name="inference_calls")
    call_key = models.CharField(max_length=160)
    request_hash = models.CharField(max_length=64)
    phase = models.CharField(max_length=80, default="optimize")
    purpose = models.CharField(max_length=40, default="optimization")
    status = models.CharField(max_length=16, default="dispatching")
    request_id = models.CharField(max_length=128, blank=True, default="")
    model = models.CharField(max_length=200)
    response = models.JSONField(null=True)
    # Reserved before dispatch, reconciled to serialized bytes only on completion.
    storage_bytes = models.PositiveBigIntegerField(default=0)
    http_status = models.IntegerField(null=True)
    usage = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    class Meta:
        constraints = [models.UniqueConstraint(fields=["job", "call_key"], name="unique_job_inference_call")]


class BenchmarkCall(models.Model):
    """One row per benchmark task/judge call: the durable billing record.

    Written once per call (fenced by the worker lease), so live progress JSON stays
    a few counters. The full `calls`/`receipts` lists are assembled from these rows
    exactly once, when the run ends (or when recovery interrupts it)."""
    job = models.ForeignKey(JobState, on_delete=models.CASCADE, related_name="benchmark_calls")
    seq = models.PositiveIntegerField()
    request_id = models.CharField(max_length=128, null=True, blank=True)
    role = models.CharField(max_length=16)
    model = models.CharField(max_length=200, null=True, blank=True)
    contender_model = models.CharField(max_length=200, null=True, blank=True)
    variant = models.CharField(max_length=16, null=True, blank=True)
    case_index = models.PositiveIntegerField(null=True, blank=True)
    completed = models.BooleanField(default=False)
    in_flight = models.BooleanField(default=False)
    observed_at = models.CharField(max_length=40)
    # Set only for a counted (valid, first-seen) receipt.
    receipt_status = models.CharField(max_length=16, blank=True, default="")
    receipt_model = models.CharField(max_length=200, null=True, blank=True)
    billed_cost_usd = models.CharField(max_length=40, blank=True, default="")
    # Wall time from dispatch to response (or error); null for calls recorded without a start.
    duration_ms = models.PositiveIntegerField(null=True, blank=True)
    # When the response (or error) arrived, captured before any bookkeeping lock.
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        constraints = [models.UniqueConstraint(fields=["job", "seq"], name="unique_benchmark_call_seq")]


class Probe(models.Model):
    """One latency measurement of a single model in a single role.

    Diagnostics probe every catalog model across the three roles and write a row
    each; "Model Standings" reads the latest row per (model, role) so users pick
    models on measured latency, not hunches. Measurements are model-intrinsic and
    shared (no owner) — a slow reflection model is slow for everyone.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)
    model = models.CharField(max_length=200)
    role = models.CharField(max_length=16)          # task | judge | reflection
    n = models.IntegerField(default=0)              # calls timed
    mean_s = models.FloatField(null=True, blank=True)
    min_s = models.FloatField(null=True, blank=True)
    max_s = models.FloatField(null=True, blank=True)
    p90_s = models.FloatField(null=True, blank=True)
    ok = models.BooleanField(default=True)
    error = models.TextField(blank=True, default="")

    class Meta:
        ordering = ["-created_at"]


# --------------------------------------------------------------------------- MiniMe
# A user's taste for one task, learned as a QCR (see ps_utils/minime). Labels are the
# source of truth; each revision stores a new QCR version next to its report.

class MiniMeTask(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    owner = models.CharField(max_length=254, db_index=True)
    name = models.CharField(max_length=200)
    definition = models.JSONField(default=dict)   # objective, input, output, requirements, goal
    inputs = models.JSONField(default=list)       # the user's real task inputs (free text)
    deployed_prompt = models.TextField(blank=True, default="")   # production prompt; {{input}} or appended
    settings = models.JSONField(default=dict)     # task_model, judge_model, round_size (used by automatic jobs)

    class Meta:
        ordering = ["-updated_at"]


class MiniMeVersion(models.Model):
    task = models.ForeignKey(MiniMeTask, on_delete=models.CASCADE, related_name="versions")
    version = models.PositiveIntegerField()
    created_at = models.DateTimeField(auto_now_add=True)
    source = models.CharField(max_length=16)          # seed | learn | edit
    qcr = models.JSONField(default=dict)               # QCR.to_dict()
    report = models.JSONField(default=dict)            # metrics, held-out metrics, flip rates, findings, gate
    frozen_at = models.DateTimeField(null=True, blank=True)   # set once the gate passed and the user froze it

    class Meta:
        ordering = ["-version"]
        constraints = [models.UniqueConstraint(fields=["task", "version"], name="unique_minime_version")]


class MiniMeLabel(models.Model):
    SPLITS = ("learn", "heldout")
    task = models.ForeignKey(MiniMeTask, on_delete=models.CASCADE, related_name="labels")
    created_at = models.DateTimeField(auto_now_add=True)
    labelled_at = models.DateTimeField(null=True, blank=True)
    round = models.PositiveIntegerField(default=0)     # 0: uploaded or entered by hand
    position = models.PositiveIntegerField(default=0)  # shuffled order within a round
    source = models.CharField(max_length=16, default="upload")   # upload | generated
    split = models.CharField(max_length=8, default="learn")
    input = models.TextField(blank=True, default="")
    output = models.TextField(blank=True, default="")
    passed = models.BooleanField(null=True)            # None: not labelled yet
    reason = models.TextField(blank=True, default="")
    # The judge's verdict (current QCR) shown to the reviewer on learn rows; hidden on held-out (blind) rows.
    judged_pass = models.BooleanField(null=True)
    judged_failed = models.JSONField(default=list)     # [{id, text}] of the statements that failed
    judged_version = models.PositiveIntegerField(null=True, blank=True)
    # When judged_pass/judged_failed were last written: verdicts from before the judge last changed
    # (settings['judge_since']) are stale. Rows from before this field fall back to created_at.
    judged_at = models.DateTimeField(null=True, blank=True)
    action = models.CharField(max_length=8, blank=True, default="")   # keep | flip on reviewed learn rows
    # Generation intent; never shown to the person labelling.
    kind = models.CharField(max_length=16, blank=True, default="")
    target = models.CharField(max_length=32, blank=True, default="")
    expected_pass = models.BooleanField(null=True)

    class Meta:
        ordering = ["round", "position", "id"]
