from __future__ import annotations
import io
import json
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch
from cryptography.fernet import Fernet
from django.test import TestCase, override_settings
from django.utils import timezone
from . import durable, services
from .models import JobState, InferenceCall
from .metered_transport import MeteredTransport
from .scored import validate_scored_payload

KEY = Fernet.generate_key().decode()

def payload():
    return {"_owner": "alice", "_api_key": "ci_test_private", "_base_url": "https://inference.example.invalid/v1",
        "mode": "training_data", "require_scored_evidence": True, "scorer_revision": "gepa-holdout-v1",
        "user_prompt": "Answer {{question}}", "task_model": "task", "judge_model": "judge", "reflection_model": "reflection",
        "max_metric_calls": 5, "max_cost_usd": "1.00",
        "rows": [{"id": "train1", "input_hash": "1" * 64, "inputs": {"question": "one"}, "expected_output": "1"}, {"id": "train2", "input_hash": "2" * 64, "inputs": {"question": "two"}, "expected_output": "2"}],
        "holdout_rows": [{"id": "held", "input_hash": "3" * 64, "inputs": {"question": "three"}, "expected_output": "3"}]}

@override_settings(PROMPT_STUDIO_CREDENTIAL_KEY=KEY, PROMPT_STUDIO_INFERENCE_BASE_URL="https://inference.example.invalid/v1",
    PROMPT_STUDIO_API_ENABLED=True, PROMPT_STUDIO_SERVICE_TOKEN="service",
    PROMPT_STUDIO_WEBHOOK_ALLOWED_HOSTS="platform.example.invalid")
class DurableJobTests(TestCase):
    headers = {"HTTP_AUTHORIZATION": "Bearer service", "HTTP_X_CI_USER_ID": "alice"}
    @override_settings(PROMPT_STUDIO_WEBHOOK_SECRET="test-only-webhook-secret")
    def test_signed_callback_uses_v1_url_bound_contract_and_omits_scored_payloads(self):
        import hashlib
        import hmac
        url = "https://platform.example.invalid/hooks/prompt-studio"
        job = SimpleNamespace(id="job_123", owner="10000000-0000-0000-0000-000000000001", status="done",
            result={"private_output": "x" * 100000}, error="", _callback_url=url)
        with patch("urllib.request.build_opener") as opener, patch("time.time", return_value=1730000000):
            services._fire_callback(job, notice_only=True)
        request = opener.return_value.open.call_args.args[0]
        expected_body = b'{"job_id": "job_123", "user_id": "10000000-0000-0000-0000-000000000001", "status": "done"}'
        self.assertEqual(request.data, expected_body)
        canonical = "\n".join(["1730000000", "POST", url, hashlib.sha256(expected_body).hexdigest()])
        self.assertEqual(request.get_header("X-ps-signature"), hmac.new(b"test-only-webhook-secret", canonical.encode(), hashlib.sha256).hexdigest())
        self.assertEqual(request.get_header("X-ps-timestamp"), "1730000000")
        self.assertEqual(opener.return_value.open.call_args.kwargs["timeout"], 5)

    def test_durable_scored_completion_sends_owner_scoped_notice_after_commit_and_secret_erasure(self):
        data = {**payload(), "_callback_url": "https://platform.example.invalid/hooks/prompt-studio"}
        row = durable.enqueue("optimize", data, "callback-job")
        def complete(job, _payload):
            JobState.objects.filter(pk=job.id, lease_token=job.lease_token).update(status="done", result={"complete": True})
        def notice(job, notice_only):
            stored = JobState.objects.get(pk=job.id)
            self.assertEqual(stored.status, "done")
            self.assertEqual(stored.credential_ciphertext, "")
            self.assertIsNone(stored.lease_token)
            self.assertEqual(job.owner, "alice")
            self.assertTrue(notice_only)
        with patch("optimizer.scored.run_scored", side_effect=complete), patch("optimizer.services._fire_callback", side_effect=notice) as callback:
            self.assertTrue(durable.process_one())
        callback.assert_called_once()
        self.assertEqual(callback.call_args.args[0].id, row.id)
    def test_enqueue_is_idempotent_before_any_dispatch_and_secrets_are_encrypted(self):
        a = durable.enqueue("optimize", payload(), "stable")
        b = durable.enqueue("optimize", payload(), "stable")
        self.assertEqual(a.id, b.id)
        self.assertEqual(a.status, "queued")
        self.assertNotIn("ci_test_private", json.dumps(a.payload))
        self.assertNotIn("ci_test_private", a.credential_ciphertext)
        self.assertEqual(durable.decrypt_secret(a.credential_ciphertext)["_api_key"], "ci_test_private")
        with self.assertRaises(durable.IdempotencyConflict):
            durable.enqueue("optimize", {**payload(), "max_cost_usd": "2"}, "stable")
        self.assertEqual(JobState.objects.count(), 1)
        self.assertEqual(InferenceCall.objects.count(), 0)

    def test_poll_and_cancel_are_owner_scoped(self):
        row = durable.enqueue("optimize", payload(), "stable")
        other = {**self.headers, "HTTP_X_CI_USER_ID": "bob"}
        self.assertEqual(self.client.get(f"/api/v1/jobs/{row.id}", **other).status_code, 404)
        self.assertEqual(self.client.post(f"/api/v1/jobs/{row.id}/cancel", **other).status_code, 404)
        self.assertEqual(self.client.get(f"/api/v1/jobs/{row.id}", **self.headers).json()["status"], "running")
        self.assertEqual(self.client.post(f"/api/v1/jobs/{row.id}/cancel", **self.headers).status_code, 200)
        row.refresh_from_db()
        self.assertEqual(row.status, "cancelled")
        self.assertEqual(row.credential_ciphertext, "")
        self.assertIsNone(durable.claim())

    def test_atomic_lease_reclaim_fences_old_worker_and_preserves_checkpoint(self):
        row = durable.enqueue("optimize", payload(), "stable")
        first = durable.claim()
        self.assertIsNone(durable.claim())
        durable.checkpoint(row.id, first.lease_token, "optimized", {"answer": "saved"})
        JobState.objects.filter(pk=row.id).update(lease_until=timezone.now() - timedelta(seconds=1))
        resumed = durable.claim()
        self.assertNotEqual(first.lease_token, resumed.lease_token)
        self.assertEqual(resumed.checkpoint["optimized"], {"answer": "saved"})
        with self.assertRaises(durable.LeaseLost):
            durable.checkpoint(row.id, first.lease_token, "bad", {})
        with self.assertRaises(durable.LeaseLost):
            services.Job(id=row.id, owner="alice", lease_token=first.lease_token)

    def test_missing_encryption_key_prevents_job_admission(self):
        with override_settings(PROMPT_STUDIO_CREDENTIAL_KEY=""):
            with self.assertRaises(ValueError):
                durable.enqueue("optimize", payload(), "stable")
        self.assertEqual(JobState.objects.count(), 0)

    def test_capability_handshake_reports_the_scored_contract(self):
        result = self.client.get("/api/v1/capabilities", **self.headers)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["durable_scored_jobs"], 1)
        self.assertTrue(result.json()["metered_inference"])
        self.assertEqual(result.json()["scorer_revisions"], ["gepa-holdout-v1"])

    def test_scored_admission_does_not_run_a_daemon_job(self):
        p = payload()
        for key in ("_owner", "_api_key", "_base_url"):
            p.pop(key)
        p["base_url"] = "https://inference.example.invalid/v1"
        headers = {**self.headers, "HTTP_X_CI_INFERENCE_KEY": "ci_test_private", "HTTP_IDEMPOTENCY_KEY": "stable"}
        with patch("threading.Thread.start") as start:
            response = self.client.post("/api/v1/optimize", data=json.dumps(p), content_type="application/json", **headers)
        self.assertEqual(response.status_code, 202)
        start.assert_not_called()
        second = self.client.post("/api/v1/optimize", data=json.dumps(p), content_type="application/json", **headers)
        self.assertEqual(response.json()["job_id"], second.json()["job_id"])

    def test_validation_rejects_overlap_unknown_scorers_and_foreign_origins(self):
        validate_scored_payload(payload())
        for override in ({"_base_url": "https://attacker.invalid/v1"}, {"scorer_revision": "fake"}, {"max_cost_usd": "NaN"},
                         {"holdout_rows": [payload()["rows"][0]]}):
            with self.assertRaises(ValueError):
                validate_scored_payload({**payload(), **override})

    def transport(self):
        row = durable.enqueue("optimize", payload(), "stable")
        return MeteredTransport(durable.claim(), payload())

    def response(self, request_id="request-1"):
        response = io.BytesIO(json.dumps({"choices": [{"message": {"content": "yes"}}], "usage": {"prompt_tokens": 5, "completion_tokens": 1}}).encode())
        response.status = 200
        response.headers = {"x-ci-request-id": request_id} if request_id else {}
        return response

    def test_every_role_is_metered_and_completed_responses_replay_without_spending(self):
        transport = self.transport()
        with patch("optimizer.metered_transport.build_opener") as opener:
            opener.return_value.open.side_effect = [self.response("task-id"), self.response("judge-id"), self.response("reflection-id")]
            for model in ("task", "judge", "reflection"):
                transport.execute({"model": model, "messages": [{"role": "user", "content": "question"}]})
            resumed = MeteredTransport(transport.job, payload())
            for model in ("task", "judge", "reflection"):
                resumed.execute({"model": model, "messages": [{"role": "user", "content": "question"}]})
            self.assertEqual(opener.return_value.open.call_count, 3)
        self.assertEqual(set(InferenceCall.objects.values_list("request_id", flat=True)), {"task-id", "judge-id", "reflection-id"})

    def test_ambiguous_dispatch_is_never_reissued_and_missing_receipts_are_not_free(self):
        transport = self.transport()
        body = {"model": "task", "messages": [{"role": "user", "content": "question"}]}
        with patch("optimizer.metered_transport.build_opener") as opener:
            opener.return_value.open.return_value = self.response(None)
            with self.assertRaises(durable.UncertainInference):
                transport.execute(body)
            resumed = MeteredTransport(transport.job, payload())
            with self.assertRaises(durable.UncertainInference):
                resumed.execute(body)
            with self.assertRaises(durable.UncertainInference):
                resumed.execute({"model": "task", "messages": [{"role": "user", "content": "different resumed candidate"}]})
            self.assertEqual(opener.return_value.open.call_count, 1)
        self.assertEqual(InferenceCall.objects.get().status, "dispatching")

    def test_cancel_stops_before_next_inference_dispatch(self):
        transport = self.transport()
        durable.cancel(transport.job.id, "alice")
        with patch("optimizer.metered_transport.build_opener") as opener:
            with self.assertRaises(durable.JobCancelled):
                transport.execute({"model": "task"})
            opener.assert_not_called()
        self.assertEqual(InferenceCall.objects.count(), 0)

    def test_worker_failure_is_durable_and_removes_credential(self):
        row = durable.enqueue("optimize", payload(), "stable")
        with patch("optimizer.scored.run_scored", side_effect=ValueError("bad config")):
            self.assertTrue(durable.process_one())
        row.refresh_from_db()
        self.assertEqual(row.status, "error")
        self.assertEqual(row.error, "ValueError")
        self.assertEqual(row.credential_ciphertext, "")

    def test_legacy_expired_job_is_interrupted_instead_of_repeating_paid_calls(self):
        row = durable.enqueue("optimize", {"_owner": "alice", "mode": "zero_shot"})
        durable.claim()
        JobState.objects.filter(pk=row.id).update(lease_until=timezone.now() - timedelta(seconds=1))
        self.assertIsNone(durable.claim())
        row.refresh_from_db()
        self.assertEqual(row.status, "interrupted")

    def test_the_short_lane_skips_learning_runs(self):
        learn = durable.enqueue("minime_learn", {"_owner": "alice", "task_id": "t"})
        score = durable.enqueue("minime_learn", {"_owner": "alice", "task_id": "t", "score_only": True})
        bench = durable.enqueue("optimize", {"_owner": "alice", "mode": "zero_shot"})
        rnd = durable.enqueue("minime_round", {"_owner": "alice", "task_id": "t"})
        self.assertEqual(durable.claim(lane="short").id, score.id)      # oldest short job, past the learning run
        self.assertEqual(durable.claim(lane="short").id, rnd.id)
        self.assertIsNone(durable.claim(lane="short"))                  # learning and other jobs are not its lane
        self.assertEqual(durable.claim(lane="any").id, learn.id)
        self.assertEqual(durable.claim(lane="any").id, bench.id)

    def test_service_lanes(self):
        import importlib.util, os, pathlib
        spec = importlib.util.spec_from_file_location("run_service", pathlib.Path(__file__).resolve().parents[2] / "run_service.py")
        rs = importlib.util.module_from_spec(spec); spec.loader.exec_module(rs)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GP_JOB_LANES", None)
            self.assertEqual((rs.job_lanes(1), rs.job_lanes(2), rs.job_lanes(3)), (["any"], ["short", "any"], ["short", "any", "any"]))
            os.environ["GP_JOB_LANES"] = "any,any"
            self.assertEqual(rs.job_lanes(2), ["any", "any"])

    def test_an_expired_minime_job_pauses_its_task_instead_of_resuming(self):
        from optimizer.models import MiniMeTask
        task = MiniMeTask.objects.create(owner="alice", name="t", settings={"optimizing": True})
        row = durable.enqueue("minime_learn", {"_owner": "alice", "task_id": str(task.id)})
        durable.claim()
        JobState.objects.filter(pk=row.id).update(lease_until=timezone.now() - timedelta(seconds=1))
        self.assertIsNone(durable.claim())                       # not resumed on its own
        row.refresh_from_db(); task.refresh_from_db()
        self.assertEqual((row.status, row.stage, task.settings["optimizing"]), ("cancelled", "Paused", False))

    def test_purpose_is_preserved_even_when_role_models_are_identical(self):
        transport = self.transport()
        body = {"model": "task", "messages": [{"role": "user", "content": "question"}]}
        with patch("optimizer.metered_transport.build_opener") as opener:
            opener.return_value.open.side_effect = [self.response("a"), self.response("b")]
            transport.execute(body, "training_generation")
            transport.execute(body, "training_judge")
        self.assertEqual(set(InferenceCall.objects.values_list("purpose", flat=True)), {"training_generation", "training_judge"})

class HeldoutBehaviorTests(TestCase):
    def test_heldout_calls_fill_nested_inputs_meter_task_usage_and_gate_constraints(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from prompt_optimizer import PromptOptimizer, OptimizationConfig, Prompt, UsageMeter
        create = Mock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="not-json"))], usage={"prompt_tokens": 5, "completion_tokens": 2}))
        endpoint = SimpleNamespace(client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), task_model="task")
        config = OptimizationConfig(name="fixture", prompts=[Prompt(role="user", template="Answer {{question}}")], mode="training_data",
            task_model="task", training_data=payload()["rows"], output_constraints={"format": "json"})
        optimizer = PromptOptimizer(endpoint=endpoint)
        optimizer._usage = UsageMeter()
        judge = Mock(return_value=1.0)
        optimizer._build_label_scorer = Mock(return_value=judge)
        result = optimizer.evaluate_prompts(config, config.prompts, payload()["holdout_rows"])
        self.assertEqual(result["scores"], [0.0])
        self.assertEqual(create.call_args.kwargs["messages"][0]["content"], "Answer three")
        judge.assert_not_called()
        self.assertEqual(optimizer._usage.as_dict()["by_role"]["task"]["input"], 5)

    def test_normalized_labeled_training_rows_retain_gold_expectations(self):
        from types import SimpleNamespace
        from prompt_optimizer import PromptOptimizer, OptimizationConfig, Prompt
        config = OptimizationConfig(name="fixture", prompts=[Prompt(role="user", template="Answer {{question}}")], mode="training_data", training_data=payload()["rows"])
        optimizer = PromptOptimizer(endpoint=SimpleNamespace())
        rows = optimizer._build_train_data(config)
        self.assertEqual(rows[0]["expectations"]["expected_response"], "1")

@override_settings(PROMPT_STUDIO_CREDENTIAL_KEY=KEY, PROMPT_STUDIO_INFERENCE_BASE_URL="https://inference.example.invalid/v1")
class ScoredJobIntegrationTests(TestCase):
    def test_persists_disjoint_per_case_evidence_and_every_request_receipt(self):
        from contextlib import contextmanager
        from types import SimpleNamespace
        from unittest.mock import Mock
        from .scored import run_scored
        row = durable.enqueue("optimize", payload(), "stable")
        row = durable.claim()
        job = services.Job(id=row.id, owner=row.owner, lease_token=row.lease_token)
        active = []
        @contextmanager
        def endpoint(transport):
            active.append(transport)
            yield SimpleNamespace()
        optimized_prompt = SimpleNamespace(role="user", template="Carefully answer {{question}}")
        result = SimpleNamespace(prompts=[optimized_prompt], to_dict=lambda: {"final_score": 1}, to_deployable=lambda: {"model": "task", "messages": []})
        optimizer = Mock()
        optimizer.optimize.return_value = result
        def evaluate(config, prompts, rows):
            self.assertEqual(rows, payload()["holdout_rows"])
            self.assertEqual(config.training_data, payload()["rows"])
            phase = active[0].phase
            InferenceCall.objects.create(job_id=job.id, call_key=phase + ":mock:0", request_hash="hash", phase=phase,
                purpose="holdout_generation", model="task", status="complete", request_id=phase + "-id", http_status=200, response={})
            return {"scores": [.5 if phase.startswith("baseline") else 1], "outputs": ["3"]}
        optimizer.evaluate_prompts.side_effect = evaluate
        with patch("optimizer.metered_transport.MeteredTransport.endpoint", endpoint), patch("prompt_optimizer.PromptOptimizer", return_value=optimizer):
            run_scored(job, payload())
        row.refresh_from_db()
        self.assertEqual(row.status, "done")
        evidence = row.result["scored_evidence"]
        self.assertTrue(evidence["completion_known"])
        self.assertFalse(evidence["cost_complete"])
        self.assertEqual(evidence["holdout"][0]["id"], "held")
        self.assertEqual(evidence["holdout"][0]["baseline_score"], .5)
        self.assertEqual(evidence["holdout"][0]["treatment_score"], 1)
        self.assertEqual(evidence["holdout"][0]["baseline_request_id"], "baseline_0-id")
        self.assertEqual(evidence["holdout"][0]["treatment_request_id"], "treatment_0-id")
        self.assertEqual(evidence["inference_request_ids"], ["baseline_0-id", "treatment_0-id"])
        self.assertIn("optimized", row.checkpoint)
        self.assertIn("holdout_0", row.checkpoint)
