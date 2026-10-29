"""API admission contracts with durable SQLite state and mocked runners only."""
from __future__ import annotations

import json
from unittest.mock import patch

from cryptography.fernet import Fernet
from django.contrib.auth.models import User
from django.test import TestCase, override_settings

from . import durable
from .models import JobState


BASE_URL = "https://inference.example.invalid/v1"


def scored_payload():
    return {
        "mode": "training_data", "require_scored_evidence": True,
        "scorer_revision": "gepa-holdout-v1", "user_prompt": "Answer {{question}}",
        "task_model": "task", "judge_model": "judge", "reflection_model": "reflection",
        "max_metric_calls": 4, "max_cost_usd": "1.00",
        "rows": [
            {"id": "a", "inputs": {"question": "one"}, "expected_output": "1", "input_hash": "1" * 64},
            {"id": "b", "inputs": {"question": "two"}, "expected_output": "2", "input_hash": "2" * 64},
        ],
        "holdout_rows": [
            {"id": "c", "inputs": {"question": "three"}, "expected_output": "3", "input_hash": "3" * 64},
        ],
    }


@override_settings(
    PROMPT_STUDIO_API_ENABLED=True, PROMPT_STUDIO_SERVICE_TOKEN="fixture-service-token",
    PROMPT_STUDIO_CREDENTIAL_KEY=Fernet.generate_key().decode(),
    PROMPT_STUDIO_INFERENCE_BASE_URL=BASE_URL + "/",
)
class ScoredApiAdmissionTests(TestCase):
    headers = {
        "HTTP_AUTHORIZATION": "Bearer fixture-service-token", "HTTP_X_CI_USER_ID": "alice",
        "HTTP_X_CI_INFERENCE_KEY": "fixture-inference-key", "HTTP_IDEMPOTENCY_KEY": "fixture-once",
    }

    def post(self, payload):
        return self.client.post("/api/v1/optimize", json.dumps(payload), content_type="application/json", **self.headers)

    def test_omitted_base_url_queues_the_configured_normalized_default(self):
        payload = {**scored_payload(), "_owner": "forged-owner", "_base_url": "https://untrusted.example/v1"}
        response = self.post(payload)
        self.assertEqual(response.status_code, 202, response.content)
        row = JobState.objects.get(pk=response.json()["job_id"])
        self.assertEqual(row.status, "queued")
        self.assertEqual(row.owner, "alice")
        self.assertEqual(row.payload["_owner"], "alice")
        self.assertEqual(row.payload["_base_url"], BASE_URL)
        self.assertNotIn("base_url", row.payload)
        self.assertNotIn("_api_key", row.payload)
        self.assertEqual(durable.decrypt_secret(row.credential_ciphertext)["_api_key"], "fixture-inference-key")
        # The frozen endpoint survives later configuration changes and restarts.
        with override_settings(PROMPT_STUDIO_INFERENCE_BASE_URL="https://new.example/v1"):
            row.refresh_from_db()
            self.assertEqual(row.payload["_base_url"], BASE_URL)

    def test_default_and_explicit_normalized_endpoint_reuse_the_same_job(self):
        first = self.post(scored_payload())
        second = self.post({**scored_payload(), "base_url": BASE_URL + "/"})
        self.assertEqual(first.status_code, 202, first.content)
        self.assertEqual(second.status_code, 202, second.content)
        self.assertEqual(first.json()["job_id"], second.json()["job_id"])
        self.assertEqual(JobState.objects.count(), 1)

    def test_nonconfigured_or_malformed_explicit_endpoint_is_rejected_before_queueing(self):
        for base in ("https://untrusted.example/v1", "http://inference.example.invalid/v1", 123, {}):
            with self.subTest(base=base):
                response = self.post({**scored_payload(), "base_url": base})
                self.assertEqual(response.status_code, 400, response.content)
        self.assertFalse(JobState.objects.exists())


class BrowserAnchorOwnershipTests(TestCase):
    def test_authenticated_anchor_owner_reaches_the_durable_runner(self):
        user = User.objects.create_user(username="alice", password="unused-fixture-password")
        self.client.force_login(user)
        response = self.client.post("/api/anchor", json.dumps({
            "mode": "zero_shot", "_owner": "forged-owner", "_target": {"model": "target"},
        }), content_type="application/json")
        self.assertEqual(response.status_code, 200, response.content)
        row = JobState.objects.get(pk=response.json()["job_id"])
        self.assertEqual(row.owner, "alice")
        self.assertEqual(row.payload["_owner"], "alice")

        def finish(job, payload):
            self.assertEqual(job.owner, "alice")
            self.assertEqual(payload["_owner"], "alice")
            self.assertEqual(payload["_target"], {"model": "target"})
            job.status = "done"

        with patch("optimizer.services._run_anchor_job", side_effect=finish) as runner, patch("threading.Thread"):
            self.assertTrue(durable.process_one())
        runner.assert_called_once()
        row.refresh_from_db()
        self.assertEqual(row.status, "done")

    def test_anonymous_anchor_does_not_start_work(self):
        response = self.client.post("/api/anchor", json.dumps({"mode": "zero_shot"}), content_type="application/json")
        self.assertEqual(response.status_code, 401)
        self.assertFalse(JobState.objects.exists())
