from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import Client, TestCase, override_settings

from .models import JobState
from .services import build_config


@override_settings(
    PROMPT_STUDIO_API_ENABLED=True,
    PROMPT_STUDIO_SERVICE_TOKEN="test-service-key",
    PROMPT_STUDIO_API_IP_ALLOWLIST="",
    PROMPT_STUDIO_INFERENCE_BASE_URL="https://inference.example.invalid/v1",
)
class ServiceApiTests(TestCase):
    headers = {
        "HTTP_AUTHORIZATION": "Bearer test-service-key",
        "HTTP_X_CI_USER_ID": "service-user",
    }

    def test_service_api_rejects_missing_or_wrong_bearer_token(self):
        self.assertEqual(self.client.get("/api/v1/capabilities").status_code, 401)
        self.assertEqual(
            self.client.get(
                "/api/v1/capabilities",
                HTTP_AUTHORIZATION="Bearer wrong-key",
                HTTP_X_CI_USER_ID="service-user",
            ).status_code,
            401,
        )
        self.assertEqual(
            self.client.get(
                "/api/v1/capabilities",
                HTTP_AUTHORIZATION="Bearer test-service-key",
            ).status_code,
            401,
        )
        self.assertEqual(
            self.client.get("/api/v1/capabilities", **self.headers).status_code,
            200,
        )

    @patch("optimizer.api_views.services.start_job")
    def test_service_api_starts_an_owned_job_without_a_csrf_cookie(self, start_job):
        start_job.return_value = SimpleNamespace(id="abc123def456")
        response = Client(enforce_csrf_checks=True).post(
            "/api/v1/optimize",
            data=json.dumps({
                "mode": "zero_shot",
                "_owner": "forged-owner",
                "name": "Support prompt",
                "system_prompt": "Be helpful.",
                "user_prompt": "",
                "task_description": "Answer support questions clearly.",
            }),
            content_type="application/json",
            **self.headers,
        )

        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json(), {
            "job_id": "abc123def456",
            "status": "running",
        })
        start_job.assert_called_once()
        payload = start_job.call_args.args[0]
        self.assertEqual(payload["_owner"], "service-user")

    @patch("optimizer.api_views.services.start_job")
    def test_service_api_rejects_an_invalid_config_before_starting_work(self, start_job):
        response = self.client.post(
            "/api/v1/optimize",
            data=json.dumps({"mode": "not-a-mode", "require_scored_evidence": True}),
            content_type="application/json",
            **self.headers,
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json()["error"],
            "Scored jobs require training_data or few_shot mode",
        )
        start_job.assert_not_called()

    def test_service_api_only_returns_its_own_jobs(self):
        JobState.objects.create(
            id="other-job", owner="someone-else", status="done",
            result={"private": "another owner's result"},
        )
        response = self.client.get(
            "/api/v1/jobs/other-job", **self.headers
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json(), {"error": "job not found"})

        result = {"deployable": {"model": "glm-5.1", "messages": []}}
        JobState.objects.create(
            id="abc123def456",
            owner="service-user",
            status="done",
            stage="Done",
            result=result,
        )
        response = self.client.get(
            "/api/v1/jobs/abc123def456", **self.headers
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "done")
        self.assertEqual(response.json()["result"], result)

    def test_browser_users_cannot_poll_another_users_job(self):
        alice = User.objects.create_user(username="alice")
        self.client.force_login(alice)
        JobState.objects.create(
            id="bob-job", owner="bob", status="done", result={"private": "Bob's result"},
        )
        response = self.client.get("/api/job/bob-job")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json(), {"error": "unknown job"})

        result = {"deployable": {"model": "glm-5.1", "messages": []}}
        JobState.objects.create(id="alice-job", owner="alice", status="done", result=result)
        response = self.client.get("/api/job/alice-job")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "done")
        self.assertEqual(response.json()["deployable"], result["deployable"])


class ConfigBridgeTests(TestCase):
    def test_a_system_only_service_job_does_not_gain_an_empty_user_prompt(self):
        config = build_config({
            "mode": "zero_shot",
            "name": "Support prompt",
            "system_prompt": "Be helpful.",
            "user_prompt": "",
            "task_description": "Answer support questions clearly.",
        })
        self.assertEqual(
            [(prompt.role, prompt.template) for prompt in config.prompts],
            [("system", "Be helpful.")],
        )
