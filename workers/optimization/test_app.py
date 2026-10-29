import os
import unittest
from unittest.mock import patch, Mock
from fastapi.testclient import TestClient
import app


class WorkerContractTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"OPTIMIZATION_WORKER_TOKEN": "test-secret", "OPTIMIZATION_OPERATIONS": "llmlingua2,sot,embed,verify"})
        self.env.start()
        self.client = TestClient(app.app)
        self.headers = {"Authorization": "Bearer test-secret"}

    def tearDown(self):
        self.env.stop()

    def test_authentication_prevents_model_dispatch(self):
        with patch.object(app, "_dispatch") as dispatch:
            self.assertEqual(self.client.post("/v1/embed", json={"text": "hello"}).status_code, 401)
            dispatch.assert_not_called()

    def test_all_four_public_operations_still_dispatch(self):
        for operation in ("llmlingua2", "sot", "embed", "verify"):
            with self.subTest(operation=operation), patch.dict(os.environ, {"OPTIMIZATION_COMPUTE_USD_PER_SECOND": ""}), patch.object(app, "_dispatch", return_value=({"ok": True}, {}, 0)) as dispatch:
                response = self.client.post(f"/v1/{operation}", headers=self.headers, json={})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(dispatch.call_args.args[0], operation)

    def test_request_limits_reject_before_execution(self):
        with patch.object(app, "_dispatch") as dispatch:
            response = self.client.post("/v1/llmlingua2", headers=self.headers, json={"texts": ["context"], "rate": 3})
            self.assertEqual(response.status_code, 422)
            dispatch.assert_not_called()

    def test_configured_operations_normalize_whitespace_and_duplicates(self):
        with patch.dict(os.environ, {"OPTIMIZATION_OPERATIONS": " embed, embed, "}), patch.object(app, "_dispatch", return_value=({"vector": [1, 0]}, {}, 3)) as dispatch:
            response = self.client.post("/v1/embed", headers=self.headers, json={"text": "hello"})
            self.assertEqual(response.status_code, 200)
            dispatch.assert_called_once()
            self.assertEqual(self.client.post("/v1/verify", headers=self.headers, json={}).status_code, 404)

    def test_unknown_configuration_fails_clearly_before_execution(self):
        with patch.dict(os.environ, {"OPTIMIZATION_OPERATIONS": "embed, typo"}), patch.object(app, "_dispatch") as dispatch:
            response = self.client.post("/v1/embed", headers=self.headers, json={"text": "hello"})
            self.assertEqual(response.status_code, 503)
            self.assertIn("Unknown OPTIMIZATION_OPERATIONS: typo", response.json()["detail"])
            self.assertEqual(self.client.post("/v1/not-an-operation", headers=self.headers, json={}).status_code, 404)
            dispatch.assert_not_called()

    def test_explicit_empty_configuration_disables_operations(self):
        with patch.dict(os.environ, {"OPTIMIZATION_OPERATIONS": " , \n,"}), patch.object(app, "_dispatch") as dispatch:
            self.assertEqual(self.client.post("/v1/embed", headers=self.headers, json={"text": "hello"}).status_code, 404)
            dispatch.assert_not_called()

    def test_budgeted_work_requires_pricing(self):
        with patch.dict(os.environ, {"OPTIMIZATION_COMPUTE_USD_PER_SECOND": ""}), patch.object(app, "_dispatch") as dispatch:
            response = self.client.post("/v1/embed", headers=self.headers, json={"text": "hello", "maxCostUsd": "1"})
            self.assertEqual(response.status_code, 503)
            dispatch.assert_not_called()

    def test_missing_compute_price_is_unknown_not_zero(self):
        with patch.dict(os.environ, {"OPTIMIZATION_COMPUTE_USD_PER_SECOND": ""}), patch.object(app, "_dispatch", return_value=({"vector": [1, 0]}, {"artifactRevision": "pinned"}, 3)):
            response = self.client.post("/v1/embed", headers=self.headers, json={"text": "hello"})
            self.assertEqual(response.status_code, 200)
            self.assertIsNone(response.json()["usage"]["costUsd"])
            self.assertEqual(response.json()["evidence"]["artifactRevision"], "pinned")

    def test_positive_compute_budget_limits_worker_deadline(self):
        with patch.dict(os.environ, {"OPTIMIZATION_COMPUTE_USD_PER_SECOND": "0.1"}), patch.object(app, "_dispatch", return_value=({"vector": [1, 0]}, {}, 3)) as dispatch:
            response = self.client.post("/v1/embed", headers=self.headers, json={"text": "hello", "maxCostUsd": "0.2"})
            self.assertEqual(response.status_code, 200)
            self.assertLessEqual(dispatch.call_args.args[2], 2)
            self.assertIsNotNone(response.json()["usage"]["costUsd"])

    def test_timeout_terminates_model_process(self):
        process, connection = Mock(), Mock()
        process.is_alive.return_value = True
        connection.poll.return_value = False
        with patch.object(app, "MODEL_PROCESS", process), patch.object(app, "MODEL_PIPE", connection):
            with self.assertRaises(app.HTTPException) as caught:
                app._dispatch("embed", app.Input(text="hello"), 0.01)
            self.assertEqual(caught.exception.status_code, 504)
            process.terminate.assert_called_once()

    def test_removed_operation_cannot_dispatch_even_when_explicitly_enabled(self):
        for selection in ("llmlingua2,sot,embed,verify", "provence"):
            with self.subTest(selection=selection), patch.dict(os.environ, {"OPTIMIZATION_OPERATIONS": selection}), patch.object(app, "_dispatch") as dispatch:
                response = self.client.post("/v1/provence", headers=self.headers, json={"question": "Q", "texts": ["Context"]})
                self.assertEqual(response.status_code, 404)
                dispatch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
