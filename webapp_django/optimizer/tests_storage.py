"""Storage and receipt fences use a temporary database and mocked HTTP only."""
import io
import json
from datetime import timedelta
from unittest.mock import patch

from cryptography.fernet import Fernet
from django.test import TestCase, override_settings
from django.utils import timezone

from . import durable, storage
from .metered_transport import MeteredTransport
from .models import InferenceCall, JobState, Run
from .scored import _complete_scored
from .tests_durable import payload


@override_settings(PROMPT_STUDIO_CREDENTIAL_KEY=Fernet.generate_key().decode())
class ReceiptStorageTests(TestCase):
    def setUp(self):
        self.data = payload()
        durable.enqueue("optimize", self.data)
        self.job = durable.claim()
        self.transport = MeteredTransport(self.job, self.data)
        self.body = {"model": "task", "messages": [{"role": "user", "content": "question"}]}

    def response(self, content="yes"):
        response = io.BytesIO(json.dumps({"choices": [{"message": {"content": content}}],
                                        "usage": {"completion_tokens": 1}}).encode())
        response.status = 200
        response.headers = {"x-ci-request-id": "request-id"}
        return response

    def test_reserves_before_network_reconciles_and_replays_without_charging_storage_twice(self):
        def open_response(*args, **kwargs):
            self.job.refresh_from_db()
            receipt = InferenceCall.objects.get()
            self.assertEqual(self.job.receipt_storage_bytes, storage.RECEIPT_RESERVATION_BYTES)
            self.assertEqual(receipt.storage_bytes, self.job.receipt_storage_bytes)
            self.assertEqual(receipt.status, "dispatching")
            return self.response()

        with patch("optimizer.metered_transport.build_opener") as opener:
            opener.return_value.open.side_effect = open_response
            original = self.transport.execute(self.body)
            receipt = InferenceCall.objects.get()
            actual = (storage.bounded_json_size(receipt.response, "response")
                      + storage.bounded_json_size(receipt.usage, "response") + storage.RECEIPT_METADATA_BYTES)
            self.job.refresh_from_db()
            self.assertEqual(receipt.storage_bytes, actual)
            self.assertEqual(self.job.receipt_storage_bytes, actual)
            resumed = MeteredTransport(self.job, self.data)
            self.assertEqual(resumed.execute(self.body), original)
            opener.return_value.open.assert_called_once()
            self.job.refresh_from_db()
            self.assertEqual(self.job.receipt_storage_bytes, actual)

    def test_aggregate_limit_stops_new_calls_after_restart_before_network_or_receipt_creation(self):
        with patch("optimizer.metered_transport.build_opener") as opener:
            opener.return_value.open.return_value = self.response()
            self.transport.execute(self.body)
            self.job.refresh_from_db()
            before = self.job.receipt_storage_bytes
            # A persisted response fits, but there is not room to reserve another.
            with patch("optimizer.metered_transport.JOB_RECEIPT_BYTES",
                       before + storage.RECEIPT_RESERVATION_BYTES - 1):
                resumed = MeteredTransport(self.job, self.data)
                with self.assertRaises(storage.StorageBudgetExceeded):
                    resumed.execute({**self.body, "messages": [{"role": "user", "content": "new"}]})
            opener.return_value.open.assert_called_once()
        self.job.refresh_from_db()
        self.assertEqual(self.job.receipt_storage_bytes, before)
        self.assertEqual(InferenceCall.objects.count(), 1)

    def test_unknown_response_keeps_reservation_and_blocks_every_new_call(self):
        with patch("optimizer.metered_transport.build_opener") as opener:
            opener.return_value.open.side_effect = TimeoutError("unknown dispatch")
            with self.assertRaises(durable.UncertainInference):
                self.transport.execute(self.body)
            resumed = MeteredTransport(self.job, self.data)
            with self.assertRaises(durable.UncertainInference):
                resumed.execute({"model": "judge"})
            opener.return_value.open.assert_called_once()
        self.job.refresh_from_db()
        self.assertEqual(self.job.receipt_storage_bytes, storage.RECEIPT_RESERVATION_BYTES)
        self.assertIsNone(InferenceCall.objects.get().response)

    def test_failed_receipt_insert_rolls_back_budget_reservation_before_dispatch(self):
        with patch.object(InferenceCall.objects, "create", side_effect=RuntimeError("database failure")), patch("optimizer.metered_transport.build_opener") as opener:
            with self.assertRaises(RuntimeError):
                self.transport.execute(self.body)
            opener.assert_not_called()
        self.job.refresh_from_db()
        self.assertEqual(self.job.receipt_storage_bytes, 0)
        self.assertFalse(InferenceCall.objects.exists())

    def test_expiry_while_network_call_is_running_cannot_commit_receipt_or_release_reservation(self):
        token = self.job.lease_token

        def expire(*args, **kwargs):
            JobState.objects.filter(pk=self.job.pk).update(lease_until=timezone.now() - timedelta(seconds=1))
            return self.response()

        with patch("optimizer.metered_transport.build_opener") as opener:
            opener.return_value.open.side_effect = expire
            with self.assertRaises(durable.LeaseLost):
                self.transport.execute(self.body)
            successor = durable.claim()
            self.assertNotEqual(successor.lease_token, token)
            with self.assertRaises(durable.UncertainInference):
                MeteredTransport(successor, self.data).execute(self.body)
            opener.return_value.open.assert_called_once()
        self.job.refresh_from_db()
        self.assertEqual(self.job.receipt_storage_bytes, storage.RECEIPT_RESERVATION_BYTES)
        self.assertEqual(InferenceCall.objects.get().status, "dispatching")

    def test_cancellation_allows_owned_paid_receipt_but_prevents_subsequent_dispatch(self):
        def cancel(*args, **kwargs):
            durable.cancel(self.job.id, self.job.owner)
            return self.response()

        with patch("optimizer.metered_transport.build_opener") as opener:
            opener.return_value.open.side_effect = cancel
            self.transport.execute(self.body)
            with self.assertRaises(durable.JobCancelled):
                self.transport.execute({"model": "judge"})
            opener.return_value.open.assert_called_once()
        self.assertEqual(InferenceCall.objects.get().status, "complete")

    def test_oversized_wire_response_keeps_only_dispatch_barrier(self):
        with patch("optimizer.metered_transport.build_opener") as opener, patch("optimizer.metered_transport.RESPONSE_BYTES", 32):
            opener.return_value.open.return_value = self.response("x" * 100)
            with self.assertRaises(durable.UncertainInference):
                self.transport.execute(self.body)
        self.assertIsNone(InferenceCall.objects.get().response)
        self.job.refresh_from_db()
        self.assertEqual(self.job.receipt_storage_bytes, storage.RECEIPT_RESERVATION_BYTES)

    def test_serialized_response_limit_prevents_persisted_expansion(self):
        with patch.dict(storage.JSON_LIMITS, response=128), patch("optimizer.metered_transport.build_opener") as opener:
            # Unicode is more expensive once Django escapes it for JSON storage.
            opener.return_value.open.return_value = self.response("😀" * 100)
            with self.assertRaises(durable.UncertainInference):
                self.transport.execute(self.body)
        self.assertIsNone(InferenceCall.objects.get().response)

    def test_oversized_request_identifier_is_not_persisted(self):
        response = self.response()
        response.headers["x-ci-request-id"] = "x" * 129
        with patch("optimizer.metered_transport.build_opener") as opener:
            opener.return_value.open.return_value = response
            with self.assertRaises(durable.UncertainInference):
                self.transport.execute(self.body)
        self.assertEqual(InferenceCall.objects.get().request_id, "")

    def test_long_model_identifier_cannot_multiply_job_payload_into_receipts(self):
        self.transport.payload = {**self.data, "task_model": "m" * 201}
        with patch("optimizer.metered_transport.build_opener") as opener:
            with self.assertRaises(ValueError):
                self.transport.execute({"model": "m" * 201})
            opener.assert_not_called()
        self.assertFalse(InferenceCall.objects.exists())

    def test_checkpoint_aggregate_and_terminal_result_limits_roll_back(self):
        with patch.dict(storage.JSON_LIMITS, checkpoint=100, result=100):
            durable.checkpoint(self.job.id, self.job.lease_token, "first", "x" * 50)
            with self.assertRaises(storage.StorageBudgetExceeded):
                durable.checkpoint(self.job.id, self.job.lease_token, "second", "x" * 50)
            with self.assertRaises(storage.StorageBudgetExceeded):
                _complete_scored(self.job, self.data, {"output": "x" * 200})
        self.job.refresh_from_db()
        self.assertEqual(self.job.checkpoint, {"first": "x" * 50})
        self.assertEqual(self.job.status, "running")
        self.assertFalse(Run.objects.exists())

    def test_input_size_is_checked_before_enqueue(self):
        before = JobState.objects.count()
        with patch.dict(storage.JSON_LIMITS, payload=100):
            with self.assertRaises(storage.StorageBudgetExceeded):
                durable.enqueue("optimize", self.data)
        self.assertEqual(JobState.objects.count(), before)
