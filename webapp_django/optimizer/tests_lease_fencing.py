from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock, patch

from cryptography.fernet import Fernet
from django.test import TestCase, override_settings
from django.utils import timezone

from . import durable, services
from .models import InferenceCall, JobState, Run
from .scored import _complete_scored


@override_settings(PROMPT_STUDIO_CREDENTIAL_KEY=Fernet.generate_key().decode())
class LeaseFencingTests(TestCase):
    def claimed(self):
        durable.enqueue("optimize", {"_owner": "alice", "_api_key": "delegated-secret", "require_scored_evidence": True})
        return durable.claim()

    def expire(self, job):
        JobState.objects.filter(pk=job.id).update(lease_until=timezone.now() - timedelta(seconds=1))

    def handle(self, row):
        return SimpleNamespace(id=row.id, owner=row.owner, lease_token=row.lease_token)

    def test_expiry_after_checkpoint_read_rejects_write(self):
        row = self.claimed()
        original_size = durable.bounded_json_size

        def expire_after_read(value, kind):
            self.expire(row)
            return original_size(value, kind)

        with patch.object(durable, "bounded_json_size", side_effect=expire_after_read):
            with self.assertRaises(durable.LeaseLost):
                durable.checkpoint(row.id, row.lease_token, "optimized", {"prompt": "stale"})
        row.refresh_from_db()
        self.assertEqual(row.checkpoint, {})

    def test_lease_expiry_at_end_rolls_back_related_receipt_write(self):
        row = self.claimed()
        with self.assertRaises(durable.LeaseLost):
            with durable.lease_write(row.id, row.lease_token):
                InferenceCall.objects.create(job=row, call_key="new", request_hash="hash", model="model")
                self.expire(row)
        self.assertFalse(InferenceCall.objects.filter(job=row).exists())

    def test_successor_token_cannot_be_overwritten_by_old_worker(self):
        row = self.claimed()
        stale_token = row.lease_token
        self.expire(row)
        successor = durable.claim()
        for action in (
            lambda: durable.checkpoint(row.id, stale_token, "late", {}),
            lambda: durable.finish_job(row.id, stale_token, "error", error="stale error"),
            lambda: durable.heartbeat(row.id, stale_token),
        ):
            with self.assertRaises(durable.LeaseLost):
                action()
        successor.refresh_from_db()
        self.assertEqual(successor.status, "running")
        self.assertEqual(successor.checkpoint, {})
        self.assertNotEqual(successor.credential_ciphertext, "")

    def test_scored_history_rolls_back_if_lease_expires_after_history_write(self):
        row = self.claimed()
        write = Run.objects.update_or_create

        def expire_after_history(*args, **kwargs):
            value = write(*args, **kwargs)
            self.expire(row)
            return value

        with patch.object(Run.objects, "update_or_create", side_effect=expire_after_history):
            with self.assertRaises(durable.LeaseLost):
                _complete_scored(self.handle(row), {"name": "test", "_api_key": "secret"}, {"scored_evidence": {"completion_known": True}})
        self.assertFalse(Run.objects.filter(job_id=row.id).exists())
        row.refresh_from_db()
        self.assertEqual(row.status, "running")
        self.assertIsNone(row.result)
        self.assertNotEqual(row.credential_ciphertext, "")

    def test_cancellation_cannot_commit_successful_scored_history(self):
        row = self.claimed()
        write = Run.objects.update_or_create

        def cancel_after_history(*args, **kwargs):
            value = write(*args, **kwargs)
            JobState.objects.filter(pk=row.id).update(cancel_requested_at=timezone.now())
            return value

        with patch.object(Run.objects, "update_or_create", side_effect=cancel_after_history):
            with self.assertRaises(durable.JobCancelled):
                _complete_scored(self.handle(row), {}, {"complete": True})
        self.assertFalse(Run.objects.filter(job_id=row.id).exists())

    def test_scored_completion_erases_credentials_before_worker_finally(self):
        row = self.claimed()
        _complete_scored(self.handle(row), {"_api_key": "secret"}, {"complete": True})
        row.refresh_from_db()
        self.assertEqual(row.status, "done")
        self.assertEqual(row.credential_ciphertext, "")
        self.assertIsNotNone(row.lease_token)
        self.assertNotIn("_api_key", Run.objects.get(job_id=row.id).config)

    def test_oversized_scored_result_is_rejected_before_history_write(self):
        from .storage import JSON_LIMITS, StorageBudgetExceeded
        row = self.claimed()
        with patch.dict(JSON_LIMITS, {"result": 32}), patch.object(Run.objects, "update_or_create") as history:
            with self.assertRaises(StorageBudgetExceeded):
                _complete_scored(self.handle(row), {}, {"output": "x" * 40})
        history.assert_not_called()
        row.refresh_from_db()
        self.assertEqual(row.status, "running")

    def test_all_terminal_job_handle_writes_erase_credentials_atomically(self):
        for status in durable.TERMINAL_STATUSES:
            with self.subTest(status=status):
                row = self.claimed()
                job = services.Job(id=row.id, owner=row.owner, lease_token=row.lease_token)
                job.status = status
                row.refresh_from_db()
                self.assertEqual(row.status, status)
                self.assertEqual(row.credential_ciphertext, "")
                self.assertEqual(row.lease_token, job.lease_token)

    def test_exception_completion_is_secret_free_even_if_finally_is_interrupted(self):
        row = durable.enqueue("optimize", {"_owner": "alice", "_api_key": "secret", "require_scored_evidence": True})
        keeper = Mock()
        keeper.join.side_effect = RuntimeError("process killed before cleanup")
        with patch("threading.Thread", return_value=keeper), patch("optimizer.scored.run_scored", side_effect=ValueError("invalid")):
            with self.assertRaisesRegex(RuntimeError, "process killed"):
                durable.process_one()
        row.refresh_from_db()
        self.assertEqual(row.status, "error")
        self.assertEqual(row.credential_ciphertext, "")

    def test_recovery_cleans_historical_terminal_secrets(self):
        for index, status in enumerate(durable.TERMINAL_STATUSES):
            JobState.objects.create(id=f"historical-{index}", owner="alice", status=status, credential_ciphertext="old ciphertext")
        self.assertIsNone(durable.claim())
        self.assertFalse(JobState.objects.exclude(credential_ciphertext="").exists())

    def test_cancelled_request_allows_receipt_fencing_but_not_checkpoints(self):
        row = self.claimed()
        durable.cancel(row.id, row.owner)
        with self.assertRaises(durable.JobCancelled):
            durable.checkpoint(row.id, row.lease_token, "late", {})
        with durable.lease_write(row.id, row.lease_token, allow_cancelled=True):
            InferenceCall.objects.create(job=row, call_key="paid", request_hash="hash", model="model", status="complete")
        self.assertTrue(InferenceCall.objects.filter(job=row, call_key="paid").exists())
