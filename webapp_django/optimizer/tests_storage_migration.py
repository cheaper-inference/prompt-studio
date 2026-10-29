"""Historical receipts must count toward a job's storage budget after upgrade."""

import json

from django.db import connection
from django.db.migrations.executor import MigrationExecutor
from django.db.models import Sum
from django.test import TransactionTestCase


class ReceiptStorageMigrationTests(TransactionTestCase):
    migrate_from = [("optimizer", "0007_interrupt_prequeue_jobs")]
    migrate_to = [("optimizer", "0008_receipt_storage_budget")]

    def setUp(self):
        super().setUp()
        executor = MigrationExecutor(connection)
        self.latest_targets = executor.loader.graph.leaf_nodes()
        executor.migrate(self.migrate_from)
        self.old_apps = executor.loader.project_state(self.migrate_from).apps

    def tearDown(self):
        try:
            MigrationExecutor(connection).migrate(self.latest_targets)
        finally:
            super().tearDown()

    def test_historical_receipts_receive_nonzero_conservative_storage_accounting(self):
        OldJob = self.old_apps.get_model("optimizer", "JobState")
        OldCall = self.old_apps.get_model("optimizer", "InferenceCall")
        job = OldJob.objects.create(id="historical-storage-job", owner="old")
        response = {"choices": [{"message": {"content": "Historical caf\u00e9 \U0001f680"}}]}
        usage = {"prompt_tokens": 15, "completion_tokens": 7, "legacy_note": "\u4f7f\u7528"}
        # SQLite accepted identifiers beyond CharField.max_length before the
        # runtime added validation. Their actual bytes must also be accounted.
        historical_model = "old-model-" + "\u754c" * 8192
        completed = OldCall.objects.create(
            job=job,
            call_key="completed-historical-call",
            request_hash="a" * 64,
            request_id="completed-historical-request",
            model=historical_model,
            status="complete",
            response=response,
            usage=usage,
        )
        dispatching = OldCall.objects.create(
            job=job,
            call_key="dispatching-historical-call",
            request_hash="b" * 64,
            model=historical_model,
            status="dispatching",
            response=None,
            usage={},
        )

        executor = MigrationExecutor(connection)
        executor.migrate(self.migrate_to)
        apps = executor.loader.project_state(self.migrate_to).apps
        Call = apps.get_model("optimizer", "InferenceCall")
        Job = apps.get_model("optimizer", "JobState")
        completed = Call.objects.get(pk=completed.pk)
        dispatching = Call.objects.get(pk=dispatching.pk)
        migrated_job = Job.objects.get(pk=job.pk)

        persisted_json_bytes = sum(
            len(json.dumps(value).encode("utf-8")) for value in (response, usage)
        )
        self.assertGreater(completed.storage_bytes, 0)
        self.assertGreaterEqual(
            completed.storage_bytes,
            persisted_json_bytes + self.identifier_bytes(completed),
        )
        # An in-flight call has no known response size. Keep room for both the
        # response and separately persisted usage at the historical 2 MiB cap.
        self.assertGreaterEqual(
            dispatching.storage_bytes,
            2 * 2 * 1024 * 1024 + self.identifier_bytes(dispatching),
        )
        total = Call.objects.filter(job_id=job.pk).aggregate(total=Sum("storage_bytes"))["total"]
        self.assertGreater(migrated_job.receipt_storage_bytes, 0)
        self.assertEqual(migrated_job.receipt_storage_bytes, total)
        self.assertEqual(completed.response, response)
        self.assertEqual(completed.usage, usage)
        self.assertEqual(completed.status, "complete")
        self.assertEqual(dispatching.status, "dispatching")
        self.assertEqual(migrated_job.owner, "old")

    @staticmethod
    def identifier_bytes(call):
        return sum(
            len(str(getattr(call, field)).encode("utf-8"))
            for field in (
                "job_id", "call_key", "request_hash", "phase", "purpose",
                "status", "request_id", "model",
            )
        )
