"""Benchmark restart recovery reuses upstream leases, not web startup hooks."""
from datetime import timedelta
from django.apps import apps
from django.test import TestCase
from django.utils import timezone
from . import durable, services
from .models import JobState, Run

class RestartRecoveryTests(TestCase):
    def setUp(self):
        self.partial = {'billing': {'calls': [{'request_id': 'paid-call'}]}}
        self.job = JobState.objects.create(id='a'*32,owner='owner',kind='benchmark',
            lease_token='old',lease_until=timezone.now()+timedelta(minutes=2),result=self.partial,
            credential_ciphertext='encrypted')
        self.run = Run.objects.create(owner='owner',kind='benchmark',job_id=self.job.id,
            status='running',result=self.partial,config={'saved':True})

    def test_web_restart_does_not_interrupt_live_worker(self):
        apps.get_app_config('optimizer').ready()
        self.job.refresh_from_db()
        self.assertEqual(self.job.status,'running')
        self.assertEqual(self.job.lease_token,'old')

    def test_worker_recovery_interrupts_without_replaying_and_preserves_receipts(self):
        JobState.objects.filter(pk=self.job.id).update(lease_until=timezone.now()-timedelta(seconds=1))
        self.assertIsNone(durable.claim())
        self.job.refresh_from_db(); self.run.refresh_from_db()
        self.assertEqual(self.job.status,'interrupted')
        self.assertEqual(self.run.status,'interrupted')
        self.assertEqual(self.job.credential_ciphertext,'')
        self.assertEqual(self.run.result,self.partial)
        self.assertEqual(self.job.result,self.partial)
        self.assertEqual(self.run.config,{'saved':True})
        self.assertIsNone(durable.claim())

    def test_cancelled_worker_expiry_keeps_cancellation(self):
        durable.cancel(self.job.id,'owner')
        JobState.objects.filter(pk=self.job.id).update(lease_until=timezone.now()-timedelta(seconds=1))
        durable.claim()
        self.job.refresh_from_db(); self.run.refresh_from_db()
        self.assertEqual(self.job.status,'cancelled')
        self.assertEqual(self.run.status,'cancelled')
        self.assertEqual(self.run.result,self.partial)

    def test_stale_worker_cannot_publish_terminal_results(self):
        JobState.objects.filter(pk=self.job.id).update(lease_until=timezone.now()-timedelta(seconds=1))
        durable.claim()
        with self.assertRaises(durable.LeaseLost):
            durable.finish_job(self.job.id,'old','done',result={'wrong':True})
        self.run.refresh_from_db()
        self.assertEqual(self.run.result,self.partial)
