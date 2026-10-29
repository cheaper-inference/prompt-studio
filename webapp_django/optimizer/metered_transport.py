"""Job-local OpenAI proxy: every SDK and GEPA/LiteLLM call has a durable barrier.

Only the trusted, configured CI origin receives the delegated credential. The
loopback listener has an ephemeral bearer token and never exposes that credential.
"""
from __future__ import annotations
import hashlib
import hmac
import json
import secrets
import threading
from collections import defaultdict
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.error import HTTPError
from urllib.request import Request, build_opener, HTTPRedirectHandler
from django.db import close_old_connections
from django.db.models import F
from .durable import assert_lease, lease_write, LeaseLost, UncertainInference
from .models import InferenceCall, JobState
from .storage import (bounded_json_size, StorageBudgetExceeded, RESPONSE_BYTES,
                      JOB_RECEIPT_BYTES, RECEIPT_RESERVATION_BYTES, RECEIPT_METADATA_BYTES)

class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None

class MeteredTransport:
    def __init__(self, job, payload):
        self.job = job
        self.payload = payload
        self.phase = "optimize"
        self.ordinals = defaultdict(int)
        self.lock = threading.Lock()
        self.token = secrets.token_urlsafe(32)
        self.failure = None

    def execute(self, body, purpose="optimization"):
        with self.lock:
            if self.failure:
                raise self.failure
            assert_lease(self.job.id, self.job.lease_token)
            if body.get("stream"):
                raise ValueError("Scored jobs require buffered inference")
            if (not isinstance(body.get("model"), str) or len(body["model"]) > 200
                    or body["model"] not in {self.payload[k] for k in ("task_model", "judge_model", "reflection_model")}):
                raise ValueError("Model is outside the frozen job configuration")
            encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
            if self.phase != "optimize":
                purpose = "holdout_judge" if purpose == "training_judge" else "holdout_generation" if purpose == "holdout_generation" else purpose
            if purpose not in ("training_generation", "training_judge", "reflection", "holdout_generation", "holdout_judge", "optimization"):
                raise ValueError("Unknown inference purpose")
            digest = hashlib.sha256(purpose.encode() + b"\n" + encoded).hexdigest()
            count_key = (self.phase, digest)
            ordinal = self.ordinals[count_key]
            self.ordinals[count_key] += 1
            key = f"{self.phase}:{digest}:{ordinal}"
            with lease_write(self.job.id, self.job.lease_token) as owned:
                prior = InferenceCall.objects.filter(job_id=self.job.id, call_key=key).first()
                if prior:
                    if prior.status != "complete":
                        raise UncertainInference("A prior inference dispatch requires reconciliation")
                    return prior.http_status, prior.response, prior.request_id
                # Unknown dispatches block every new request, including changed
                # candidates after restart. Their storage reservation is retained.
                if InferenceCall.objects.filter(job_id=self.job.id).exclude(status="complete").exists():
                    raise UncertainInference("A prior inference dispatch requires reconciliation")
                # Includes task, judge, reflection, repairs and held-out evaluation.
                maximum = 8 * self.payload["max_metric_calls"] + 8 * len(self.payload["holdout_rows"]) + 50
                if InferenceCall.objects.filter(job_id=self.job.id).count() >= maximum:
                    raise ValueError("Total inference call ceiling reached")
                if owned.receipt_storage_bytes + RECEIPT_RESERVATION_BYTES > JOB_RECEIPT_BYTES:
                    raise StorageBudgetExceeded("Job aggregate receipt storage limit reached")
                JobState.objects.filter(pk=self.job.id).update(
                    receipt_storage_bytes=F("receipt_storage_bytes") + RECEIPT_RESERVATION_BYTES)
                receipt = InferenceCall.objects.create(job_id=self.job.id, call_key=key,
                    request_hash=digest, model=body["model"], phase=self.phase, purpose=purpose,
                    storage_bytes=RECEIPT_RESERVATION_BYTES)
            assert_lease(self.job.id, self.job.lease_token)
            url = self.payload["_base_url"].rstrip("/") + "/chat/completions"
            request = Request(url, data=encoded, method="POST", headers={"Content-Type": "application/json",
                "Authorization": "Bearer " + self.payload["_api_key"]})
            try:
                try:
                    response = build_opener(NoRedirect()).open(request, timeout=100)
                except HTTPError as error:
                    response = error
                with response:
                    status = response.status
                    request_id = response.headers.get("x-ci-request-id") or response.headers.get("x-request-id") or ""
                    raw = response.read(RESPONSE_BYTES + 1)
                if len(raw) > RESPONSE_BYTES:
                    raise UncertainInference("Inference response exceeded the bounded receipt size")
                data = json.loads(raw)
                if not isinstance(data, dict) or not isinstance(data.get("usage", {}), dict) or len(request_id) > 128:
                    raise UncertainInference("Inference returned invalid receipt metadata")
                if (200 <= status < 300 or status >= 500) and not request_id:
                    raise UncertainInference("Successful inference omitted its canonical request receipt")
                usage = data.get("usage", {})
                actual_bytes = (bounded_json_size(data, "response") + bounded_json_size(usage, "response")
                                + RECEIPT_METADATA_BYTES)
                # Cancellation stops new calls; a paid response may still settle
                # under its live lease. Lease expiry rolls back receipt and bytes.
                with lease_write(self.job.id, self.job.lease_token, allow_cancelled=True):
                    changed = InferenceCall.objects.filter(pk=receipt.pk, status="dispatching").update(
                        status="complete", http_status=status, response=data, request_id=request_id,
                        usage=usage, storage_bytes=actual_bytes)
                    if changed != 1:
                        raise UncertainInference("Inference receipt already changed")
                    JobState.objects.filter(pk=self.job.id).update(
                        receipt_storage_bytes=F("receipt_storage_bytes") - receipt.storage_bytes + actual_bytes)
                return status, data, request_id
            except LeaseLost:
                raise
            except Exception as error:
                # The committed dispatching row deliberately remains unknown.
                raise UncertainInference("Inference dispatch has no durable terminal receipt") from error

    @contextmanager
    def endpoint(self):
        transport = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                close_old_connections()
                try:
                    if self.path != "/v1/chat/completions" or not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + transport.token):
                        self.send_error(403)
                        return
                    size = int(self.headers.get("Content-Length", "0"))
                    if size <= 0 or size > 8 * 1024 * 1024:
                        raise ValueError("Invalid inference payload size")
                    status, value, request_id = transport.execute(json.loads(self.rfile.read(size)), self.headers.get("x-gepa-purpose", "optimization"))
                except Exception as error:
                    transport.failure = error
                    status, value, request_id = 503, {"error": {"message": type(error).__name__, "type": "job_stopped"}}, ""
                finally:
                    close_old_connections()
                data = json.dumps(value).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                if request_id:
                    self.send_header("x-ci-request-id", request_id)
                self.end_headers()
                self.wfile.write(data)
        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, name="job-inference-receipts")
        thread.start()
        try:
            from llm_endpoint import configure
            endpoint = configure(api_key=self.token, base_url=f"http://127.0.0.1:{server.server_port}/v1", set_openai_env=False, env_file=None)
            import openai
            object.__setattr__(endpoint, "_client", openai.OpenAI(api_key=self.token, base_url=endpoint.base_url, max_retries=0, timeout=110))
            yield endpoint
            if self.failure:
                raise self.failure
        except Exception:
            if self.failure:
                raise self.failure
            raise
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
