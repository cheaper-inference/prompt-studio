"""Zero-shot rewrites must fail loudly: MLflow's MetaPromptOptimizer returns the original
prompts as a success when the rewriter call fails (e.g. insufficient wallet balance)."""
from types import SimpleNamespace
from unittest import mock

from django.test import SimpleTestCase
from mlflow.exceptions import MlflowException
from mlflow.genai.optimize import MetaPromptOptimizer

import prompt_optimizer as po

BALANCE = MlflowException("Failed to call reflection model openai:/claude-opus-4.8: litellm.BadRequestError: "
                          "OpenAIException - Insufficient wallet balance for this request.")
PROMPTS = {"sys": "You write follow-ups.", "user": "Lead from {{city}} wants {{goal}}."}


def strict(reply=None, error=None):
    opt = po.strict_metaprompt_optimizer(reflection_model="openai:/claude-opus-4.8", guidelines="follow-ups")
    opt._call_reflection_model = mock.Mock(side_effect=error, return_value=reply)
    return opt


class ZeroShotFailureTests(SimpleTestCase):
    def test_classification(self):
        self.assertEqual(po.classify_rewriter_failure(BALANCE).code, "insufficient_balance")
        self.assertEqual(po.classify_rewriter_failure(Exception("Error code: 403 - model not allowed")).code,
                         "rewriter_unavailable")
        self.assertEqual(po.classify_rewriter_failure(
            MlflowException("Failed to parse reflection model response as JSON: x")).code, "rewrite_rejected")
        for mlflow_text in ("Unexpected prompts found in improved prompts: ['extra']",
                            "Prompts missing from improved prompts: ['user']"):
            self.assertEqual(po.classify_rewriter_failure(MlflowException(mlflow_text)).code, "rewrite_rejected")
        self.assertEqual(po.classify_rewriter_failure(TimeoutError("read timed out")).code, "rewriter_failed")

    def test_failed_retry_is_not_reported_as_key_access(self):
        """Production, Nov 4: a timed-out rewrite whose retry crashed (tenacity missing) said
        "Your key can't use the rewriter model" because ModuleNotFoundError matched NotFoundError."""
        call = MlflowException("Failed to call reflection model openai:/claude-opus-4.8: tenacity import failed "
                               "please run `pip install tenacity`. ErrorNo module named 'tenacity'")
        missing = ModuleNotFoundError("No module named 'tenacity'")
        missing.__cause__ = Exception("litellm.Timeout: APITimeoutError - Request timed out.")
        call.__cause__ = missing
        failure = po.classify_rewriter_failure(call)
        self.assertEqual(failure.code, "rewriter_failed")
        self.assertIn("timed out", failure.message)
        for text in ("litellm.InternalServerError: OpenAIException - boom", "Error code: 503 - overloaded",
                     "APIConnectionError: Connection error.", "Request timed out."):
            failure = po.classify_rewriter_failure(Exception(text))
            self.assertEqual((failure.code, "timed out" in failure.message), ("rewriter_failed", True), text)
        # The API's own not-found error still means the key can't use the model.
        self.assertEqual(po.classify_rewriter_failure(
            Exception("litellm.NotFoundError: OpenAIException - model x")).code, "rewriter_unavailable")

    def test_zero_shot_rewrite_retries_are_capped(self):
        """Zero-shot rewrites send a RetryPolicy capping retries at ZERO_SHOT_RETRIES (MLflow's default is 5)."""
        endpoint = mock.Mock(uri=lambda model: f"openai:/{model}", reflection_model="m",
                             lm_kwargs=lambda **extra: extra)
        optimizer = po.PromptOptimizer.__new__(po.PromptOptimizer)
        optimizer.endpoint = endpoint
        config = po.OptimizationConfig(name="z", prompts="Say hi", mode=po.Mode.ZERO_SHOT, task_description="greet")
        policy = optimizer._build_optimizer(config).lm_kwargs["retry_policy"]
        self.assertEqual((policy.TimeoutErrorRetries, policy.InternalServerErrorRetries),
                         (po.ZERO_SHOT_RETRIES, po.ZERO_SHOT_RETRIES))
        self.assertFalse(policy.AuthenticationErrorRetries)

    def test_timed_out_rewrite_is_sent_at_most_retries_plus_one_times(self):
        """End to end through MLflow and litellm (nothing mocked above the HTTP socket): a local
        server that never answers within the timeout sees 2..ZERO_SHOT_RETRIES + 1 requests."""
        import threading
        import time
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        import litellm

        hits = []

        class Slow(BaseHTTPRequestHandler):
            def do_POST(self):   # noqa: N802 (http.server API)
                """Count the request, then answer only after the client has timed out."""
                hits.append(time.monotonic())
                time.sleep(2)
                try:
                    self.send_response(500)
                    self.end_headers()
                except OSError:
                    pass   # the client already gave up

            def log_message(self, *args):
                """Keep the test output quiet."""

        server = ThreadingHTTPServer(("127.0.0.1", 0), Slow)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base = f"http://127.0.0.1:{server.server_address[1]}/v1"
        endpoint = mock.Mock(uri=lambda model: f"openai:/{model}", reflection_model="m",
                             lm_kwargs=lambda **extra: {"api_base": base, "api_key": "test", **extra})
        optimizer = po.PromptOptimizer.__new__(po.PromptOptimizer)
        optimizer.endpoint = endpoint
        config = po.OptimizationConfig(name="z", prompts="Say hi", mode=po.Mode.ZERO_SHOT, task_description="greet",
                                       request_timeout=1)
        rewriter = optimizer._build_optimizer(config)
        with self.assertRaises(MlflowException) as caught:
            rewriter._call_reflection_model("Rewrite: Say hi", False)
        # The client's timeout, not the server's late 500, must be what failed (litellm re-raises the last attempt's error).
        self.assertIsInstance(caught.exception.__cause__, litellm.Timeout)
        self.assertGreater(len(hits), 1)
        self.assertLessEqual(len(hits), po.ZERO_SHOT_RETRIES + 1)

    def test_retry_counts_per_error_as_litellm_resolves_them(self):
        """litellm picks the retry count by exception class: 500, 503, timeouts and rate limits get
        ZERO_SHOT_RETRIES; a bad key is not retried (DefaultRetries is unset)."""
        import litellm
        from litellm.router_utils.get_retry_from_policy import get_num_retries_from_retry_policy

        endpoint = mock.Mock(uri=lambda model: f"openai:/{model}", reflection_model="m",
                             lm_kwargs=lambda **extra: extra)
        optimizer = po.PromptOptimizer.__new__(po.PromptOptimizer)
        optimizer.endpoint = endpoint
        config = po.OptimizationConfig(name="z", prompts="Say hi", mode=po.Mode.ZERO_SHOT, task_description="greet")
        policy = optimizer._build_optimizer(config).lm_kwargs["retry_policy"]
        args = dict(message="x", llm_provider="openai", model="m")
        cases = {"500": (litellm.InternalServerError(**args), po.ZERO_SHOT_RETRIES),
                 "503": (litellm.ServiceUnavailableError(**args), po.ZERO_SHOT_RETRIES),
                 "timeout": (litellm.Timeout(**args), po.ZERO_SHOT_RETRIES),
                 "429": (litellm.RateLimitError(**args), po.ZERO_SHOT_RETRIES),
                 "401": (litellm.AuthenticationError(**args), None)}
        for name, (error, expected) in cases.items():
            self.assertEqual(get_num_retries_from_retry_policy(exception=error, retry_policy=policy), expected, name)

    def test_litellm_logging_thread_does_not_import_through_mlflow_hooks(self):
        """The module litellm's logging thread imports after every call is marked absent, so the
        import fails at once instead of taking the import lock (the Nov 8 worker deadlock)."""
        import importlib
        import sys

        self.assertIn(po.LITELLM_LAZY_PROXY_MODULE, sys.modules)
        self.assertIsNone(sys.modules[po.LITELLM_LAZY_PROXY_MODULE])
        with self.assertRaises(ImportError):
            importlib.import_module(po.LITELLM_LAZY_PROXY_MODULE)
        # The litellm code path itself: it catches the ImportError and reports no savings.
        from litellm.litellm_core_utils import litellm_logging
        self.assertIsNone(litellm_logging._autorouter_savings_for_payload(
            request_metadata={}, model="m", custom_llm_provider="openai", model_id=None,
            usage_object=None, cost_breakdown=None))

    def test_rewriter_reply_text_does_not_look_like_a_transport_failure(self):
        """MLflow's parse error quotes the rewriter's reply; words in it must not decide the code."""
        reply = "Sure! Handle HTTP 402 and permission errors, and say when the model is not available."
        parse = MlflowException(f"Failed to parse reflection model response as JSON: Expecting value: "
                                f"line 1 column 1 (char 0)\nResponse: {reply}")
        self.assertEqual(po.classify_rewriter_failure(parse).code, "rewrite_rejected")
        # A real transport failure is still classified from its cause chain.
        call = MlflowException("Failed to call reflection model openai:/claude-opus-4.8: boom")
        call.__cause__ = Exception("Error code: 403 - model not allowed for this key")
        self.assertEqual(po.classify_rewriter_failure(call).code, "rewriter_unavailable")

    def test_status_codes_need_http_context(self):
        """A JSON offset like 'column 402' is not a status code; a real one still classifies."""
        for column in (401, 402, 403, 404):
            parse = MlflowException("Failed to parse reflection model response as JSON: "
                                    f"Expecting ',' delimiter: line 1 column {column} (char {column - 1})")
            self.assertEqual(po.classify_rewriter_failure(parse).code, "rewrite_rejected", column)
        cases = {"Error code: 402 - payment required": "insufficient_balance",
                 "Error code: 401 - invalid key": "rewriter_unavailable",
                 "status code 403": "rewriter_unavailable",
                 "Error code: 404 - {'error': 'no such model'}": "rewriter_unavailable",
                 "HTTP 402": "insufficient_balance"}
        for text, code in cases.items():
            self.assertEqual(po.classify_rewriter_failure(Exception(text)).code, code, text)

    def test_balance_failure_raises_instead_of_returning_the_original(self):
        with self.assertRaises(po.OptimizerFailed) as caught:
            strict(error=BALANCE)._optimize_zero_shot(PROMPTS, {}, False)
        self.assertEqual(caught.exception.code, "insufficient_balance")
        self.assertIn("Add credit", caught.exception.message)
        self.assertIn("Insufficient wallet balance", caught.exception.detail)

    def test_dropped_variable_is_rejected(self):
        dropped = {"sys": "You write follow-ups.", "user": "Lead from {{city}}."}
        with self.assertRaises(po.OptimizerFailed) as caught:
            strict(reply=dropped)._optimize_zero_shot(PROMPTS, {}, False)
        self.assertEqual(caught.exception.code, "rewrite_rejected")

    def test_dropped_spaced_variable_is_rejected(self):
        """MLflow's own check does not see {{ name }}; GP fills it, so a rewrite must keep it."""
        prompts = {"user": "Greet {{ name }}."}
        with self.assertRaises(po.OptimizerFailed) as caught:
            strict(reply={"user": "Greet the customer warmly."})._optimize_zero_shot(prompts, {}, False)
        self.assertEqual(caught.exception.code, "rewrite_rejected")
        kept = strict(reply={"user": "Greet {{ name }} warmly."})._optimize_zero_shot(prompts, {}, False)
        self.assertEqual(kept.optimized_prompts, {"user": "Greet {{ name }} warmly."})
        # MLflow sees {{name}} as a new variable and strips it from the rewrite ("Greet  warmly."),
        # so a rewrite that only tightened the spacing is rejected too, rather than served broken.
        with self.assertRaises(po.OptimizerFailed):
            strict(reply={"user": "Greet {{name}} warmly."})._optimize_zero_shot(prompts, {}, False)

    def test_added_spaced_variable_is_rejected(self):
        """MLflow strips an added {{new}} but not {{ new }}; GP would then ask for an input the original never had."""
        with self.assertRaises(po.OptimizerFailed) as caught:
            strict(reply={"user": "Greet {{ name }} from {{ new_input }}."})._optimize_zero_shot(
                {"user": "Greet {{ name }}."}, {}, False)
        self.assertEqual(caught.exception.code, "rewrite_rejected")
        self.assertIn("new_input", caught.exception.detail)

    def test_success_returns_the_rewrite(self):
        rewrite = {"sys": "You write short WhatsApp follow-ups.", "user": "City: {{city}}\nGoal: {{goal}}"}
        out = strict(reply=dict(rewrite))._optimize_zero_shot(PROMPTS, {}, False)
        self.assertEqual(out.optimized_prompts, rewrite)

    def test_zero_shot_uses_the_strict_optimizer_and_mlflows_output_limit(self):
        endpoint = SimpleNamespace(uri=lambda m: f"openai:/{m}", reflection_model="claude-opus-4.8",
                                   lm_kwargs=lambda **kw: kw)
        opt = po.PromptOptimizer.__new__(po.PromptOptimizer)
        opt.endpoint = endpoint
        config = po.OptimizationConfig(name="t", prompts="Suggest keywords for {{service}}.",
                                       mode=po.Mode.ZERO_SHOT, task_description="keywords")
        built = opt._build_optimizer(config)
        self.assertIsInstance(built, MetaPromptOptimizer)
        self.assertEqual(type(built).__name__, "StrictMetaPromptOptimizer")
        # MLflow sets the output limit itself; a second one is rejected by the CI proxy.
        self.assertFalse({"max_tokens", "max_completion_tokens", "max_output_tokens"} & set(built.lm_kwargs))

    def test_job_records_the_user_facing_error(self):
        from optimizer import services
        job = SimpleNamespace(id="j1", stage="", status="running", result=None, error=None)
        failure = po.classify_rewriter_failure(BALANCE)
        with mock.patch("prompt_optimizer.PromptOptimizer") as optimizer, \
             mock.patch.object(services, "_endpoint_for"), mock.patch.object(services, "_tracking_uri"), \
             mock.patch.object(services, "build_config"), mock.patch.object(services, "record_run") as record, \
             mock.patch("traceback.print_exc"):
            optimizer.return_value.optimize.side_effect = failure
            services._run_job(job, {"mode": "zero_shot", "name": "t", "_owner": "alice"})
        self.assertEqual(job.status, "error")
        self.assertEqual(job.error, "Your wallet balance can't cover this optimization. Add credit and run it again.")
        recorded = record.call_args.kwargs["error"]
        self.assertTrue(recorded.startswith("insufficient_balance: "))
        self.assertIn("Insufficient wallet balance", recorded)

    def test_wallet_error_from_the_real_call_chain_fails_the_job(self):
        """No mocks inside MLflow: litellm itself raises the proxy's 402-style error, as in production."""
        import litellm
        from optimizer import services
        error = litellm.BadRequestError("OpenAIException - Insufficient wallet balance for this request.",
                                        model="claude-opus-4.8", llm_provider="openai")
        seen = {}

        def refuse(*args, **kwargs):
            seen.update({k: v for k, v in kwargs.items() if "token" in k})
            raise error

        from mlflow.genai.utils.llm_utils import _DEFAULT_MAX_TOKENS
        opt = po.strict_metaprompt_optimizer(reflection_model="openai:/claude-opus-4.8", guidelines="follow-ups")
        with mock.patch.object(litellm, "completion", side_effect=refuse), \
             self.assertRaises(po.OptimizerFailed) as caught:
            opt._optimize_zero_shot(PROMPTS, {}, False)
        self.assertEqual(caught.exception.code, "insufficient_balance")
        self.assertEqual(seen, {"max_completion_tokens": _DEFAULT_MAX_TOKENS})   # exactly one output limit, MLflow's

        job = SimpleNamespace(id="j2", stage="", status="running", result=None, error=None)
        with mock.patch("prompt_optimizer.PromptOptimizer") as optimizer, \
             mock.patch.object(services, "_endpoint_for"), mock.patch.object(services, "_tracking_uri"), \
             mock.patch.object(services, "build_config"), mock.patch.object(services, "record_run"), \
             mock.patch("traceback.print_exc"):
            optimizer.return_value.optimize.side_effect = caught.exception
            services._run_job(job, {"mode": "zero_shot", "name": "t", "_owner": "alice"})
        self.assertEqual(job.status, "error")
        self.assertEqual(job.error, caught.exception.message)
        self.assertIsNone(job.result)

    def test_job_records_the_error_when_prompt_optimizer_cannot_be_imported(self):
        import builtins
        from optimizer import services
        real_import = builtins.__import__

        def broken(name, *args, **kwargs):
            if name == "prompt_optimizer":
                raise ImportError("No module named 'some_runtime_dep'")
            return real_import(name, *args, **kwargs)

        job = SimpleNamespace(id="j3", stage="", status="running", result=None, error=None)
        with mock.patch("builtins.__import__", side_effect=broken), \
             mock.patch.object(services, "record_run") as record, mock.patch("traceback.print_exc"):
            services._run_job(job, {"mode": "zero_shot", "name": "t", "_owner": "alice"})
        self.assertEqual(job.status, "error")
        self.assertIn("some_runtime_dep", job.error)
        self.assertIn("some_runtime_dep", record.call_args.kwargs["error"])
