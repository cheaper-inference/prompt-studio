"""A timed-out or failed task/judge call scores its example 0 instead of ending the run
(production, Nov 8: one gpt-5-nano judge timeout failed a 3-row few-shot run); a model that
keeps failing stops the run with a clear message."""
from unittest import mock

import httpx
import openai
from django.test import SimpleTestCase
from mlflow.entities import Feedback
from mlflow.exceptions import MlflowException
from mlflow.genai.scorers import scorer

import prompt_optimizer as po


def timeout():
    return openai.APITimeoutError(request=httpx.Request("POST", "https://ci.example/v1/chat/completions"))


def status(code):
    return openai.APIStatusError(f"Error code: {code}", response=httpx.Response(code, request=httpx.Request("POST", "https://x")),
                                 body=None)


class TransientTests(SimpleTestCase):
    def test_timeouts_connection_errors_and_5xx_are_transient(self):
        """Worth scoring as a miss: timeouts, dropped connections, 5xx, also when wrapped."""
        wrapped = MlflowException("Failed to call the judge")
        wrapped.__cause__ = timeout()
        for exc in (timeout(), openai.APIConnectionError(request=httpx.Request("POST", "https://x")), status(503),
                    status(500), TimeoutError("read timed out"), wrapped):
            self.assertTrue(po._is_transient(exc), exc)

    def test_bad_requests_auth_and_bugs_are_not(self):
        """A 4xx or a code bug is not absorbed: it would fail every call the same way."""
        for exc in (status(400), status(401), status(402), ValueError("bad"), KeyError("x")):
            self.assertFalse(po._is_transient(exc), exc)


class GuardTests(SimpleTestCase):
    def guard(self):
        return po._CallGuard({"task": "claude-sonnet-4.5", "judge": "gpt-5-nano"}, 90)

    def test_a_few_failures_keep_the_run_going(self):
        """Four failures among many calls: no stop."""
        guard = self.guard()
        for _ in range(4):
            guard.failed("judge", timeout())
            guard.ok("judge")
        guard.check()
        self.assertEqual(guard.summary(), {"judge": 4})

    def test_mostly_failing_model_stops_the_run_with_a_clear_message(self):
        """Five failures out of seven calls: stop, naming the role, the model and the timeout."""
        guard = self.guard()
        guard.ok("judge")
        guard.ok("judge")
        for _ in range(5):
            guard.failed("judge", timeout())
        with self.assertRaises(po.OptimizerFailed) as caught:
            guard.check()
        self.assertEqual(caught.exception.code, "model_timeout")
        self.assertEqual(caught.exception.message, "The judge model (gpt-5-nano) didn't answer within 90 s on 5 of 7 "
                                                   "calls, so the run stopped. Try again, or choose a faster judge model.")

    def test_five_failures_in_a_row_stop_even_after_many_successes(self):
        """A model that starts failing for good (an outage) stops the run after five in a row."""
        guard = self.guard()
        for _ in range(20):
            guard.ok("task")
        for _ in range(5):
            guard.failed("task", timeout())
        self.assertIsNotNone(guard.stopped)
        self.assertIn("task model (claude-sonnet-4.5)", guard.stopped.message)


class WrapperTests(SimpleTestCase):
    def setUp(self):
        self.guard = po._CallGuard({"task": "t", "judge": "j"}, 90)

    def test_failed_task_call_answers_the_marker_and_others_raise(self):
        """A transient task failure becomes TASK_CALL_FAILED; any other error still raises."""
        flaky = po._guard_predict(mock.Mock(side_effect=timeout()), self.guard)
        self.assertEqual(flaky(question="q"), po.TASK_CALL_FAILED)
        with self.assertRaises(ValueError):
            po._guard_predict(mock.Mock(side_effect=ValueError("bug")), self.guard)(question="q")
        self.assertEqual(po._guard_predict(lambda **kw: "answer", self.guard)(question="q"), "answer")
        self.assertEqual(self.guard.summary(), {"task": 1})

    def test_marker_scores_zero_without_calling_the_judge(self):
        """No answer, no judge call: the example scores 0 with the reason."""
        judge = mock.Mock(return_value=1.0)

        @scorer(name="quality")
        def quality(outputs=None):
            return judge(outputs)

        result = po._guard_scorer(quality, self.guard).run(outputs=po.TASK_CALL_FAILED)
        self.assertIsInstance(result, Feedback)
        self.assertEqual(result.value, 0.0)
        judge.assert_not_called()

    def test_judge_timeout_scores_zero_then_stops_once_it_keeps_failing(self):
        """Each judge timeout scores 0; the fifth in a row raises the run's OptimizerFailed."""
        @scorer(name="quality")
        def quality(outputs=None):
            raise timeout()

        guarded = po._guard_scorer(quality, self.guard)
        for _ in range(4):
            result = guarded.run(outputs="an answer")
            self.assertEqual((result.value, "judge model call" in result.rationale), (0.0, True))
        with self.assertRaises(po.OptimizerFailed) as caught:
            guarded.run(outputs="an answer")
        self.assertIn("judge model (j)", caught.exception.message)

    def test_judge_bugs_and_deterministic_scorers_pass_through(self):
        """A non-transient judge error still raises; the constraint scorer is not counted as a judge call."""
        @scorer(name="quality")
        def broken(outputs=None):
            raise KeyError("result")

        @scorer(name="output_constraints")
        def constraints(outputs=None):
            return 1.0

        with self.assertRaises(KeyError):
            po._guard_scorer(broken, self.guard).run(outputs="x")
        self.assertEqual(po._guard_scorer(constraints, self.guard).run(outputs="x"), 1.0)
        self.assertEqual(self.guard.counts["judge"]["calls"], 0)
