"""Offline endpoint-kwargs contract tests; never instantiate an inference client."""
import unittest

from llm_endpoint import DEFAULT_MODEL, Endpoint, configure


class ModelDefaultTests(unittest.TestCase):
    def test_endpoint_and_configuration_use_available_model_defaults(self):
        self.assertEqual(DEFAULT_MODEL, "deepseek-v4.1-flash")
        endpoints = (
            Endpoint(base_url="http://127.0.0.1:1234/v1", api_key="fixture-key"),
            configure(base_url="http://127.0.0.1:1234/v1", api_key="fixture-key",
                      env_file=None, set_openai_env=False),
        )
        for endpoint in endpoints:
            with self.subTest(endpoint=type(endpoint).__name__):
                self.assertEqual(endpoint.task_model, DEFAULT_MODEL)
                self.assertEqual(endpoint.judge_model, DEFAULT_MODEL)
                self.assertEqual(endpoint.reflection_model, DEFAULT_MODEL)

    def test_explicit_model_choices_are_preserved(self):
        endpoint = configure(base_url="http://127.0.0.1:1234/v1", api_key="fixture-key",
                             task_model="custom-task", judge_model="custom-judge",
                             reflection_model="custom-reflection",
                             env_file=None, set_openai_env=False)
        self.assertEqual(endpoint.task_model, "custom-task")
        self.assertEqual(endpoint.judge_model, "custom-judge")
        self.assertEqual(endpoint.reflection_model, "custom-reflection")


class ReflectionKwargsTests(unittest.TestCase):
    def setUp(self):
        self.endpoint = Endpoint(base_url="http://127.0.0.1:1234/v1", api_key="fixture-key")

    def test_meta_reflection_calls_have_a_default_metering_purpose(self):
        self.assertEqual(self.endpoint.lm_kwargs(), {
            "api_base": self.endpoint.base_url,
            "api_key": "fixture-key",
            "extra_headers": {"x-gepa-purpose": "reflection"},
        })

    def test_custom_headers_merge_without_losing_the_default_or_mutating_the_caller(self):
        headers = {"x-correlation-id": "correlation"}
        result = self.endpoint.lm_kwargs(temperature=0.5, extra_headers=headers)
        self.assertEqual(result["temperature"], 0.5)
        self.assertEqual(result["extra_headers"], {
            "x-gepa-purpose": "reflection", "x-correlation-id": "correlation"})
        self.assertEqual(headers, {"x-correlation-id": "correlation"})

    def test_explicit_purpose_override_is_preserved_for_both_optimizer_paths(self):
        options = {"extra_headers": {"x-gepa-purpose": "training_judge", "x-extra": "value"}}
        self.assertEqual(self.endpoint.lm_kwargs(**options)["extra_headers"], options["extra_headers"])
        result = self.endpoint.gepa_kwargs(reflection_lm_kwargs=options, reflection_minibatch_size=3)
        self.assertEqual(result["reflection_lm_kwargs"]["extra_headers"], options["extra_headers"])
        self.assertEqual(result["reflection_lm_kwargs"]["num_retries"], 0)
        self.assertEqual(result["reflection_minibatch_size"], 3)
        self.assertEqual(options, {"extra_headers": {"x-gepa-purpose": "training_judge", "x-extra": "value"}})

    def test_gepa_custom_headers_also_retain_default_reflection_attribution(self):
        result = self.endpoint.gepa_kwargs(reflection_lm_kwargs={
            "extra_headers": {"x-extra": "value"}, "num_retries": 2})
        self.assertEqual(result["reflection_lm_kwargs"]["extra_headers"], {
            "x-gepa-purpose": "reflection", "x-extra": "value"})
        self.assertEqual(result["reflection_lm_kwargs"]["num_retries"], 2)


if __name__ == "__main__":
    unittest.main()
