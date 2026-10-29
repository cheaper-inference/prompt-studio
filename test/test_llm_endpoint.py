"""Offline endpoint-kwargs contract tests; never instantiate an inference client."""
import unittest

from llm_endpoint import Endpoint


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
