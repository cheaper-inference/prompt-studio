from html.parser import HTMLParser

from django.template.loader import render_to_string
from django.test import SimpleTestCase

from llm_endpoint import DEFAULT_MODEL


class _Inputs(HTMLParser):
    def __init__(self):
        super().__init__()
        self.values = {}

    def handle_starttag(self, tag, attrs):
        if tag == "input":
            attrs = dict(attrs)
            self.values[attrs.get("id")] = attrs.get("value")


class StudioModelDefaultTests(SimpleTestCase):
    def test_rendered_form_defaults_match_shared_endpoint(self):
        inputs = _Inputs()
        inputs.feed(render_to_string("optimizer/index.html"))
        for field in ("taskModel", "judgeModel", "targetName"):
            with self.subTest(field=field):
                self.assertEqual(inputs.values[field], DEFAULT_MODEL)
        self.assertEqual(DEFAULT_MODEL, "deepseek-v4.1-flash")

    def test_unavailable_model_error_points_to_settings(self):
        from optimizer.minime.jobs import _plain_error

        message = _plain_error(
            "Error code: 404 - Model 'unavailable-model' is not currently available.")
        self.assertEqual(message,
                         "model unavailable (404): choose another model in Settings")
        self.assertEqual(_plain_error("HTTP 404: unrelated missing resource"),
                         "HTTP 404: unrelated missing resource")
