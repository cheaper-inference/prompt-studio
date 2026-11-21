"""The quote scanner agrees with the pattern it replaced and stays fast on hostile text."""
import random
import re
import time

from django.test import SimpleTestCase

from ps_utils.rubric_parser import _PLACEHOLDER, _protect_quotes

# The pattern the scanner replaced, kept as the reference behaviour.
_QUOTED_REFERENCE = re.compile(r'"([^"]+)"|“([^”]+)”|(?<!\w)\'(.+?)\'(?!\w)|(?<!\w)‘(.+?)’(?!\w)')

# Far above what a linear scan needs here (milliseconds, even on a busy CI runner) and far below
# what the quadratic pattern took on the same text (minutes).
BUDGET_SECONDS = 5.0


def _protect_quotes_reference(text):
    terms = []

    def keep(match):
        terms.append(next(group for group in match.groups() if group is not None))
        return _PLACEHOLDER.format(len(terms) - 1)
    return _QUOTED_REFERENCE.sub(keep, text), terms


def _random_texts(alphabet, count, longest, seed=20261121):
    rng = random.Random(seed)
    return [''.join(rng.choice(alphabet) for _ in range(rng.randint(0, longest))) for _ in range(count)]


def _seconds(function, text):
    started = time.perf_counter()
    function(text)
    return time.perf_counter() - started


class QuoteScannerTests(SimpleTestCase):
    def test_finds_the_terms_the_pattern_found(self):
        written = [
            'Mention "urgent care" and “walk-in clinic” once.',
            "Don't use 'cheap' or the customer's name.",
            'Use ‘same day’ but don’t say ‘now’.',
            'An empty "" pair, a lone " and a lone “ are left alone.',
            "Quotes 'do not\nspan' lines, \"but these\ndo\".",
            "''a' 'b'' it's 'x'y' ‘‘a’ ’b’",
        ]
        for text in written + _random_texts('"“”\'‘’ab1_ é.\n-', count=5000, longest=24):
            self.assertEqual(_protect_quotes(text), _protect_quotes_reference(text), repr(text))

    def test_unclosed_quotes_do_not_slow_the_scan(self):
        # Every quote here can open a term and none can close one, so the pattern rescanned the
        # rest of the text from each of them.
        hostile = {
            'single quotes that open and never close': " 'a" * 50000,
            'curly single quotes that open and never close': ' ‘a' * 50000,
            'curly double quotes that never close': '“' * 100000,
            'single quotes closed only after a line break': " 'a" * 50000 + "\n'",
        }
        for name, text in hostile.items():
            with self.subTest(name):
                self.assertLess(_seconds(_protect_quotes, text), BUDGET_SECONDS)

