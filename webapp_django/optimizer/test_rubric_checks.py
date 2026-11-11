"""Rubric compilation (mandatory code checks), answer parsing, and mixed scoring."""
import json
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
from django.test import SimpleTestCase
from ps_utils.rubric_checks import combined_scorer, RubricEvaluationError
from ps_utils.rubric_parser import compile_rubric, extract_items, extract_judged_text, group_criteria, run_check
from . import benchmarks as b
from .test_benchmarks import payload

KEYWORDS = '1. urgent care clinic near me\n2. walk-in clinic same day appointment\n3. emergency room alternative open now'


def checks(criterion):
    entry = compile_rubric([criterion])[0]
    return entry['method'], entry['checks']


class RubricParserTests(SimpleTestCase):
    def test_counts_and_lengths_compile_to_code(self):
        cases = {
            '3 keywords in total.': [{'kind': 'item_count', 'min': 3, 'max': 3}],
            'Return exactly three headlines': [{'kind': 'item_count', 'min': 3, 'max': 3}],
            'Between 3 and 5 items': [{'kind': 'item_count', 'min': 3, 'max': 5}],
            'At least 2 suggestions': [{'kind': 'item_count', 'min': 2, 'max': None}],
            'Each keyword must contain less than or equal to 5 words.': [{'kind': 'length', 'unit': 'words', 'scope': 'item', 'min': None, 'max': 5}],
            'Keywords must be ≤ 5 words': [{'kind': 'length', 'unit': 'words', 'scope': 'item', 'min': None, 'max': 5}],
            'Each headline: 4-6 words': [{'kind': 'length', 'unit': 'words', 'scope': 'item', 'min': 4, 'max': 6}],
            'Keep the answer under 50 words': [{'kind': 'length', 'unit': 'words', 'scope': 'answer', 'min': None, 'max': 49}],
            'The response must be fewer than 100 characters': [{'kind': 'length', 'unit': 'characters', 'scope': 'answer', 'min': None, 'max': 99}],
            'Return a JSON array': [{'kind': 'json', 'shape': 'array'}],
            'No duplicates': [{'kind': 'unique_items'}],
            "Must not mention 'free'": [{'kind': 'contains', 'term': 'free', 'negate': True, 'scope': 'answer'}],
            'The answer must not exceed 200 chars': [{'kind': 'length', 'unit': 'characters', 'scope': 'answer', 'min': None, 'max': 200}],
            'Keep the reply within 280 characters': [{'kind': 'length', 'unit': 'characters', 'scope': 'answer', 'min': None, 'max': 280}],
        }
        for criterion, expected in cases.items():
            self.assertEqual(checks(criterion), ('deterministic', expected), criterion)

    def test_unrecognised_limits_go_to_the_judge_not_exact_counts(self):
        for criterion in ['Around 200 characters', 'Roughly 50 words', 'Must not ever exceed 100 words']:
            self.assertFalse(any(c['kind'] == 'length' for c in checks(criterion)[1]), criterion)

    def test_distant_negations_never_become_minimum_lengths(self):
        for criterion in [
            'Do not under any circumstances exceed 100 words',
            'The answer must never under any circumstances go beyond 50 words',
            'The answer cannot under any circumstances surpass 200 characters',
        ]:
            with self.subTest(criterion=criterion):
                entry = compile_rubric([criterion])[0]
                self.assertFalse(any(c['kind'] == 'length' for c in entry['checks']))
                self.assertEqual(entry['method'], 'llm_judge')
                self.assertEqual(entry['judge_text'], criterion)

    def test_limit_phrasings_compile_to_code(self):
        def length(low, high, unit='characters'):
            return [{'kind': 'length', 'unit': unit, 'scope': 'answer', 'min': low, 'max': high}]
        cases = {'The answer must not surpass 200 characters': length(None, 200),
                 'Must not go beyond 50 words': length(None, 50, 'words'),
                 "Can't top 100 characters": length(None, 100),
                 'Should not be above 30 words': length(None, 30, 'words'),
                 'Must not be over 280 characters': length(None, 280),
                 "Don't exceed 120 characters": length(None, 120),
                 'Must not be greater than 40 characters': length(None, 40),
                 '200 characters max': length(None, 200),
                 '60 characters maximum': length(None, 60),
                 'Must exceed 100 words': length(101, None, 'words'),
                 'Surpasses 20 words': length(21, None, 'words')}
        for criterion, expected in cases.items():
            self.assertEqual(checks(criterion), ('deterministic', expected), criterion)

    def test_negated_one_sided_limits_flip_instead_of_inverting(self):
        def length(low, high, unit='words'):
            return [{'kind': 'length', 'unit': unit, 'scope': 'answer', 'min': low, 'max': high}]
        def count(low, high):
            return [{'kind': 'item_count', 'min': low, 'max': high}]
        cases = {'The answer should not have more than 3 bullet points.': count(None, 3),
                 'No fewer than 3 items.': count(3, None),
                 'Do not write fewer than 3 items.': count(3, None),
                 'No less than 20 words.': length(20, None),
                 'Must not have less than 20 words.': length(20, None),
                 'Never more than 50 words.': length(None, 50),
                 'Do not use more than 50 words.': length(None, 50),
                 'Must not contain over 100 characters.': length(None, 100, 'characters'),
                 'Do not use 5 words or more.': length(None, 4),
                 "Don't go under 10 words.": length(10, None),
                 'Should never exceed 50 words.': length(None, 50),
                 'More than 3 items.': count(4, None),
                 'Under 50 words.': length(None, 49)}
        for criterion, expected in cases.items():
            self.assertEqual(checks(criterion), ('deterministic', expected), criterion)
        # A negation further away is not guessed at: the limit goes to the judge.
        for criterion in ['Do not under any circumstances use more than 50 words.',
                          'No reply should have more than 3 items.', 'No reply, ever, should have more than 3 items.',
                          'No answer should, under any circumstances, exceed 50 words.',
                          'No reply, ever, exceeds 50 words.', 'No emojis, under 50 words.',
                          # Negations that are not "not": none of these is a minimum.
                          'Answer without exceeding 50 words.', 'Avoid more than 50 words.', 'Nothing over 50 words.',
                          'More than 50 words is not allowed.', 'Answers over 50 words fail.',
                          'None of the keywords should exceed 5 words.']:
            entry = compile_rubric([criterion])[0]
            self.assertEqual((entry['method'], entry['checks'], entry['judge_text']), ('llm_judge', [], criterion), criterion)

    def test_limits_code_cannot_place_stay_with_the_judge(self):
        cases = {'Do not use more than 50 words or exceed 200 characters.':
                     [{'kind': 'length', 'unit': 'words', 'scope': 'answer', 'min': None, 'max': 50}],
                 'Never include "spam", keep under 50 words.':
                     [{'kind': 'contains', 'term': 'spam', 'negate': True, 'scope': 'answer'}],
                 'No duplicates, the answer should be under 50 words.': [{'kind': 'unique_items'}]}
        for criterion, expected in cases.items():
            with self.subTest(criterion=criterion):
                entry = compile_rubric([criterion])[0]
                self.assertEqual((entry['method'], entry['checks']), ('hybrid', expected))
                self.assertTrue(all(run_check('A short answer.', c)[0] for c in entry['checks']))
                # The judge is told to ignore only what code verified, never the limit left to it.
                note = entry['judge_text'][len(criterion):]
                self.assertNotIn('characters', note)
                self.assertNotIn('lengths', note)
                if not any(c['kind'] == 'length' for c in expected):
                    self.assertNotIn('words', note)

    def test_negated_limits_pass_compliant_answers(self):
        check = checks('The answer should not have more than 3 bullet points.')[1][0]
        self.assertTrue(run_check('- a\n- b', check)[0])
        self.assertFalse(run_check('- a\n- b\n- c\n- d\n- e', check)[0])
        check = checks('No fewer than 3 items.')[1][0]
        self.assertTrue(run_check('- a\n- b\n- c\n- d', check)[0])
        self.assertFalse(run_check('- a\n- b', check)[0])

    def test_apostrophes_do_not_end_quoted_terms(self):
        def contains(term, negate):
            return [{'kind': 'contains', 'term': term, 'negate': negate, 'scope': 'answer'}]
        cases = {"Never include 'we're sorry'.": contains("we're sorry", True),
                 "Must include the phrase 'don't worry'.": contains("don't worry", False),
                 "Don't mention 'price'.": contains('price', True),
                 'Don’t mention ‘price’.': contains('price', True),
                 "It's required to include 'refund policy'.": contains('refund policy', False),
                 "Do not use the word 'can't'.": contains("can't", True)}
        for criterion, expected in cases.items():
            self.assertEqual(checks(criterion)[1], expected, criterion)
        check = checks("Never include 'we're sorry'.")[1][0]
        self.assertTrue(run_check('We will ship your order tomorrow.', check)[0])
        self.assertFalse(run_check("We're sorry for the delay.", check)[0])

    def test_negation_and_coordination_reach_every_quoted_term(self):
        def contains(*terms, negate=True):
            return [{'kind': 'contains', 'term': t, 'negate': negate, 'scope': 'answer'} for t in terms]
        self.assertEqual(checks("Don't mention 'price' or 'discount'."), ('deterministic', contains('price', 'discount')))
        self.assertEqual(checks("Never use 'a', 'b' or 'c'."), ('deterministic', contains('a', 'b', 'c')))
        self.assertEqual(checks("Include 'a', 'b', 'c'."), ('deterministic', contains('a', 'b', 'c', negate=False)))
        # A negation the pattern does not reach, or "either one" wording, is left to the judge whole.
        for criterion in ["Don't ever mention 'price'.", "Include 'price' or 'discount'.", "You may mention 'price'.",
                          "Including 'price' is optional.", "Including 'price' is not required.",
                          "Answers that don't mention 'refund' fail.", "Reject any reply that doesn't include 'refund'.",
                          "Don't use 'x' unless asked.", "Don't mention 'cheap', 'affordable' is fine.",
                          "Don't use 'x' more than 3 times."]:
            entry = compile_rubric([criterion])[0]
            self.assertEqual((entry['method'], entry['checks'], entry['judge_text']), ('llm_judge', [], criterion), criterion)

    def test_numeric_diagnostics_do_not_log_private_rubric_text(self):
        criterion = 'Include the private Customer-DEMO-123 launch plan in approximately 200 words'
        with self.assertLogs('ps_utils.rubric_parser', 'INFO') as logs:
            compile_rubric([criterion])
        self.assertIn('1 numeric clause(s) left to the judge', logs.output[0])
        self.assertNotIn('customer-demo-123', '\n'.join(logs.output).lower())
        self.assertNotIn(criterion.lower(), '\n'.join(logs.output).lower())

    def test_answer_length_ignores_quotes_wrapping_the_whole_answer(self):
        check = checks('The answer is at most 10 characters')[1][0]
        for answer, ok in [('"0123456789"', True), ('“0123456789”', True), ("'0123456789'", True),
                           ('0123456789', True), ('"012345678"9', False), ('say "hi" 0123', False),
                           ('"0123456789A"', False)]:
            self.assertEqual(run_check(answer, check)[0], ok, answer)
        self.assertIn('quotation marks wrapping the whole answer not counted', run_check('"0123"', check)[1])

    def test_wrapping_quote_rules_are_a_code_check(self):
        for criterion in ['Any wrapping quotes FAIL.', 'The answer must be presented without any wrapper quotes',
                          'The whole answer must be presented without any wrapping quotes.',
                          'The answer must not be wrapped in quotation marks.', 'Do not wrap the answer in quotes.',
                          'Quotation marks must not wrap the whole answer.', 'No surrounding quotes.']:
            self.assertEqual(checks(criterion), ('deterministic', [{'kind': 'unquoted', 'quotes': 'any'}]), criterion)
        self.assertEqual(checks('At most 120 characters, without wrapping quotes')[1],
                         [{'kind': 'length', 'unit': 'characters', 'scope': 'answer', 'min': None, 'max': 120},
                          {'kind': 'unquoted', 'quotes': 'any'}])
        for criterion in ['No quotes.', 'Use quotes from customers', 'Wrap key terms in quotes',
                          'Each tagline must not be wrapped in quotes.']:   # not about wrapping the answer: the judge
            self.assertFalse(any(c['kind'] == 'unquoted' for c in checks(criterion)[1]), criterion)
        check = {'kind': 'unquoted', 'quotes': 'any'}
        for answer, ok in [('"Save 40% on inference"', False), ('“Save big”', False), ("'Save big'", False),
                           ('  "Save big"\n', False), ('Save "big" now', True), ('Save big', True), ('"Save" big', True),
                           ("'Don't wait'", False), ('‘Don’t wait’', False), ("Don't wait", True),   # apostrophes are not quotes
                           ("'Customers' choice'", False), ('‘Customers’ choice’', False),          # nor plural possessives
                           ("'Save' or 'win'", True), ("'Bosses' and 'workers'", True),            # two quoted spans
                           ('"Say \\"hi\\""', False), ('"ends with \\"', True),                   # escaped quotes are content
                           ('""', False), ('"a\\\\"', False), ('"a\\\\\\"', True)]:              # empty; \\ before the end is not an escape
            self.assertEqual(run_check(answer, check)[0], ok, answer)
        self.assertEqual(checks('No surrounding double quotes.')[1], [{'kind': 'unquoted', 'quotes': 'double'}])
        double = {'kind': 'unquoted', 'quotes': 'double'}
        self.assertEqual((run_check('"Save big"', double)[0], run_check("'Save big'", double)[0]), (False, True))
        length = checks('The answer is at most 10 characters')[1][0]
        self.assertTrue(run_check("'Don't wait'", length)[0])          # 10 characters inside the quotes
        self.assertTrue(run_check('"Say \\"hi\\""', checks('The answer is at most 10 characters')[1][0])[0])   # Say \"hi\" is 10
        from ps_utils.spec_draft import render
        for c in (check, double, {'kind': 'unquoted', 'quotes': 'single'}):
            self.assertEqual(checks(render(c, None))[1], [c])          # its canonical wording compiles back to it

    def test_every_character_is_how_to_count_not_a_list(self):
        self.assertEqual(checks('The tweet is at most 280 characters long, counting every character including emojis')[1],
                         [{'kind': 'length', 'unit': 'characters', 'scope': 'answer', 'min': None, 'max': 280}])

    def test_scope_survives_clause_splitting(self):
        method, found = checks("Each tagline must include 'Keak' and be no more than 8 words")
        self.assertEqual(method, 'deterministic')
        self.assertEqual({(c['kind'], c['scope']) for c in found}, {('contains', 'item'), ('length', 'item')})

    def test_semantic_criteria_go_to_the_judge_unchanged(self):
        for criterion in ['Each keyword must be relevant to the healthcare domain.', 'Be concise and friendly',
                          'Highlight the unique value proposition', 'Answer in at most 2 sentences', 'Use at least 2 hashtags']:
            entry = compile_rubric([criterion])[0]
            self.assertEqual((entry['method'], entry['checks'], entry['judge_text']), ('llm_judge', [], criterion), criterion)

    def test_mixed_clause_is_hybrid_and_scopes_the_judge(self):
        entry = compile_rubric(['Return exactly 3 headlines about urgent care'])[0]
        self.assertEqual(entry['method'], 'hybrid')
        self.assertEqual(entry['checks'], [{'kind': 'item_count', 'min': 3, 'max': 3}])
        self.assertIn('verified separately in code', entry['judge_text'])


class AnswerParsingTests(SimpleTestCase):
    def test_formats_are_detected(self):
        self.assertEqual(extract_items('["a b", "c"]'), (['a b', 'c'], 'json_array'))
        self.assertEqual(extract_items('```json\n["a", "b"]\n```'), (['a', 'b'], 'json_array'))
        self.assertEqual(extract_items('a\n\nb'), (['a', 'b'], 'lines'))
        self.assertEqual(extract_items('Here are three:\n1. a\n2) b\n- "c"'), (['a', 'b', 'c'], 'marked_lines'))
        for bad in ['{"a": 1}', '[1, 2]', '[]', '', '1.\n2.']:
            self.assertIsNone(extract_items(bad)[0], bad)

    def test_regression_numbered_keywords_pass_in_code(self):
        # An LLM judge failed "<= 5 words" here while its own explanation said all were 5 or fewer.
        compiled = compile_rubric(['3 keywords in total.', 'Each keyword must contain less than or equal to 5 words.'])
        results = [run_check(KEYWORDS, check) for entry in compiled for check in entry['checks']]
        self.assertTrue(all(passed for passed, _, _ in results))
        self.assertEqual(results[1][2]['words_per_item'], [5, 5, 5])

    def test_check_failures_explain_the_measurement(self):
        passed, explanation, observed = run_check('a b c d e f', {'kind': 'length', 'unit': 'words', 'scope': 'answer', 'min': None, 'max': 5})
        self.assertFalse(passed); self.assertEqual(observed, {'words': 6}); self.assertIn('at most 5', explanation)
        passed, explanation, _ = run_check('free trial\nsale', {'kind': 'contains', 'term': 'free', 'negate': True, 'scope': 'item'})
        self.assertFalse(passed); self.assertIn('1 of 2', explanation)
        self.assertFalse(run_check('carefree', {'kind': 'contains', 'term': 'free', 'negate': False, 'scope': 'answer'})[0])
        self.assertFalse(run_check('a\nA', {'kind': 'unique_items'})[0])
        self.assertFalse(run_check('not json', {'kind': 'json', 'shape': 'any'})[0])


class CombinedScorerTests(SimpleTestCase):
    def test_all_code_criteria_skip_the_judge(self):
        make_judge = Mock(side_effect=AssertionError('Judge must not be called'))
        evidence = []
        compiled = compile_rubric(['3 keywords in total.', 'Each keyword must contain less than or equal to 5 words.'])
        self.assertEqual(combined_scorer(compiled, make_judge, evidence.append)(outputs=KEYWORDS), 1)
        make_judge.assert_not_called()
        self.assertEqual([c['method'] for c in evidence[-1]['criteria']], ['deterministic'] * 2)
        self.assertTrue(evidence[-1]['overall_pass'])

    def test_judge_sees_only_semantic_criteria_in_order(self):
        evidence = []
        def factory(texts, receive):
            self.assertEqual(texts, ['Each keyword must be relevant to the healthcare domain.'])
            return lambda **kw: receive({'criteria': [dict(passed=False, status='fail', explanation='Off topic', evidence=None)]})
        compiled = compile_rubric(['3 keywords in total.', 'Each keyword must be relevant to the healthcare domain.'])
        self.assertEqual(combined_scorer(compiled, factory, evidence.append)(outputs=KEYWORDS), .5)
        self.assertEqual([c['passed'] for c in evidence[-1]['criteria']], [True, False])
        self.assertEqual(evidence[-1]['criteria'][1]['criterion'], 'Each keyword must be relevant to the healthcare domain.')

    def test_hybrid_fails_in_code_without_judging_and_combines_explanations(self):
        compiled = compile_rubric(['Return exactly 3 headlines about urgent care'])
        make_judge = Mock(side_effect=AssertionError('Judge must not be called'))
        evidence = []
        self.assertEqual(combined_scorer(compiled, make_judge, evidence.append)(outputs='a\nb'), 0)
        self.assertEqual(evidence[-1]['criteria'][0]['method'], 'deterministic')
        evidence = []
        factory = lambda texts, receive: (lambda **kw: receive({'criteria': [dict(passed=True, status='pass', explanation='On topic', evidence='a')]}))
        self.assertEqual(combined_scorer(compiled, factory, evidence.append)(outputs='a\nb\nc'), 1)
        detail = evidence[-1]['criteria'][0]
        self.assertEqual((detail['method'], detail['compiled_method']), ('llm_judge', 'hybrid'))
        self.assertTrue(detail['explanation'].startswith('Code: Observed 3 items') and 'Judge: On topic' in detail['explanation'])

    def test_judge_failure_keeps_code_evidence_and_is_not_a_fail(self):
        for error in [RubricEvaluationError('invalid_evidence'), TimeoutError('secret must not be published')]:
            evidence = []
            compiled = compile_rubric(['3 keywords in total.', 'Be relevant'])
            judge = combined_scorer(compiled, lambda *args: Mock(side_effect=error), evidence.append)
            with self.assertRaises(type(error)): judge(outputs=KEYWORDS)
            self.assertIsNone(evidence[-1]['overall_pass'])
            self.assertEqual([d['passed'] for d in evidence[-1]['criteria']], [True, None])
            self.assertNotIn('secret', json.dumps(evidence))

    def test_benchmark_records_compiled_rubric_and_ignores_client_checks(self):
        from llm_endpoint import Endpoint
        p = payload(); p.update(evaluation_mode='rubric', criteria=['Exactly 4 items'],
                                deterministic_checks=[{'criterion_index': 0, 'criterion': 'stale', 'kind': 'bogus'}])
        manifest = b.validate(p)
        self.assertEqual(manifest['scoring']['rubric_compiler'], 'rubric-parser-v8')
        self.assertEqual(manifest['scoring']['compiled_rubric'][0]['method'], 'deterministic')
        self.assertNotIn('deterministic_checks', manifest['scoring'])
        endpoint = Endpoint(base_url='https://unused.invalid/v1', api_key='never-sent')
        response = NS(usage=NS(prompt_tokens=10, completion_tokens=20, total_tokens=30, completion_tokens_details=None),
                      choices=[NS(message=NS(content='["a","b","c","d"]'))])
        with patch.object(endpoint.client.chat.completions, 'create', return_value=response) as create:
            result = b.execute(manifest, endpoint, Mock())
        self.assertEqual(create.call_count, 8)   # task calls only: 2 models x 2 variants x 2 rows
        for arm in result['arms']:
            self.assertEqual(arm['quality']['rubric']['pass_rate'], 1)
            self.assertTrue(all(case['judge_usage']['calls'] == 0 for case in arm['cases']))

    def test_quality_rate_excludes_errors_but_keeps_real_failures(self):
        rows = [dict(graded_by='rubric', score=score, error=error, usage=None) for score, error in [(1, None), (0, None), (None, 'TimeoutError')]]
        result = b.summarize(rows, {})['quality']['rubric']
        self.assertEqual((result['n'], result['evaluated'], result['evaluation_errors'], result['failed']), (3, 2, 1, 1))
        self.assertEqual(result['pass_rate'], .5)
        self.assertIsNone(b.summarize(rows[2:], {})['quality']['rubric']['pass_rate'])



class VerdictUnderJudgeErrorTests(SimpleTestCase):
    """A failed criterion decides the all-criteria verdict even if the judge errors."""
    CRITERIA = ['Exactly 4 items', 'Each item must be relevant to the healthcare domain.']

    def test_code_fail_plus_judge_error_is_a_fail(self):
        evidence = []
        compiled = compile_rubric(['3 keywords in total.', 'Be relevant'])
        judge = combined_scorer(compiled, lambda *args: Mock(side_effect=TimeoutError('down')), evidence.append)
        with self.assertRaises(TimeoutError): judge(outputs='a\nb')
        self.assertEqual([d['passed'] for d in evidence[-1]['criteria']], [False, None])
        self.assertIs(evidence[-1]['overall_pass'], False)
        self.assertEqual(evidence[-1]['evaluation_errors'], 1)

    def run_benchmark(self, answer):
        from llm_endpoint import Endpoint
        p = payload(); p.update(evaluation_mode='rubric', models=['small'], optimized=[], criteria=self.CRITERIA,
                                rows=[{'inputs': {'q': 'a'}}])
        endpoint = Endpoint(base_url='https://unused.invalid/v1', api_key='never-sent')
        def create(**kw):
            if kw['model'] == 'judge':
                raise RuntimeError('judge down')
            return NS(model=kw['model'], usage=NS(prompt_tokens=1, completion_tokens=1, total_tokens=2, completion_tokens_details=None),
                      choices=[NS(message=NS(content=answer))])
        with patch.object(endpoint.client.chat.completions, 'create', side_effect=create):
            return b.execute(b.validate(p), endpoint, Mock())['arms'][0]

    def test_benchmark_counts_known_fail_in_strict_rate_not_in_mean(self):
        arm = self.run_benchmark('["a","b"]')                     # 2 items: fails in code
        case, quality = arm['cases'][0], arm['quality']['rubric']
        self.assertIsNone(case['error'])
        self.assertIs(case['rubric_evaluation']['overall_pass'], False)
        self.assertEqual(case['evaluations']['rubric'],
                         {'score': None, 'error': None, 'overall_pass': False, 'judge_error': 'RuntimeError'})
        self.assertEqual((quality['evaluated'], quality['failed'], quality['evaluation_errors']), (1, 1, 0))
        self.assertEqual(quality['pass_rate'], 0)
        self.assertEqual(quality['mean_cases'], 0)
        self.assertIsNone(quality['mean_score'])
        self.assertEqual(arm['errors'], 0)

    def test_benchmark_unknown_outcome_stays_an_evaluation_error(self):
        arm = self.run_benchmark('["a","b","c","d"]')             # passes code: verdict depends on the judge
        case, quality = arm['cases'][0], arm['quality']['rubric']
        self.assertEqual(case['error'], 'RuntimeError')
        self.assertIsNone(case['rubric_evaluation']['overall_pass'])
        self.assertEqual((quality['evaluated'], quality['evaluation_errors']), (0, 1))
        self.assertIsNone(quality['pass_rate'])

    def test_summarize_strict_rate_and_mean_cover_different_cases(self):
        rows = [dict(graded_by='rubric', score=1, error=None, usage=None),
                dict(graded_by='rubric', score=1/3, error=None, usage=None),
                dict(graded_by='rubric', score=None, error=None, overall_pass=False, usage=None),
                dict(graded_by='rubric', score=None, error='RuntimeError', usage=None)]
        q = b.summarize(rows, {})['quality']['rubric']
        self.assertEqual((q['n'], q['evaluated'], q['passed'], q['failed'], q['evaluation_errors']), (4, 3, 1, 2, 1))
        self.assertAlmostEqual(q['pass_rate'], 1/3)
        self.assertEqual(q['mean_cases'], 2)
        self.assertAlmostEqual(q['mean_score'], 2/3)

    def test_benchmark_judge_prompt_is_strict_optimizer_prompt_unchanged(self):
        from llm_endpoint import Endpoint
        from prompt_optimizer import PromptOptimizer, OptimizationConfig
        endpoint = Endpoint(base_url='https://unused.invalid/v1', api_key='never-sent')
        opt, cfg = PromptOptimizer(endpoint), OptimizationConfig(name='test', prompts=[])
        prompts = {}
        for label, kwargs, content in [('benchmark', {'on_judgment': lambda d: None},
                                        {'criteria': [{'explanation': 'ok', 'evidence': None, 'passed': True}]}),
                                       ('optimizer', {}, {'per_guideline': [True]})]:
            response = NS(usage=None, choices=[NS(message=NS(content=json.dumps(content)))])
            with patch.object(endpoint.client.chat.completions, 'create', return_value=response) as call:
                opt._build_guidelines_scorer(cfg, ['Short'], **kwargs)(inputs={}, outputs='answer')
            prompts[label] = call.call_args.kwargs['messages'][0]['content']
        self.assertIn('passes only if the response fully satisfies it', prompts['benchmark'])
        self.assertNotIn('do not require perfection', prompts['benchmark'])
        self.assertIn('independently and fairly — do not require perfection.', prompts['optimizer'])

    def test_evidence_matches_up_to_formatting_but_not_invented_text(self):
        from prompt_optimizer import _evidence_in
        out = 'Stop guessing, start watching. Export **custom metrics** to Prometheus \u2014 with @cheaperinfer.'
        for quote in ['custom metrics to Prometheus', '\u201cStop guessing,  start watching\u201d', 'Prometheus - with',
                      'STOP GUESSING', 'start watching...']:
            self.assertTrue(_evidence_in(quote, out), quote)
        for quote in ['fabricated', 'custom metrics to Grafana', '...', '""']:
            self.assertFalse(_evidence_in(quote, out), quote)

    def test_a_whole_answer_guideline_may_quote_several_excerpts(self):
        """No duplicate hashtags across ten tweets: the evidence lists pieces from different tweets.
        Accepted when every piece is in the answer; one invented piece still fails."""
        from prompt_optimizer import _evidence_in
        out = ('1. Talk to your devices #AIRevolution #VoiceTech\n2. Smarter homes #SmartTech\n'
               '3. Work less, do more #Productivity #AIEfficiency\n4. Create freely #CreativeAI')
        for quote in ['#AIRevolution, #VoiceTech, #SmartTech, #Productivity', '#AIRevolution \u2026 #CreativeAI',
                      '#AIRevolution | #SmartTech', '#AIRevolution\n#CreativeAI', 'Work less, do more',
                      '\u201cwork less\u201d; do more; #creativeai']:
            self.assertTrue(_evidence_in(quote, out), quote)
        for quote in ['#AIRevolution, #MadeUpTag', '#MadeUpTag', 'Smarter homes, invented claim', '... | ;',
                      'do, more, less', 'Smarter homes, do']:     # fragments too short to be evidence of anything
            self.assertFalse(_evidence_in(quote, out), quote)

    def test_invalid_evidence_gets_a_repair_that_says_how_to_quote(self):
        from llm_endpoint import Endpoint
        from prompt_optimizer import PromptOptimizer, OptimizationConfig
        endpoint = Endpoint(base_url='https://unused.invalid/v1', api_key='never-sent')
        opt, cfg = PromptOptimizer(endpoint), OptimizationConfig(name='test', prompts=[])
        bad = {'explanation': 'ok', 'evidence': 'not in the answer at all', 'passed': True}
        good = {'explanation': 'ok', 'evidence': 'answer', 'passed': True}
        replies = [NS(usage=None, choices=[NS(message=NS(content=json.dumps({'criteria': [c]})))]) for c in (bad, good)]
        got = []
        with patch.object(endpoint.client.chat.completions, 'create', side_effect=replies) as call:
            opt._build_guidelines_scorer(cfg, ['Short'], on_judgment=got.append)(inputs={}, outputs='answer')
        repair = call.call_args_list[1].kwargs['messages'][-1]['content']
        self.assertIn("several exact excerpts separated by ' \u2026 '", repair)
        self.assertEqual(got[0]['passed'], 1)

    def test_an_unusable_reply_is_kept_on_the_error_for_the_log(self):
        from llm_endpoint import Endpoint
        from prompt_optimizer import PromptOptimizer, OptimizationConfig
        endpoint = Endpoint(base_url='https://unused.invalid/v1', api_key='never-sent')
        opt, cfg = PromptOptimizer(endpoint), OptimizationConfig(name='test', prompts=[])
        cut = NS(usage=None, choices=[NS(finish_reason='length', message=NS(content='{"criteria": [{"explanation": "The'))])
        with patch.object(endpoint.client.chat.completions, 'create', return_value=cut), \
                self.assertRaises(RubricEvaluationError) as raised:
            opt._build_guidelines_scorer(cfg, ['Short'], on_judgment=lambda d: None)(inputs={}, outputs='answer')
        self.assertEqual((raised.exception.code, raised.exception.finish), ('invalid_json', 'length'))
        self.assertTrue(raised.exception.reply.startswith('{"criteria"'))

    def test_wrong_criterion_count_gets_a_specific_repair_and_then_passes(self):
        from llm_endpoint import Endpoint
        from prompt_optimizer import PromptOptimizer, OptimizationConfig
        endpoint = Endpoint(base_url='https://unused.invalid/v1', api_key='never-sent')
        opt, cfg = PromptOptimizer(endpoint), OptimizationConfig(name='test', prompts=[])
        one = {'explanation': 'ok', 'evidence': None, 'passed': True}
        replies = [NS(usage=None, choices=[NS(message=NS(content=json.dumps({'criteria': c})))]) for c in ([one], [one, one])]
        got = []
        with patch.object(endpoint.client.chat.completions, 'create', side_effect=replies) as call:
            opt._build_guidelines_scorer(cfg, ['Short', 'Polite'], on_judgment=got.append)(inputs={}, outputs='answer')
        repair = call.call_args_list[1].kwargs['messages'][-1]['content']
        self.assertIn('return exactly 2 criterion objects', repair)
        self.assertIn('(you returned 1)', repair)
        self.assertEqual(got[0]['passed'], 2)


CASE_7 = ('Here are 3 prospective keywords for occupational health services:\n\n'
          '1. **"pre-employment physical near me"** — High-intent, location-based keyword capturing job candidates.\n\n'
          '2. **"DOT physical exam"** — Targets commercial drivers who are required by law.\n\n'
          '3. **"workplace drug testing services"** — Captures employers (B2B demand).\n\n'
          '**Bonus considerations:** You could expand into *"workers comp doctor,"*.')
CASE_7_CRITERIA = ['3 keywords in total.', 'Each keyword must contain less than or equal to 5 words.',
                   'Each keyword must be relevant to the healthcare domain.']


class ItemScopedJudgeTests(SimpleTestCase):
    """Code extracts the keywords; the judge decides relevance per keyword; code applies "each"."""

    def test_per_item_semantic_criteria_are_item_scoped(self):
        scopes = {c: compile_rubric([c])[0]['judge_scope'] for c in
                  CASE_7_CRITERIA + ['Be concise and friendly', 'Return exactly 3 headlines about urgent care']}
        self.assertEqual(list(scopes.values()), [None, None, 'item', 'answer', 'answer'])

    def test_short_units_drop_commentary(self):
        cases = {'**"DOT physical exam"** — Targets drivers': ('DOT physical exam', 'quoted'),
                 '**teeth cleaning near me** — captures intent': ('teeth cleaning near me', 'bold'),
                 'DOT physical exam - Targets drivers': ('DOT physical exam', 'head'),
                 'pre-employment physical': ('pre-employment physical', 'whole'),
                 '*emergency room alternative*': ('emergency room alternative', 'whole'),
                 '"walk-in" clinic open now': ('"walk-in" clinic open now', 'whole'),   # quote is content
                 'Keyword: urgent care': ('Keyword: urgent care', 'whole')}   # a colon may precede the keyword
        for item, expected in cases.items():
            self.assertEqual(extract_judged_text(item, 'keyword'), expected, item)

    def test_other_units_are_judged_whole(self):
        for unit in ['headline', 'tagline', 'item', None]:
            for item in ['Save 20% — today only', '"Feel better faster": same-day visits', 'Care that fits your day - open 7 days']:
                self.assertEqual(extract_judged_text(item, unit), (item, 'whole'), (unit, item))
        self.assertEqual(extract_judged_text('"Same-day care, no appointment"', 'headline'), ('Same-day care, no appointment', 'whole'))

    def test_compiled_rubric_records_the_item_unit(self):
        units = [compile_rubric([c])[0]['item_unit'] for c in
                 ['Each keyword must be relevant to the healthcare domain.', 'Each headline must be catchy.', 'Be concise and friendly']]
        self.assertEqual(units, ['keyword', 'headline', None])

    def scorer(self, verdict, batch=None):
        seen = {}
        def make_item_judge(requirement, items, receive):
            seen.update(requirement=requirement, items=items)
            return lambda **kw: verdict(items, receive)
        batch = batch or Mock(side_effect=AssertionError('Batch judge must not be called'))
        return seen, lambda compiled, evidence: combined_scorer(compiled, batch, evidence.append, make_item_judge=make_item_judge)

    def test_case_7_judge_sees_only_extracted_keywords(self):
        seen, build = self.scorer(lambda items, receive: receive([dict(passed=True, explanation='Healthcare') for _ in items]))
        evidence = []
        self.assertAlmostEqual(build(compile_rubric(CASE_7_CRITERIA), evidence)(outputs=CASE_7), 2/3)
        self.assertEqual(seen['items'], ['pre-employment physical near me', 'DOT physical exam', 'workplace drug testing services'])
        self.assertEqual(seen['requirement'], CASE_7_CRITERIA[2])
        relevance = evidence[-1]['criteria'][2]
        self.assertEqual((relevance['status'], relevance['judge_scope'], relevance['evidence']), ('pass', 'item', None))
        self.assertEqual([r['rule'] for r in relevance['items']], ['quoted'] * 3)
        self.assertEqual(relevance['explanation'], 'All 3 items pass: "pre-employment physical near me"; '
                                                   '"DOT physical exam"; "workplace drug testing services".')
        self.assertIs(evidence[-1]['overall_pass'], False)          # criterion 2 still fails on whole items

    def test_one_failing_item_fails_the_criterion_with_that_item_as_evidence(self):
        _, build = self.scorer(lambda items, receive: receive(
            [dict(passed=i != 1, explanation='Not healthcare' if i == 1 else 'Healthcare') for i in range(len(items))]))
        evidence = []
        build(compile_rubric(CASE_7_CRITERIA[2:]), evidence)(outputs='1. urgent care\n2. pizza delivery\n3. walk-in clinic')
        detail = evidence[-1]['criteria'][0]
        self.assertEqual((detail['status'], detail['evidence']), ('fail', 'pizza delivery'))
        self.assertIn('2 ("pizza delivery"): Not healthcare', detail['explanation'])

    def test_unusable_item_verdict_keeps_other_criteria_verdicts(self):
        def bad(items, receive): raise RubricEvaluationError('invalid_items')
        batch = lambda texts, receive: (lambda **kw: receive({'criteria': [dict(passed=True, status='pass', explanation='Fine', evidence=None)]}))
        _, build = self.scorer(bad, batch)
        evidence = []
        with self.assertRaises(RubricEvaluationError):
            build(compile_rubric(CASE_7_CRITERIA[2:] + ['Be professional']), evidence)(outputs='1. urgent care\n2. walk-in clinic')
        self.assertEqual([(d['status'], d.get('error_code')) for d in evidence[-1]['criteria']],
                         [('evaluation_error', 'invalid_items'), ('pass', None)])

    def test_independent_judge_calls_run_at_the_same_time(self):
        import threading
        both_running = threading.Barrier(2, timeout=5)   # breaks if the calls run one after another
        threads = set()
        def item(items, receive):
            threads.add(threading.get_ident()); both_running.wait()
            receive([dict(passed=True, explanation='Healthcare') for _ in items])
        def batch(texts, receive):
            def call(**kw):
                threads.add(threading.get_ident()); both_running.wait()
                receive({'criteria': [dict(passed=True, explanation='Fine', evidence=None)]})
            return call
        _, build = self.scorer(item, batch)
        evidence = []
        score = build(compile_rubric(CASE_7_CRITERIA[2:] + ['Be professional']), evidence)(outputs='1. urgent care\n2. walk-in clinic')
        self.assertEqual(score, 1)
        self.assertEqual(len(threads), 2)
        self.assertEqual([d['status'] for d in evidence[-1]['criteria']], ['pass', 'pass'])

    def test_concurrency_never_exceeds_the_limit(self):
        import threading, time
        from ps_utils.rubric_checks import MAX_CONCURRENT_JUDGE_CALLS, _run_concurrently
        lock, running, peak = threading.Lock(), [0], [0]
        def call():
            with lock:
                running[0] += 1; peak[0] = max(peak[0], running[0])
            time.sleep(0.05)
            with lock:
                running[0] -= 1
        self.assertEqual(_run_concurrently([call] * 10), [None] * 10)
        self.assertEqual(peak[0], MAX_CONCURRENT_JUDGE_CALLS)                  # caller included

    def test_fatal_error_is_raised_after_the_other_call_finishes(self):
        import time
        finished = []
        def item(items, receive):
            time.sleep(0.05)
            receive([dict(passed=True, explanation='Healthcare') for _ in items]); finished.append('item')
        def batch(texts, receive):
            def call(**kw): raise TimeoutError('judge down')
            return call
        _, build = self.scorer(item, batch)
        evidence = []
        with self.assertRaises(TimeoutError):
            build(compile_rubric(CASE_7_CRITERIA[2:] + ['Be professional']), evidence)(outputs='1. urgent care\n2. walk-in clinic')
        self.assertEqual(finished, ['item'])                                   # not abandoned mid-call
        self.assertEqual(evidence[-1]['criteria'][0]['status'], 'pass')         # its verdict is kept

    def test_no_items_falls_back_to_whole_answer_judging(self):
        batch = Mock(return_value=lambda **kw: None)
        _, build = self.scorer(lambda items, receive: self.fail('Item judge must not be called'), batch)
        build(compile_rubric(CASE_7_CRITERIA[2:]), [])(outputs='{"not": "a list"}')
        self.assertEqual(batch.call_args.args[0], [CASE_7_CRITERIA[2]])

    def item_judge(self, *contents):
        from llm_endpoint import Endpoint
        from prompt_optimizer import PromptOptimizer, OptimizationConfig
        endpoint = Endpoint(base_url='https://unused.invalid/v1', api_key='never-sent')
        received = []
        judge = PromptOptimizer(endpoint)._build_item_judge(OptimizationConfig(name='test', prompts=[]),
                                                           CASE_7_CRITERIA[2], ['urgent care', 'pizza delivery'], received.append)
        responses = [NS(usage=None, choices=[NS(message=NS(content=json.dumps(c)))]) for c in contents]
        return judge, received, patch.object(endpoint.client.chat.completions, 'create', side_effect=responses)

    def test_item_judge_prompt_and_valid_response(self):
        judge, received, create = self.item_judge({'items': [{'item': 1, 'explanation': 'Clinic', 'passed': True},
                                                            {'item': 2, 'explanation': 'Food', 'passed': False}]})
        with create as call:
            judge(inputs={}, outputs='ignored')
        prompt = call.call_args.kwargs['messages'][0]['content']
        self.assertIn('1. urgent care\n2. pizza delivery', prompt)
        self.assertLess(prompt.index('"explanation"'), prompt.index('"passed"'))
        self.assertEqual(received, [[{'passed': True, 'explanation': 'Clinic'}, {'passed': False, 'explanation': 'Food'}]])

    def test_item_judge_rejects_wrong_count_or_numbering_after_one_repair(self):
        bad = [{'items': [{'item': 1, 'explanation': 'x', 'passed': True}]},
               {'items': [{'item': 2, 'explanation': 'x', 'passed': True}, {'item': 1, 'explanation': 'x', 'passed': True}]},
               {'items': [{'item': 1, 'explanation': 'x', 'passed': 'true'}, {'item': 2, 'explanation': 'x', 'passed': True}]}]
        for obj in bad:
            judge, received, create = self.item_judge(obj, obj)
            with create as call, self.assertRaises(RubricEvaluationError) as raised:
                judge(inputs={}, outputs='ignored')
            self.assertEqual((call.call_count, raised.exception.code, received), (2, 'invalid_items', []))



class OnlyItemsTests(SimpleTestCase):
    """"Only the items" is enforced in code, and only when a criterion asks for it."""

    def only(self, criterion):
        entry = compile_rubric([criterion])[0]
        return entry['method'], next(c for c in entry['checks'] if c['kind'] == 'only_items')

    def test_wording_decides_the_list_style(self):
        cases = {'Output only the keywords, one per line.': 'none', 'Just the keywords, no numbering': 'none',
                 'No other text': 'none', 'Return a numbered list of 3 headlines, nothing else': 'numbered',
                 'Return a bulleted list of taglines with no commentary': 'bulleted'}
        for criterion, style in cases.items():
            method, check = self.only(criterion)
            self.assertEqual((method, check['markers']), ('deterministic', style), criterion)

    def test_not_enforced_unless_asked(self):
        for criterion in CASE_7_CRITERIA + ['Only list keywords related to dentistry', 'Be concise and friendly']:
            kinds = [c['kind'] for c in compile_rubric([criterion])[0]['checks']]
            self.assertNotIn('only_items', kinds, criterion)

    def test_bare_keywords(self):
        _, check = self.only('Output only the keywords, one per line.')
        verdicts = {CASE_7: False,
                    'pre-employment physical near me\nDOT physical exam\nworkplace drug testing services': True,
                    '["urgent care near me", "walk-in clinic"]': True,
                    '1. urgent care near me\n2. walk-in clinic': False,             # marker
                    'urgent care near me - local intent\nwalk-in clinic': False,    # commentary
                    'Keywords:\nurgent care\nwalk-in clinic': False,                # lead-in
                    '**urgent care**\nwalk-in clinic': False,                       # emphasis
                    '```\nurgent care\n```': False}                                 # fence
        for output, expected in verdicts.items():
            self.assertIs(run_check(output, check)[0], expected, output)
        passed, explanation, observed = run_check(CASE_7, check)
        self.assertEqual(observed['violations'][0]['reason'], 'lead-in text')
        self.assertIn('Required bare lines and nothing else', explanation)

    def test_numbered_headlines_keep_dashes_and_colons_as_content(self):
        _, check = self.only('Return a numbered list of 3 headlines, nothing else')
        self.assertTrue(run_check(HEADLINES, check)[0])
        self.assertFalse(run_check('Save 20% — today only\nSame-day care', check)[0])   # markers were required
        self.assertFalse(run_check('["a", "b"]', check)[0])


# Second and third running examples, so the design is not keyword-specific.
HEADLINES = '1. Save 20% — today only\n2. Feel better faster: book same-day care\n3. "No wait" urgent care, open late'
HEADLINE_CRITERIA = ['Return a numbered list of 3 headlines, nothing else', 'Each headline must be no more than 8 words.',
                     'Each headline must be catchy and specific to urgent care.']
CONTACT = '{"name": "Ana Silva", "email": "ana@example.com", "phone": null}'
CONTACT_CRITERIA = ['Return a JSON object', 'The extracted email must match the one in the input.']


class RunningExampleTests(SimpleTestCase):
    def test_headlines_are_judged_whole(self):
        compiled = compile_rubric(HEADLINE_CRITERIA)
        self.assertEqual([c['method'] for c in compiled], ['deterministic', 'deterministic', 'llm_judge'])
        self.assertEqual((compiled[2]['judge_scope'], compiled[2]['item_unit']), ('item', 'headline'))
        seen, evidence = {}, []
        def make_item_judge(requirement, items, receive):
            seen['items'] = items
            return lambda **kw: receive([dict(passed=True, explanation='Catchy') for _ in items])
        score = combined_scorer(compiled, Mock(side_effect=AssertionError('no batch')), evidence.append,
                                make_item_judge=make_item_judge)(outputs=HEADLINES)
        self.assertEqual(seen['items'], ['Save 20% — today only', 'Feel better faster: book same-day care',
                                         '"No wait" urgent care, open late'])   # dash, colon, inner quotes kept
        self.assertEqual(score, 1)
        self.assertTrue(evidence[-1]['overall_pass'])

    def test_json_extraction_checks_structure_in_code_and_content_with_the_judge(self):
        compiled = compile_rubric(CONTACT_CRITERIA)
        self.assertEqual([(c['method'], c['judge_scope']) for c in compiled], [('deterministic', None), ('llm_judge', 'answer')])
        batch = lambda texts, receive: (lambda **kw: receive({'criteria': [dict(passed=True, status='pass', explanation='Matches', evidence='ana@example.com')]}))
        evidence = []
        self.assertEqual(combined_scorer(compiled, batch, evidence.append)(outputs=CONTACT), 1)
        evidence = []
        fenced = '```json\n' + CONTACT + '\n```'
        self.assertEqual(combined_scorer(compiled, batch, evidence.append)(outputs=fenced), .5)   # strict JSON: no fences
        self.assertEqual(evidence[-1]['criteria'][0]['status'], 'fail')



NESTED_QCR = ['Ensure each keyword:', '- Is 5 words or fewer.', '- Is directly relevant to the healthcare domain.',
              'Return only the numbered list of 3 keywords, with no additional text.']


class LeadInGroupingTests(SimpleTestCase):
    """Regression: a nested QCR arrives line by line; sub-lines must keep their lead-in's scope."""

    def test_lead_in_scopes_its_sub_lines(self):
        compiled = compile_rubric(NESTED_QCR)
        self.assertEqual([c['criterion'] for c in compiled],
                         ['Ensure each keyword: is 5 words or fewer.',
                          'Ensure each keyword: is directly relevant to the healthcare domain.', NESTED_QCR[3]])
        self.assertEqual([c['source_lines'] for c in compiled], [[0, 1], [0, 2], [3]])
        self.assertEqual(compiled[0]['checks'], [{'kind': 'length', 'unit': 'words', 'scope': 'item', 'min': None, 'max': 5}])
        self.assertEqual((compiled[1]['judge_scope'], compiled[1]['item_unit']), ('item', 'keyword'))

    def test_keywords_within_the_limit_pass_per_item_not_per_answer(self):
        answer = '1. walk in clinic open now\n2. urgent care no appointment\n3. weekend urgent care near me'
        passed, explanation, _ = run_check(answer, compile_rubric(NESTED_QCR)[0]['checks'][0])
        self.assertTrue(passed)
        self.assertIn('Words per item: 5, 4, 5', explanation)

    def test_grouping_edge_cases(self):
        self.assertEqual(group_criteria(['Be concise', '- Be polite', 'Each headline:', 'Ends with a colon:', '3 keywords in total.']),
                         [('Be concise', [0]), ('Be polite', [1]), ('Each headline:', [2]),
                          ('Ends with a colon:', [3]), ('3 keywords in total.', [4])])
        self.assertEqual(group_criteria(['The JSON must:', '1. Be an object', '2) Have a name field']),
                         [('The JSON must: be an object', [0, 1]), ('The JSON must: have a name field', [0, 2])])



BULLET_QCR = ['- Provide exactly 3 keywords.', '- Present them as a numbered list (1., 2., 3.).',
              '- Each keyword must be 5 words or fewer.', '- Each keyword must be clearly relevant to the healthcare domain.',
              '- Output ONLY the numbered keywords. Do NOT include any introductory text, explanations, or commentary.']


class FormatPhrasingTests(SimpleTestCase):
    """Regression: format rules phrased as instructions compile to code, not the judge."""

    def test_bullet_qcr_compiles_format_rules_to_code(self):
        compiled = compile_rubric(BULLET_QCR)
        self.assertEqual([c['criterion'] for c in compiled], [c[2:] for c in BULLET_QCR])   # markers stripped
        self.assertEqual([(c['method'], [k['kind'] for k in c['checks']]) for c in compiled],
                         [('deterministic', ['item_count']), ('deterministic', ['list_style']), ('deterministic', ['length']),
                          ('llm_judge', []), ('deterministic', ['only_items'])])
        self.assertEqual(compiled[1]['checks'][0]['markers'], 'numbered')
        self.assertEqual(compiled[4]['checks'][0]['markers'], 'numbered')   # "only the numbered keywords"

    def test_list_style_and_only_items_split_the_work(self):
        compiled = compile_rubric(BULLET_QCR)
        style, only = compiled[1]['checks'][0], compiled[4]['checks'][0]
        verdicts = {'1. urgent care open now\n2. walk in UTI treatment\n3. urgent care for sprained ankle': (True, True),
                    'Here are some:\n1. a\n2. b\n3. c': (True, False),     # list is fine, lead-in is not
                    '1. a\n3. b\n4. c': (False, True),                     # numbering must count 1, 2, 3
                    '- a\n- b\n- c': (False, False),
                    '["a", "b", "c"]': (False, False)}
        for answer, expected in verdicts.items():
            self.assertEqual((run_check(answer, style)[0], run_check(answer, only)[0]), expected, answer)

    def test_do_not_include_phrasing_is_an_only_items_rule(self):
        for criterion in ['Do not include any explanations or commentary.', "Don't add any introductory text",
                          'Output only the bulleted taglines, no introductory text']:
            kinds = [c['kind'] for c in compile_rubric([criterion])[0]['checks']]
            self.assertEqual(kinds, ['only_items'], criterion)



# Real answer shapes (claude-sonnet-5.5, 20-case run): a numbered table, then bulleted tips.
TABLE_ANSWER = """Here are 3 prospective keywords for sleep medicine:

| # | Keyword | Words | Why it's a good prospect |
|---|---------|-------|--------------------------|
| 1 | **sleep apnea testing near me** | 4 | High local and transactional intent. |
| 2 | **home sleep study cost** | 4 | Price research signals readiness to book. |
| 3 | **chronic insomnia treatment without medication** | 5 | Reaches patients seeking non-drug options. |

- Pair each keyword with a dedicated landing page.
- Use location modifiers for clinic-level targeting.
- Check search volume in Keyword Planner."""


class MarkdownTableTests(SimpleTestCase):
    def test_table_rows_are_the_items_not_the_surrounding_bullets(self):
        items, fmt = extract_items(TABLE_ANSWER)
        self.assertEqual(fmt, 'table')
        self.assertEqual(len(items), 3)
        self.assertTrue(items[0].startswith('**sleep apnea testing near me** | '))   # key column first, index and counts dropped

    def test_key_column_comes_from_the_header_else_the_first_non_index_column(self):
        self.assertEqual([extract_judged_text(i, 'keyword')[0] for i in extract_items(TABLE_ANSWER)[0]],
                         ['sleep apnea testing near me', 'home sleep study cost', 'chronic insomnia treatment without medication'])
        plain = '| Term | Intent |\n|---|---|\n| acne doctor | Condition |\n| rash clinic | Symptom |'
        self.assertEqual([extract_judged_text(i, 'keyword') for i in extract_items(plain)[0]],
                         [('acne doctor', 'cell'), ('rash clinic', 'cell')])
        headline = extract_items('| # | Headline |\n|---|---|\n| 1 | Save 20% — today only |')[0]
        self.assertEqual(extract_judged_text(headline[0], 'headline'), ('Save 20% — today only', 'whole'))

    def test_case_7_criteria_on_a_table_answer(self):
        compiled = compile_rubric(CASE_7_CRITERIA)
        count, length = compiled[0]['checks'][0], compiled[1]['checks'][0]
        self.assertTrue(run_check(TABLE_ANSWER, count)[0])                   # 3 keywords, not 6 lines
        passed, explanation, _ = run_check(TABLE_ANSWER, length)
        self.assertFalse(passed)                                              # strict: the rows carry commentary
        self.assertNotIn(' | ', explanation)
        seen = {}
        def make_item_judge(requirement, items, receive):
            seen['items'] = items
            return lambda **kw: receive([dict(passed=True, explanation='Healthcare') for _ in items])
        combined_scorer(compiled, Mock(side_effect=AssertionError('no batch')), [].append, make_item_judge=make_item_judge)(outputs=TABLE_ANSWER)
        self.assertEqual(seen['items'], ['sleep apnea testing near me', 'home sleep study cost', 'chronic insomnia treatment without medication'])

    def test_only_items_names_table_lines(self):
        check = next(c for c in compile_rubric(['Output only the keywords, one per line.'])[0]['checks'])
        passed, _, observed = run_check(TABLE_ANSWER, check)
        self.assertFalse(passed)
        self.assertIn('table', {v['reason'] for v in observed['violations']})

    def test_a_pipe_inside_a_plain_list_is_not_a_table(self):
        self.assertEqual(extract_items('1. a | b\n2. c\n3. d'), (['a | b', 'c', 'd'], 'marked_lines'))


class QualityAggregateTests(SimpleTestCase):
    def case(self, verdicts, score):
        crit = ['3 keywords', '<= 5 words', 'Relevant']
        methods = ['deterministic', 'deterministic', 'llm_judge']
        return {'graded_by': 'rubric', 'score': score, 'error': None if None not in verdicts else 'RubricEvaluationError',
                'usage': None, 'rubric_evaluation': {'criteria': [
                    {'criterion': c, 'passed': v, 'method': m, 'compiled_method': m} for c, v, m in zip(crit, verdicts, methods)]}}

    def test_per_criterion_rates_count_errors_separately_from_failures(self):
        cases = [self.case([True, False, True], 2/3), self.case([True, False, None], None), self.case([True, True, True], 1)]
        rows = b.summarize(cases, {})['quality']['rubric']['criteria']
        self.assertEqual([(r['criterion'], r['evaluated'], r['passed'], r['evaluation_errors']) for r in rows],
                         [('3 keywords', 3, 3, 0), ('<= 5 words', 3, 1, 0), ('Relevant', 2, 2, 1)])
        self.assertAlmostEqual(rows[1]['pass_rate'], 1/3)
        self.assertEqual(rows[1]['interval_95'], b.interval(1, 3))
        self.assertEqual(rows[2]['method'], 'llm_judge')

    def test_mean_score_is_partial_credit_with_reproducible_bootstrap_interval(self):
        cases = [self.case([True, False, True], 2/3), self.case([True, True, True], 1), self.case([True, False, False], 1/3)]
        quality = b.summarize(cases, {})['quality']['rubric']
        self.assertAlmostEqual(quality['mean_score'], 2/3)
        self.assertAlmostEqual(quality['pass_rate'], 1/3)          # strict headline unchanged
        low, high = quality['mean_interval_95']
        self.assertTrue(1/3 <= low <= 2/3 <= high <= 1)
        self.assertEqual(quality['mean_interval_95'], b.summarize(cases, {})['quality']['rubric']['mean_interval_95'])
        self.assertIsNone(b.bootstrap_mean_interval([1.0]))
        self.assertEqual(b.bootstrap_mean_interval([.5, .5, .5]), [.5, .5])


class JudgeRetryTests(SimpleTestCase):
    def make_client(self, outcomes):
        import httpx, openai
        calls = []
        def create(**kw):
            calls.append(kw)
            outcome = outcomes.pop(0)
            if outcome == 500:
                raise openai.InternalServerError('upstream', response=httpx.Response(500, request=httpx.Request('POST', 'https://x.invalid')), body=None)
            if outcome == 400:
                raise openai.BadRequestError('bad', response=httpx.Response(400, request=httpx.Request('POST', 'https://x.invalid')), body=None)
            return NS(usage=None, choices=[NS(message=NS(content='ok'))])
        raw = NS(chat=NS(completions=NS(create=create)))
        billing = b.BillingCollector(b.validate(payload()))
        billing.model, billing.variant, billing.case_index = 'small', 'baseline', 1
        return b.CancellationClient(raw, lambda: None, billing), calls, billing

    def test_judge_retries_once_on_transient_error_and_records_both_attempts(self):
        import openai
        judge = {'extra_headers': {'x-gepa-purpose': 'training_judge'}, 'model': 'judge'}
        client, calls, billing = self.make_client([500, 'ok'])
        self.assertEqual(client.chat.completions.create(**judge).choices[0].message.content, 'ok')
        self.assertEqual((len(calls), len(billing.calls)), (2, 2))
        client, calls, _ = self.make_client([500, 500])
        with self.assertRaises(openai.InternalServerError): client.chat.completions.create(**judge)
        self.assertEqual(len(calls), 2)                      # one retry only

    def test_no_retry_for_task_calls_or_request_errors(self):
        import openai
        client, calls, _ = self.make_client([500])
        with self.assertRaises(openai.InternalServerError):
            client.chat.completions.create(model='small', extra_headers={'x-gepa-purpose': 'holdout_generation'})
        self.assertEqual(len(calls), 1)
        client, calls, _ = self.make_client([400])
        with self.assertRaises(openai.BadRequestError):
            client.chat.completions.create(model='judge', extra_headers={'x-gepa-purpose': 'training_judge'})
        self.assertEqual(len(calls), 1)


class JudgeSelectionRecordTests(SimpleTestCase):
    def test_manifest_keeps_a_bounded_health_check_record(self):
        p = payload(); p['judge_selection'] = {'method': 'highest_priced_healthy_v1', 'tried': [
            {'model': 'fast', 'ok': False, 'reason': 'http_503 on call 5/6', 'ms': 12, 'calls': 5}, {'model': 'judge', 'ok': True, 'ms': 8, 'calls': 6, 'extra': 'dropped'}]}
        self.assertEqual(b.validate(p)['scoring']['judge_selection'], {'method': 'highest_priced_healthy_v1', 'tried': [
            {'model': 'fast', 'ok': False, 'reason': 'http_503 on call 5/6', 'ms': 12, 'calls': 5}, {'model': 'judge', 'ok': True, 'ms': 8, 'calls': 6}]})
        for bad in [None, 'x', {'tried': 'x'}, {'tried': [{'ok': True}]}]:
            p['judge_selection'] = bad
            self.assertIsNone(b.validate(p)['scoring']['judge_selection'])

class JudgeIntegrityTests(SimpleTestCase):
    def optimizer(self, *contents):
        from llm_endpoint import Endpoint
        from prompt_optimizer import PromptOptimizer
        endpoint=Endpoint(base_url='https://unused.invalid/v1',api_key='never-sent')
        responses=[NS(usage=None,choices=[NS(message=NS(content=c))]) for c in contents]
        return PromptOptimizer(endpoint), patch.object(endpoint.client.chat.completions,'create',side_effect=responses)

    def test_judge_must_differ_from_contenders(self):
        p=payload(); p['judge_model']='small'
        with self.assertRaisesRegex(ValueError,'differ from every contender'): b.validate(p)

    def test_strict_label_judge_raises_on_unusable_verdict_but_legacy_scores_zero(self):
        from prompt_optimizer import OptimizationConfig
        cfg=OptimizationConfig(name='test',prompts=[])
        for bad in ['not json', '{"rationale":"x","match":"yes"}']:
            opt,create=self.optimizer(bad,bad)
            with create, self.assertRaises(RubricEvaluationError) as ctx:
                opt._build_label_scorer(cfg,strict=True)('gold','answer')
            self.assertEqual(ctx.exception.code,'invalid_label')
            opt,create=self.optimizer(bad,bad)
            with create: self.assertEqual(opt._build_label_scorer(cfg)('gold','answer'),0.0)
        opt,create=self.optimizer('{"rationale":"same meaning","match":true}')
        with create as mock:
            self.assertEqual(opt._build_label_scorer(cfg,strict=True)('gold','answer'),1.0)
        prompt=mock.call_args.kwargs['messages'][0]['content']
        self.assertLess(prompt.index('"rationale"'),prompt.index('"match"'))

    def test_benchmark_reference_judge_failure_is_evaluation_error_not_fail(self):
        from llm_endpoint import Endpoint
        p=payload(); p.update(evaluation_mode='reference',models=['small'],optimized=[],
                              rows=[{'inputs':{'q':'a'},'expected_output':'yes'}])
        endpoint=Endpoint(base_url='https://unused.invalid/v1',api_key='never-sent')
        def create(**kw):
            content='yes' if kw['model']=='small' else 'no verdict here'
            return NS(model=kw['model'],usage=NS(prompt_tokens=1,completion_tokens=1,total_tokens=2,completion_tokens_details=None),
                      choices=[NS(message=NS(content=content))])
        with patch.object(endpoint.client.chat.completions,'create',side_effect=create):
            arm=b.execute(b.validate(p),endpoint,Mock())['arms'][0]
        ref=arm['quality']['reference']
        self.assertEqual((ref['evaluated'],ref['evaluation_errors'],ref['failed']),(0,1,0))
        self.assertIsNone(ref['pass_rate'])
        self.assertEqual(arm['cases'][0]['error'],'RubricEvaluationError')
        self.assertIn('boolean match verdict',arm['cases'][0]['error_detail'])

    def test_rubric_judge_writes_explanation_before_verdict(self):
        from prompt_optimizer import OptimizationConfig
        opt,create=self.optimizer(json.dumps({'criteria':[{'explanation':'All five words or fewer.','evidence':None,'passed':True}]}))
        evidence=[]
        judge=opt._build_guidelines_scorer(OptimizationConfig(name='test',prompts=[]),['Short'],on_judgment=evidence.append)
        with create as mock: self.assertEqual(judge(inputs={},outputs='answer'),1.0)
        prompt=mock.call_args.kwargs['messages'][0]['content']
        self.assertLess(prompt.index('"explanation"'),prompt.index('"passed"'))
        self.assertTrue(evidence[-1]['criteria'][0]['passed'])
