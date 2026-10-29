"""Spec-from-baseline v2: QCR drafted deterministically from the baseline prompt-set."""
import json
from django.test import SimpleTestCase, override_settings
from ps_utils.rubric_parser import compile_rubric, run_check
from ps_utils.spec_draft import draft_criteria, render

# The 08:03 baseline (QCR embedded in the system prompt) and the 06:48 one (no QCR).
EMBEDDED = [{'role': 'system', 'content': 'Search marketing keywords exploration for a healthcare chain. Your goal is to identify '
             'new prospective keywords to generate new user demand for the healthcare clinics owned by the chain. \n\n'
             '3 keywords in total.\nEach keyword must contain less than or equal to 5 words.\n'
             'Each keyword must be relevant to the healthcare domain.'},
            {'role': 'user', 'content': 'Suggest 3 keywords for {{service}}.'}]
PLAIN = [EMBEDDED[0] | {'content': EMBEDDED[0]['content'].split('\n\n')[0]}, EMBEDDED[1]]
NESTED = [{'role': 'user', 'content': 'Suggest keywords for {{service}}.\n\nFollow these rules:\n- Return exactly 3 keywords.\n'
           '- Each keyword must be 5 words or fewer.\n- Do not include any explanations.\n'
           '- Present them as a numbered list (1., 2., 3.).'}]


class DraftCriteriaTests(SimpleTestCase):
    def test_embedded_qcr_drafts_every_stated_criterion_with_its_span(self):
        drafts = draft_criteria(EMBEDDED)
        self.assertEqual([(d['kind'], d['criterion'], d['source_span'], d['source_role']) for d in drafts],
                         [('code', '3 keywords in total.', '3 keywords in total.', 'system'),
                          ('code', 'Each keyword must contain at most 5 words.', 'Each keyword must contain less than or equal to 5 words.', 'system'),
                          ('judge', 'Each keyword must be relevant to the healthcare domain.',
                           'Each keyword must be relevant to the healthcare domain.', 'system')])
        self.assertTrue(all(d['origin'] == 'stated' for d in drafts))
        self.assertEqual(compile_rubric([drafts[2]['criterion']])[0]['judge_scope'], 'item')

    def test_context_goals_task_verbs_and_templated_sentences_are_not_judged_criteria(self):
        prompt = [{'role': 'user', 'content': 'Write 3 taglines for Keak. Your goal is brand awareness. '
                   'Each tagline must mention {{product}}. Avoid jargon.'}]
        self.assertEqual([(d['kind'], d['criterion']) for d in draft_criteria(prompt)],
                         [('code', '3 taglines in total.'), ('judge', 'Avoid jargon.')])

    def test_hybrid_rule_is_drafted_verbatim_without_a_duplicate_code_check(self):
        prompt = [{'role': 'user', 'content': 'Follow these rules:\n- Return exactly 3 keywords.\n'
                   '- Each keyword must contain at most 5 words and be relevant to healthcare.\n- Keep the tone professional.'}]
        drafts = draft_criteria(prompt)
        self.assertEqual([(d['kind'], d['criterion']) for d in drafts],
                         [('code', '3 keywords in total.'),
                          ('hybrid', 'Each keyword must contain at most 5 words and be relevant to healthcare.'),
                          ('judge', 'Keep the tone professional.')])
        self.assertEqual([e['method'] for e in compile_rubric([d['criterion'] for d in drafts])],
                         ['deterministic', 'hybrid', 'llm_judge'])

    def test_judged_sub_rule_keeps_the_lead_in_scope(self):
        drafts = draft_criteria([{'role': 'user', 'content': 'Each keyword must:\n- be relevant to healthcare'}])
        self.assertEqual([(d['criterion'], d['source_span']) for d in drafts],
                         [('Each keyword must: be relevant to healthcare', '- be relevant to healthcare')])
        self.assertEqual(compile_rubric([drafts[0]['criterion']])[0]['judge_scope'], 'item')

    def test_spec_is_only_as_complete_as_the_baseline(self):
        self.assertEqual([(d['criterion'], d['source_span']) for d in draft_criteria(PLAIN)],
                         [('3 keywords in total.', 'Suggest 3 keywords for {{service}}.')])

    def test_every_span_is_verbatim_prompt_text(self):
        for messages in (EMBEDDED, PLAIN, NESTED):
            text = '\n'.join(m['content'] for m in messages)
            self.assertTrue(all(d['source_span'] in text for d in draft_criteria(messages)))

    def test_nested_rules_keep_scope_and_do_not_contradict(self):
        drafts = draft_criteria(NESTED)
        self.assertEqual([d['criterion'] for d in drafts],
                         ['3 keywords in total.', 'Each keyword must contain at most 5 words.',
                          'Output only the numbered keywords, nothing else.', 'Present the keywords as a numbered list.'])
        self.assertEqual(drafts[2]['source_span'], '- Do not include any explanations.')
        compliant = '1. urgent care near me\n2. walk-in clinic open now\n3. pediatric urgent care'
        for entry in compile_rubric([d['criterion'] for d in drafts]):
            self.assertTrue(all(run_check(compliant, check)[0] for check in entry['checks']), entry['criterion'])

    def test_rendered_criteria_compile_back_to_the_same_check(self):
        prompt = [{'role': 'user', 'content': "Write between 2 and 4 taglines. Each tagline must include 'Keak'. "
                   'Return a JSON array. No duplicates. Keep the answer under 50 words.'}]
        drafts = draft_criteria(prompt)
        self.assertEqual([d['criterion'] for d in drafts],
                         ['Between 2 and 4 taglines.', 'Each tagline must include "Keak".', 'Return a JSON array.',
                          'No duplicate taglines.', 'The answer must contain at most 49 words.'])
        for d in drafts:
            back = compile_rubric([d['criterion']])[0]['checks']
            self.assertTrue(any({k: v for k, v in c.items() if k != 'unit'} == {k: v for k, v in d['check'].items() if k != 'unit'}
                                for c in back), d['criterion'])

    def test_render_rejects_unknown_kinds(self):
        with self.assertRaises(ValueError):
            render({'kind': 'bogus'}, 'keyword')


@override_settings(PROMPT_STUDIO_API_ENABLED=True, PROMPT_STUDIO_SERVICE_TOKEN='fixture-service-token')
class DraftCriteriaApiTests(SimpleTestCase):
    headers = {'HTTP_AUTHORIZATION': 'Bearer fixture-service-token', 'HTTP_X_CI_USER_ID': 'alice'}

    def post(self, body, **headers):
        return self.client.post('/api/v1/benchmarks/draft-criteria', json.dumps(body), content_type='application/json',
                                **(headers or self.headers))

    def test_drafts_from_the_baseline(self):
        response = self.post({'baseline': EMBEDDED})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body['version'], 'spec-draft-v2')
        self.assertEqual([(c['kind'], c['criterion']) for c in body['criteria']],
                         [('code', '3 keywords in total.'), ('code', 'Each keyword must contain at most 5 words.'),
                          ('judge', 'Each keyword must be relevant to the healthcare domain.')])

    def test_rejects_bad_input_and_missing_auth(self):
        self.assertEqual(self.post({'baseline': 'not a list'}).status_code, 400)
        self.assertEqual(self.post({'baseline': [{'role': 'assistant', 'content': 'x'}]}).status_code, 400)
        self.assertEqual(self.post({'baseline': EMBEDDED}, HTTP_X_CI_USER_ID='alice').status_code, 401)
