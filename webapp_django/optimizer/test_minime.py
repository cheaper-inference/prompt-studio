"""MiniMe foundation: labels, seed QCR, evaluation metrics and flip rates (judge stubbed, no network)."""
import itertools
import os
import tempfile
from django.test import SimpleTestCase
from ps_utils.minime.evaluate import evaluate, metrics
from ps_utils.minime.labels import LabelError, load_labels
from ps_utils.minime.qcr import QCR, Statement, seed_qcr

RELEVANT = 'Each keyword must be relevant to the healthcare domain.'
COUNT = '3 keywords in total.'
GOOD = '1. urgent care clinic near me\n2. walk-in clinic same day appointment\n3. emergency room alternative open now'
OFF = '1. urgent care clinic near me\n2. cheap flights to paris\n3. walk-in clinic open now'
SHORT = '1. urgent care clinic near me\n2. walk-in clinic open now'

# Keyword-sheet layout: ';' delimiter, 'service'/'answer' columns.
SHEET = ('id;service;answer;verdict;reason\n'
         f'1;urgent care;"{GOOD}";Pass;All three are intents a patient would search.\n'
         f'2;urgent care;"{OFF}";Fail;cheap flights to paris is not healthcare.\n'
         f'3;urgent care;"{SHORT}";Fail;Only two keywords; three were asked for.\n')


def write(text):
    f = tempfile.NamedTemporaryFile('w', suffix='.csv', delete=False, encoding='utf-8')
    f.write(text); f.close()
    return f.name


def stub_judge(decide):
    """make_judge stand-in: decide(criterion_text, output) -> bool, one verdict per text."""
    def factory(texts, receive):
        def call(inputs=None, outputs=None):
            receive({'criteria': [dict(passed=decide(t, outputs), explanation='stub', evidence=None) for t in texts]})
        return call
    return factory


def qcr(*texts):
    return QCR(task={'objective': 'Healthcare SEM keywords'},
               statements=[Statement(id=f's{n}', text=t) for n, t in enumerate(texts, start=1)])


class LabelTests(SimpleTestCase):
    def test_keyword_sheet_loads_with_column_mapping(self):
        path = write(SHEET)
        rows = load_labels(path, input_col='service', output_col='answer')
        os.unlink(path)
        self.assertEqual([(r['id'], r['passed']) for r in rows], [('1', True), ('2', False), ('3', False)])
        self.assertEqual(rows[1]['output'], OFF)

    def test_missing_verdict_or_reason_rejects_the_file(self):
        path = write('input,output,verdict,reason\nq,a,Pass,\nq,a,maybe,why\nq,a,Fail,ok\n')
        with self.assertRaises(LabelError) as caught:
            load_labels(path)
        os.unlink(path)
        self.assertIn('row 1: reason is empty', str(caught.exception))
        self.assertIn("row 2: verdict 'maybe' is not Pass/Fail", str(caught.exception))

    def test_missing_column_is_named(self):
        path = write('input,output,verdict\nq,a,Pass\n')
        with self.assertRaisesRegex(LabelError, 'missing column.*reason'):
            load_labels(path)
        os.unlink(path)


class SeedTests(SimpleTestCase):
    def test_seed_is_one_statement_per_line_of_the_quality_description(self):
        task = {'objective': 'Find new search keywords for a healthcare chain.',
                'requirements': f'- {COUNT}\n\n2. {RELEVANT}\n{COUNT}'}
        seeded = seed_qcr(task)
        self.assertEqual([(s.id, s.text) for s in seeded.statements], [('s1', COUNT), ('s2', RELEVANT)])
        self.assertEqual({(s.status, s.origin) for s in seeded.statements}, {('active', 'seed')})
        self.assertEqual(seed_qcr({'objective': 'x'}).statements, [])

    def test_seed_requires_an_objective_and_known_fields(self):
        with self.assertRaises(ValueError):
            seed_qcr({'requirements': COUNT})
        with self.assertRaises(ValueError):
            seed_qcr({'objective': 'x', 'tone': 'warm'})

    def test_round_trip(self):
        original = qcr(COUNT, RELEVANT)
        original.statements[1].support = ['2']
        path = write('')
        original.save(path)
        self.assertEqual(QCR.load(path).to_dict(), original.to_dict())
        os.unlink(path)


class EvaluateTests(SimpleTestCase):
    def setUp(self):
        self.path = write(SHEET)
        self.rows = load_labels(self.path, input_col='service', output_col='answer')

    def tearDown(self):
        os.unlink(self.path)

    def test_code_and_judge_statements_are_anded(self):
        judge = stub_judge(lambda text, output: 'paris' not in output)
        report = evaluate(qcr(COUNT, RELEVANT), self.rows, judge)
        self.assertEqual([r['predicted'] for r in report['results']], [True, False, False])
        self.assertEqual(report['results'][2]['statements'], {'s1': False, 's2': True})   # count fails in code
        m = report['metrics']
        self.assertEqual((m['power'], m['fail_catch'], m['fdp'], m['kappa']), (1.0, 1.0, 0.0, 1.0))

    def test_missing_statement_shows_as_a_missed_fail(self):
        report = evaluate(qcr(COUNT), self.rows, stub_judge(lambda *a: True))
        m = report['metrics']
        self.assertEqual(m['confusion'], {'pass_kept': 1, 'fail_caught': 1, 'fail_missed': 1, 'pass_failed': 0})
        self.assertEqual((m['power'], m['fail_catch'], m['fdp']), (1.0, 0.5, 0.0))

    def test_empty_qcr_passes_everything_without_judging(self):
        def never(*a):
            raise AssertionError('judge must not be called')
        report = evaluate(qcr(), self.rows, never)
        self.assertEqual([r['predicted'] for r in report['results']], [True] * 3)
        self.assertEqual(report['metrics']['fail_catch'], 0.0)

    def test_flip_rate_separates_code_from_a_noisy_judge(self):
        coin = itertools.cycle([True, False])
        report = evaluate(qcr(COUNT, RELEVANT), self.rows, stub_judge(lambda *a: next(coin)), repeats=2)
        self.assertEqual(report['flip_rates']['s1'], 0.0)
        self.assertGreater(report['flip_rates']['s2'], 0.0)

    def test_metrics_handle_empty_denominators(self):
        rows = [{'id': '1', 'passed': True}]
        m = metrics(rows, [{'id': '1', 'predicted': True}])
        self.assertEqual((m['power'], m['fail_catch'], m['fdp']), (1.0, None, None))


NEW_RULE = "Each keyword must not use the word 'first'."


# ---- active learning ------------------------------------------------------------
from collections import Counter
from ps_utils.minime.candidates import generate_round, plan_round


def kinds(plan, split='learn'):
    return Counter((a['kind'], a['target']) for a in plan if a['split'] == split)


class PlanRoundTests(SimpleTestCase):
    def test_rule_breaking_asks_go_to_untested_rules_first_and_rotate(self):
        from ps_utils.minime.qcr import QCR, Statement
        q = QCR(task={'objective': 'x'}, statements=[Statement(id=f's{i}', text=f'rule {i}') for i in range(1, 7)])
        targets = lambda p: [a['target'] for a in p if a['split'] == 'learn' and a['kind'] == 'violate']
        plan = plan_round(q, size=10, heldout_share=0.2, random_share=0.3, comply_share=0.3, offset=1,
                          tested={'s1': 3, 's2': 2, 's3': 1, 's4': 1})
        self.assertEqual(targets(plan)[:2], ['s5', 's6'])                 # the untested rules come first
        seen = set()
        for r in range(4):                                                 # all tested equally: every rule gets turns
            seen |= set(targets(plan_round(q, size=10, heldout_share=0.2, random_share=0.3, comply_share=0.3,
                                           offset=r, tested={})))
        self.assertEqual(seen, {f's{i}' for i in range(1, 7)})

    def test_fresh_qcr_splits_heldout_and_covers_every_statement(self):
        plan = plan_round(qcr(COUNT, RELEVANT), size=25)
        self.assertEqual(Counter(a['split'] for a in plan), {'learn': 20, 'heldout': 5})
        self.assertEqual(kinds(plan), {('violate', 's1'): 10, ('violate', 's2'): 10})
        self.assertEqual(kinds(plan, 'heldout'), {('representative', None): 5})

    def test_priorities_gaps_then_weak_then_flipping(self):
        q = qcr(COUNT, RELEVANT)
        q.statements.append(Statement(id='s3', text=NEW_RULE, status='candidate', support=['1']))
        report = {'uncovered_fails': [{'id': '6', 'reason': 'too generic'}]}
        plan = plan_round(q, report, flip_rates={'s2': 0.2, 's1': 0.0}, size=10)
        learn = [(a['kind'], a['target'] or a['reason']) for a in plan if a['split'] == 'learn']
        self.assertEqual(learn, [('gap', 'too generic'), ('violate', 's3'), ('violate', 's3'), ('near_miss', 's3'),
                                 ('near_miss', 's2'), ('near_miss', 's2'), ('violate', 's1'), ('violate', 's2')])
        self.assertEqual(len(plan), 10)

    def test_missed_fails_target_their_statement(self):
        report = {'missed_fails': [{'id': '5', 'statements': ['s2'], 'reason': 'x'}]}
        plan = plan_round(qcr(COUNT, RELEVANT), report, size=5)
        self.assertEqual(plan[0], {'kind': 'violate', 'split': 'learn', 'target': 's2', 'reason': None})

    def test_gaps_take_at_most_half_the_learning_slots(self):
        report = {'uncovered_fails': [{'id': str(i), 'reason': f'r{i}'} for i in range(30)]}
        plan = plan_round(qcr(COUNT), report, size=10)
        self.assertEqual(sum(a['kind'] == 'gap' for a in plan), 4)

    def test_empty_qcr_asks_for_representative_answers(self):
        self.assertEqual({a['kind'] for a in plan_round(qcr(), size=5)}, {'representative'})

    def test_deterministic(self):
        q = qcr(COUNT, RELEVANT)
        self.assertEqual(plan_round(q, size=25), plan_round(q, size=25))


class GenerateRoundTests(SimpleTestCase):
    def test_candidates_are_blind_shuffled_and_reproducible(self):
        q = qcr(COUNT, RELEVANT)
        q.statements.append(Statement(id='s3', text=NEW_RULE, status='candidate', support=[]))
        plan = plan_round(q, size=10)
        prompts = []
        def generate(prompt):
            prompts.append(prompt)
            return f'  answer {len(prompts)}  '
        first = generate_round(q, plan, ['urgent care', 'cancer screening'], generate, seed=7)
        again = generate_round(q, plan, ['urgent care', 'cancer screening'], lambda p: 'x', seed=7)
        self.assertEqual([c['id'] for c in first], [c['id'] for c in again])
        self.assertNotEqual([c['id'] for c in first], sorted((c['id'] for c in first), key=lambda i: int(i[1:])))
        self.assertTrue(all(c['output'].startswith('answer') and c['output'] == c['output'].strip() for c in first))
        self.assertTrue({c['input'] for c in first} <= {'urgent care', 'cancer screening'})
        violate = next(p for p in prompts if 'must clearly break' in p)
        self.assertIn(NEW_RULE, violate)                       # target stated
        self.assertIn(f'- {COUNT}', violate)                   # active statements listed
        rep = next(c for c in first if c['split'] == 'heldout')
        self.assertEqual((rep['kind'], rep['expected_pass']), ('representative', None))

    def test_inputs_are_required(self):
        with self.assertRaises(ValueError):
            generate_round(qcr(COUNT), plan_round(qcr(COUNT), size=2), [], lambda p: 'x')


# ---- fit check & deployed prompt ---------------------------------------------------
from ps_utils.minime.evaluate import regressions


class FitAndDeployedTests(SimpleTestCase):
    def test_regressions_use_todays_labels(self):
        rows = [{'id': '1', 'passed': True}, {'id': '2', 'passed': False}, {'id': '3', 'passed': False}]
        before = {'1': True, '2': False, '3': True, '4': True}           # 1, 2 right; 3 wrong; 4 gone
        now = [{'id': '1', 'predicted': False}, {'id': '2', 'predicted': None}, {'id': '3', 'predicted': False}]
        self.assertEqual(regressions(rows, before, now), ['1'])          # 2 unscored now, 3 improved
        self.assertEqual(regressions(rows, None, now), [])

    def test_representative_answers_use_the_deployed_prompt(self):
        q = qcr(COUNT)
        plan = [{'kind': 'representative', 'split': 'heldout', 'target': None, 'reason': None},
                {'kind': 'violate', 'split': 'learn', 'target': 's1', 'reason': None}]
        seen = []
        generate_round(q, plan, ['flu shot'], lambda p: seen.append(p) or 'x', deployed_prompt='Suggest 3 keywords for {{input}}.')
        self.assertIn('Suggest 3 keywords for flu shot.', seen)
        self.assertTrue(any('must clearly break' in p for p in seen))   # learning asks unchanged
        seen.clear()
        generate_round(q, plan[:1], ['flu shot'], lambda p: seen.append(p) or 'x', deployed_prompt='Suggest keywords.')
        self.assertEqual(seen, ['Suggest keywords.\n\nInput:\nflu shot'])


# ---- the QCR as GEPA's text component ------------------------------------------------
from ps_utils.minime.rubric_text import MAX_EDITS, RubricRejected, from_text, parse, to_text


class RubricTextTests(SimpleTestCase):
    def setUp(self):
        self.q = qcr(COUNT, RELEVANT, 'No brand names.')

    def test_round_trip_is_no_edit(self):
        new, edits = from_text(to_text(self.q), self.q)
        self.assertEqual((edits, [s.text for s in new.active()]), ([], [COUNT, RELEVANT, 'No brand names.']))

    def test_add_sharpen_retire(self):
        text = f'1. {COUNT}\n2. Each keyword must be relevant to healthcare clinics.\n3. {NEW_RULE}'
        new, edits = from_text(text, self.q)
        self.assertEqual([(e['op'], e['id']) for e in edits], [('sharpen', 's2'), ('retire', 's3'), ('add', 's4')])
        self.assertEqual([(s.id, s.status) for s in new.statements],
                         [('s1', 'active'), ('s2', 'active'), ('s4', 'active'), ('s3', 'retired')])
        self.assertEqual(new.statements[2].origin, 'learned')

    def test_too_many_edits_or_nothing_is_rejected(self):
        with self.assertRaises(RubricRejected):
            from_text('```\n```', self.q)
        many = '\n'.join(f'{n}. brand new rule {n}' for n in range(1, MAX_EDITS + 2))
        with self.assertRaises(RubricRejected):
            from_text(many, qcr())

    def test_parse_accepts_numbers_and_bullets(self):
        self.assertEqual(parse('Rubric:\n1) a\n- b\n* a\nnot a line'), ['a', 'b'])
        self.assertEqual(to_text(qcr()), 'No requirements yet.')

    def test_parse_reads_the_formats_models_write(self):
        """Bold numbering and colons are read; only the last fenced block is the rubric, and if it
        holds no list the reply is unusable: an earlier list (analysis notes) never stands in."""
        self.assertEqual(parse('**1.** a\n__2.__ b\n3: c'), ['a', 'b', 'c'])
        self.assertEqual(parse('Notes:\n```\n1. Drop the count rule.\n```\nRubric:\n```\nAt most 5 keywords.\nNo brands.\n```'), [])
        self.assertEqual(parse('Thinking:\n1. note one\n```\n1. a\n2. b\n```'), ['a', 'b'])
        self.assertEqual(parse('Here: \n- **Bold statement** stays'), ['**Bold statement** stays'])
        with self.assertRaisesRegex(RubricRejected, 'no usable reply'):
            from_text('', self.q)

    def test_random_share_reserves_untargeted_learning_asks(self):
        plan = plan_round(qcr(COUNT), size=10, heldout_share=0.2, random_share=0.25)
        self.assertEqual(Counter((a['split'], a['kind']) for a in plan),
                         {('learn', 'violate'): 6, ('learn', 'representative'): 2, ('heldout', 'representative'): 2})



# ---- input pool ---------------------------------------------------------------------
from ps_utils.minime.candidates import blind_only, expand_inputs


class InputPoolTests(SimpleTestCase):
    def test_expand_parses_dedupes_and_caps(self):
        reply = '1. Flu Shot\n- flu shot\n* urgent care\n"travel vaccines"\n\n' + 'x' * 400 + '\nstd testing'
        seen = []
        new = expand_inputs('Task: keywords', ['urgent care'], ['dermatology'], lambda p: seen.append(p) or reply, n=2)
        self.assertEqual(new, ['Flu Shot', 'travel vaccines'])
        self.assertIn('- urgent care', seen[0])
        self.assertIn('Write 2 new inputs', seen[0])

    def test_blind_slice_is_stable_and_partial(self):
        names = [f'service {n}' for n in range(200)]
        blind = [x for x in names if blind_only(x)]
        self.assertTrue(30 < len(blind) < 70)
        self.assertEqual(blind, [x for x in names if blind_only(x.upper())])   # case-insensitive, deterministic

    def test_generate_round_keeps_blind_asks_on_their_own_inputs(self):
        plan = plan_round(qcr(COUNT), size=10)
        out = generate_round(qcr(COUNT), plan, {'learn': ['a', 'b'], 'heldout': ['z']}, lambda p: 'x', seed=3)
        self.assertEqual({c['input'] for c in out if c['split'] == 'heldout'}, {'z'})
        self.assertTrue({c['input'] for c in out if c['split'] == 'learn'} <= {'a', 'b'})


class ComplyTests(SimpleTestCase):
    """Answers written to pass: good answers get seen, in learning and in the blind checks."""
    def test_learning_slots_include_answers_written_to_pass(self):
        plan = plan_round(qcr(COUNT, RELEVANT), size=10, heldout_share=0.2, random_share=0.3, comply_share=0.3)
        learn = [a['kind'] for a in plan if a['split'] == 'learn']
        self.assertEqual((len(learn), learn.count('comply'), learn.count('representative')), (8, 2, 2))
        self.assertNotIn('comply', [a['kind'] for a in plan_round(qcr(), size=10, comply_share=0.3)])  # no rules yet

    def test_blind_cycle_rotates_over_rounds(self):
        cycle = ('violate', 'representative', 'violate', 'comply')
        kinds = [[a['kind'] for a in plan_round(qcr(COUNT, RELEVANT), size=10, heldout_share=0.2, blind_cycle=cycle,
                                                offset=n) if a['split'] == 'heldout'] for n in (1, 2)]
        self.assertEqual(kinds, [['violate', 'comply'], ['violate', 'representative']])

    def test_comply_prompt_lists_every_rule_and_expects_pass(self):
        seen = []
        out = generate_round(qcr(COUNT, RELEVANT), [{'kind': 'comply', 'split': 'learn', 'target': None, 'reason': None}],
                             ['flu shot'], lambda p: seen.append(p) or 'x')
        self.assertIn('meets every requirement', seen[0])
        self.assertIn(COUNT, seen[0]); self.assertIn(RELEVANT, seen[0])
        self.assertIs(out[0]['expected_pass'], True)


class AdversarialBlindTests(SimpleTestCase):
    def test_half_the_blind_asks_break_a_rule(self):
        plan = plan_round(qcr(COUNT, RELEVANT), size=20, heldout_share=0.2, adversarial_share=0.5)
        blind = [(a['kind'], a['target']) for a in plan if a['split'] == 'heldout']
        self.assertEqual(blind, [('violate', 's1'), ('violate', 's2'), ('representative', None), ('representative', None)])
        self.assertEqual({a['kind'] for a in plan_round(qcr(), size=10, adversarial_share=0.5) if a['split'] == 'heldout'},
                         {'representative'})          # nothing to break yet


class JudgePromptTests(SimpleTestCase):
    def test_judge_prompt_lists_active_statements_with_input_and_answer_slots(self):
        from ps_utils.minime.qcr import judge_prompt
        q = qcr(COUNT, RELEVANT)
        q.statements[1].status = 'retired'
        text = judge_prompt(q)
        self.assertIn('1. 3 keywords in total.', text)
        self.assertNotIn(RELEVANT, text)
        self.assertIn('Objective: Healthcare SEM keywords', text)
        self.assertTrue(text.rstrip().endswith('{{output}}') and '{{input}}' in text)
        self.assertIn('(no requirements yet)', judge_prompt(qcr()))


class MiniMeDataDirTests(SimpleTestCase):
    def test_data_dir_follows_minime_data_dir(self):
        """Production keeps the judge cache and GEPA checkpoints on the persistent volume."""
        import os
        from unittest.mock import patch
        from optimizer.minime import judging, learning
        with patch.dict(os.environ, {"MINIME_DATA_DIR": "/vol/minime"}):
            self.assertEqual(judging._judge_cache_path(), "/vol/minime/minime_judge_cache.sqlite")
            self.assertEqual(learning._gepa_dir("r1"), "/vol/minime/minime_gepa/r1")
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MINIME_DATA_DIR", None)
            self.assertTrue(judging._judge_cache_path().endswith("mlruns/minime_judge_cache.sqlite"))


class ReviewFixRoundTests(SimpleTestCase):
    """Copilot review of PR #48: round planning and the prompts of targeted asks."""
    CYCLE = ("violate", "representative", "violate", "comply")

    def test_a_round_of_two_keeps_a_blind_check(self):
        """#7: round(2 * 0.2) is 0; a positive share keeps one blind check."""
        plan = plan_round(qcr(COUNT, RELEVANT), size=2, heldout_share=0.2)
        self.assertEqual(Counter(a['split'] for a in plan)['heldout'], 1)
        self.assertEqual(Counter(a['split'] for a in plan_round(qcr(COUNT), size=2, heldout_share=0))['heldout'], 0)

    def test_blind_violations_rotate_over_every_rule(self):
        """#8: violations sit at fixed cycle positions; their targets still cover every rule."""
        for n in (2, 3, 4):
            q = QCR(task={'objective': 'x'}, statements=[Statement(id=f's{i}', text=f'rule {i}') for i in range(1, n + 1)])
            seen = {a['target'] for r in range(6)
                    for a in plan_round(q, size=10, heldout_share=0.2, blind_cycle=self.CYCLE, offset=r)
                    if a['split'] == 'heldout' and a['kind'] == 'violate'}
            self.assertEqual(seen, {f's{i}' for i in range(1, n + 1)}, n)

    def test_targeted_asks_state_the_rules_as_they_are_now(self):
        """#9: an edited rule ('exactly 5') is not contradicted by the seed requirement ('exactly 3')."""
        q = QCR(task={'objective': 'Healthcare SEM keywords', 'requirements': 'Exactly 3 keywords.'},
                statements=[Statement(id='s1', text='Exactly 5 keywords.')])
        plan = [{'kind': 'violate', 'split': 'learn', 'target': 's1', 'reason': None},
                {'kind': 'representative', 'split': 'heldout', 'target': None, 'reason': None}]
        prompts = []
        generate_round(q, plan, ['urgent care'], lambda p: prompts.append(p) or 'x', seed=1)
        targeted = next(p for p in prompts if 'must clearly break' in p)
        self.assertNotIn('Exactly 3', targeted)
        self.assertIn('Exactly 5 keywords.', targeted)
        self.assertIn('Healthcare SEM keywords', targeted)
        plain = next(p for p in prompts if p is not targeted)
        self.assertIn('Exactly 3', plain)              # the plain task, as described, for representative answers


class RewriterReplyTests(SimpleTestCase):
    def test_a_cut_off_or_empty_reply_is_asked_for_again_with_a_larger_cap(self):
        from types import SimpleNamespace as NS
        from unittest.mock import patch
        from optimizer import minime as mm
        reply = lambda content, finish: NS(choices=[NS(finish_reason=finish, message=NS(content=content))])
        calls = []
        def create(**kw):
            calls.append(kw['max_tokens'])
            return reply('', 'length') if len(calls) == 1 else reply('```\n1. a\n```', 'stop')
        ep = NS(client=NS(chat=NS(completions=NS(create=create))))
        with patch('optimizer.services._endpoint_for', lambda p: ep):
            text = mm.learning.reflect_once({}, 'prompt')
        self.assertEqual((text, calls), ('```\n1. a\n```', [mm.REWRITER_MAX_TOKENS, 2 * mm.REWRITER_MAX_TOKENS]))
        calls.clear()
        with patch('optimizer.services._endpoint_for', lambda p: NS(client=NS(chat=NS(completions=NS(
                create=lambda **kw: calls.append(kw['max_tokens']) or reply('1. a', 'stop')))))):
            self.assertEqual(mm.learning.reflect_once({}, 'prompt'), '1. a')
        self.assertEqual(len(calls), 1)                                   # a good reply: one call
        calls.clear()
        with patch('optimizer.services._endpoint_for', lambda p: NS(client=NS(chat=NS(completions=NS(
                create=lambda **kw: calls.append(kw['max_tokens']) or reply('1. a\n2. Do not', 'length')))))):
            self.assertEqual(mm.learning.reflect_once({}, 'prompt'), '')   # cut off twice: no rewrite at all
        self.assertEqual(calls, [mm.REWRITER_MAX_TOKENS, 2 * mm.REWRITER_MAX_TOKENS])

