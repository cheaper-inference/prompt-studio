import os
import tempfile
"""MiniMe v2 on the site: the judge works, the user keeps or flips, GEPA learns (LLM stubbed)."""
import json
import types
from types import SimpleNamespace
from unittest import skipUnless
from unittest.mock import patch
from django.contrib.auth.models import User
from django.test import TestCase
from django.utils import timezone
from . import minime as mm
from .models import JobState, MiniMeLabel, MiniMeTask

try:
    import gepa, mlflow  # noqa: F401
    HAS_GEPA = True
except ImportError:
    HAS_GEPA = False

DEFINITION = {'objective': 'Suggest search keywords for a healthcare service.',
              'requirements': '3 keywords in total.\nEach keyword must be relevant to the healthcare domain.'}
RULE = "Each keyword must not use the word 'first'."
GOOD = '1. urgent care near me\n2. walk-in clinic\n3. same day doctor'
FIRST = '1. first time flu shot\n2. walk-in clinic\n3. same day doctor'
CSV = ('service;answer;verdict;reason\n'
       f'urgent care;"{GOOD}";Pass;real patient searches\n'
       f'urgent care;"{FIRST}";Fail;nobody searches with first\n')


def judge_fn(texts, receive):
    """Stub judge: a statement about 'first' fails answers that use it; everything else passes."""
    return lambda inputs=None, outputs=None: receive(
        {'criteria': [dict(passed=not ('first' in t and 'first' in outputs), explanation='', evidence=None) for t in texts]})


def stub_llm(payload):
    return (lambda prompt: FIRST if 'must clearly break' in prompt else GOOD), judge_fn, None, 'stub-judge'


def job():
    return SimpleNamespace(stage='', status='running', result=None, error='')


LEARNING_PHASE = mm.learning_phase      # the real one (MiniMeV2Tests patches it to 'gepa')


class MiniMeV2Tests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user('ahmet', password='x')
        self.client.force_login(self.user)
        no_draft = patch.object(mm.rounds, 'draft_rubric', lambda *a: (_ for _ in ()).throw(RuntimeError('no endpoint in tests')))
        no_draft.start()
        self.addCleanup(no_draft.stop)
        import tempfile
        scratch = tempfile.mkdtemp()                                     # GEPA checkpoints never land in the repo
        no_repo = patch.object(mm.learning, '_gepa_dir', lambda run_id: f'{scratch}/{run_id}')
        no_repo.start()
        self.addCleanup(no_repo.stop)
        no_cache = patch.object(mm.judging, '_judge_cache_path', lambda: f'{scratch}/judge_cache.sqlite')
        no_cache.start()
        self.addCleanup(no_cache.stop)
        quiet = patch.object(mm.learning, 'reflect_once', lambda payload, prompt: 'NONE')   # the coach makes no real call
        quiet.start()
        self.addCleanup(quiet.stop)
        gepa_now = patch.object(mm.learning, 'learning_phase', lambda rows: 'gepa')   # most tests exercise the GEPA path
        gepa_now.start()
        self.addCleanup(gepa_now.stop)

    def post(self, url, body):
        return self.client.post(url, json.dumps(body), content_type='application/json')

    def create(self, start=True, **extra):
        """A new task, and (as the user would) Start optimizing: nothing starts on its own."""
        r = self.post('/api/minime/tasks', {'name': 'kw', 'definition': DEFINITION, 'inputs': ['urgent care'], **extra})
        self.assertEqual(r.status_code, 200, r.content)
        if start:
            r = self.post(f"/api/minime/tasks/{r.json()['id']}/start", {})
            self.assertEqual(r.status_code, 200, r.content)
        return r.json(), MiniMeTask.objects.get(pk=r.json()['id'])

    def round(self, task):
        stages = []
        class J(SimpleNamespace):
            def __setattr__(self, k, v):
                if k == 'stage':
                    stages.append(v)
                super().__setattr__(k, v)
        j = J(stage='', status='running', result=None, error='')
        self.last_stages = stages
        with patch.object(mm.judging, 'llm', stub_llm):
            mm.run_round(j, {'task_id': str(task.id), '_owner': 'ahmet'})
        self.assertEqual(j.status, 'done', j.error)
        return j

    def test_create_starts_nothing_until_start_optimizing(self):
        t, task = self.create(start=False)
        self.assertEqual((t['optimizing'], JobState.objects.count()), (False, 0))
        self.client.get(f'/api/minime/tasks/{task.id}/queue')                   # opening it starts nothing
        self.assertEqual(JobState.objects.count(), 0)
        r = self.post(f'/api/minime/tasks/{task.id}/start', {}).json()
        self.assertEqual((r['optimizing'], list(JobState.objects.values_list('kind', flat=True))), (True, ['minime_round']))
        r = self.post(f'/api/minime/tasks/{task.id}/stop', {}).json()
        self.assertEqual((r['optimizing'], JobState.objects.get().status), (False, 'cancelled'))
        l = MiniMeLabel.objects.create(task=task, input='i', output=FIRST, judged_pass=True, judged_version=0)
        for n in range(6):                                                       # paused: reviews kept, no job
            l = MiniMeLabel.objects.create(task=task, input='i', output=FIRST, judged_pass=True, judged_version=0)
            self.post(f'/api/minime/tasks/{task.id}/labels/{l.pk}', {'action': 'flip', 'reason': 'first'})
        self.assertEqual(JobState.objects.exclude(status='cancelled').count(), 0)

    def test_create_seeds_from_plain_words_and_starts_a_round(self):
        t, task = self.create()
        self.assertEqual([s['text'] for s in t['statements']], DEFINITION['requirements'].split('\n'))
        self.assertEqual(t['settings']['round_size'], mm.ROUND_SIZE)
        self.assertEqual(list(JobState.objects.values_list('kind', flat=True)), ['minime_round'])
        self.assertEqual(t['jobs']['minime_round']['stage'], 'Queued')
        r = self.post('/api/minime/tasks', {'definition': {'objective': ' '}, 'inputs': ['x']})
        self.assertEqual((r.status_code, r.json()['error']), (400, 'describe the task'))

    def test_round_judges_every_answer_and_hides_blind_verdicts(self):
        _, task = self.create(deployed_prompt='Keywords for {{input}}')
        self.round(task)
        writing = [x for x in self.last_stages if x.startswith('Writing')]
        self.assertEqual([w.split(' · ')[0] for w in writing], ['Writing 0%', 'Writing 80%'])
        self.assertEqual([x for x in self.last_stages if not x.startswith('Drafting')][0], 'Finding inputs')
        self.assertTrue(any(x.startswith('Judging 80%') for x in self.last_stages))
        labels = list(task.labels.all())
        self.assertEqual(len(labels), 10)
        self.assertEqual(sum(l.split == 'heldout' for l in labels), 2)
        self.assertTrue(all(l.judged_pass is not None and l.judged_version == 0 for l in labels))
        shown = [mm.next_candidate(task)] + [None]
        nxt = self.client.get(f'/api/minime/tasks/{task.id}/queue').json()['next']
        self.assertIn('blind', nxt)
        blind = task.labels.filter(split='heldout').first()
        MiniMeLabel.objects.filter(task=task).exclude(pk=blind.pk).update(passed=True)
        self.assertEqual(mm.next_candidate(task), {'id': blind.pk, 'input': blind.input, 'output': blind.output, 'blind': True})

    def test_rounds_grow_an_input_pool_and_blind_checks_use_unseen_inputs(self):
        _, task = self.create()
        pool = ['dermatology clinic', 'flu shot', 'pediatric urgent care', 'sports physical', 'covid test',
                'travel vaccines', 'std testing', 'allergy shots', 'physical therapy', 'x-ray near me']
        stub = lambda payload: ((lambda p: '\n'.join(f'{n}. {x}' for n, x in enumerate(pool + ['urgent care'], 1))
                                 if p.startswith('You are helping test') else GOOD), judge_fn, None, 'stub')
        j = job()
        with patch.object(mm.judging, 'llm', stub):
            mm.run_round(j, {'task_id': str(task.id), '_owner': 'ahmet'})
        self.assertEqual(j.status, 'done', j.error)
        task.refresh_from_db()
        grown = task.settings['input_pool']
        self.assertEqual(len(grown), mm.POOL_GROWTH)                        # capped per round
        self.assertNotIn('urgent care', grown)                              # a seed is never re-added
        pools = mm.input_pools(task)
        self.assertTrue(set(pools['heldout']).isdisjoint(pools['learn']))   # blind inputs stay unseen
        used = {l.split: set() for l in task.labels.all()}
        for l in task.labels.all():
            used[l.split].add(l.input)
        self.assertTrue(used['heldout'] <= set(pools['heldout']))
        self.assertTrue(used['learn'] <= set(pools['learn']))
        self.assertIn('Found 8 new inputs similar to yours', mm.summary(task)['log'][1]['text'])
        r = self.post(f'/api/minime/tasks/{task.id}', {'input_pool': grown[1:] + ['sneaky new one']}).json()
        self.assertEqual([x['text'] for x in r['input_pool']], grown[1:])  # prune only, never add

    def test_keep_flip_and_blind_review(self):
        _, task = self.create()
        shown = MiniMeLabel.objects.create(task=task, input='i', output=FIRST, judged_pass=True,
                                           judged_failed=[], judged_version=0)
        blind = MiniMeLabel.objects.create(task=task, split='heldout', input='i', output=GOOD, judged_pass=True)
        url = lambda l: f'/api/minime/tasks/{task.id}/labels/{l.pk}'
        r = self.post(url(shown), {'action': 'flip', 'reason': ' '})        # a No needs no comment
        self.assertEqual(r.status_code, 200)
        shown.refresh_from_db()
        self.assertEqual((shown.passed, shown.reason), (False, ''))
        self.post(url(shown), {'action': 'flip', 'reason': 'uses first'})
        shown.refresh_from_db()
        self.assertEqual((shown.passed, shown.action, shown.reason), (False, 'flip', 'uses first'))
        self.assertEqual(self.post(url(blind), {'action': 'keep'}).status_code, 400)   # nothing to keep: verdict hidden
        self.post(url(blind), {'action': 'label', 'verdict': 'Pass'})                    # reason optional
        blind.refresh_from_db()
        self.assertEqual((blind.passed, blind.action), (True, ''))

    def test_comments_count_as_feedback_and_must_be_met(self):
        _, task = self.create()
        mk = lambda **kw: MiniMeLabel.objects.create(task=task, input='i', output=FIRST, judged_pass=False,
                                                     judged_failed=[{'id': 's1', 'text': '3 keywords in total.'}], **kw)
        mk(passed=False, action='keep')                                   # plain Yes: no feedback
        note = mk(passed=False, action='keep', reason="fails, but because it uses 'first', not the count")
        mk(passed=True, action='flip')                                    # a No, without a comment
        mk(split='heldout', passed=False, reason='blind: never learned from')
        self.assertEqual(mm.feedback_since_learn(task), 2)
        row = next(r for r in mm.confirmed(task) if r['id'] == str(note.pk))
        self.assertTrue(row['note'])
        asked = []
        check = lambda output, said, comment: asked.append(said) or 'first' in said
        self.assertFalse(mm._right(row, False, [{'text': '3 keywords in total.'}], check))   # right verdict, wrong reason
        self.assertTrue(mm._right(row, False, [{'text': RULE}], check))
        self.assertFalse(mm._right(row, True, [], check))                                     # wrong verdict: no need to ask
        self.assertEqual(asked, ['FAIL (breaks: 3 keywords in total.)', f'FAIL (breaks: {RULE})'])

    def test_a_failed_round_is_reported_until_one_succeeds(self):
        t, task = self.create()
        job = JobState.objects.get(kind='minime_round')
        self.assertEqual(mm.summary(task)['failed'], {})                 # queued, not failed
        job.status, job.error = 'error', "InternalServerError: Error code: 503 - {'message': 'No compatible route completed this request.'}"
        job.save()
        failed = mm.summary(task)['failed']
        self.assertEqual(list(failed), ['minime_round'])
        self.assertIn('unavailable (503)', failed['minime_round']['error'])
        later = mm.start('minime_round', 'ahmet', {'task_id': str(task.id)})
        self.assertEqual(mm.summary(task)['failed'], {})                 # a retry is running
        JobState.objects.filter(pk=later.id).update(status='done')
        self.assertEqual(mm.summary(task)['failed'], {})                 # and it worked

    def test_judge_quality_is_agreement_on_blind_checks_only(self):
        _, task = self.create()
        q = mm.summary(task)['quality']
        self.assertEqual({k: q[k] for k in ('pct', 'agreed', 'n')}, {'pct': None, 'agreed': 0, 'n': 0})
        self.assertIn('Not enough', q['detail']['action'])
        for judged, mine in [(True, True), (False, False), (True, False), (False, False)]:
            l = MiniMeLabel.objects.create(task=task, split='heldout', input='i', output=GOOD, judged_pass=judged, judged_version=0)
            mm.review(task, l.pk, 'label', 'Pass' if mine else 'Fail')
        shown = MiniMeLabel.objects.create(task=task, input='i', output=GOOD, judged_pass=True)
        mm.review(task, shown.pk, 'flip', reason='x')          # shown rows never count (anchoring)
        q = mm.summary(task)['quality']
        self.assertEqual({k: q[k] for k in ('pct', 'agreed', 'n')}, {'pct': 75, 'agreed': 3, 'n': 4})

    def test_each_rule_version_is_scored_on_all_blind_checks(self):
        _, task = self.create()
        for out, mine in [(FIRST, 'Fail'), (GOOD, 'Pass')] * 3:          # v0 passes all: right on GOOD only
            l = MiniMeLabel.objects.create(task=task, split='heldout', input='i', output=out, judged_pass=True, judged_version=0)
            mm.review(task, l.pk, 'label', mine)
        self.assertEqual(mm.quality(task)['detail']['representative'], {'pct': 50, 'hit': 3, 'of': 6})
        mm.edit_qcr(task, [{'id': s.id, 'text': s.text} for s in mm.current_qcr(task).active()] + [{'text': RULE}])
        self.assertEqual(mm.quality(task)['detail']['representative']['of'], 0)     # v1 not scored yet
        judged = []
        def counting(texts, receive):
            judged.append(tuple(texts))
            return judge_fn(texts, receive)
        self.assertEqual(mm.score_blind(task, counting, None), 6)
        self.assertTrue(all(t == (RULE,) for t in judged))                 # only the new rule (here: checked in code)
        q = mm.quality(task)
        self.assertEqual((q['version'], q['detail']['representative']), (1, {'pct': 100, 'hit': 6, 'of': 6}))
        self.assertEqual(mm.score_blind(task, counting, None), 0)   # already scored
        vs = mm.quality_series(task)['versions']
        self.assertEqual([(v['v'], v['rep']['pct']) for v in vs], [(0, 50), (1, 100)])
        self.assertEqual([(v['type'], v['changes']) for v in vs],
             [('seed', []), ('edit', [{'op': 'add', 'before': '', 'after': RULE}])])   # what changed per version
        l = MiniMeLabel.objects.create(task=task, split='heldout', input='i', output=GOOD, judged_pass=True, judged_version=1)
        mm.review(task, l.pk, 'label', 'Pass')                              # reviewed under v1: counts for v1
        self.assertEqual(mm.quality(task)['detail']['representative']['of'], 7)
        S = mm.quality_series(task)
        self.assertEqual((S['to_score'], [v['rep']['of'] for v in S['versions']]), (1, [6, 7]))   # v0 not scored on it yet
        self.assertEqual(mm.score_blind(task, counting, None), 1)            # v0 gets a verdict on the new check
        S = mm.quality_series(task)                                          # now the same checks for every version
        self.assertEqual((S['to_score'], [(v['rep']['of'], v['rep']['pct']) for v in S['versions']]), (0, [(7, 57), (7, 100)]))

    def test_quality_reports_representative_and_adversarial_agreement_separately(self):
        _, task = self.create()
        def blind(judged, mine, kind='representative'):
            l = MiniMeLabel.objects.create(task=task, split='heldout', input='i', output=GOOD, judged_pass=judged, kind=kind,
                                           judged_version=0)
            mm.review(task, l.pk, 'label', 'Pass' if mine else 'Fail')
        for _ in range(9):
            blind(True, True)                          # representative: agrees 9 of 10
        blind(False, True)
        for _ in range(3):
            blind(False, False, 'violate')             # adversarial: agrees 3 of 4
        blind(True, False, 'violate')
        q = mm.summary(task)['quality']
        self.assertEqual((q['pct'], q['n']), (90, 10))                                  # headline = representative
        self.assertEqual(q['detail']['representative'], {'pct': 90, 'hit': 9, 'of': 10})
        self.assertEqual(q['detail']['adversarial'], {'pct': 75, 'hit': 3, 'of': 4})
        self.assertIn('tougher outputs', q['detail']['action'])

    def test_flips_start_learning_and_a_low_queue_starts_a_round(self):
        _, task = self.create()
        JobState.objects.all().delete()
        for n in range(mm.GEPA_EVERY):
            l = MiniMeLabel.objects.create(task=task, input='i', output=FIRST, judged_pass=True, judged_version=0)
            mm.review(task, l.pk, 'flip', reason='uses first')
        mm.autostart(task, 'ahmet')
        self.assertEqual(sorted(JobState.objects.values_list('kind', flat=True)), ['minime_learn', 'minime_round'])
        mm.autostart(task, 'ahmet')                                                    # already running: no duplicates
        self.assertEqual(JobState.objects.count(), 2)

    def learn_with(self, rewrite, flips=5, settled=True):
        _, task = self.create()
        for n in range(flips):
            l = MiniMeLabel.objects.create(task=task, input='i', output=FIRST, judged_pass=True, judged_version=0)
            mm.review(task, l.pk, 'flip', reason='uses first')
        keep = MiniMeLabel.objects.create(task=task, input='i', output=GOOD, judged_pass=True, judged_version=0)
        mm.review(task, keep.pk, 'keep')
        waiting = MiniMeLabel.objects.create(task=task, input='i', output=FIRST, judged_pass=True, judged_version=0)
        j = job()
        mode = {'settled': settled, 'every': mm.LEARN_EVERY if settled else mm.LEARN_EVERY_EARLY,
                'max_edits': mm.MAX_EDITS if settled else mm.MAX_EDITS_EARLY}
        with patch.object(mm.judging, 'llm', stub_llm), patch.object(mm.learning, 'run_gepa', lambda *a, **k: rewrite), \
                patch.object(mm.scoring, 'learning_mode', lambda task: mode):
            mm.run_learn(j, {'task_id': str(task.id), '_owner': 'ahmet'})
        self.assertEqual(j.status, 'done', j.error)
        if (j.result or {}).get('test'):                     # the follow-up job: check the pick on the reviews
            task.refresh_from_db()
            self.assertTrue(next(e for e in mm.summary(task)['log'] if e.get('kind') == 'learn')['checking'])
            j.result.update(mm.test_pick(task, j.result['test'], judge_fn, None, {'task_id': str(task.id)}))
        waiting.refresh_from_db()
        task.refresh_from_db()
        return task, j, waiting

    def test_learn_applies_a_safe_edit_and_rejudges_waiting_answers(self):
        task, j, waiting = self.learn_with(f'1. 3 keywords in total.\n2. Each keyword must be relevant to the healthcare domain.\n3. {RULE}')
        self.assertTrue(j.result['accepted'])
        self.assertEqual([e['op'] for e in j.result['edits']], ['add'])
        self.assertEqual(len(j.result['fixed']), 5)
        s = mm.summary(task)
        self.assertEqual((s['version'], s['statements'][-1]['text'], s['feedback_since_learn']), (1, RULE, 0))
        self.assertIn('Learned: added', s['log'][0]['text'])
        self.assertEqual((s['log'][0]['accepted'], s['log'][0]['edits'][0]['op'], s['log'][0]['version']), (True, 'add', 1))
        self.assertEqual((waiting.judged_pass, waiting.judged_version), (False, 1))

    def test_round_saves_each_judged_batch_at_once(self):
        _, task = self.create(deployed_prompt='Keywords for {{input}}')
        saved, real = [], mm._verdicts
        def counting(*a, **k):
            saved.append(MiniMeLabel.objects.filter(task=task).count())
            return real(*a, **k)
        with patch.object(mm.judging, '_verdicts', counting):
            self.round(task)
        self.assertEqual(saved, [0, mm.PARALLEL])           # the first batch was reviewable before the second was judged

    def test_learn_applies_the_pick_even_when_it_breaks_a_reviewed_answer(self):
        task, j, _ = self.learn_with("1. 3 keywords in total.\n2. Each keyword must not use the word 'walk-in'.\n3. " + RULE)
        follow = [x.payload for x in JobState.objects.filter(kind='minime_learn') if x.payload.get('test')]
        self.assertEqual([p['test'] for p in follow], [{'version': 1, 'based_on': 0}])   # the check runs as its own job
        self.assertTrue(j.result['accepted'])                                  # no gate after GEPA
        self.assertEqual(len(j.result['broke']), 1)
        s = mm.summary(task)
        self.assertEqual((s['version'], s['feedback_since_learn']), (1, 0))   # the run used those flips
        self.assertIn('Learned:', s['log'][0]['text'])
        self.assertIn('gets 1 wrong', s['log'][0]['text'])
        event = s['log'][0]
        self.assertEqual((event['kind'], event['accepted'], len(event['fixed']), len(event['broke'])), ('learn', True, 5, 1))
        self.assertEqual((event['rewriter'], event['judge']), ('claude-opus-4.8', 'stub-judge'))
        self.assertEqual({k: event['broke'][0][k] for k in ('output', 'verdict')}, {'output': GOOD, 'verdict': 'PASS'})

    def test_while_new_a_net_gain_is_applied(self):
        task, j, _ = self.learn_with("1. 3 keywords in total.\n2. Each keyword must not use the word 'walk-in'.\n3. " + RULE,
                                     settled=False)
        self.assertTrue(j.result['accepted'])                                 # fixes 5, gets 1 wrong
        event = next(e for e in mm.summary(task)['log'] if e.get('kind') == 'learn')
        self.assertEqual((event['kind'], event['accepted'], event['bold'], len(event['fixed']), len(event['broke'])),
                         ('learn', True, True, 5, 1))

    def test_bold_until_settled(self):
        _, task = self.create()
        self.assertEqual(mm.learning_mode(task), {'settled': False, 'every': 1, 'max_edits': None})
        for n in range(mm.READY_STABLE):
            l = MiniMeLabel.objects.create(task=task, input='i', output=GOOD, judged_pass=True, judged_version=0)
            mm.review(task, l.pk, 'keep')
        self.assertEqual(mm.learning_mode(task), {'settled': True, 'every': 3, 'max_edits': 3})
        self.assertEqual(mm.summary(task)['learn_every'], mm.GEPA_EVERY)     # once GEPA runs: every GEPA_EVERY

    def test_your_lines_are_kept_as_written(self):
        t, task = self.create()
        with patch.object(mm.rounds, 'draft_rubric', lambda *a: self.fail('must not draft over the user\'s own rules')):
            self.round(task)
        task.refresh_from_db()
        s = mm.summary(task)
        self.assertEqual((s['version'], [x['text'] for x in s['statements']]), (0, DEFINITION['requirements'].split('\n')))

    def test_with_no_rules_a_small_rubric_is_drafted_once(self):
        r = self.post('/api/minime/tasks', {'name': 'kw', 'definition': {'objective': DEFINITION['objective']},
                                            'inputs': ['urgent care']})
        task = MiniMeTask.objects.get(pk=r.json()['id'])
        drafted = "1. Exactly 3 keywords.\n2. Relevant to healthcare.\n3. " + RULE + "\n4. No brand names.\n5. Lowercase."
        with patch.object(mm.rounds, 'draft_rubric', lambda payload, task_text, req: drafted):
            self.round(task)
            self.round(task)
        task.refresh_from_db()
        s = mm.summary(task)
        self.assertEqual(s['version'], 1)                                   # drafted once, not per round
        self.assertEqual([x['text'] for x in s['statements']], parse_lines(drafted)[:mm.DRAFT_MAX])   # small start
        entry = next(e for e in s['log'] if e.get('kind') == 'draft')
        self.assertEqual([e['op'] for e in entry['edits']], ['add'] * mm.DRAFT_MAX)

    def test_a_failed_draft_starts_with_no_rules(self):
        r = self.post('/api/minime/tasks', {'name': 'kw', 'definition': {'objective': DEFINITION['objective']},
                                            'inputs': ['urgent care']})
        task = MiniMeTask.objects.get(pk=r.json()['id'])
        self.round(task)                                                     # draft_rubric raises in tests
        task.refresh_from_db()
        s = mm.summary(task)
        self.assertEqual((s['version'], s['statements']), (0, []))
        self.assertTrue(any('Could not draft' in e['text'] for e in s['log']))

    def test_a_run_chains_the_next_when_feedback_arrived_meanwhile(self):
        _, task = self.create()
        JobState.objects.all().delete()
        for n in range(2):
            l = MiniMeLabel.objects.create(task=task, input='i', output=FIRST, judged_pass=True, judged_version=0)
            mm.review(task, l.pk, 'flip', reason='uses first')
        def gepa_while_user_reviews(*a, **k):
            l = MiniMeLabel.objects.create(task=task, input='i', output=FIRST, judged_pass=True, judged_version=0)
            mm.review(task, l.pk, 'flip', reason='still uses first')
            return 'Sorry, nothing.'
        _p = patch.object(mm.config, 'GEPA_EVERY', 1); _p.start(); self.addCleanup(_p.stop)
        j = job()
        with patch.object(mm.judging, 'llm', stub_llm), patch.object(mm.learning, 'run_gepa', gepa_while_user_reviews):
            mm.run_learn(j, {'task_id': str(task.id), '_owner': 'ahmet'})
        self.assertEqual(j.status, 'done', j.error)
        self.assertEqual(list(JobState.objects.values_list('kind', 'status')), [('minime_learn', 'queued')])

    def test_checking_a_change_judges_only_the_statements_that_changed(self):
        from ps_utils.minime.qcr import QCR, Statement
        current = QCR(task=dict(DEFINITION), statements=[Statement('s1', '3 keywords in total.'),
                                                         Statement('s2', 'Each keyword must be relevant to the healthcare domain.')])
        candidate = QCR(task=dict(DEFINITION), statements=[Statement('s1', '3 keywords in total.'),
                                                           Statement('s2', 'Each keyword must be relevant to the healthcare domain.'),
                                                           Statement('s3', RULE)])
        rows = [{'id': '1', 'input': 'i', 'output': FIRST}, {'id': '2', 'input': 'i', 'output': GOOD}]
        before = {'1': (True, []), '2': (True, [])}
        judged = []
        def make_judge(texts, receive):
            judged.append(tuple(texts))
            return judge_fn(texts, receive)
        out = mm._reuse_verdicts(current, candidate, rows, before, make_judge, None)
        self.assertEqual({k: v[0] for k, v in out.items()}, {'1': False, '2': True})
        self.assertEqual(out['1'][1], [{'id': 's3', 'text': RULE}])
        self.assertFalse(any('relevant to the healthcare' in x for t in judged for x in t))   # kept LLM statement: not re-judged
        kept_fail = mm._reuse_verdicts(current, current, rows, {'1': (False, [{'id': 's2', 'text': 'Each keyword must be relevant to the healthcare domain.'}]),
                                                              '2': (True, [])}, make_judge, None)
        self.assertEqual({k: v[0] for k, v in kept_fail.items()}, {'1': False, '2': True})   # nothing changed: no calls needed

    def test_rule_edits_keep_ids_and_a_no_op_save_is_refused(self):
        _, task = self.create()
        same = [{'id': s.id, 'text': s.text, 'status': 'active'} for s in mm.current_qcr(task).active()]
        v = task.versions.count()
        with self.assertRaises(mm.MiniMeError):
            mm.edit_qcr(task, same)                                        # nothing changed: no new version
        self.assertEqual(task.versions.count(), v)
        mm.edit_qcr(task, [{**same[0], 'text': 'Exactly 3 keywords.'}, {**same[1], 'status': 'retired'}] + same[2:])
        q = mm.current_qcr(task)
        self.assertEqual(next(s for s in q.statements if s.id == same[0]['id']).text, 'Exactly 3 keywords.')   # reworded, same id
        self.assertEqual(next(s for s in q.statements if s.id == same[1]['id']).status, 'retired')

    def test_reviewed_answers_from_older_versions_reuse_unchanged_statements(self):
        _, task = self.create()
        l = MiniMeLabel.objects.create(task=task, input='i', output=FIRST, judged_pass=True, judged_failed=[],
                                       judged_version=0, passed=False, action='flip', reason='uses first')
        mm.edit_qcr(task, [{'id': 's1', 'text': '3 keywords in total.'},
                           {'id': 's2', 'text': 'Each keyword must be relevant to the healthcare domain.'}, {'text': RULE}])
        judged = []
        def make_judge(texts, receive):
            judged.append(tuple(texts))
            return judge_fn(texts, receive)
        out = mm._current_verdicts(task, mm.current_qcr(task), mm.confirmed(task), make_judge, None)
        self.assertEqual(out[str(l.pk)][0], False)                         # the added rule, judged now
        self.assertFalse(any('relevant to the healthcare' in x for t in judged for x in t))   # unchanged: reused

    def test_a_gepa_result_survives_a_cut_off_run(self):
        _, task = self.create()
        for n in range(2):
            l = MiniMeLabel.objects.create(task=task, input='i', output=FIRST, judged_pass=True, judged_version=0)
            mm.review(task, l.pk, 'flip', reason='uses first')
        rewrite = f'1. 3 keywords in total.\n2. Each keyword must be relevant to the healthcare domain.\n3. {RULE}'
        calls = []
        def gepa(*a, **k):
            calls.append(1)
            return rewrite
        def cut_off(*a, **k):
            raise RuntimeError('worker stopped')
        j = job()
        with patch.object(mm.judging, 'llm', stub_llm), patch.object(mm.learning, 'run_gepa', gepa), patch.object(mm.store, '_new_version', cut_off):
            mm.run_learn(j, {'task_id': str(task.id), '_owner': 'ahmet'})       # GEPA ran, then the job died
        task.refresh_from_db()
        self.assertEqual(j.status, 'error')
        self.assertEqual((task.settings['pending_rewrite']['text'], task.settings['pending_rewrite']['based_on']), (rewrite, 0))
        self.assertEqual(mm.feedback_since_learn(task), 2)                      # feedback not used up
        j = job()
        with patch.object(mm.judging, 'llm', stub_llm), patch.object(mm.learning, 'run_gepa', gepa):
            mm.run_learn(j, {'task_id': str(task.id), '_owner': 'ahmet'})       # the next run picks it up
        self.assertEqual((j.status, j.result['accepted'], len(calls)), ('done', True, 1))   # GEPA not run again
        task.refresh_from_db()
        self.assertNotIn('pending_rewrite', task.settings)
        self.assertEqual(mm.summary(task)['statements'][-1]['text'], RULE)

    def test_a_pending_rewrite_for_an_older_rubric_is_ignored(self):
        _, task = self.create()
        l = MiniMeLabel.objects.create(task=task, input='i', output=FIRST, judged_pass=True, judged_version=0)
        mm.review(task, l.pk, 'flip', reason='uses first')
        mm._update_settings(task, lambda s: {**s, "pending_rewrite": {"text": "1. old", "based_on": 99, "mark": "x"}})
        calls = []
        j = job()
        with patch.object(mm.judging, 'llm', stub_llm), patch.object(mm.learning, 'run_gepa', lambda *a, **k: calls.append(1) or 'Sorry'):
            mm.run_learn(j, {'task_id': str(task.id), '_owner': 'ahmet'})
        self.assertEqual(len(calls), 1)

    def test_d_train_splits_into_disjoint_d_feedback_and_d_pareto(self):
        rows = ([{'id': str(n), 'passed': n % 2 == 0, 'flip': n % 5 == 0, 'note': False} for n in range(150)])
        feedback, pareto = mm.split_train(rows)
        self.assertEqual(len(pareto), mm.PARETO_SIZE)                                   # capped at n_pareto
        self.assertEqual(len(feedback) + len(pareto), 150)
        self.assertFalse({r['id'] for r in feedback} & {r['id'] for r in pareto})        # disjoint
        self.assertTrue({True, False} <= {r['passed'] for r in pareto})                 # both verdicts
        self.assertTrue(any(r['flip'] for r in pareto) and any(r['flip'] for r in feedback))
        self.assertEqual(mm.split_train(rows), (feedback, pareto))                       # deterministic
        small_f, small_p = mm.split_train(rows[:12])
        self.assertEqual((len(small_f), len(small_p)), (8, 4))                           # 1/3 while small
        self.assertEqual(mm.split_train(rows[:1]), (rows[:1], rows[:1]))                  # too few to split

    def test_cold_start_reflects_once_without_gepa(self):
        _, task = self.create()
        l = MiniMeLabel.objects.create(task=task, input='i', output=FIRST, judged_pass=True, judged_version=0)
        mm.review(task, l.pk, 'flip', reason='nobody searches with first')
        prompts = []
        j = job()
        with patch.object(mm.learning, 'learning_phase', LEARNING_PHASE), \
                patch.object(mm.judging, 'llm', stub_llm), patch.object(mm.learning, 'run_gepa', lambda *a, **k: self.fail('no GEPA at cold start')), \
                patch.object(mm.learning, 'reflect_once', lambda payload, prompt: prompts.append(prompt) or
                             f'```\n1. 3 keywords in total.\n2. Each keyword must be relevant to the healthcare domain.\n3. {RULE}\n```'):
            mm.run_learn(j, {'task_id': str(task.id), '_owner': 'ahmet'})
        self.assertEqual((j.status, j.result['accepted']), ('done', True), j.error)
        self.assertIn('nobody searches with first', prompts[0])                       # your feedback is the ASI
        self.assertIn('The rubric said: PASS', prompts[0])
        task.refresh_from_db()
        entry = next(e for e in mm.summary(task)['log'] if e.get('kind') == 'learn')
        self.assertEqual(entry['phase'], 'direct')

    def test_gepa_starts_once_there_is_enough_to_split(self):
        row = lambda n, passed, fb=False: {'id': str(n), 'passed': passed, 'flip': fb, 'note': False}
        few = [row(n, n % 2 == 0, n < 3) for n in range(10)]
        many = [row(n, n % 2 == 0, n < 3) for n in range(20)]
        one_sided = [row(n, True, n < 3) for n in range(25)]
        self.assertEqual([LEARNING_PHASE(x) for x in (few, many, one_sided)], ['direct', 'gepa', 'direct'])

    def test_the_rewriters_notes_before_the_rubric_never_become_rules(self):
        from ps_utils.minime.rubric_text import parse
        reply = ("Looking at the examples:\n1. **Example 1**: the expert passes it.\n2. **Example 2**: fails.\n\n"
                 "```\n1. At most 280 characters.\n2. Emojis accompany content inline.\n```")
        self.assertEqual(parse(reply), ['At most 280 characters.', 'Emojis accompany content inline.'])
        self.assertEqual(parse('1. a\n2. b'), ['a', 'b'])                        # no block: the list itself

    def test_undo_an_applied_change(self):
        task, j, _ = self.learn_with(f'1. 3 keywords in total.\n2. Each keyword must be relevant to the healthcare domain.\n3. {RULE}')
        at = next(e for e in mm.summary(task)['log'] if e.get('kind') == 'learn')['at']
        r = self.post(f'/api/minime/tasks/{task.id}/undo', {'at': at}).json()
        self.assertEqual((r['version'], [x['text'] for x in r['statements']]), (2, DEFINITION['requirements'].split('\n')))
        self.assertEqual(next(e for e in r['log'] if e.get('at') == at)['undone'], 2)
        again = self.post(f'/api/minime/tasks/{task.id}/undo', {'at': at})
        self.assertEqual(again.status_code, 400)

    def test_deleting_a_task_ends_its_jobs_quietly(self):
        _, task = self.create()
        queued = JobState.objects.get(kind='minime_round')
        l = MiniMeLabel.objects.create(task=task, input='i', output=FIRST, judged_pass=True, judged_version=0)
        mm.review(task, l.pk, 'flip', reason='uses first')
        tid = str(task.id)
        def gepa_then_delete(*a, **k):
            MiniMeTask.objects.filter(pk=tid).delete()          # the user deletes the task meanwhile
            return f'1. 3 keywords in total.\n2. {RULE}'
        j = job()
        with patch.object(mm.judging, 'llm', stub_llm), patch.object(mm.learning, 'run_gepa', gepa_then_delete):
            mm.run_learn(j, {'task_id': tid, '_owner': 'ahmet'})
        self.assertEqual((j.status, j.result), ('done', {'skipped': 'the task was deleted'}))
        _, other = self.create()
        mm.delete_task(other)
        self.assertEqual(JobState.objects.filter(payload__task_id=str(other.id)).exclude(status='cancelled').count(), 0)

    def test_feedback_alone_is_read_as_agreeing_or_not(self):
        _, task = self.create()
        url = lambda l: f'/api/minime/tasks/{task.id}/labels/{l.pk}'
        a = MiniMeLabel.objects.create(task=task, input='i', output=FIRST, judged_pass=False, judged_version=0)
        b = MiniMeLabel.objects.create(task=task, input='i', output=GOOD, judged_pass=False, judged_version=0)
        with patch.object(mm.tasks, 'interpret_feedback', lambda task, label, seen, c: 'fine' not in c):
            self.assertEqual(self.post(url(a), {'action': 'comment', 'reason': ' '}).status_code, 400)
            self.post(url(a), {'action': 'comment', 'reason': 'right, it uses first', 'shown_pass': False})
            self.post(url(b), {'action': 'comment', 'reason': 'this one is fine', 'shown_pass': False})
        a.refresh_from_db(); b.refresh_from_db()
        self.assertEqual((a.passed, a.action, a.reason), (False, 'keep', 'right, it uses first'))   # agrees: kept, a note
        self.assertEqual((b.passed, b.action), (True, 'flip'))                                       # disagrees: flipped
        def boom(*args):
            raise RuntimeError('endpoint down')
        c = MiniMeLabel.objects.create(task=task, input='i', output=GOOD, judged_pass=False, judged_version=0)
        with patch.object(mm.tasks, 'interpret_feedback', boom):
            r = self.post(url(c), {'action': 'comment', 'reason': 'hmm'})
        self.assertEqual(r.status_code, 400)
        self.assertIn('could not read your feedback', r.json()['error'])

    def test_a_gepa_run_resumes_from_its_checkpoint_with_more_feedback(self):
        import os, tempfile
        _, task = self.create()
        for n in range(4):
            l = MiniMeLabel.objects.create(task=task, input=f'i{n}', output=FIRST, judged_pass=True, judged_version=0)
            mm.review(task, l.pk, 'flip', reason='uses first')
        tmp = tempfile.mkdtemp()
        seen = []
        def gepa(current, rows, *a, split=None, run_dir=None, **k):
            seen.append(([r['id'] for r in split[0]], [r['id'] for r in split[1]], run_dir))
            os.makedirs(run_dir, exist_ok=True)
            if len(seen) == 1:
                raise RuntimeError('Ctrl-C')                       # cut off mid-search
            return f'1. 3 keywords in total.\n2. {RULE}'
        with patch.object(mm.learning, '_gepa_dir', lambda run_id: os.path.join(tmp, run_id)), \
                patch.object(mm.judging, 'llm', stub_llm), patch.object(mm.learning, 'run_gepa', gepa):
            mm.run_learn(job(), {'task_id': str(task.id), '_owner': 'ahmet'})
            task.refresh_from_db()
            run = task.settings['gepa_run']
            extra = MiniMeLabel.objects.create(task=task, input='new', output=FIRST, judged_pass=True, judged_version=0)
            mm.review(task, extra.pk, 'flip', reason='uses first again')          # reviewed meanwhile
            j = job()
            mm.run_learn(j, {'task_id': str(task.id), '_owner': 'ahmet'})
        self.assertEqual(j.status, 'done', j.error)
        (fb1, par1, dir1), (fb2, par2, dir2) = seen
        self.assertEqual((par2, dir2), (par1, dir1))                 # same frozen D_pareto, same checkpoint: resumed
        self.assertIn(str(extra.pk), fb2); self.assertNotIn(str(extra.pk), fb1)   # D_feedback grew
        task.refresh_from_db()
        self.assertNotIn('gepa_run', task.settings)                 # done: checkpoint dropped
        self.assertFalse(os.path.exists(dir1))

    def test_judge_results_survive_a_new_job(self):
        calls = []
        def make_judge(texts, receive):
            def call(inputs=None, outputs=None):
                calls.append(1)
                receive({'criteria': [{'passed': True}], 'overall_pass': True})
            return call
        for n in range(2):                                        # two jobs: two fresh in-memory caches
            judge, _ = mm._cached(make_judge, None, 'm')
            got = []
            judge(('rule',), got.append)(inputs='i', outputs='o')
            self.assertEqual(got, [{'criteria': [{'passed': True}], 'overall_pass': True}])
        self.assertEqual(len(calls), 1)                           # the second job paid nothing
        def failing(texts, receive):
            return lambda inputs=None, outputs=None: (calls.append(1), receive({'criteria': [{'status': 'evaluation_error'}]}))
        for n in range(2):
            mm._cached(failing, None, 'm')[0](('other',), lambda r: None)(inputs='i', outputs='o')
        self.assertEqual(len(calls), 3)                           # errors are never stored

    def test_the_judge_store_is_safe_across_threads(self):
        import threading
        writer = mm._JudgeStore('m')
        for i in range(40):                                       # rule results and item results, side by side
            writer[('batch', (f'r{i}',), 'i', 'o')] = {'criteria': [{'passed': True, 'n': i}]}
            writer[('item', 'req', (f'x{i}',), 'i', 'o')] = [{'passed': True, 'n': i}]
        reader, wrong = mm._JudgeStore('m'), []                   # a fresh store reads them from the file
        def work(t):
            for i in range(40):
                b, it = ('batch', (f'r{i}',), 'i', 'o'), ('item', 'req', (f'x{i}',), 'i', 'o')
                if not (b in reader and isinstance(reader[b], dict) and reader[b]['criteria'][0]['n'] == i):
                    wrong.append(('batch', i))
                if not (it in reader and isinstance(reader[it], list) and reader[it][0]['n'] == i):
                    wrong.append(('item', i))
                reader[('batch', (f'w{t}-{i}',), 'i', 'o')] = {'criteria': []}   # writes interleave with reads
        threads = [threading.Thread(target=work, args=(t,)) for t in range(8)]
        [t.start() for t in threads]; [t.join() for t in threads]
        self.assertEqual(wrong, [])

    def test_a_cached_result_of_the_wrong_shape_is_judged_again(self):
        calls = []
        def make_judge(texts, receive):
            return lambda inputs=None, outputs=None: (calls.append(1), receive({'criteria': [{'passed': True}]}))
        mm._JudgeStore('m')[('batch', ('rule',), 'i', 'o')] = [{'passed': True, 'explanation': 'an item verdict'}]
        got = []
        mm._cached(make_judge, None, 'm')[0](('rule',), got.append)(inputs='i', outputs='o')
        self.assertEqual((len(calls), got), (1, [{'criteria': [{'passed': True}]}]))
        mm._cached(make_judge, None, 'm')[0](('rule',), got.append)(inputs='i', outputs='o')
        self.assertEqual(len(calls), 1)                           # the good result replaced it

    def test_the_coach_gives_tips_after_a_run_that_traded(self):
        _, task = self.create()
        for n in range(3):
            l = MiniMeLabel.objects.create(task=task, input='i', output=FIRST, judged_pass=False, judged_version=0,
                                           judged_failed=[{'id': 's2', 'text': 'Each keyword must be relevant to the healthcare domain.'}])
            mm.review(task, l.pk, 'flip')                                     # disagreements without a comment
        prompts = []
        def coach(payload, prompt):
            prompts.append(prompt)
            return "Here you go:\n- Say why when you disagree: name the word that is off.\n- Rule 2 keeps being overruled.\n- third"
        j = SimpleNamespace(status='done', result={'accepted': False, 'edits': [{'op': 'add'}], 'fixed': ['1'], 'broke': ['2', '3']})
        with patch.object(mm.learning, 'reflect_once', coach):
            mm._coach(j, {'task_id': str(task.id), '_owner': 'ahmet'})
        self.assertIn('3 of 3 disagreements came without a comment', prompts[0])
        self.assertIn('relevant to the healthcare domain', prompts[0])
        self.assertIn('fixed 1 and broke 2', prompts[0])
        task.refresh_from_db()
        tip = next(e for e in mm.summary(task)['log'] if e.get('kind') == 'tip')
        self.assertEqual(len(tip['tips']), 2)                                 # at most 2
        r = self.post(f'/api/minime/tasks/{task.id}/tip', {'at': tip['at']}).json()
        self.assertTrue(next(e for e in r['log'] if e.get('kind') == 'tip')['dismissed'])
        with patch.object(mm.learning, 'reflect_once', coach):                        # little evidence, nothing traded: no tip
            mm._coach(SimpleNamespace(status='done', result={'accepted': True}), {'task_id': str(task.id), '_owner': 'ahmet'})
        self.assertEqual(len(prompts), 1)

    def test_a_transient_judge_failure_is_retried(self):
        import json as _json
        tries = []
        def flaky(texts, receive):
            def call(inputs=None, outputs=None):
                tries.append(1)
                if len(tries) < 3:
                    _json.loads('')                                   # what an empty reply body raises
                receive({'criteria': [{'passed': True}], 'overall_pass': True})
            return call
        got = []
        with patch.object(mm.config, 'JUDGE_RETRY_PAUSE', 0):
            mm._cached(flaky, None, 'm')[0](('r',), got.append)(inputs='i', outputs='o')
        self.assertEqual((len(tries), len(got)), (3, 1))
        def dead(texts, receive):
            return lambda inputs=None, outputs=None: _json.loads('')
        with patch.object(mm.config, 'JUDGE_RETRY_PAUSE', 0), self.assertRaises(ValueError):
            mm._cached(dead, None, 'm')[0](('q',), got.append)(inputs='i', outputs='o')

    def test_an_interrupted_run_is_reported(self):
        _, task = self.create()
        JobState.objects.filter(kind='minime_round').update(status='interrupted', error='Job cannot be safely replayed')
        self.assertIn('stopped', mm.summary(task)['failed']['minime_round']['error'])

    def test_an_interrupted_learn_run_keeps_the_feedback_for_the_next_one(self):
        _, task = self.create()
        for n in range(2):
            l = MiniMeLabel.objects.create(task=task, input='i', output=FIRST, judged_pass=True, judged_version=0)
            mm.review(task, l.pk, 'flip', reason='uses first')
        j = job()
        def boom(*a, **k):
            raise RuntimeError('worker stopped')
        with patch.object(mm.judging, 'llm', stub_llm), patch.object(mm.learning, 'run_gepa', boom):
            mm.run_learn(j, {'task_id': str(task.id), '_owner': 'ahmet'})
        self.assertEqual(j.status, 'error')
        self.assertEqual(mm.feedback_since_learn(MiniMeTask.objects.get(pk=task.id)), 2)

    def old_proposal(self, rewrite):
        """A "Not applied" proposal as earlier versions logged it (runs are no longer gated)."""
        _, task = self.create()
        current = mm.current_qcr(task)
        candidate, edits = mm.from_text(rewrite, current, None)
        mm.log(task, "Learning tried: x. Not applied: y.", kind="learn", accepted=False, edits=edits,
               proposal=candidate.to_dict(), based_on=current.version)
        return task, mm.summary(task)['log'][0]['at']

    def test_user_can_apply_a_rejected_proposal_while_it_still_fits(self):
        rewrite = "1. 3 keywords in total.\n2. Each keyword must not use the word 'walk-in'.\n3. " + RULE
        task, at = self.old_proposal(rewrite)
        JobState.objects.all().delete()
        r = self.post(f'/api/minime/tasks/{task.id}/apply', {'at': at}).json()
        self.assertEqual((r['version'], [s['text'] for s in r['statements']][-1]), (1, RULE))
        applied = next(e for e in r['log'] if e['at'] == at)
        self.assertEqual((applied['applied'], applied['version']), (True, 1))
        self.assertIn('You applied the proposed change', r['log'][0]['text'])
        self.assertEqual(JobState.objects.get().payload.get('rejudge_only'), True)       # waiting answers re-judged
        again = self.post(f'/api/minime/tasks/{task.id}/apply', {'at': at})
        self.assertEqual((again.status_code, again.json()['error']), (400, 'already applied'))

    def test_a_stale_proposal_cannot_be_applied(self):
        rewrite = "1. 3 keywords in total.\n2. Each keyword must not use the word 'walk-in'.\n3. " + RULE
        task, at = self.old_proposal(rewrite)
        mm.edit_qcr(task, [{'id': 's1', 'text': '3 keywords in total.'}])        # the rubric moved on
        r = self.post(f'/api/minime/tasks/{task.id}/apply', {'at': at})
        self.assertEqual(r.status_code, 400)
        self.assertIn('no longer applies', r.json()['error'])

    def test_learn_rejects_an_unusable_rewrite(self):
        task, j, _ = self.learn_with('Sorry, I cannot help.')
        self.assertEqual((j.result['accepted'], j.result['why']), (False, 'the rewrite has no statements'))
        self.assertIn('GEPA search', j.result['timing']['phases'])     # timed even when nothing is applied

    def test_hand_edit_makes_a_version_and_rejudges(self):
        t, task = self.create()
        JobState.objects.all().delete()
        statements = [{'id': 's1', 'text': '3 keywords in total.'}, {'id': 's2', 'text': '', 'status': 'active'},
                      {'text': 'No brand names.'}]
        r = self.post(f'/api/minime/tasks/{task.id}/qcr', {'statements': statements}).json()
        self.assertEqual(([s['text'] for s in r['statements']], r['version']), (['3 keywords in total.', 'No brand names.'], 1))
        row = JobState.objects.get()
        self.assertEqual((row.kind, row.payload['rejudge_only']), ('minime_learn', True))

    def test_settings_upload_delete_and_owner_scope(self):
        t, task = self.create()
        r = self.post(f'/api/minime/tasks/{task.id}', {'settings': {'judge_model': 'm2', 'round_size': '7'}}).json()
        self.assertEqual(r['settings'], {'task_model': None, 'judge_model': 'm2', 'rewriter_model': None, 'round_size': 7})
        self.assertEqual(r['rewriter_default'], 'claude-opus-4.8')
        self.assertEqual(r['models'], {'task_model': mm.TASK_MODEL, 'judge_model': 'm2', 'rewriter_model': 'claude-opus-4.8'})
        self.assertEqual(self.post(f'/api/minime/tasks/{task.id}/labels',
                                   {'csv': CSV, 'input_col': 'service', 'output_col': 'answer'}).json(), {'added': 2})
        self.client.force_login(User.objects.create_user('other', password='x'))
        self.assertEqual(self.client.get(f'/api/minime/tasks/{task.id}/queue').status_code, 400)
        self.client.force_login(self.user)
        self.assertEqual(self.client.delete(f'/api/minime/tasks/{task.id}').json(), {'deleted': str(task.id)})
        self.assertEqual(MiniMeLabel.objects.count(), 0)


    # ---- Copilot review of PR #48 (fix/minime-review) ---------------------------------
    def test_a_task_gets_at_most_one_job_of_each_kind(self):
        """#1: a second start (two quick reviews, two clicks) returns the job already queued."""
        _, task = self.create(start=False)
        tid = {'task_id': str(task.id)}
        a, b = mm.start('minime_round', 'ahmet', tid), mm.start('minime_round', 'ahmet', tid)
        self.assertEqual(a.id, b.id)
        learn, score = mm.start('minime_learn', 'ahmet', tid), mm.start('minime_learn', 'ahmet', {**tid, 'score_only': True})
        self.assertNotEqual(learn.id, score.id)                           # a scoring job is its own lane
        self.assertEqual(mm.start('minime_learn', 'ahmet', tid).id, learn.id)
        test = mm.start('minime_learn', 'ahmet', {**tid, 'score_only': True, 'test': {'version': 1}})
        self.assertNotEqual(test.id, score.id)                            # testing a new version always starts
        JobState.objects.filter(pk=a.id).update(cancel_requested_at=timezone.now())
        self.assertNotEqual(mm.start('minime_round', 'ahmet', tid).id, a.id)   # one being stopped does not count
        self.assertEqual(JobState.objects.filter(kind='minime_round').count(), 2)

    def test_a_stored_fail_whose_failed_rules_are_dropped_is_judged_again(self):
        """#2: only failed rules are stored, so another rule's result is unknown: never promoted to PASS."""
        from ps_utils.minime.qcr import QCR, Statement
        current = QCR(task={'objective': 'x'}, statements=[Statement(id='s1', text='3 keywords in total.'),
                                                           Statement(id='s2', text=RULE)])
        candidate = QCR(task={'objective': 'x'}, statements=[Statement(id='s2', text=RULE)])
        row = {'id': 'r1', 'input': 'i', 'output': FIRST}
        before = {'r1': (False, [{'id': 's1', 'text': '3 keywords in total.'}])}   # s2's result was not stored
        verdict, failed = mm.judging._reuse_verdicts(current, candidate, [row], before, judge_fn, None)['r1']
        self.assertEqual((verdict, [f['text'] for f in failed]), (False, [RULE]))   # judged: FIRST breaks s2
        kept = {'r1': (False, [{'id': 's2', 'text': RULE}])}                        # a kept failed rule: reused
        self.assertFalse(mm.judging._reuse_verdicts(current, candidate, [row], kept,
                                                    lambda *a: self.fail('not judged again'), None)['r1'][0])

    def test_a_judge_change_drops_the_checkpoint_and_the_saved_rewrite(self):
        """#3: both hold the old judge's scores."""
        import tempfile, os
        _, task = self.create(start=False)
        run_dir = tempfile.mkdtemp()
        mm._update_settings(task, lambda s: {**s, 'gepa_run': {'based_on': 0, 'run_dir': run_dir},
                                             'pending_rewrite': {'text': 'x', 'based_on': 0}})
        mm._judge_epoch(task)
        task.refresh_from_db()
        self.assertFalse({'gepa_run', 'pending_rewrite'} & set(task.settings))
        self.assertFalse(os.path.exists(run_dir))
        since = task.settings['judge_since']
        self.assertTrue(mm.learning._same_epoch(task, {'judge_since': since}))
        self.assertTrue(mm.learning._same_epoch(task, {}))                 # recorded before the epoch key existed
        self.assertFalse(mm.learning._same_epoch(task, {'judge_since': '2020-01-01T00:00:00+00:00'}))

    def test_a_rewrite_is_not_published_over_a_newer_version(self):
        """#4: the base check and the creation are one transaction."""
        _, task = self.create(start=False)
        base = mm.current_qcr(task)
        mm._new_version(task, mm.current_qcr(task), 'edit')                # your edit lands first
        self.assertIsNone(mm._new_version(task, base, 'learn', base=base.version))
        self.assertIsNotNone(mm._new_version(task, mm.current_qcr(task), 'learn', base=base.version + 1))

    def blind_check(self, task, **kw):
        return MiniMeLabel.objects.create(task=task, split='heldout', input='i', output=FIRST, passed=False,
                                          labelled_at=timezone.now(), **kw)

    def test_unjudged_blind_checks_are_scored(self):
        """#12: an uploaded or unjudged blind check still gets every version's verdict."""
        _, task = self.create(start=False)
        check = self.blind_check(task, source='upload')
        self.assertEqual(mm.score_blind(task, judge_fn, None), 1)
        self.assertIn(str(check.pk), (task.versions.get(version=0).report or {}).get('blind', {}))

    def test_scores_from_a_judge_replaced_mid_scoring_are_not_written(self):
        """#5: the judge changed while scoring ran: its epoch reset wins."""
        _, task = self.create(start=False)
        self.blind_check(task, judged_pass=False, judged_version=99)       # judged, but not with v0's rules: v0 scores it
        real = mm.judging._current_verdicts
        def changed_meanwhile(t, *a, **k):
            out = real(t, *a, **k)
            mm._judge_epoch(MiniMeTask.objects.get(pk=t.pk))
            return out
        with patch.object(mm.judging, '_current_verdicts', changed_meanwhile):
            self.assertEqual(mm.score_blind(task, judge_fn, None), 0)
        self.assertNotIn('blind', task.versions.get(version=0).report or {})

    def test_verdicts_from_before_a_judge_change_are_judged_again(self):
        """#6: a waiting answer's verdict, and a reviewed one's, from the previous judge are stale."""
        _, task = self.create(start=False)
        old = timezone.now()
        waiting = MiniMeLabel.objects.create(task=task, input='i', output=FIRST, judged_pass=True, judged_version=0,
                                             judged_at=old)
        mm._judge_epoch(task)
        task.refresh_from_db()
        self.assertEqual(mm.rejudge_waiting(task, mm.current_qcr(task), judge_fn, None), 1)
        waiting.refresh_from_db()
        self.assertGreaterEqual(waiting.judged_at, mm.judge_since(task))
        self.assertEqual(mm.rejudge_waiting(task, mm.current_qcr(task), judge_fn, None), 0)   # now current
        row = {'id': 'x', 'input': 'i', 'output': FIRST, 'judged_pass': True, 'judged_failed': [], 'judged_version': 0,
               'judged_at': old}
        with patch.object(mm.judging, '_verdicts', lambda q, rows, *a, **k: {r['id']: ('judged', []) for r in rows}):
            self.assertEqual(mm.judging._current_verdicts(task, mm.current_qcr(task), [row], judge_fn, None)['x'][0],
                             'judged')

    def test_a_restart_pause_cancels_the_tasks_queued_jobs(self):
        """#10: paused means no paid calls: the queued round ends too."""
        _, task = self.create()                                           # Start optimizing queues a round
        self.assertTrue(JobState.objects.filter(kind='minime_round', status='queued').exists())
        mm.pause_task(task.id)
        task.refresh_from_db()
        self.assertFalse(task.settings.get('optimizing'))
        self.assertFalse(JobState.objects.filter(kind='minime_round', status='queued', cancel_requested_at=None).exists())

    def test_the_search_result_is_saved_before_its_checkpoint_is_dropped(self):
        """#11: cut off between the two, the result or the resumable search survives."""
        seen = []
        real = mm.learning._end_gepa_run
        def spy(task):
            seen.append(bool((MiniMeTask.objects.get(pk=task.pk).settings or {}).get('pending_rewrite')))
            real(task)
        with patch.object(mm.learning, '_end_gepa_run', spy):
            self.learn_with('x')                 # any GEPA result; it is saved, then checked
        self.assertEqual(seen[:1], [True])

    # ---- Copilot review of PR #50 ------------------------------------------------------
    def test_verdicts_judged_across_a_judge_change_stay_stale(self):
        """judged_at is when judging started: a judge change mid-batch leaves the batch stale."""
        _, task = self.create(start=False)
        MiniMeLabel.objects.create(task=task, input='i', output=FIRST)
        real = mm.judging._verdicts
        def judge_then_change(*a, **k):
            out = real(*a, **k)
            mm._judge_epoch(MiniMeTask.objects.get(pk=task.pk))
            return out
        with patch.object(mm.judging, '_verdicts', judge_then_change):
            mm.rejudge_waiting(task, mm.current_qcr(task), judge_fn, None)
        task.refresh_from_db()
        self.assertEqual(len(mm.judging.stale_waiting(task, mm.current_qcr(task).version)), 1)

    def test_a_learning_run_spanning_a_judge_change_publishes_nothing(self):
        """Its rewrite was chosen by the old judge: set aside, and its feedback stays unconsumed."""
        _, task = self.create()
        for n in range(5):
            l = MiniMeLabel.objects.create(task=task, input='i', output=FIRST, judged_pass=True, judged_version=0)
            mm.review(task, l.pk, 'flip', reason='uses first')
        def gepa_then_judge_change(*a, **k):
            mm._judge_epoch(MiniMeTask.objects.get(pk=task.pk))
            return f'1. 3 keywords in total.\n2. {RULE}'
        j = job()
        with patch.object(mm.judging, 'llm', stub_llm), patch.object(mm.learning, 'run_gepa', gepa_then_judge_change):
            mm.run_learn(j, {'task_id': str(task.id), '_owner': 'ahmet'})
        task.refresh_from_db()
        self.assertEqual((j.result or {}).get('why'), 'the judge changed meanwhile', j.result)
        self.assertEqual(task.versions.count(), 1)
        self.assertNotIn('learned_through', task.settings)
        self.assertEqual(mm.feedback_since_learn(task), 5)

    def test_a_rejudge_job_catches_up_with_an_edit_made_while_it_ran(self):
        """A second edit merges into the running rejudge job (one per task): it judges again before finishing."""
        _, task = self.create(start=False)
        waiting = MiniMeLabel.objects.create(task=task, input='i', output=FIRST, judged_pass=True, judged_version=0)
        mm._new_version(task, mm.current_qcr(task), 'edit')                # v1: the first edit
        real, calls = mm.judging.rejudge_waiting, []
        def rejudge_then_edit(t, q, *a, **k):
            n = real(t, q, *a, **k)
            if not calls:
                mm._new_version(t, mm.current_qcr(t), 'edit')               # v2 arrives while v1 is judged
            calls.append(q.version)
            return n
        j = job()
        with patch.object(mm.judging, 'llm', stub_llm), patch.object(mm.judging, 'rejudge_waiting', rejudge_then_edit):
            mm.run_learn(j, {'task_id': str(task.id), '_owner': 'ahmet', 'rejudge_only': True})
        waiting.refresh_from_db()
        self.assertEqual((calls, waiting.judged_version), ([1, 2], 2))

    def test_a_judge_change_rejudges_the_waiting_answers(self):
        """While optimizing it starts a rejudge; Start counts the previous judge's verdicts as stale."""
        _, task = self.create()
        MiniMeLabel.objects.create(task=task, input='i', output=FIRST, judged_pass=True, judged_version=0,
                                   judged_at=timezone.now())
        JobState.objects.all().delete()
        mm.update_task(task, settings={'judge_model': 'another-judge'})
        self.assertTrue(any((j.payload or {}).get('rejudge_only') for j in JobState.objects.filter(kind='minime_learn')))
        JobState.objects.all().delete()
        mm.stop_jobs(task, quiet=True)
        mm.start_optimizing(MiniMeTask.objects.get(pk=task.pk), 'ahmet')
        self.assertTrue(any((j.payload or {}).get('rejudge_only') for j in JobState.objects.filter(kind='minime_learn')))

    def test_comments_on_uploaded_labels_are_notes(self):
        """They reach direct rewriting and grading like a comment on a kept verdict."""
        _, task = self.create(start=False)
        mm.upload_labels(task, CSV, input_col='service', output_col='answer')
        notes = {r['output']: r['note'] for r in mm.confirmed(task)}
        self.assertEqual(notes, {GOOD: True, FIRST: True})

    # ---- Copilot's third review of PR #50 (fix/minime-judge-epoch) ----------------------
    def test_a_rejudge_stops_when_its_judge_is_replaced_and_its_verdicts_stay_stale(self):
        """Verdicts carry when the judge was configured; a later batch after a judge change is not judged."""
        from datetime import timedelta
        _, task = self.create(start=False)
        for _ in range(2):
            MiniMeLabel.objects.create(task=task, input='i', output=FIRST)
        configured = timezone.now() - timedelta(seconds=5)
        real, batches = mm.judging._verdicts, []
        def judge_then_change(*a, **k):
            out = real(*a, **k)
            batches.append(1)
            mm._judge_epoch(MiniMeTask.objects.get(pk=task.pk))       # after the first batch
            return out
        with patch.object(mm.judging, '_verdicts', judge_then_change), patch.object(mm.judging._config, 'PARALLEL', 1):
            n = mm.rejudge_waiting(task, mm.current_qcr(task), judge_fn, None, configured=configured)
        task.refresh_from_db()
        self.assertEqual((n, len(batches)), (1, 1))                        # the second batch never ran
        self.assertEqual(len(mm.judging.stale_waiting(task, mm.current_qcr(task).version)), 2)   # both still stale

    def test_round_verdicts_carry_when_the_judge_was_configured(self):
        _, task = self.create(start=False)
        at = '2026-01-01T00:00:00+00:00'
        real = mm.judging._payload
        with patch.object(mm.judging, '_payload', lambda t, p: {**real(t, p), '_configured_at': at}):
            self.round(task)
        self.assertEqual({l.judged_at.isoformat() for l in task.labels.all()}, {at})

    def test_rejected_rewrite_diagnostics_do_not_echo_model_text(self):
        import io
        from contextlib import redirect_stderr, redirect_stdout
        _, task = self.create()
        for n in range(5):
            label = MiniMeLabel.objects.create(task=task, input='i', output=FIRST,
                                              judged_pass=True, judged_version=0)
            mm.review(task, label.pk, 'flip', reason='uses first')
        private_reply = 'Private synthetic customer launch plan: Customer-DEMO-456'
        stderr, stdout, j = io.StringIO(), io.StringIO(), job()
        with patch.object(mm.judging, 'llm', stub_llm), \
                patch.object(mm.learning, 'run_gepa', return_value=private_reply), \
                redirect_stderr(stderr), redirect_stdout(stdout):
            mm.run_learn(j, {'task_id': str(task.id), '_owner': 'ahmet'})
        self.assertEqual(j.status, 'done')
        self.assertFalse(j.result['accepted'])
        self.assertIn('rewrite unusable', stderr.getvalue())
        self.assertNotIn(private_reply, stderr.getvalue() + stdout.getvalue())
        self.assertNotIn(private_reply, json.dumps(j.result))
        self.assertNotIn('reply_head', j.result)

    def test_a_judge_change_just_before_the_watermark_keeps_the_feedback(self):
        """learned_through is advanced only if the judge epoch is unchanged, checked under the settings lock."""
        _, task = self.create()
        for n in range(5):
            l = MiniMeLabel.objects.create(task=task, input='i', output=FIRST, judged_pass=True, judged_version=0)
            mm.review(task, l.pk, 'flip', reason='uses first')
        real = mm.store._update_settings
        def change_judge_first(t, change):
            if 'learned_through' in change(dict(MiniMeTask.objects.get(pk=t.pk).settings or {})):   # the watermark write: the judge changes first
                mm._judge_epoch(MiniMeTask.objects.get(pk=t.pk))
            return real(t, change)
        j = job()
        with patch.object(mm.judging, 'llm', stub_llm), patch.object(mm.learning, 'run_gepa', lambda *a, **k: 'Sorry'), \
                patch.object(mm.store, '_update_settings', change_judge_first):
            mm.run_learn(j, {'task_id': str(task.id), '_owner': 'ahmet'})
        task.refresh_from_db()
        self.assertNotIn('learned_through', task.settings)
        self.assertEqual(mm.feedback_since_learn(task), 5)

    def test_a_rejudge_request_queues_behind_a_running_rejudge(self):
        """A running rejudge may be past its last check: a new request is not absorbed by it."""
        _, task = self.create(start=False)
        tid = {'task_id': str(task.id), 'rejudge_only': True}
        running = mm.start('minime_learn', 'ahmet', tid)
        JobState.objects.filter(pk=running.id).update(status='running')
        queued = mm.start('minime_learn', 'ahmet', tid)
        self.assertNotEqual(queued.id, running.id)
        self.assertEqual(mm.start('minime_learn', 'ahmet', tid).id, queued.id)   # but one queued is enough
        main = mm.start('minime_round', 'ahmet', {'task_id': str(task.id)})
        JobState.objects.filter(pk=main.id).update(status='running')
        self.assertEqual(mm.start('minime_round', 'ahmet', {'task_id': str(task.id)}).id, main.id)   # rounds: never two

    # ---- Copilot's review of PR #51 (fix/minime-claim-serial) --------------------------
    def test_two_workers_never_run_rejudges_of_one_task_at_once(self):
        """A rejudge queued behind a running one waits for it; other tasks and kinds are not held up."""
        from .durable import claim, enqueue
        _, task = self.create(start=False)
        _, other = self.create(start=False)
        rejudge = lambda t: enqueue('minime_learn', {'task_id': str(t.id), 'rejudge_only': True, '_owner': 'ahmet'})
        first, second, elsewhere = rejudge(task), rejudge(task), rejudge(other)
        a = claim(lane='any')                                              # worker A
        b = claim(lane='any')                                              # worker B, at the same time
        self.assertEqual((a.id, b.id), (first.id, elsewhere.id))           # B skips the second: same task, same kind
        self.assertIsNone(claim(lane='any'))
        JobState.objects.filter(pk=a.id).update(status='done')
        self.assertEqual(claim(lane='any').id, second.id)                  # once the first ends, it runs

    def test_jobs_blocked_behind_a_running_one_do_not_starve_other_work(self):
        """210 scorers queued behind a running one (tests bypass coalescing): more than either lane's window
        (20 any, 200 short) and three 100-row pages; work queued after them still runs on both lanes."""
        from datetime import timedelta
        from .durable import claim, enqueue
        _, task = self.create(start=False)
        _, other = self.create(start=False)
        score = {'task_id': str(task.id), 'score_only': True, '_owner': 'ahmet'}
        running = enqueue('minime_learn', score)
        JobState.objects.filter(pk=running.id).update(status='running', lease_until=timezone.now() + timedelta(minutes=2))
        for n in range(210):
            enqueue('minime_learn', {**score, 'test': {'version': n}})
        later_round = enqueue('minime_round', {'task_id': str(other.id), '_owner': 'ahmet'})
        later_any = enqueue('optimize', {'_owner': 'ahmet', 'mode': 'zero_shot'})
        self.assertEqual(claim(lane='short').id, later_round.id)
        self.assertEqual(claim(lane='any').id, later_any.id)
        self.assertEqual(JobState.objects.filter(status='queued').count(), 210)   # the blocked ones still wait

    def test_verdicts_are_stamped_with_the_epoch_of_the_settings_they_were_read_from(self):
        """A judge change after the task was fetched, before the payload is built, still makes them stale."""
        _, task = self.create(start=False)
        mm._judge_epoch(task)
        snapshot = MiniMeTask.objects.get(pk=task.pk)                      # the job fetched the task
        mm._judge_epoch(MiniMeTask.objects.get(pk=task.pk))               # the judge changes meanwhile
        stamped = mm.judging.configured_at(mm.judging._payload(snapshot, {}))
        self.assertLess(stamped, mm.judge_since(MiniMeTask.objects.get(pk=task.pk)))   # so its verdicts are stale

    def test_a_new_default_judge_starts_a_new_epoch_for_tasks_that_use_the_default(self):
        """Tasks on MiniMe's default judge switch judge when the default changes: handled like a
        Settings change. A task that chose its judge is not affected."""
        _, default = self.create(start=False)
        _, chosen = self.create(start=False)
        mm._update_settings(chosen, lambda s: {**s, 'judge_model': 'deepseek-v4.1-flash'})
        for t in (default, chosen):                                        # epochs from before the key existed
            mm._update_settings(t, lambda s: {**{k: v for k, v in s.items() if k != 'judge_epoch_model'},
                                              'judge_since': '2026-10-01T00:00:00+00:00'})
        mm._ensure_judge_epoch({'task_id': str(default.id)})
        mm._ensure_judge_epoch({'task_id': str(chosen.id)})
        default.refresh_from_db(); chosen.refresh_from_db()
        self.assertNotEqual(default.settings['judge_since'], '2026-10-01T00:00:00+00:00')         # new epoch
        self.assertEqual(default.settings['judge_epoch_model'], mm.JUDGE_MODEL)
        self.assertTrue(any('MiniMe\'s default' in e.get('text', '') for e in default.settings.get('log', [])))
        self.assertEqual(chosen.settings['judge_since'], '2026-10-01T00:00:00+00:00')             # unchanged
        self.assertEqual(chosen.settings['judge_epoch_model'], 'deepseek-v4.1-flash')
        before = default.settings['judge_since']
        mm._ensure_judge_epoch({'task_id': str(default.id)})                                     # once only
        default.refresh_from_db()
        self.assertEqual(default.settings['judge_since'], before)

    def test_uploaded_labels_count_as_feedback(self):
        """#13: a labelled sheet carries no keep/flip action, yet its FAILs and reasons are feedback."""
        _, task = self.create(start=False)
        self.assertEqual(mm.upload_labels(task, CSV, input_col='service', output_col='answer'), 2)
        self.assertEqual(mm.feedback_since_learn(task), 2)                # a FAIL, and a PASS with a reason


@skipUnless(HAS_GEPA, 'needs mlflow and gepa')
def parse_lines(text):
    from ps_utils.minime.rubric_text import parse
    return parse(text)


class DescribeTests(TestCase):
    """Changing the task description: a minor rewording keeps the task, a substantial change clones it."""
    setUp, post, create, round = MiniMeV2Tests.setUp, MiniMeV2Tests.post, MiniMeV2Tests.create, MiniMeV2Tests.round

    def test_reword_keeps_reviews_and_logs_the_change(self):
        t, task = self.create()
        self.round(task)
        label = task.labels.first()
        mm.review(task, label.id, 'keep', reason='fine')
        r = self.post(f'/api/minime/tasks/{task.id}/describe',
                      {'mode': 'reword', 'definition': {'objective': 'Suggest search keywords for a health service.',
                                                        'requirements': 'ignored'}})
        self.assertEqual(r.status_code, 200, r.content)
        body = r.json()
        self.assertEqual(body['version'], t['version'] + 1)
        self.assertEqual(body['definition']['objective'], 'Suggest search keywords for a health service.')
        self.assertEqual(body['definition']['requirements'], DEFINITION['requirements'])  # the statements own them
        self.assertEqual([s['text'] for s in body['statements']], [s['text'] for s in t['statements']])
        self.assertIn('Objective: Suggest search keywords for a health service.', body['judge_prompt'])
        self.assertEqual(body['reviewed'], 1)
        entry = body['log'][0]
        self.assertEqual((entry['kind'], entry['changes'][0]['field']), ('describe', 'objective'))
        self.assertEqual(entry['changes'][0]['before'], DEFINITION['objective'])
        again = self.post(f'/api/minime/tasks/{task.id}/describe', {'mode': 'reword', 'definition': {
            'objective': 'Suggest search keywords for a health service.'}}).json()
        self.assertEqual(again['version'], body['version'])          # no change, no new version
        self.assertEqual(self.post(f'/api/minime/tasks/{task.id}/describe', {'mode': 'reword', 'definition': {
            'objective': ' '}}).status_code, 400)
        self.assertEqual(self.post(f'/api/minime/tasks/{task.id}/describe', {'mode': 'x'}).status_code, 400)

    def test_clone_carries_rules_and_settings_not_reviews(self):
        t, task = self.create(deployed_prompt='Keywords for {{input}}', settings={'judge_model': 'j-1'})
        self.round(task)
        mm.review(task, task.labels.first().id, 'keep', reason='fine')
        mm.edit_qcr(task, [{'id': 's1', 'text': '3 keywords in total.'}, {'text': RULE}])
        r = self.post(f'/api/minime/tasks/{task.id}/describe',
                      {'mode': 'clone', 'definition': {'objective': 'Write ad headlines for a clinic.'}, 'name': ''})
        self.assertEqual(r.status_code, 200, r.content)
        new = r.json()
        self.assertNotEqual(new['id'], str(task.id))
        self.assertEqual(new['name'], 'kw (new)')
        self.assertEqual([s['text'] for s in new['statements']], ['3 keywords in total.', RULE])
        self.assertEqual((new['version'], new['reviewed']), (0, 0))
        self.assertEqual((new['deployed_prompt'], new['inputs'], new['settings']['judge_model']),
                         ('Keywords for {{input}}', ['urgent care'], 'j-1'))
        self.assertIn('Cloned from', new['log'][0]['text'])
        old = mm.summary(MiniMeTask.objects.get(pk=task.id))
        self.assertEqual((old['definition']['objective'], old['reviewed']), (DEFINITION['objective'], 1))
        self.assertIn('Cloned into', old['log'][0]['text'])

    def test_keep_and_flip_refer_to_the_verdict_shown(self):
        t, task = self.create()
        self.round(task)
        label = task.labels.filter(passed=None, split='learn').exclude(judged_pass=None).first()
        shown = label.judged_pass
        MiniMeLabel.objects.filter(pk=label.pk).update(judged_pass=not shown)    # re-judged meanwhile
        r = self.post(f'/api/minime/tasks/{task.id}/labels/{label.id}', {'action': 'keep', 'shown_pass': shown})
        self.assertEqual(r.status_code, 200, r.content)
        self.assertEqual(MiniMeLabel.objects.get(pk=label.pk).passed, shown)

    def test_prompt_placeholder_any_name_but_only_one(self):
        from ps_utils.minime.candidates import deployed
        self.assertEqual(deployed('Tweet on {{theme}}. Again: {{ theme }}.', 'GPUs'), 'Tweet on GPUs. Again: GPUs.')
        self.assertEqual(deployed('Tweet on {{input}}', 'GPUs'), 'Tweet on GPUs')
        self.assertEqual(deployed('Write a tweet.', 'GPUs'), 'Write a tweet.\n\nInput:\nGPUs')
        t, task = self.create(deployed_prompt='Tweet on {{theme}}')
        r = self.post(f'/api/minime/tasks/{task.id}', {'deployed_prompt': 'Tweet on {{theme}} for {{audience}}'})
        self.assertEqual(r.status_code, 400)
        self.assertIn('{{theme}}, {{audience}}', r.json()['error'])
        self.assertEqual(MiniMeTask.objects.get(pk=task.id).deployed_prompt, 'Tweet on {{theme}}')
        self.assertEqual(self.post('/api/minime/tasks', {'definition': DEFINITION, 'inputs': ['x'],
                                                         'deployed_prompt': '{{a}} {{b}}'}).status_code, 400)

    def test_stop_ends_queued_work_and_asks_a_running_job_to_stop(self):
        from datetime import timedelta
        from django.utils import timezone
        from optimizer import durable
        t, task = self.create()
        JobState.objects.all().delete()
        queued = mm.start('minime_round', 'ahmet', {'task_id': str(task.id)})
        running = mm.start('minime_learn', 'ahmet', {'task_id': str(task.id)})
        JobState.objects.filter(pk=running.id).update(status='running', lease_token='t', stage='GEPA 40%',
                                                       lease_until=timezone.now() + timedelta(seconds=60))
        other = mm.start('minime_round', 'ahmet', {'task_id': 'another-task'})
        r = self.post(f'/api/minime/tasks/{task.id}/stop', {}).json()
        self.assertEqual(JobState.objects.get(pk=queued.id).status, 'cancelled')
        self.assertEqual(r['jobs']['minime_learn']['stage'], 'Stopping')       # until it notices
        self.assertTrue(r['jobs']['minime_learn']['stopping'])
        self.assertIsNone(JobState.objects.get(pk=other.id).cancel_requested_at)   # other tasks untouched
        self.assertTrue(mm._cancel_requested(SimpleNamespace(id=running.id)))
        JobState.objects.filter(pk=running.id).update(lease_until=timezone.now() - timedelta(seconds=1))
        durable.claim()                                                         # its worker is gone: not replayed
        self.assertEqual(JobState.objects.get(pk=running.id).status, 'cancelled')
        self.assertNotIn('minime_learn', mm.failed_jobs(task))                  # a stop is not a failure

    def test_a_stopped_learning_run_keeps_its_gepa_checkpoint_and_ends_cancelled(self):
        from optimizer.durable import JobCancelled
        _, task = self.create()
        for n in range(25):
            l = MiniMeLabel.objects.create(task=task, input='i', output=FIRST if n % 2 else GOOD, judged_pass=True, judged_version=0)
            mm.review(task, l.pk, 'flip' if n % 2 else 'keep', reason='uses first' if n % 2 else '')
        j = SimpleNamespace(id=None, stage='', status='running', result=None, error='')
        with patch.object(mm.judging, 'llm', stub_llm), patch.object(mm.learning, 'run_gepa', lambda *a, **k: "1. x"), \
                patch.object(mm.jobs, '_cancel_requested', lambda job: True), self.assertRaises(JobCancelled):
            mm._learn(j, {'task_id': str(task.id), '_owner': 'ahmet'})
        task.refresh_from_db()
        self.assertIn('gepa_run', task.settings)                                # resumable later
        self.assertNotIn('pending_rewrite', task.settings)
        self.assertEqual(j.error, '')

    def test_versions_behind_on_blind_checks_are_scored_by_a_job_the_page_starts(self):
        _, task = self.create()
        for out, mine in [(FIRST, 'Fail'), (GOOD, 'Pass')] * 2:
            l = MiniMeLabel.objects.create(task=task, split='heldout', input='i', output=out, judged_pass=True, judged_version=0)
            mm.review(task, l.pk, 'label', mine)
        mm.edit_qcr(task, [{'id': s.id, 'text': s.text} for s in mm.current_qcr(task).active()] + [{'text': RULE}])
        JobState.objects.all().delete()
        self.client.get(f'/api/minime/tasks/{task.id}/queue')
        jobs = list(JobState.objects.all())
        self.assertEqual([(j.kind, j.payload.get('score_only')) for j in jobs], [('minime_learn', True)])
        self.assertEqual(mm.active_jobs(task)['minime_learn']['rejudge'], True)    # a job line, not the learning banner
        self.client.get(f'/api/minime/tasks/{task.id}/queue')
        self.assertEqual(JobState.objects.count(), 1)                              # one at a time
        j = job()
        with patch.object(mm.judging, 'llm', stub_llm):
            mm.run_learn(j, jobs[0].payload | {'_owner': 'ahmet'})
        self.assertEqual((j.status, mm.quality_series(task)['to_score']), ('done', 0))
        mm.edit_qcr(task, [{'id': s.id, 'text': s.text} for s in mm.current_qcr(task).active()][:1])   # a new version
        JobState.objects.all().delete()
        with patch.object(mm.judging, 'llm', stub_llm):
            mm.run_learn(job(), {'task_id': str(task.id), '_owner': 'ahmet', 'rejudge_only': True})
        self.assertEqual([j.payload.get('score_only') for j in JobState.objects.all()], [True])   # scoring: a job of its own
        JobState.objects.all().delete()
        self.client.get(f'/api/minime/tasks/{task.id}/queue')
        self.assertEqual(JobState.objects.count(), 0)                              # caught up: nothing to start

    def test_the_judge_works_without_mlflow(self):
        import sys
        from llm_endpoint import Endpoint
        endpoint = Endpoint(base_url='http://stub.invalid/v1', api_key='x', task_model='m', judge_model='j', reflection_model='m')
        reply = SimpleNamespace(usage=None, choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace(
            content='{"criteria": [{"explanation": "short", "evidence": null, "passed": true}]}'))])
        blocked = {k: None for k in list(sys.modules) if k == 'mlflow' or k.startswith('mlflow.')} | {'mlflow': None}
        got = []
        with patch.dict(sys.modules, blocked), patch('optimizer.services._endpoint_for', lambda p: endpoint), \
                patch.object(endpoint.client.chat.completions, 'create', return_value=reply), \
                patch.object(mm.judging, '_judge_client', lambda ep: ep.client), \
                patch.object(mm.judging, '_judge_cache_path', lambda: os.path.join(tempfile.mkdtemp(), 'jc.sqlite')):
            _, make_judge, _, model = mm.llm({})
            make_judge(('Short',), got.append)(inputs='i', outputs='answer')
        self.assertEqual((model, got[0]['criteria'][0]['passed']), ('j', True))

    def test_the_judge_client_is_bounded(self):
        """A hung judge call ends after JUDGE_TIMEOUT, and the SDK does not retry it (MiniMe does)."""
        from llm_endpoint import Endpoint
        client = mm.judging._judge_client(Endpoint(base_url='http://stub.invalid/v1', api_key='x'))
        self.assertEqual((client.timeout, client.max_retries), (mm.JUDGE_TIMEOUT, 0))

    def test_the_judge_runs_at_its_default_and_a_judge_change_rescores_the_versions(self):
        _, task = self.create()
        captured = []
        from llm_endpoint import Endpoint
        endpoint = Endpoint(base_url='http://stub.invalid/v1', api_key='x', task_model='m', judge_model='j', reflection_model='m')
        ok = SimpleNamespace(usage=None, choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace(
            content='{"criteria": [{"explanation": "short", "evidence": null, "passed": true}]}'))])
        def create(**k):
            captured.append('reasoning_effort' in k)
            return ok
        with patch('optimizer.services._endpoint_for', lambda p: endpoint), \
                patch.object(endpoint.client.chat.completions, 'create', side_effect=create), \
                patch.object(mm.judging, '_judge_client', lambda ep: ep.client), \
                patch.object(mm.judging, '_judge_cache_path', lambda: os.path.join(tempfile.mkdtemp(), 'jc.sqlite')):
            _, make_judge, _, _ = mm.llm(mm._payload(task, {}))
            make_judge(('Short',), lambda r: None)(inputs='i', outputs='answer')
        self.assertEqual(captured, [False])                                   # no reasoning setting: the model's default
        v = task.versions.get(version=0)
        v.report = {'blind': {'1': True}}; v.save()
        r = self.post(f'/api/minime/tasks/{task.id}', {'settings': {'judge_model': 'other-judge'}}).json()
        self.assertNotIn('blind', task.versions.get(version=0).report)       # scored again with the new judge
        self.assertIn('Judge changed to other-judge', r['log'][0]['text'])

    def test_a_stale_verdict_is_marked_and_rejudged_first(self):
        _, task = self.create()
        JobState.objects.all().delete()
        first = MiniMeLabel.objects.create(task=task, input='i', output=FIRST, judged_pass=True, judged_version=0)
        MiniMeLabel.objects.create(task=task, input='i', output=GOOD, judged_pass=True, judged_version=0)
        mm.edit_qcr(task, [{'id': s.id, 'text': s.text} for s in mm.current_qcr(task).active()] + [{'text': RULE}])
        nxt = mm.next_candidate(task)
        self.assertEqual((nxt['id'], nxt['judged_version']), (first.pk, 0))   # older than v1: the page says so
        self.assertEqual(mm.rejudge_waiting(task, mm.current_qcr(task), judge_fn, None), 2)
        self.assertEqual(mm.next_candidate(task)['judged_version'], 1)
        self.assertEqual(mm.rejudge_waiting(task, mm.current_qcr(task), judge_fn, None), 0)   # nothing stale left

    def test_the_clock_counts_overlapping_judge_calls_once(self):
        c = mm._Clock()
        for a, b in [(0, 10), (5, 15), (20, 30), (40, 41)]:       # 4 calls: 0-15 overlap, then 20-30, 40-41
            c.call(a, b)
        w = c.window(0, 35)
        self.assertEqual((w['calls'], w['busy'], w['total'], w['longest']), (3, 25, 30, 10))
        self.assertEqual(c.window(36, 50)['busy'], 1)
        self.assertIn('judge 25 s busy (3 calls, avg 10 s, longest 10 s)', mm._judge_line(w))

    def test_rename(self):
        t, task = self.create()
        self.assertEqual(self.post(f'/api/minime/tasks/{task.id}', {'name': ' SEM v2 '}).json()['name'], 'SEM v2')


class ReadyTests(TestCase):
    """The four "good enough" checks, from what the user did."""
    setUp, post, create = MiniMeV2Tests.setUp, MiniMeV2Tests.post, MiniMeV2Tests.create

    def add(self, task, n, **kw):
        from django.utils import timezone
        from datetime import timedelta
        base = timezone.now() - timedelta(days=1) + timedelta(minutes=len(self.made))   # reviewed before now
        for i in range(n):
            self.made.append(MiniMeLabel.objects.create(task=task, labelled_at=base + timedelta(seconds=i), **{'judged_version': 0, **kw}))

    def test_reaching_ready_pauses_the_task_once(self):
        self.made = []
        t, task = self.create()                                     # optimizing; a round is queued
        both = [{'id': 's1'}, {'id': 's2'}]
        self.add(task, 5, split='heldout', kind='representative', passed=True, judged_pass=True)
        self.add(task, 5, split='heldout', kind='representative', passed=False, judged_pass=False, judged_failed=both)
        self.add(task, 10, split='heldout', kind='violate', passed=False, judged_pass=False, judged_failed=both)
        self.add(task, 20, split='learn', kind='comply', action='keep', passed=True, judged_pass=True)
        self.assertTrue(mm.readiness(task)['ready'])
        self.assertTrue(mm.pause_if_ready(task))
        task.refresh_from_db()
        self.assertEqual((mm.optimizing(task), JobState.objects.get(kind='minime_round').status), (False, 'cancelled'))
        self.assertIn('Ready to use: MiniMe paused', mm.summary(task)['log'][0]['text'])
        self.post(f'/api/minime/tasks/{task.id}/start', {})            # the user keeps optimizing while ready
        self.assertFalse(mm.pause_if_ready(task))                       # not paused again
        task.refresh_from_db()
        self.assertTrue(mm.optimizing(task))

    def test_ready_says_what_is_left(self):
        self.made = []
        t, task = self.create()
        self.add(task, 9, split='heldout', kind='representative', passed=True, judged_pass=True)
        self.add(task, 9, split='heldout', kind='violate', passed=False, judged_pass=False, judged_failed=[{'id': 's1'}])
        r = mm.readiness(task)
        self.assertEqual([(t['status'], t['short']) for t in r['targets'].values()],
                         [('open', '100% · 9/10'), ('open', '100% · 9/10')])
        self.assertEqual(r['todo'][0], '1 more blind check of each kind')

    def test_checks_turn_green_one_by_one(self):
        self.made = []
        t, task = self.create()
        r = mm.readiness(task)
        self.assertEqual((r['ready'], r['done'], r['of']), (False, 0, 5))
        self.assertEqual({c['key']: c['status'] for c in r['checks']}, {'both': 'open', 'stable': 'open', 'rules': 'open'})
        both = lambda failed: [{'id': 's1'}, {'id': 's2'}] if failed else []
        # blind: 10 representative (5 pass, 5 fail) and 10 adversarial, all agreed
        self.add(task, 5, split='heldout', kind='representative', passed=True, judged_pass=True)
        self.add(task, 5, split='heldout', kind='representative', passed=False, judged_pass=False, judged_failed=both(1))
        self.add(task, 10, split='heldout', kind='violate', passed=False, judged_pass=False, judged_failed=both(1))
        r = mm.readiness(task)
        self.assertEqual({c['key']: c['status'] for c in r['checks']}, {'both': 'met', 'stable': 'open', 'rules': 'met'})
        self.assertTrue(r['targets']['representative']['ok'] and r['targets']['adversarial']['ok'])
        self.assertEqual(r['checks'][0]['short'], '5 · 15')
        self.add(task, 19, split='learn', kind='comply', action='keep', passed=True, judged_pass=True)
        self.add(task, 1, split='learn', kind='violate', action='flip', passed=True, judged_pass=False, judged_failed=both(1))
        r = mm.readiness(task)
        self.assertTrue(r['ready'], r)
        self.add(task, 1, split='learn', kind='violate', action='flip', passed=True, judged_pass=False)
        r = mm.readiness(task)
        self.assertFalse(r['ready'])                             # two flips in the last 20
        self.assertEqual((r['checks'][1]['status'], r['checks'][1]['short']), ('not', 'still moving'))
        self.add(task, 19, split='learn', kind='comply', action='keep', passed=True, judged_pass=True)
        self.assertTrue(mm.readiness(task)['ready'])             # the older flip left the window
        mm.edit_qcr(task, [{'id': 's1', 'text': '3 keywords in total.'}, {'text': RULE}])
        r = {c['key']: c for c in mm.readiness(task)['checks']}
        self.assertEqual(r['stable']['status'], 'open')          # the rules changed since: too early to tell
        self.assertEqual(r['stable']['short'], '0/20 since v1')
        self.assertFalse(r['rules']['ok']); self.assertIn('s3', r['rules']['detail'])   # the new rule is untested
        body = self.client.get(f'/api/minime/tasks/{task.id}').json()
        self.assertIn('ready', body)
        S = body['series']
        self.assertEqual(len(S['points']), 20)                   # one per blind check
        self.assertEqual((S['points'][-1]['adv'], S['points'][-1]['adv_of']), (100, 10))
        self.assertEqual(S['marks'], [{'v': 1, 'n': 60, 'type': 'edit'}])   # after all 60 reviews
        self.assertEqual(S['reviewed'], 60)


class GepaLearnTests(TestCase):
    """A real GEPA run (the gepa library, no MLflow) with the rewriter and the judge stubbed."""

    def test_gepa_adds_the_rule_the_flips_teach(self):
        import os
        import sys
        import tempfile
        from llm_endpoint import Endpoint
        task = mm.create_task('ahmet', 'kw', {'objective': 'SEM keywords', 'requirements': '3 keywords in total.'}, ['x'])
        for n in range(6):
            l = MiniMeLabel.objects.create(task=task, input='x', output=FIRST if n % 2 else GOOD,
                                           judged_pass=True, judged_version=0)
            mm.review(task, l.pk, 'flip' if n % 2 else 'keep', reason='uses first: nobody searches that way' if n % 2 else '')
        seen = []

        models, caps = [], []

        def reflection(*a, **k):
            seen.append(k['messages'][0]['content'])
            models.append(k.get('model')); caps.append(k)
            text = f'```\n1. 3 keywords in total.\n2. {RULE}\n```'
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=text),
                                                                        finish_reason='stop')], usage=None)
        endpoint = Endpoint(base_url='http://stub.invalid/v1', api_key='x', task_model='m', judge_model='m', reflection_model='m')
        j = job()
        blocked = {k: None for k in list(sys.modules) if k == 'mlflow' or k.startswith('mlflow.')} | {'mlflow': None}
        with patch.dict(sys.modules, blocked), \
                patch.object(endpoint.client.chat.completions, 'create', reflection), \
                patch('optimizer.services._endpoint_for', lambda p: endpoint), \
                patch.object(mm.judging, 'llm', stub_llm), patch.object(mm.learning, 'learning_phase', lambda rows: 'gepa'), \
                patch.object(mm.learning, '_gepa_dir', lambda run_id: os.path.join(tempfile.mkdtemp(), run_id)), \
                patch.object(mm.judging, '_judge_cache_path', lambda: os.path.join(tempfile.mkdtemp(), 'jc.sqlite')):
            mm.run_learn(j, {'task_id': str(task.id), '_owner': 'ahmet'})
        self.assertEqual(j.status, 'done', j.error)
        self.assertTrue(j.result['accepted'], j.result)
        self.assertEqual(j.result['edits'], [{'op': 'add', 'id': 's2', 'after': RULE}])
        self.assertIn('quality rubric', seen[0])                        # our reflection template reached GEPA
        self.assertIn('uses first: nobody searches that way', seen[0])  # the flip's reason is the ASI
        self.assertEqual(models, ['claude-opus-4.8'])                   # the rewriter, with its output cap
        self.assertNotIn('Completions', seen[0])                        # no traces in the rewriter's prompt
        self.assertIn("Rubric's verdict", seen[0])
        # (any `import mlflow` during the run would have raised: MLflow is blocked above)
        self.assertEqual((caps[0]['max_tokens'], caps[0]['timeout']), (mm.REWRITER_MAX_TOKENS, mm.REWRITER_TIMEOUT))
