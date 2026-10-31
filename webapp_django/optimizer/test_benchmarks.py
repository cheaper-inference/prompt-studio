"""Mocked inference: no paid calls; benchmarks write nothing to MLflow."""
import json
import os
import tempfile
from types import SimpleNamespace as NS
from unittest.mock import patch
from django.test import SimpleTestCase, TestCase, override_settings
from . import benchmarks as b
from .models import Run

# Benchmark cases run one at a time here, whatever the shell exports: parallel cases write job state
# from several threads, which Django's in-memory test database cannot take ("table is locked"), and
# call-order assertions assume serial cases. Tests of concurrency set GP_BENCHMARK_CONCURRENCY themselves.
_SERIAL = patch.dict(os.environ, {'GP_BENCHMARK_CONCURRENCY': '1'})


def setUpModule():
    _SERIAL.start()


def tearDownModule():
    _SERIAL.stop()


def payload():
    return {'name':'Review','dataset_name':'test','baseline':[{'role':'user','content':'{{q}}'}],
            'optimized':[{'role':'user','content':'Answer: {{q}}'}],'models':['small','large'],
            'rows':[{'inputs':{'q':'a'},'expected_output':'yes'},{'inputs':{'q':'b'}}],
            'criteria':['Be concise', 'Be polite'],'judge_model':'judge','holdout_confirmed':True,
            'price_snapshot':{'captured_at':'2026-09-15T00:00:00Z','models':{'small':{'input_per_million':1,'output_per_million':2}}}}

class BillingTests(SimpleTestCase):
    def test_receipts_deduplicate_and_missing_is_not_zero(self):
        snapshots = []
        collector = b.BillingCollector(b.validate(payload()), snapshots.append)
        collector.model, collector.variant, collector.case_index = 'small', 'baseline', 1
        def response(request_id, amount, status='settled', currency='USD'):
            return NS(cheaper_inference={'request_id': request_id, 'billing': {
                'status': status, 'billed_cost_usd': amount, 'currency': currency}})
        collector.record(response('task', '0.10'), 'small')
        collector.record(response('task', '0.10'), 'small')
        collector.role = 'judge'
        collector.record(response('judge', '0.02'), 'judge')
        collector.record(response('repair', '0.03'), 'judge')
        collector.record(response('cached', '0', 'cached'), 'judge')
        for item in [None, NS(cheaper_inference={'billing': []}), response('bad', 'NaN'),
                     response('negative', '-1'), response('failed', '0', 'failed'), response('currency', '1', currency='EUR')]:
            collector.record(item, 'judge')
        collector.finish_answer()
        result = collector.snapshot()
        self.assertEqual(result['reported_spend_usd'], '0.15')
        self.assertEqual(result['calls_with_receipts'], 4)
        self.assertEqual(result['calls_missing_receipts'], 6)
        self.assertEqual(result['models']['small']['completed_answers'], 1)
        # An answer with an unpriced call does not sample the projection (CI's rule).
        self.assertEqual(result['models']['small']['priced_answers'], 0)
        self.assertEqual(result['models']['small']['completed_spend_usd'], '0')
        self.assertEqual(result['receipts'][2]['role'], 'judge')
        self.assertEqual(snapshots[0]['models']['small']['completed_answers'], 0)

    def test_flattened_ci_receipt_is_priced(self):
        """CI's KEA-7852 moved request_id and billing to the top level of the body."""
        from openai.types.chat import ChatCompletion
        bill = {'status': 'settled', 'billed_cost_usd': '0.004', 'currency': 'USD'}
        parsed = ChatCompletion.model_validate({'id': 'c', 'object': 'chat.completion', 'created': 0, 'model': 'small',
            'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': 'hi'}}],
            'request_id': 'flat-1', 'billing': bill})
        self.assertEqual(b.ci_receipt(parsed), {'request_id': 'flat-1', 'billing': bill})
        self.assertEqual(b.ci_receipt(NS(cheaper_inference={'request_id': 'old', 'billing': bill}))['request_id'], 'old')
        self.assertEqual(b.ci_receipt(NS()), {})
        collector = b.BillingCollector(b.validate(payload()))
        collector.model, collector.variant, collector.case_index = 'small', 'baseline', 1
        collector.record(parsed, 'small')
        self.assertEqual((collector.snapshot()['reported_spend_usd'], collector.missing), ('0.004', 0))

    def test_live_snapshots_omit_receipts_and_the_final_one_keeps_them(self):
        published = []
        collector = b.BillingCollector(b.validate(payload()), published.append)
        collector.model, collector.variant, collector.case_index = 'small', 'baseline', 1
        index = collector.begin_call('small')
        collector.record(NS(cheaper_inference={'request_id': 'r1', 'billing': {
            'status': 'settled', 'billed_cost_usd': '0.01', 'currency': 'USD'}}), 'small', call_index=index)
        collector.finish_answer()
        self.assertTrue(published and all('receipts' not in snap and 'calls' not in snap for snap in published))
        self.assertEqual(published[-1]['calls_with_receipts'], 1)
        self.assertEqual(published[-1]['models']['small']['priced_answers'], 1)
        self.assertEqual(published[-1]['models']['small']['completed_spend_usd'], '0.01')
        self.assertLess(max(len(json.dumps(snap)) for snap in published), 600)   # counters only
        final = collector.snapshot()
        self.assertEqual(final['calls'][0]['request_id'], 'r1')                  # CI reconciles from calls, once
        self.assertEqual([r['request_id'] for r in final['receipts']], ['r1'])

    def test_unstorable_request_id_or_amount_counts_as_missing(self):
        collector = b.BillingCollector(b.validate(payload()))
        collector.model, collector.variant, collector.case_index = 'small', 'baseline', 1
        bill = lambda amount: {'status': 'settled', 'billed_cost_usd': amount, 'currency': 'USD'}
        collector.record(NS(cheaper_inference={'request_id': 'x' * 129, 'billing': bill('0.01')}), 'small')
        collector.record(NS(cheaper_inference={'request_id': 'long-amount', 'billing': bill('0.' + '1' * 45)}), 'small')
        collector.record(NS(cheaper_inference={'request_id': 'y' * 128, 'billing': bill('0.02')}), 'small')
        snap = collector.snapshot()
        self.assertEqual((snap['calls_with_receipts'], snap['calls_missing_receipts']), (1, 2))
        self.assertIsNone(snap['calls'][0]['request_id'])
        self.assertEqual(snap['reported_spend_usd'], '0.02')

    def test_concurrent_calls_keep_unique_rows_and_consistent_counters(self):
        import threading
        ops, collector = [], None
        collector = b.BillingCollector(b.validate(payload()), persist=lambda op, snap: ops.append(op))
        collector.model, collector.variant, collector.case_index, collector.role = 'small', 'baseline', 1, 'judge'
        start = threading.Barrier(8, timeout=5)
        def judge_call(n):
            start.wait()
            seq = collector.begin_call('judge')
            collector.record(NS(cheaper_inference={'request_id': f'r{n}', 'billing': {
                'status': 'settled', 'billed_cost_usd': '0.01', 'currency': 'USD'}}), 'judge', call_index=seq)
        threads = [threading.Thread(target=judge_call, args=(n,)) for n in range(8)]
        for t in threads: t.start()
        for t in threads: t.join()
        final = collector.snapshot()
        self.assertEqual(sorted(c['request_id'] for c in final['calls']), sorted(f'r{n}' for n in range(8)))
        self.assertEqual((final['calls_with_receipts'], final['calls_missing_receipts'], final['reported_spend_usd']), (8, 0, '0.08'))
        self.assertEqual(len(ops), 16)                                          # one row write per begin and per record

    def test_recorded_latency_excludes_waiting_for_the_bookkeeping_lock(self):
        import threading, time
        captured = []
        collector = b.BillingCollector(b.validate(payload()),
                                       persist=lambda op, snap: (op('job') if op else None, time.sleep(0.15)))
        collector.model, collector.variant, collector.case_index, collector.role = 'small', 'baseline', 1, 'judge'
        fast = lambda **kw: NS(cheaper_inference={'request_id': f"r{threading.get_ident()}", 'billing': {
            'status': 'settled', 'billed_cost_usd': '0.01', 'currency': 'USD'}}, choices=[])
        client = b.CancellationClient(NS(chat=NS(completions=NS(create=fast))), lambda: None, collector)
        with patch('optimizer.benchmark_calls.insert'), \
                patch('optimizer.benchmark_calls.update', side_effect=lambda job, seq, **kw: captured.append(kw)):
            threads = [threading.Thread(target=lambda: client.chat.completions.create(model='judge')) for _ in range(3)]
            for t in threads: t.start()
            for t in threads: t.join()
        latencies = [kw['duration_ms'] for kw in captured if 'duration_ms' in kw]
        self.assertEqual(len(latencies), 3)
        self.assertTrue(all(ms < 100 for ms in latencies), latencies)   # instant responses; lock waits were ~150 ms each
        self.assertTrue(all(kw.get('completed_at') is not None for kw in captured if 'duration_ms' in kw))   # arrival time recorded

    def test_percentiles_interpolate(self):
        from .benchmark_timing import percentile
        values = list(range(1, 11))
        self.assertEqual((percentile(values, .5), percentile(values, .9), percentile([4.0], .9), percentile([], .5)),
                         (5.5, 9.1, 4.0, None))

    def test_live_projection_counters_follow_ci_rules(self):
        from decimal import Decimal
        collector = b.BillingCollector(b.validate(payload()))
        def answer(case, *amounts):
            collector.model, collector.variant, collector.case_index = 'small', 'baseline', case
            for n, amount in enumerate(amounts):
                bill = {'status': 'settled', 'billed_cost_usd': amount, 'currency': 'USD'} if amount else {}
                collector.record(NS(cheaper_inference={'request_id': f'{case}-{n}', 'billing': bill}), 'small')
            collector.finish_answer()
        answer(1, '0.01', '0.02')       # fully priced
        answer(2, '0.05', None)         # one call unpriced
        bucket = collector.snapshot(live=True)['models']['small']
        self.assertEqual((bucket['completed_answers'], bucket['priced_answers']), (2, 1))
        self.assertEqual(Decimal(bucket['completed_spend_usd']), Decimal('0.03'))
        self.assertEqual(Decimal(bucket['reported_spend_usd']), Decimal('0.08'))

    def test_wrapper_records_success_and_lost_response_without_retrying(self):
        from unittest.mock import Mock
        collector = b.BillingCollector(b.validate(payload()))
        collector.model = 'small'
        response = NS(cheaper_inference={'request_id': 'one', 'billing': {'status': 'settled', 'currency': 'USD', 'billed_cost_usd': '0.001'}})
        create = Mock(side_effect=[response, TimeoutError('no response')])
        client = b.CancellationClient(NS(chat=NS(completions=NS(create=create))), lambda: None, collector)
        self.assertIs(client.chat.completions.create(model='small'), response)
        with self.assertRaises(TimeoutError): client.chat.completions.create(model='judge')
        self.assertEqual(create.call_count, 2)
        self.assertEqual(collector.snapshot()['calls_missing_receipts'], 1)
        self.assertEqual(collector.snapshot()['reported_spend_usd'], '0.001')

    def test_raw_headers_capture_ids_without_inline_prices_and_freeze_completion(self):
        from unittest.mock import Mock
        collector = b.BillingCollector(b.validate(payload()))
        collector.model, collector.variant, collector.case_index = 'small', 'baseline', 1
        response = NS(choices=[])
        raw = NS(headers={'x-ci-request-id': '22222222-2222-4222-8222-222222222222'}, parse=Mock(return_value=response))
        create = Mock(return_value=raw)
        completions = NS(with_raw_response=NS(create=create))
        client = b.CancellationClient(NS(chat=NS(completions=completions)), lambda: None, collector)
        self.assertIs(client.chat.completions.create(model='small'), response)
        before = collector.snapshot()
        collector.finish_answer()
        self.assertFalse(before['calls'][0]['completed'])
        self.assertTrue(collector.snapshot()['calls'][0]['completed'])
        self.assertEqual(before['calls'][0]['request_id'], raw.headers['x-ci-request-id'])
        self.assertEqual(before['calls_missing_receipts'], 1)
        self.assertEqual(before['reported_spend_usd'], '0')
        create.assert_called_once()

    def test_real_sdk_raw_response_captures_canonical_header(self):
        import httpx
        from openai import OpenAI
        collector = b.BillingCollector(b.validate(payload()))
        collector.model, collector.variant, collector.case_index = 'small', 'baseline', 1
        request_id = '44444444-4444-4444-8444-444444444444'
        def transport(request):
            return httpx.Response(200, headers={'x-ci-request-id': request_id}, json={
                'id': 'provider-id', 'object': 'chat.completion', 'created': 1, 'model': 'small',
                'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'answer'}, 'finish_reason': 'stop'}]})
        with OpenAI(api_key='fixture', base_url='https://fixture.invalid/v1', max_retries=0,
                    http_client=httpx.Client(transport=httpx.MockTransport(transport))) as sdk:
            client = b.CancellationClient(sdk, lambda: None, collector)
            response = client.chat.completions.create(model='small', messages=[{'role':'user','content':'q'}])
        self.assertEqual(response.choices[0].message.content, 'answer')
        self.assertEqual(collector.snapshot()['calls'][0]['request_id'], request_id)
        self.assertFalse(collector.snapshot()['calls'][0]['in_flight'])

    def test_failed_response_retains_header_id_without_retry(self):
        from unittest.mock import Mock
        collector = b.BillingCollector(b.validate(payload()))
        collector.model, collector.variant, collector.case_index = 'small', 'baseline', 1
        error = RuntimeError('upstream failure')
        error.response = NS(headers={'x-ci-request-id': '33333333-3333-4333-8333-333333333333'})
        create = Mock(side_effect=error)
        client = b.CancellationClient(NS(chat=NS(completions=NS(with_raw_response=NS(create=create)))), lambda: None, collector)
        with self.assertRaises(RuntimeError): client.chat.completions.create(model='small')
        self.assertEqual(collector.snapshot()['calls'][0]['request_id'], error.response.headers['x-ci-request-id'])
        create.assert_called_once()

    def test_real_evaluator_captures_judge_repairs(self):
        from llm_endpoint import Endpoint
        from unittest.mock import Mock
        p = payload(); p['evaluation_mode'] = 'rubric'; p['models'] = ['small']; p['rows'] = p['rows'][:1]
        manifest = b.validate(p)
        collector = b.BillingCollector(manifest)
        calls = []
        def completion(**kwargs):
            role = 'task' if kwargs['model'] == 'small' else 'judge'
            calls.append(role)
            content = 'yes' if role == 'task' else ('invalid JSON' if len(calls) % 3 == 2 else json.dumps({'per_guideline': [True, True], 'details': [{'explanation':'ok','evidence':'yes'}] * 2}))
            return NS(cheaper_inference={'request_id': str(len(calls)), 'billing': {'status':'settled','currency':'USD','billed_cost_usd':'0.01'}}, model=kwargs['model'], usage=None, choices=[NS(message=NS(content=content))])
        endpoint = Endpoint(base_url='https://unused.invalid/v1',api_key='never-sent')
        object.__setattr__(endpoint,'_client',b.CancellationClient(NS(chat=NS(completions=NS(create=completion))),lambda:None,collector))
        tracker = NS(arm=lambda *args:'child',finish_arm=Mock(),run_id='parent',experiment_id='experiment')
        report = b.execute(manifest,endpoint,tracker,billing=collector)
        self.assertEqual(calls, ['task','judge','judge'] * 2)
        self.assertEqual(report['billing']['reported_spend_usd'],'0.06')
        self.assertEqual([r['role'] for r in report['billing']['receipts']], calls)
        self.assertEqual(report['billing']['models']['small']['completed_answers'],2)
        self.assertTrue(all(arm['errors'] == 0 for arm in report['arms']))

class ContractTests(SimpleTestCase):
    def test_cancel_stops_before_judge_without_scoring_failure(self):
        from llm_endpoint import Endpoint
        from unittest.mock import Mock
        p=payload(); p['evaluation_mode']='rubric'
        endpoint=Endpoint(base_url='https://unused.invalid/v1',api_key='never-sent')
        response=NS(model='small',usage=NS(prompt_tokens=1,completion_tokens=1,total_tokens=2,completion_tokens_details=None),choices=[NS(message=NS(content='answer'))])
        create=Mock(return_value=response)
        client=NS(chat=NS(completions=NS(create=create)))
        def check():
            if create.call_count:
                raise b.BenchmarkCancelled()
        object.__setattr__(endpoint,'_client',b.CancellationClient(client,check))
        tracker=NS(arm=lambda *args:'child',finish_arm=Mock(),run_id='parent',experiment_id='experiment')
        with self.assertRaises(b.BenchmarkCancelled):
            b.execute(b.validate(p),endpoint,tracker)
        self.assertEqual(create.call_count,1)
        tracker.finish_arm.assert_not_called()

    def test_synthetic_provenance_never_attests_holdout(self):
        p = payload()
        p['holdout_confirmed'] = True
        p['synthetic_provenance'] = {'source': 'synthetic', 'holdout_status': 'verified', 'model': 'generator'}
        manifest = b.validate(p)
        self.assertEqual(manifest['dataset']['synthetic_provenance']['model'], 'generator')
        self.assertEqual(manifest['dataset']['synthetic_provenance']['holdout_status'], 'unverified')
        self.assertEqual(manifest['dataset']['holdout_status'], 'not_attested')
        self.assertEqual(manifest['dataset']['split'], 'validation')
        p['synthetic_provenance'] = {'large': 'x' * 20001}
        with self.assertRaises(ValueError): b.validate(p)

    def test_three_model_limit(self):
        p=payload(); p['models']=['a','b','c']
        self.assertEqual(len(b.validate(p)['models']),3)
        p['models'].append('d')
        with self.assertRaises(ValueError): b.validate(p)

    def test_identity_and_allowlist(self):
        p=payload(); p['_api_key']='secret'; p['tracking_uri']='untrusted'
        m=b.validate(p)
        self.assertNotIn('secret',json.dumps(m)); self.assertNotIn('untrusted',json.dumps(m))
        p['rows'][0]['expected_output']='no'
        self.assertNotEqual(m['dataset']['digest'],b.validate(p)['dataset']['digest'])

    def test_invalid_inputs(self):
        for edit in (lambda p:p.update(models=[]),lambda p:p.update(holdout_confirmed='false'),
                     lambda p:p['rows'].append(p['rows'][0]),lambda p:p['rows'][1].update(inputs={}),
                     lambda p:p.update(criteria=[]),lambda p:p.update(baseline=[42])):
            p=payload(); edit(p)
            with self.assertRaises(ValueError): b.validate(p)

    def test_explicit_modes(self):
        p=payload(); p['evaluation_mode']='both'
        with self.assertRaisesRegex(ValueError, 'case 2'): b.validate(p)
        p['rows'][1]['expected_output']='yes'
        self.assertEqual(b.validate(p)['scoring']['evaluation_mode'],'both')
        p['criteria']=[]
        with self.assertRaisesRegex(ValueError, 'criterion'): b.validate(p)
        p['evaluation_mode']='reference'
        b.validate(p)
        p['evaluation_mode']='rubric'
        with self.assertRaisesRegex(ValueError, 'criterion'): b.validate(p)

    def test_both_scores_same_output_and_isolates_judge_failures(self):
        from llm_endpoint import Endpoint
        for failure in (None, 'reference', 'rubric'):
            p=payload(); p['evaluation_mode']='both'; p['rows']=p['rows'][:1]; p['models']=['small']
            manifest=b.validate(p)
            endpoint=Endpoint(base_url='https://unused.invalid/v1',api_key='never-sent')
            calls=[]
            def completion(**kw):
                kind='task' if kw['model'] != 'judge' else ('rubric' if 'Guidelines (in order)' in kw['messages'][0]['content'] else 'reference')
                calls.append(kind)
                if kind == failure: raise RuntimeError('Mock judge failure')
                content={'task':'yes','reference':'{"match":true}', 'rubric':json.dumps({'per_guideline':[True,False], 'details':[
                    {'explanation':'Short answer','evidence':'yes'}, {'explanation':'Not JSON','evidence':'yes'}]})}[kind]
                return NS(model=kw['model'],usage=NS(prompt_tokens=100,completion_tokens=10,total_tokens=110,completion_tokens_details=None),choices=[NS(message=NS(content=content))])
            tracker=NS(arm=lambda *args:'child',finish_arm=lambda *args:None,run_id='parent',experiment_id='experiment')
            stages=[]
            with patch.object(endpoint.client.chat.completions,'create',side_effect=completion):
                report=b.execute(manifest,endpoint,tracker,stage=lambda message: stages.append((message,len(calls))))
            completed=[(message,count) for message,count in stages if 'answers evaluated · Continuing' in message or 'answers evaluated · Saving results' in message]
            self.assertEqual(completed, [('Benchmarking · 1 of 2 answers evaluated · Continuing',3), ('Benchmarking · 2 of 2 answers evaluated · Saving results',6)])
            self.assertEqual(calls.count('task'),2)
            self.assertEqual(calls.count('reference'),2)
            self.assertEqual(calls.count('rubric'),2)
            arm=report['arms'][0]
            self.assertEqual(arm['usage']['total_cases'],1)
            self.assertEqual(arm['quality']['reference']['n'],1)
            self.assertEqual(arm['quality']['rubric']['n'],1)
            self.assertEqual(arm['quality']['reference']['mean_score'],None if failure=='reference' else 1)
            self.assertEqual(arm['quality']['rubric']['mean_score'],None if failure=='rubric' else .5)

    def test_array_references(self):
        p=payload()
        for reference in (['urgent care nearby', '  exact text  '], []):
            p['rows'][0]['expected_output']=reference
            self.assertEqual(b.validate(p)['dataset']['rows'][0]['expected_output'],reference)
        for reference in ([42], [['nested']], {'answer':'no'}):
            p['rows'][0]['expected_output']=reference
            with self.assertRaises(ValueError): b.validate(p)

    def test_validation_and_held_out_labels(self):
        p=payload()
        self.assertEqual(b.validate(p)['dataset']['split'],'test')
        p['holdout_confirmed']=False
        data=b.validate(p)['dataset']
        self.assertEqual(data['split'],'validation')
        self.assertEqual(data['holdout_status'],'not_attested')

    def test_missing_usage_and_failure_denominators(self):
        s=b.summarize([{'graded_by':'reference','score':1,'usage':None},
                       {'graded_by':'reference','error':'TimeoutError','usage':None}],
                      {'input_per_million':1,'output_per_million':2})
        self.assertIsNone(s['estimated_cost_per_call']); self.assertIsNone(s['usage']['avg_input'])
        self.assertEqual(s['quality']['reference']['pass_rate'],1)
        self.assertEqual(s['quality']['reference']['evaluated'],1)
        self.assertEqual(s['quality']['reference']['evaluation_errors'],1)
        self.assertEqual(s['quality']['rubric']['n'],0); self.assertEqual(s['errors'],1)

    def test_real_evaluator_writes_nothing_to_mlflow(self):
        from llm_endpoint import Endpoint
        from prompt_optimizer import PromptOptimizer
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,{'MLFLOW_TRACKING_URI':f'sqlite:///{tmp}/tracking.db','MLFLOW_ENABLE_ASYNC_LOGGING':'false'}):
            p=payload(); p['holdout_confirmed']=False; p['rows'][0]['expected_output']=['yes','indeed']
            m=b.validate(p); tracker=b.Tracker('customer-a','b1',m)
            endpoint=Endpoint(base_url='https://unused.invalid/v1',api_key='never-sent')
            judge_requests=[]
            def completion(**kw):
                content='yes'
                if kw['model']=='judge':
                    judge_requests.append(kw['messages'])
                    content=json.dumps({'per_guideline':[True,False], 'details':[
                        {'explanation':'The answer is one word.', 'evidence':'yes'},
                        {'explanation':'A bare word is not valid JSON.', 'evidence':'yes'}]}) if 'Guidelines (in order)' in kw['messages'][0]['content'] else '{"match": true}'
                return NS(model=kw['model'],usage=NS(prompt_tokens=100,completion_tokens=10,total_tokens=110,completion_tokens_details=None),choices=[NS(message=NS(content=content))])
            with patch.object(endpoint.client.chat.completions,'create',side_effect=completion):
                result=b.execute(m,endpoint,tracker)
            tracker.finish(result)
            self.assertEqual(len(result['arms']),4)
            self.assertEqual(result['arms'][0]['cases'][0]['expected_output'],['yes','indeed'])
            self.assertIn('["yes", "indeed"]', str(judge_requests))
            self.assertEqual(result['arms'][0]['quality']['reference']['pass_rate'],1)
            self.assertEqual(result['arms'][0]['quality']['rubric']['mean_score'],.5)
            self.assertAlmostEqual(result['arms'][0]['estimated_cost_per_call'],.00012)
            self.assertIsNone(result['arms'][2]['estimated_cost_per_call'])
            # No MLflow store is created; the ids CI's report schema requires are GP's own.
            self.assertFalse(os.path.exists(f'{tmp}/tracking.db'))
            self.assertEqual(len({tracker.run_id, *(a['mlflow_run_id'] for a in result['arms'])}), 5)
            self.assertTrue(tracker.experiment_id.startswith('ci-benchmarks-'))
            self.assertEqual(set(m['runtime']), {'benchmark_sha256', 'evaluator_sha256'})
            evidence=result['arms'][0]['cases'][1]['rubric_evaluation']
            self.assertEqual(evidence['passed'],1)
            self.assertFalse(evidence['overall_pass'])
            self.assertEqual(evidence['criteria'][1]['criterion'],'Be polite')

    def test_invalid_rubric_evidence_retries_then_errors(self):
        from llm_endpoint import Endpoint
        from prompt_optimizer import PromptOptimizer, OptimizationConfig
        endpoint=Endpoint(base_url='https://unused.invalid/v1',api_key='never-sent')
        opt=PromptOptimizer(endpoint)
        captured=[]
        judge=opt._build_guidelines_scorer(OptimizationConfig(name='test',prompts=[]), ['Return JSON'], on_judgment=captured.append)
        bad=[{'per_guideline':[True], 'details':[{'explanation':'Good', 'evidence':'fabricated'}]},
             {'per_guideline':['true'], 'details':[{'explanation':'Good', 'evidence':None}]},
             {'per_guideline':[False], 'details':[]}]
        for obj in bad:
            response=NS(usage=None,choices=[NS(message=NS(content=json.dumps(obj)))])
            with patch.object(endpoint.client.chat.completions,'create',return_value=response) as call:
                with self.assertRaises(ValueError):
                    judge(inputs={},outputs='yes')
                self.assertEqual(call.call_count,2)
        self.assertEqual(len(captured),3)
        self.assertTrue(all(item['overall_pass'] is None for item in captured))
        self.assertEqual(captured[0]['criteria'][0]['error_code'],'invalid_evidence')

    def test_rubric_missing_evidence_and_legacy_scoring(self):
        from llm_endpoint import Endpoint
        from prompt_optimizer import PromptOptimizer, OptimizationConfig
        endpoint=Endpoint(base_url='https://unused.invalid/v1',api_key='never-sent')
        opt=PromptOptimizer(endpoint); captured=[]
        config=OptimizationConfig(name='test',prompts=[])
        response=NS(usage=None,choices=[NS(message=NS(content=json.dumps({'per_guideline':[False],
            'details':[{'explanation':'The required closing statement is missing.', 'evidence':None}]})))])
        with patch.object(endpoint.client.chat.completions,'create',return_value=response):
            opt._build_guidelines_scorer(config,['Include a closing statement'],on_judgment=captured.append)(inputs={},outputs='yes')
        self.assertIsNone(captured[0]['criteria'][0]['evidence'])
        response.choices[0].message.content='{"per_guideline":[true,false],"rationale":"one passed"}'
        with patch.object(endpoint.client.chat.completions,'create',return_value=response):
            value=opt._build_guidelines_scorer(config,['a','b'])(inputs={},outputs='yes')
        from prompt_optimizer import _scorer_value
        self.assertEqual(_scorer_value(value),.5)

@override_settings(PROMPT_STUDIO_API_ENABLED=True,PROMPT_STUDIO_SERVICE_TOKEN='test',PROMPT_STUDIO_API_IP_ALLOWLIST='',
    PROMPT_STUDIO_CREDENTIAL_KEY='MDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDA=')
class ApiTests(TestCase):
    def leased_job(self, stored):
        from . import services
        from .models import JobState
        from django.utils import timezone
        from datetime import timedelta
        JobState.objects.create(id=stored.job_id,owner=stored.owner,kind='benchmark',
            lease_token='lease',lease_until=timezone.now()+timedelta(minutes=2))
        return services.Job(id=stored.job_id,owner=stored.owner,lease_token='lease')

    def test_owner_isolation(self):
        h={'HTTP_AUTHORIZATION':'Bearer test','HTTP_X_CI_USER_ID':'owner','HTTP_X_CI_INFERENCE_KEY':'secret'}
        p=payload(); p['base_url']='https://unused.invalid/v1'
        r=self.client.post('/api/v1/benchmarks',data=json.dumps(p),content_type='application/json',**h)
        self.assertEqual(r.status_code,200,r.content)
        job=r.json()['job_id']; stored=Run.objects.get(job_id=job)
        self.assertNotIn('secret',json.dumps(stored.config))
        self.assertEqual(self.client.get(f'/api/v1/jobs/{job}',**h).status_code,200)
        h['HTTP_X_CI_USER_ID']='other'
        self.assertEqual(self.client.get(f'/api/v1/jobs/{job}',**h).status_code,404)
        self.assertEqual(self.client.get(f'/api/v1/runs/{stored.id}',**h).status_code,404)

    def test_billing_survives_polling_and_keeps_receipts_private_to_owner(self):
        from .models import JobState
        h = {'HTTP_AUTHORIZATION':'Bearer test','HTTP_X_CI_USER_ID':'owner'}
        stored = Run.objects.create(kind='benchmark',owner='owner',job_id='b'*32,status='running')
        snapshot = {'reported_spend_usd':'0.12','receipts':[{'request_id':'receipt'}]}
        JobState.objects.create(id=stored.job_id,owner='owner',kind='benchmark',result={'billing':snapshot})
        url = f'/api/v1/jobs/{stored.job_id}'
        body = self.client.get(url,**h).json()
        self.assertEqual(body['billing'],{'reported_spend_usd':'0.12'})
        self.assertIsNone(body['result'])
        self.assertEqual(self.client.get(url,**{**h,'HTTP_X_CI_USER_ID':'other'}).status_code,404)
        JobState.objects.filter(pk=stored.job_id).update(status='done')
        self.assertEqual(self.client.get(url,**h).json()['result']['billing'],snapshot)

    def test_cancel_is_owner_scoped_and_idempotent(self):
        from .models import JobState
        h={'HTTP_AUTHORIZATION':'Bearer test','HTTP_X_CI_USER_ID':'owner'}
        stored=Run.objects.create(kind='benchmark',owner='owner',job_id='c'*32,status='running')
        JobState.objects.create(id=stored.job_id,owner='owner',kind='benchmark',status='queued')
        url=f'/api/v1/jobs/{stored.job_id}/cancel'
        self.assertEqual(self.client.post(url,**{**h,'HTTP_X_CI_USER_ID':'other'}).status_code,404)
        self.assertEqual(self.client.get(url,**h).status_code,405)
        for _ in range(2):
            self.assertEqual(self.client.post(url,**h).json()['status'],'cancelled')
        self.assertEqual(self.client.get(f'/api/v1/jobs/{stored.job_id}',**h).json()['status'],'cancelled')
        stored.refresh_from_db(); self.assertEqual(stored.status,'cancelled')
        stored.status='done'; stored.save()
        JobState.objects.filter(pk=stored.job_id).update(status='done')
        self.assertEqual(self.client.post(url,**h).json()['status'],'done')

    def test_inflight_charge_survives_cancellation_in_existing_json(self):
        from . import services
        from .models import JobState
        from llm_endpoint import Endpoint
        from unittest.mock import Mock
        stored = Run.objects.create(kind='benchmark',owner='owner',job_id='e'*32,status='running')
        job = self.leased_job(stored)
        endpoint = Endpoint(base_url='https://unused.invalid/v1',api_key='never-sent')
        def completion(**kwargs):
            from .durable import cancel
            cancel(stored.job_id, 'owner')
            return NS(cheaper_inference={'request_id':'paid-before-cancel','billing':{
                'status':'settled','currency':'USD','billed_cost_usd':'0.0123'}},
                model=kwargs['model'],usage=None,choices=[NS(message=NS(content='yes'))])
        create = Mock(side_effect=completion)
        client = NS(chat=NS(completions=NS(create=create)))
        client.with_options = lambda **kwargs: client
        object.__setattr__(endpoint,'_client',client)
        with patch('django.db.connection.close'), patch('optimizer.benchmarks.Tracker'), patch('optimizer.benchmarks.services._endpoint_for',return_value=endpoint):
            b.run(job,str(stored.pk),b.validate(payload()),{'_owner':'owner'})
        stored.refresh_from_db()
        self.assertEqual(stored.status,'cancelled')
        self.assertEqual(stored.result['billing']['reported_spend_usd'],'0.0123')
        self.assertEqual(JobState.objects.get(pk=job.id).result['billing']['calls_with_receipts'],1)
        self.assertEqual(create.call_count,1)

    def test_cancelled_worker_updates_job_and_history(self):
        from . import services
        from .models import JobState
        stored=Run.objects.create(kind='benchmark',owner='owner',job_id='d'*32,status='cancelling')
        job=self.leased_job(stored)
        from .models import JobState
        from django.utils import timezone
        JobState.objects.filter(pk=stored.job_id).update(cancel_requested_at=timezone.now())
        with patch('django.db.connection.close'), patch('optimizer.benchmarks.Tracker') as tracker:
            b.run(job,str(stored.id),{}, {})
        tracker.assert_not_called()
        stored.refresh_from_db()
        self.assertEqual(stored.status,'cancelled')
        self.assertEqual(JobState.objects.get(id=job.id).status,'cancelled')

    def assert_queue_completion(self, include_optimized):
        from . import durable
        from .models import JobState
        from llm_endpoint import Endpoint
        from unittest.mock import Mock
        p = {**payload(), 'models':['small'], 'rows':[{'inputs':{'q':'a'}}],
             'evaluation_mode':'rubric', '_owner':'owner', '_api_key':'private-key',
             '_base_url':'https://unused.invalid/v1', '_idempotency_key':'benchmark-once'}
        if not include_optimized:
            p.pop("optimized")
        queued = b.start(p)
        self.assertEqual(b.start(p).id, queued.id)
        self.assertEqual(Run.objects.filter(job_id=queued.id).count(), 1)
        self.assertEqual(queued.status, 'queued')
        self.assertNotIn('private-key', json.dumps(queued.payload))
        self.assertNotIn('private-key', queued.credential_ciphertext)
        self.assertEqual(durable.decrypt_secret(queued.credential_ciphertext)['_api_key'], 'private-key')
        endpoint = Endpoint(base_url='https://unused.invalid/v1', api_key='never-sent')
        def completion(**kw):
            content = 'yes' if kw['model'] == 'small' else json.dumps({'per_guideline':[True,True],
                'details':[{'explanation':'It meets the criterion.', 'evidence':'yes'}]*2})
            return NS(model=kw['model'],usage=NS(prompt_tokens=10,completion_tokens=2,total_tokens=12,completion_tokens_details=None),
                      choices=[NS(message=NS(content=content))])
        count = 0
        def raw_completion(**kw):
            nonlocal count
            count += 1
            response = completion(**kw)
            return NS(headers={'x-ci-request-id':f'paid-{count}'},parse=lambda:response)
        create = Mock(side_effect=raw_completion)
        client = NS(chat=NS(completions=NS(with_raw_response=NS(create=create))))
        client.with_options = lambda **kw: client
        object.__setattr__(endpoint,'_client',client)
        tracker = NS(arm=lambda *args:'child',finish_arm=lambda *args:None,run_id='parent',
                     experiment_id='experiment',finish=Mock(),fail=Mock())
        with patch('optimizer.benchmarks.Tracker',return_value=tracker), patch('optimizer.benchmarks.services._endpoint_for',return_value=endpoint):
            self.assertTrue(durable.process_one())
        queued.refresh_from_db()
        stored = Run.objects.get(job_id=queued.id)
        self.assertEqual((queued.status,stored.status),('done','done'))
        self.assertEqual(queued.result,stored.result)
        self.assertEqual(queued.credential_ciphertext,'')
        self.assertIsNone(queued.lease_token)
        variants = 2 if include_optimized else 1
        self.assertEqual(create.call_count,variants * 2)
        self.assertEqual(len(stored.result['arms']),variants)
        self.assertEqual(stored.result['billing']['models']['small']['total_answers'],variants)
        if not include_optimized:
            self.assertEqual(list(stored.result['manifest']['variants']),['baseline'])
        self.assertTrue(all(a['quality']['rubric']['pass_rate'] == 1 for a in stored.result['arms']))
        calls = stored.result['billing']['calls']
        self.assertEqual([c['request_id'] for c in calls],[f'paid-{i+1}' for i in range(variants * 2)])
        self.assertTrue(all(c['completed'] for c in calls))
        tracker.finish.assert_called_once()
        self.assertFalse(durable.process_one())

    def test_queued_cancellation_never_dispatches(self):
        from . import durable
        queued = b.start({**payload(), '_owner':'owner', '_api_key':'private-key', '_base_url':'https://unused.invalid/v1'})
        durable.cancel(queued.id,'owner')
        with patch('optimizer.benchmarks.services._endpoint_for') as endpoint:
            self.assertFalse(durable.process_one())
        endpoint.assert_not_called()
        queued.refresh_from_db()
        self.assertEqual(queued.credential_ciphertext,'')
        self.assertEqual(Run.objects.get(job_id=queued.id).status,'cancelled')

    def test_worker_failure_updates_history_and_clears_key(self):
        from . import durable
        queued = b.start({**payload(), '_owner':'owner', '_api_key':'private-key', '_base_url':'https://unused.invalid/v1'})
        with patch('optimizer.benchmarks.run_queued',side_effect=RuntimeError('failed')):
            self.assertTrue(durable.process_one())
        queued.refresh_from_db()
        self.assertEqual(queued.status,'error')
        self.assertEqual(queued.credential_ciphertext,'')
        self.assertEqual(Run.objects.get(job_id=queued.id).status,'error')

    def test_stale_worker_cannot_publish_late_paid_response(self):
        from . import durable
        from .models import JobState
        from llm_endpoint import Endpoint
        from django.utils import timezone
        from datetime import timedelta
        from unittest.mock import Mock
        stored = Run.objects.create(kind='benchmark',owner='owner',job_id='f'*32,status='running')
        job = self.leased_job(stored)
        endpoint = Endpoint(base_url='https://unused.invalid/v1',api_key='never-sent')
        def completion(**kw):
            JobState.objects.filter(pk=job.id).update(lease_until=timezone.now()-timedelta(seconds=1))
            durable.claim()
            return NS(model=kw['model'],usage=None,choices=[NS(message=NS(content='yes'))])
        create = Mock(side_effect=completion)
        client = NS(chat=NS(completions=NS(create=create)))
        client.with_options = lambda **kw: client
        object.__setattr__(endpoint,'_client',client)
        with patch('optimizer.benchmarks.Tracker'), patch('optimizer.benchmarks.services._endpoint_for',return_value=endpoint):
            with self.assertRaises(durable.LeaseLost):
                b.run(job,str(stored.id),b.validate(payload()),{'_owner':'owner'})
        stored.refresh_from_db()
        state = JobState.objects.get(pk=job.id)
        self.assertEqual((stored.status,state.status),('interrupted','interrupted'))
        self.assertEqual(stored.result,state.result)
        self.assertNotIn('calls',stored.result['billing'])                  # live JSON: counters only
        from .models import BenchmarkCall
        row = BenchmarkCall.objects.get(job_id=job.id)                       # the durable record
        self.assertTrue(row.in_flight); self.assertIsNone(row.request_id)    # late response not recorded
        h = {'HTTP_AUTHORIZATION':'Bearer test','HTTP_X_CI_USER_ID':'owner'}
        polled = self.client.get(f'/api/v1/jobs/{job.id}',**h).json()['billing']
        self.assertTrue(polled['calls'][0]['in_flight'])                     # assembled from rows for CI
        detail = self.client.get(f'/api/v1/runs/{stored.id}',**h).json()['result']['billing']
        self.assertTrue(detail['calls'][0]['in_flight'])
        self.assertEqual(create.call_count,1)

    def test_cancelled_run_keeps_full_billing_receipts(self):
        from .models import JobState
        from llm_endpoint import Endpoint
        from django.utils import timezone
        from unittest.mock import Mock
        stored = Run.objects.create(kind='benchmark',owner='owner',job_id='c'*32,status='running')
        job = self.leased_job(stored)
        endpoint = Endpoint(base_url='https://unused.invalid/v1',api_key='never-sent')
        def completion(**kw):
            # The first answer is billed; then the user cancels before the next call.
            JobState.objects.filter(pk=job.id).update(cancel_requested_at=timezone.now())
            return NS(model=kw['model'],usage=None,choices=[NS(message=NS(content='yes'))],
                      cheaper_inference={'request_id':'paid-1','billing':{'status':'settled','billed_cost_usd':'0.02','currency':'USD'}})
        create = Mock(side_effect=completion)
        client = NS(chat=NS(completions=NS(create=create)))
        client.with_options = lambda **kw: client
        object.__setattr__(endpoint,'_client',client)
        with patch('optimizer.benchmarks.Tracker'), patch('optimizer.benchmarks.services._endpoint_for',return_value=endpoint):
            b.run(job,str(stored.id),b.validate(payload()),{'_owner':'owner'})
        stored.refresh_from_db()
        self.assertEqual(stored.status,'cancelled')
        self.assertEqual([r['request_id'] for r in stored.result['billing']['receipts']],['paid-1'])
        self.assertEqual(create.call_count,1)

    def test_zero_call_terminal_ledger_has_empty_lists(self):
        from .models import JobState
        from .benchmark_calls import assemble
        stored = Run.objects.create(kind='benchmark',owner='owner',job_id='9'*32,status='error',
                                    result={'billing':{'reported_spend_usd':'0'}})
        JobState.objects.create(id=stored.job_id,owner='owner',kind='benchmark',status='error',result=stored.result)
        self.assertEqual(assemble(stored.job_id,{'reported_spend_usd':'0'}),{'reported_spend_usd':'0','calls':[],'receipts':[]})
        h = {'HTTP_AUTHORIZATION':'Bearer test','HTTP_X_CI_USER_ID':'owner'}
        self.assertEqual(self.client.get(f'/api/v1/jobs/{stored.job_id}',**h).json()['billing']['calls'],[])
        detail = self.client.get(f'/api/v1/runs/{stored.id}',**h).json()['result']['billing']
        self.assertEqual((detail['calls'],detail['receipts']),([],[]))

    def test_failed_run_keeps_full_billing_receipts(self):
        from .models import JobState
        from llm_endpoint import Endpoint
        from unittest.mock import Mock
        stored = Run.objects.create(kind='benchmark',owner='owner',job_id='e'*32,status='running')
        job = self.leased_job(stored)
        endpoint = Endpoint(base_url='https://unused.invalid/v1',api_key='never-sent')
        def completion(**kw):
            n = create.call_count
            return NS(model=kw['model'],usage=None,choices=[NS(message=NS(content='yes'))],
                      cheaper_inference={'request_id':f'paid-{n}','billing':{'status':'settled','billed_cost_usd':'0.02','currency':'USD'}})
        create = Mock(side_effect=completion)
        client = NS(chat=NS(completions=NS(create=create)))
        client.with_options = lambda **kw: client
        object.__setattr__(endpoint,'_client',client)
        # Paid answers are recorded, then a non-cancellation error ends the run.
        with patch('optimizer.benchmarks.Tracker') as tracker, patch('optimizer.benchmarks.services._endpoint_for',return_value=endpoint):
            tracker.return_value.finish.side_effect = RuntimeError('mlflow down')
            b.run(job,str(stored.id),b.validate(payload()),{'_owner':'owner'})
        stored.refresh_from_db()
        state = JobState.objects.get(pk=job.id)
        self.assertEqual((stored.status,state.status),('error','error'))
        self.assertGreater(create.call_count,0)
        expected = [f'paid-{n}' for n in range(1,create.call_count+1)]
        for result in (stored.result,state.result):
            self.assertEqual(sorted(r['request_id'] for r in result['billing']['receipts']),sorted(expected))
        # Every dispatched call records its latency.
        from .models import BenchmarkCall
        self.assertFalse(BenchmarkCall.objects.filter(job_id=job.id, duration_ms__isnull=True).exists())
        # Rows hold exactly what the final snapshot holds.
        from .benchmark_calls import assemble
        counters = {k: v for k, v in stored.result['billing'].items() if k not in ('calls','receipts')}
        rebuilt = assemble(job.id, counters)
        self.assertEqual(rebuilt['calls'],stored.result['billing']['calls'])
        self.assertEqual(rebuilt['receipts'],stored.result['billing']['receipts'])

    def test_stage_updates_never_rewrite_the_billing_snapshot(self):
        from .models import JobState
        from llm_endpoint import Endpoint
        from unittest.mock import Mock
        stored = Run.objects.create(kind='benchmark',owner='owner',job_id='6'*32,status='running')
        job = self.leased_job(stored)
        endpoint = Endpoint(base_url='https://unused.invalid/v1',api_key='never-sent')
        stages = []
        def completion(**kw):
            stages.append(JobState.objects.get(pk=job.id).stage)            # the live stage reaches the DB
            return NS(model=kw['model'],usage=None,choices=[NS(message=NS(content='yes'))],
                      cheaper_inference={'request_id':f'p{len(stages)}','billing':{'status':'settled','billed_cost_usd':'0.01','currency':'USD'}})
        client = NS(chat=NS(completions=NS(create=Mock(side_effect=completion))))
        client.with_options = lambda **kw: client
        object.__setattr__(endpoint,'_client',client)
        # A whole-snapshot save during the run could overwrite newer billing counters: forbid it.
        with patch('optimizer.benchmarks.Tracker') as tracker, patch('optimizer.benchmarks.services._endpoint_for',return_value=endpoint), \
                patch('optimizer.services._save_job_state', side_effect=AssertionError('stage must not save the whole job')):
            tracker.return_value.configure_mock(run_id='parent', experiment_id='experiment', **{'arm.return_value': 'child'})
            b.run(job,str(stored.id),b.validate(payload()),{'_owner':'owner'})
        stored.refresh_from_db()
        self.assertEqual(stored.status,'done')
        self.assertTrue(any(s.startswith('Benchmarking ·') for s in stages), stages)

    def test_lost_lease_terminates_the_mlflow_runs_it_created(self):
        from . import durable
        from .models import JobState
        from llm_endpoint import Endpoint
        from django.utils import timezone
        from datetime import timedelta
        from unittest.mock import Mock
        stored = Run.objects.create(kind='benchmark',owner='owner',job_id='4'*32,status='running')
        job = self.leased_job(stored)
        endpoint = Endpoint(base_url='https://unused.invalid/v1',api_key='never-sent')
        def completion(**kw):
            JobState.objects.filter(pk=job.id).update(lease_until=timezone.now()-timedelta(seconds=1))
            durable.claim()
            return NS(model=kw['model'],usage=None,choices=[NS(message=NS(content='yes'))])
        client = NS(chat=NS(completions=NS(create=Mock(side_effect=completion))))
        client.with_options = lambda **kw: client
        object.__setattr__(endpoint,'_client',client)
        with patch('optimizer.benchmarks.Tracker') as tracker, patch('optimizer.benchmarks.services._endpoint_for',return_value=endpoint):
            with self.assertRaises(durable.LeaseLost):
                b.run(job,str(stored.id),b.validate(payload()),{'_owner':'owner'})
        tracker.return_value.fail.assert_called_once()                        # parent and child runs terminated

    def test_oversized_final_billing_still_ends_the_run(self):
        from .models import JobState
        from .storage import StorageBudgetExceeded
        from llm_endpoint import Endpoint
        from unittest.mock import Mock
        stored = Run.objects.create(kind='benchmark',owner='owner',job_id='d'*32,status='running')
        job = self.leased_job(stored)
        endpoint = Endpoint(base_url='https://unused.invalid/v1',api_key='never-sent')
        create = Mock(side_effect=lambda **kw: NS(model=kw['model'],usage=None,choices=[NS(message=NS(content='yes'))],
                      cheaper_inference={'request_id':'paid','billing':{'status':'settled','billed_cost_usd':'0.02','currency':'USD'}}))
        client = NS(chat=NS(completions=NS(create=create)))
        client.with_options = lambda **kw: client
        object.__setattr__(endpoint,'_client',client)
        def budget(value, kind):
            if 'receipts' in (value.get('billing') or {}):
                raise StorageBudgetExceeded('too large')
            return 0
        with patch('optimizer.benchmarks.Tracker') as tracker, patch('optimizer.benchmarks.services._endpoint_for',return_value=endpoint), \
                patch('optimizer.durable.bounded_json_size',side_effect=budget):
            tracker.return_value.finish.side_effect = RuntimeError('mlflow down')
            b.run(job,str(stored.id),b.validate(payload()),{'_owner':'owner'})
        stored.refresh_from_db()
        state = JobState.objects.get(pk=job.id)
        self.assertEqual((stored.status,state.status),('error','error'))   # terminal despite the oversized result
        self.assertNotIn('calls',stored.result['billing'])                  # last live counters kept
        h = {'HTTP_AUTHORIZATION':'Bearer test','HTTP_X_CI_USER_ID':'owner'}
        detail = self.client.get(f'/api/v1/runs/{stored.id}',**h).json()['result']['billing']
        self.assertEqual(len(detail['calls']),create.call_count)             # rebuilt from rows on read
        self.assertEqual([r['request_id'] for r in detail['receipts']],['paid'])   # one id, deduplicated

    def test_original_only_queue_runs_one_variant_without_duplicate_calls(self):
        self.assert_queue_completion(include_optimized=False)

    def test_queue_completion_encrypts_credentials_and_records_billing(self):
        self.assert_queue_completion(include_optimized=True)



class ParallelCaseTests(SimpleTestCase):
    WORDS = ['alpha', 'bravo', 'charlie', 'delta']

    def manifest(self, words):
        p = payload(); p.update(evaluation_mode='rubric', models=['small'], optimized=[], criteria=['Be relevant'],
                                rows=[{'inputs': {'q': w}, 'expected_output': 'yes'} for w in words])
        return b.validate(p)

    def run_execute(self, words, create, check=lambda: None, concurrency='4'):
        from llm_endpoint import Endpoint
        from unittest.mock import Mock
        self.words = words
        manifest = self.manifest(words)
        collector = b.BillingCollector(manifest)
        endpoint = Endpoint(base_url='https://unused.invalid/v1', api_key='never-sent')
        object.__setattr__(endpoint, '_client', b.CancellationClient(NS(chat=NS(completions=NS(create=create))), check, collector))
        with patch.dict(os.environ, {'GP_BENCHMARK_CONCURRENCY': concurrency}):
            report = b.execute(manifest, endpoint, Mock(), billing=collector)
        return report, collector

    def word_in(self, kw):
        text = json.dumps(kw['messages'])
        return next(w for w in self.words if w in text)

    def answer(self, kw, word):
        if kw['model'] == 'judge':
            content = '{"criteria": [{"explanation": "ok", "evidence": null, "passed": true}]}'
        else:
            content = word
        return NS(model=kw['model'], usage=None, choices=[NS(message=NS(content=content))],
                  cheaper_inference={'request_id': f"{kw['model']}-{word}", 'billing': {'status': 'settled', 'billed_cost_usd': '0.01', 'currency': 'USD'}})

    def test_cases_overlap_keep_order_and_bill_their_own_case(self):
        import threading
        all_tasks_at_once = threading.Barrier(4, timeout=5)   # breaks if cases run one after another
        def create(**kw):
            word = self.word_in(kw)
            if kw['model'] != 'judge':
                all_tasks_at_once.wait()
            return self.answer(kw, word)
        report, collector = self.run_execute(self.WORDS, create)
        cases = report['arms'][0]['cases']
        self.assertEqual([c['output'] for c in cases], self.WORDS)                 # case order kept
        self.assertTrue(all(c['error'] is None and c['score'] == 1 for c in cases))
        final = collector.snapshot()
        for call in final['calls']:                                               # each call billed to its case
            word = call['request_id'].split('-', 1)[1]
            self.assertEqual(call['case_index'], self.WORDS.index(word) + 1, call)
            self.assertEqual(call['role'], 'task' if call['request_id'].startswith('small') else 'judge')
        bucket = final['models']['small']
        self.assertEqual((bucket['completed_answers'], bucket['priced_answers'], final['calls_with_receipts']), (4, 4, 8))

    def test_cancellation_stops_new_cases_and_waits_for_running_ones(self):
        import threading
        from .durable import JobCancelled
        cancelled, tasks = threading.Event(), []
        def check():
            if cancelled.is_set():
                raise JobCancelled('cancelled')
        def create(**kw):
            word = self.word_in(kw)
            if kw['model'] != 'judge':
                tasks.append(word)
                cancelled.set()                 # the user cancels during the first answers
            return self.answer(kw, word)
        with self.assertRaises(JobCancelled):
            self.run_execute(self.WORDS + ['echo', 'foxtrot'], create, check, concurrency='2')
        self.assertLessEqual(len(tasks), 2)                                      # no case started after the stop

    def test_work_is_queued_case_by_case_across_models_so_every_model_is_priced_early(self):
        from llm_endpoint import Endpoint
        from unittest.mock import Mock
        self.words = ['alpha', 'bravo']
        p = payload(); p.update(evaluation_mode='rubric', models=['small', 'other'], optimized=[], criteria=['Be relevant'],
                                rows=[{'inputs': {'q': w}, 'expected_output': 'yes'} for w in self.words])
        p['price_snapshot']['models']['other'] = {'input_per_million': 1, 'output_per_million': 2}
        manifest = b.validate(p)
        snapshots = []
        collector = b.BillingCollector(manifest, snapshots.append)
        endpoint = Endpoint(base_url='https://unused.invalid/v1', api_key='never-sent')
        object.__setattr__(endpoint, '_client', b.CancellationClient(NS(chat=NS(completions=NS(
            create=lambda **kw: self.answer(kw, self.word_in(kw))))), lambda: None, collector))
        with patch.dict(os.environ, {'GP_BENCHMARK_CONCURRENCY': '1'}):
            report = b.execute(manifest, endpoint, Mock(), billing=collector)
        tasks = [(c['contender_model'], c['case_index']) for c in collector.snapshot()['calls'] if c['role'] == 'task']
        self.assertEqual(tasks, [('small', 1), ('other', 1), ('small', 2), ('other', 2)])
        # After the first round (2 answers) every contender has a priced answer to project from.
        after_round = next(s for s in snapshots if sum(m['completed_answers'] for m in s['models'].values()) == 2)
        self.assertTrue(all(m['priced_answers'] == 1 for m in after_round['models'].values()))
        self.assertEqual([(a['model'], [c['output'] for c in a['cases']]) for a in report['arms']],
                         [('small', self.words), ('other', self.words)])       # arms keep case order

    def test_concurrency_setting_is_bounded(self):
        for value, expected in {None: 1, '4': 4, '0': 1, '99': 8, 'four': 1}.items():
            env = {k: v for k, v in os.environ.items() if k != 'GP_BENCHMARK_CONCURRENCY'}
            if value is not None:
                env['GP_BENCHMARK_CONCURRENCY'] = value
            with patch.dict(os.environ, env, clear=True):
                self.assertEqual(b.benchmark_concurrency(), expected, value)


class TimingCommandTests(TestCase):
    def setUp(self):
        from datetime import datetime, timedelta, timezone
        from .models import BenchmarkCall, JobState
        t0 = datetime(2026, 10, 11, 12, 0, tzinfo=timezone.utc)
        manifest = b.validate(payload())
        manifest['scoring']['judge_selection'] = {'method': 'm', 'tried': [{'model': 'judge', 'ok': True, 'ms': 4000}]}
        self.job = JobState.objects.create(id='7' * 32, owner='owner', kind='benchmark', status='done')
        self.run = Run.objects.create(kind='benchmark', owner='owner', job_id=self.job.id, status='done', config=manifest,
            result={'billing': {'reported_spend_usd': '0.03', 'models': {'small': {'total_answers': 4, 'completed_answers': 1}}}})
        # One task (10 s), then two judge calls at once (5 s each): 20 s of calls in a 15 s window.
        for seq, role, offset, ms in [(0, 'task', 0, 10000), (1, 'judge', 10, 5000), (2, 'judge', 10, 5000)]:
            BenchmarkCall.objects.create(job=self.job, seq=seq, role=role, model='small' if role == 'task' else 'judge',
                contender_model='small', variant='baseline', case_index=1,
                observed_at=(t0 + timedelta(seconds=offset)).isoformat(), duration_ms=ms,
                completed_at=t0 + timedelta(seconds=offset, milliseconds=ms))

    def test_benchmark_timings_reports_latency_and_concurrency(self):
        import io
        from django.core.management import call_command
        out = io.StringIO()
        call_command('benchmark_timings', '5', '--json', stdout=out)
        row = json.loads(out.getvalue())[0]
        self.assertEqual((row['task_call_s']['median'], row['judge_call_s']['median'], row['concurrency']), (10.0, 5.0, 1.3))
        self.assertEqual((row['calls']['task'], row['calls']['judge'], row['calls']['timed'], row['judge_check_s']), (1, 2, 3, 4.0))
        self.assertIn('to_first_call_s', row)
        self.assertEqual(row['answers'], '1/4')
        table = io.StringIO()
        call_command('benchmark_timings', stdout=table)
        self.assertIn('judge', table.getvalue().splitlines()[1])

    def test_benchmark_costs_splits_spend_by_role_and_totals_runs(self):
        import io
        from django.core.management import call_command
        from .models import BenchmarkCall
        BenchmarkCall.objects.filter(job=self.job, role='task').update(receipt_status='settled', billed_cost_usd='0.02')
        BenchmarkCall.objects.filter(job=self.job, role='judge').update(receipt_status='settled', billed_cost_usd='0.005')
        self.run.result['billing'].update(calls_with_receipts=3, calls_missing_receipts=0)
        self.run.save()
        out = io.StringIO()
        call_command('benchmark_costs', '5', '--json', stdout=out)
        row = json.loads(out.getvalue())[0]
        self.assertEqual((row['spend_usd'], row['task_usd'], row['judge_usd'], row['per_answer_usd'], row['unpriced_calls']),
                         ('0.03', '0.02', '0.010', '0.030000', 0))
        table = io.StringIO()
        call_command('benchmark_costs', stdout=table)
        self.assertIn('total over 1 runs: $0.0300', table.getvalue())
        Run.objects.create(kind='benchmark', owner='owner', job_id='5' * 32, status='error', config={}, result={})
        partial = io.StringIO()
        call_command('benchmark_costs', stdout=partial)
        self.assertIn('total over 1 of 2 fully priced runs (1 excluded; CI prices them): $0.0300', partial.getvalue())

    def test_benchmark_costs_shows_na_not_zero_for_calls_ci_prices(self):
        import io
        from django.core.management import call_command
        self.run.result['billing'].update(reported_spend_usd='0', calls_with_receipts=0, calls_missing_receipts=3)
        self.run.save()
        table = io.StringIO()
        call_command('benchmark_costs', stdout=table)
        text = table.getvalue()
        self.assertNotIn('$0.0000', text)
        self.assertIn('n/a', text.splitlines()[1])
        self.assertIn('total: n/a', text)

    def test_watch_benchmark_rejects_an_ambiguous_prefix(self):
        from django.core.management import call_command, CommandError
        from .models import JobState
        JobState.objects.create(id='7' * 31 + '0', owner='owner', kind='benchmark', status='done')
        with self.assertRaisesRegex(CommandError, 'matches several benchmarks'):
            call_command('watch_benchmark', '777', '--interval', '0')
        with self.assertRaisesRegex(CommandError, 'No matching benchmark'):
            call_command('watch_benchmark', 'nope', '--interval', '0')

    def test_concurrency_ignores_idle_gaps(self):
        from datetime import timedelta
        from .benchmark_timing import summarize
        from .models import BenchmarkCall
        BenchmarkCall.objects.filter(job=self.job).delete()
        t0 = self.run.created_at
        # Strictly sequential calls with idle gaps between them: concurrency must be 1.0.
        for seq, offset in enumerate([0, 30, 90]):
            BenchmarkCall.objects.create(job=self.job, seq=seq, role='task', model='small', contender_model='small',
                variant='baseline', case_index=seq + 1, observed_at=(t0 + timedelta(seconds=offset)).isoformat(),
                duration_ms=10000, completed_at=t0 + timedelta(seconds=offset + 10))
        self.assertEqual(summarize(self.run)['concurrency'], 1.0)

    def test_cost_split_falls_back_to_receipts_and_is_unavailable_without_history(self):
        from .benchmark_timing import cost
        older = Run.objects.create(kind='benchmark', owner='owner', job_id='8' * 32, status='done', config={'models': ['small']},
            result={'billing': {'reported_spend_usd': '0.05', 'models': {'small': {'total_answers': 1, 'completed_answers': 1}},
                                'receipts': [{'role': 'task', 'billed_cost_usd': '0.04'}, {'role': 'judge', 'billed_cost_usd': '0.01'}]}})
        row = cost(older)                                                        # no rows: stored receipts
        self.assertEqual((row['spend_usd'], row['task_usd'], row['judge_usd']), ('0.05', '0.04', '0.01'))
        del older.result['billing']['receipts']
        row = cost(older)                                                        # no rows, no receipts
        self.assertEqual((row['spend_usd'], row['task_usd'], row['judge_usd']), ('0.05', None, None))

    def test_watch_benchmark_prints_finished_calls_and_stops_when_the_run_has_ended(self):
        import io
        from django.core.management import call_command
        out = io.StringIO()
        call_command('watch_benchmark', self.job.id[:8], '--interval', '0', stdout=out)
        lines = out.getvalue().splitlines()
        self.assertEqual([line.split()[0] for line in lines[2:5]], ['0', '1', '2'])
        self.assertIn('10.0s', lines[2])
        self.assertTrue(lines[-1].startswith('ended: done'))


class TaskFailureTests(SimpleTestCase):
    """A failed model call is the configuration's failure: an evaluated Fail, not a gap."""

    def run_arm(self, task, mode='rubric', criteria=('Exactly 4 items', 'Be relevant')):
        from llm_endpoint import Endpoint
        from types import SimpleNamespace as NS
        from unittest.mock import Mock, patch
        p = payload(); p.update(evaluation_mode=mode, models=['small'], optimized=[], criteria=list(criteria),
                                rows=[{'inputs': {'q': 'a'}, 'expected_output': 'yes'}, {'inputs': {'q': 'b'}, 'expected_output': 'yes'}])
        endpoint = Endpoint(base_url='https://unused.invalid/v1', api_key='never-sent')
        calls = {'task': 0}
        def create(**kw):
            if kw['model'] == 'judge':
                if mode == 'reference':
                    content = '{"rationale": "Same", "match": true}'
                else:
                    content = '{"items": [' + ','.join('{"item": %d, "explanation": "ok", "passed": true}' % i for i in range(1, 5)) + ']}' \
                        if 'Items (' in kw['messages'][0]['content'] else \
                        '{"criteria": [{"explanation": "ok", "evidence": null, "passed": true}]}'
                return NS(model='judge', usage=None, choices=[NS(message=NS(content=content))])
            calls['task'] += 1
            return task(calls['task'], kw)
        with patch.object(endpoint.client.chat.completions, 'create', side_effect=create):
            return b.execute(b.validate(p), endpoint, Mock())['arms'][0]

    @staticmethod
    def answer(content='["a","b","c","d"]'):
        from types import SimpleNamespace as NS
        return NS(model='small', usage=NS(prompt_tokens=10, completion_tokens=20, total_tokens=30, completion_tokens_details=None),
                  choices=[NS(message=NS(content=content))])

    @staticmethod
    def timeout():
        import httpx, openai
        return openai.APITimeoutError(request=httpx.Request('POST', 'https://unused.invalid/v1/chat/completions'))

    def test_timeout_is_an_evaluated_fail_and_cost_uses_answered_calls(self):
        def task(n, kw):
            if n == 2:
                raise self.timeout()
            return self.answer()
        arm = self.run_arm(task)
        q, failed = arm['quality']['rubric'], arm['cases'][1]
        self.assertEqual((q['n'], q['evaluated'], q['evaluation_errors'], q['passed'], q['failed']), (2, 2, 0, 1, 1))
        self.assertEqual(q['pass_rate'], .5)
        self.assertEqual(q['mean_score'], .5)                       # no answer earns no partial credit
        self.assertIsNone(failed['error'])
        self.assertEqual(failed['task_error'], 'APITimeoutError')
        self.assertEqual(failed['evaluations']['rubric'], {'score': 0.0, 'error': None, 'overall_pass': False, 'task_error': 'APITimeoutError'})
        self.assertIs(failed['rubric_evaluation']['overall_pass'], False)
        self.assertEqual([c['status'] for c in failed['rubric_evaluation']['criteria']], ['fail', 'fail'])
        self.assertEqual(arm['errors'], 0)
        self.assertEqual((arm['usage']['measured_cases'], arm['usage']['total_cases']), (1, 2))
        self.assertEqual((arm['usage']['avg_input'], arm['usage']['avg_output']), (10, 20))
        self.assertEqual(arm['estimated_cost_per_call'], (10 * 1 + 20 * 2) / 1_000_000)

    def test_reference_mode_timeout_is_a_fail(self):
        def task(n, kw):
            if n == 1:
                raise self.timeout()
            return self.answer('yes')
        arm = self.run_arm(task, mode='reference')
        q = arm['quality']['reference']
        self.assertEqual((q['evaluated'], q['passed'], q['failed'], q['evaluation_errors']), (2, 1, 1, 0))
        self.assertEqual(arm['cases'][0]['score'], 0.0)

    def test_non_provider_errors_stay_evaluation_errors(self):
        def task(n, kw):
            if n == 2:
                raise KeyError('template bug')                       # our failure, not the model's
            return self.answer()
        arm = self.run_arm(task)
        q = arm['quality']['rubric']
        self.assertEqual((q['evaluated'], q['evaluation_errors']), (1, 1))
        self.assertEqual(arm['cases'][1]['error'], 'KeyError')
        self.assertNotIn('task_error', arm['cases'][1])

    def test_cost_needs_at_least_one_answered_call(self):
        s = b.summarize([{'graded_by': 'rubric', 'score': 0.0, 'error': None, 'usage': None}],
                        {'input_per_million': 1, 'output_per_million': 2})
        self.assertIsNone(s['estimated_cost_per_call'])
        self.assertEqual(s['usage']['measured_cases'], 0)
