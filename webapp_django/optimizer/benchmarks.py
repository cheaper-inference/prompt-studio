"""Versioned benchmarks around the existing evaluator; no optimization behavior changes.

All tracking uses explicit MlflowClient run IDs, never process-global active runs.
Customer authorization stays in Django. Tracking artifacts contain no credentials.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

import sentry_sdk

from . import services

SCHEMA = 'ps-benchmark-v1'


class BenchmarkCancelled(Exception):
    pass


def ci_receipt(response):
    """CI's per-response receipt: {'request_id', 'billing'}, or {} when absent.
    Since CI's KEA-7852 (2026-10-13) both keys sit at the top level of the body; before
    that they were nested under `cheaper_inference`. Both shapes are read."""
    legacy = getattr(response, 'cheaper_inference', None)
    if isinstance(legacy, dict):
        return legacy
    bill = getattr(response, 'billing', None)
    if not isinstance(bill, dict):
        extra = getattr(response, 'model_extra', None) or {}
        bill = extra.get('billing') if isinstance(extra, dict) else None
    if not isinstance(bill, dict):
        return {}
    request_id = getattr(response, 'request_id', None)
    if not isinstance(request_id, str):
        extra = getattr(response, 'model_extra', None) or {}
        request_id = extra.get('request_id') if isinstance(extra, dict) else None
    return {'request_id': request_id, 'billing': bill}


def is_transient(exc):
    """Provider-side or network failures worth one retry; never 4xx request errors."""
    import openai
    if isinstance(exc, (openai.APITimeoutError, openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError)):
        return True
    status = getattr(exc, 'status_code', None)
    return isinstance(status, int) and status >= 500


class CancellationClient:
    """Check durable cancellation before every task/judge request, including repairs."""
    def __init__(self, client, check, billing=None):
        from types import SimpleNamespace
        self._client = client
        def attempt(*args, **kwargs):
            check()
            request_id = None
            call_index = billing.begin_call(kwargs.get("model")) if billing is not None else None
            import time
            started = time.monotonic()   # dispatch; measured before the collector lock
            elapsed_ms = lambda: round((time.monotonic() - started) * 1000)
            arrived = lambda: datetime.now(timezone.utc)   # response time, before the collector lock
            try:
                completions = client.chat.completions
                raw_api = getattr(completions, 'with_raw_response', None)
                if billing is not None and raw_api is not None:
                    raw = raw_api.create(*args, **kwargs)
                    request_id = raw.headers.get('x-ci-request-id')
                    response = raw.parse()
                else:
                    response = completions.create(*args, **kwargs)
            except Exception as exc:
                if billing is not None:
                    headers = getattr(getattr(exc, 'response', None), 'headers', {})
                    billing.record(None, kwargs.get('model'), request_id or headers.get('x-ci-request-id'), call_index,
                                   duration_ms=elapsed_ms(), completed_at=arrived())
                raise
            if billing is not None:
                billing.record(response, kwargs.get('model'), request_id, call_index, duration_ms=elapsed_ms(),
                               completed_at=arrived())
            return response
        def create(*args, **kwargs):
            # Judge calls get one retry on a transient failure (5xx, 429, timeout, connection).
            # Task calls are never retried: a repeated answer is a different sample.
            # Each attempt is a separately recorded, cancellation-checked call.
            judge = (kwargs.get('extra_headers') or {}).get('x-gepa-purpose') == 'training_judge'
            try:
                return attempt(*args, **kwargs)
            except Exception as exc:
                if not (judge and is_transient(exc)):
                    raise
            return attempt(*args, **kwargs)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

    def __getattr__(self, name):
        return getattr(self._client, name)




class BillingCollector:
    """Response receipts only: never confuse absent/failed receipts with free calls.

    `persist(op, snapshot)`, when given, writes one call's row change (`op(job_id)`)
    together with the live counters, in one lease-fenced transaction. Live snapshots
    carry counters only; the full `calls`/`receipts` lists appear once, at the end."""
    def __init__(self, manifest, publish=lambda snapshot: None, persist=None):
        import threading
        # An answer's independent judge calls run concurrently: one lock keeps call
        # numbering, counters and their row writes consistent and in order.
        self._lock = threading.RLock()
        self.publish = publish
        self.persist = persist
        self.receipts = []
        self.calls = []
        self.call_ids = set()
        self.context = manifest.get('billing_context')
        self.seen = set()
        self.missing = 0
        self.spend = Decimal('0')
        self.answer_spend = {}
        self.unpriced_answers = set()
        import contextvars
        # The call context (contender, variant, case, role) is per case: cases may run
        # concurrently, and a case's judge threads inherit it via contextvars.copy_context.
        self._context = contextvars.ContextVar(f'billing_context_{id(self)}',
            default={'model': None, 'variant': None, 'case_index': None, 'role': 'task'})
        self.models = {model: {'total_answers': len(manifest['variants']) * len(manifest['dataset']['rows']),
                              'completed_answers': 0, 'priced_answers': 0, 'reported_spend_usd': '0',
                              'completed_spend_usd': '0'} for model in manifest['models']}

    def _context_property(name):
        return property(lambda self: self._context.get()[name],
                        lambda self, value: self._context.set({**self._context.get(), name: value}))
    model = _context_property('model')
    variant = _context_property('variant')
    case_index = _context_property('case_index')
    role = _context_property('role')
    del _context_property

    def _emit(self, op=None):
        snapshot = self.snapshot(live=True)
        if self.persist is not None:
            self.persist(op, snapshot)
        else:
            self.publish(snapshot)

    def _answer(self):
        return (self.model, self.variant, self.case_index)

    def begin_call(self, model):
        with self._lock:
            return self._begin_call(model)

    def _begin_call(self, model):
        call = {'request_id': None, 'role': self.role, 'model': model,
            'contender_model': self.model, 'variant': self.variant, 'case_index': self.case_index,
            'completed': False, 'in_flight': True, 'observed_at': datetime.now(timezone.utc).isoformat()}
        self.calls.append(call)
        seq = len(self.calls) - 1
        from . import benchmark_calls
        self._emit(lambda job_id: benchmark_calls.insert(job_id, seq, call))
        return seq

    def record(self, response, returned_model, header_request_id=None, call_index=None, duration_ms=None,
               completed_at=None):
        with self._lock:
            return self._record(response, returned_model, header_request_id, call_index, duration_ms, completed_at)

    def _record(self, response, returned_model, header_request_id=None, call_index=None, duration_ms=None,
                completed_at=None):
        from . import benchmark_calls
        receipt = ci_receipt(response)
        bill = receipt.get('billing')
        bill = bill if isinstance(bill, dict) else {}
        candidate_id = header_request_id or receipt.get('request_id')
        # Bounded like the durable inference path (BenchmarkCall.request_id is varchar(128)):
        # an oversized ID is treated as missing rather than breaking the fenced write.
        request_id = candidate_id if isinstance(candidate_id, str) and 0 < len(candidate_id) <= 128 else None
        ops = []
        # Keep correlation even when the response carries no inline price.
        if call_index is not None:
            self.calls[call_index].update(request_id=request_id, in_flight=False)
            if request_id:
                self.call_ids.add(request_id)
            seq = call_index
            # The caller measures latency around the provider call (excluding lock waits);
            # fall back to wall time since begin_call for direct callers.
            if duration_ms is None:
                try:
                    started = datetime.fromisoformat(self.calls[call_index]['observed_at'])
                    duration_ms = max(0, round((datetime.now(timezone.utc) - started).total_seconds() * 1000))
                except (KeyError, TypeError, ValueError):
                    duration_ms = None
            completed_at = completed_at or datetime.now(timezone.utc)
            ops.append(lambda job_id: benchmark_calls.update(job_id, seq, request_id=request_id, in_flight=False,
                                                             duration_ms=duration_ms, completed_at=completed_at))
        elif not request_id or request_id not in self.call_ids:
            if request_id:
                self.call_ids.add(request_id)
            call = {'request_id': request_id, 'role': self.role,
                'model': returned_model, 'contender_model': self.model,
                'variant': self.variant, 'case_index': self.case_index, 'completed': False,
                'observed_at': datetime.now(timezone.utc).isoformat()}
            self.calls.append(call)
            seq = len(self.calls) - 1
            ops.append(lambda job_id: benchmark_calls.insert(job_id, seq, call))
        else:
            seq = None
        valid = isinstance(request_id, str) and bool(request_id) and bill.get('currency') == 'USD' and bill.get('status') in ('settled', 'cached')
        try:
            amount = Decimal(str(bill.get('billed_cost_usd')))
            # BenchmarkCall.billed_cost_usd is varchar(40): an amount that cannot be stored is not a receipt.
            valid = valid and amount.is_finite() and amount >= 0 and len(str(amount)) <= 40
        except (InvalidOperation, ValueError):
            valid = False
        if not valid:
            self.missing += 1
            self.unpriced_answers.add(self._answer())
        elif request_id not in self.seen:
            self.seen.add(request_id)
            self.spend += amount
            self.answer_spend[self._answer()] = self.answer_spend.get(self._answer(), Decimal('0')) + amount
            bucket = self.models.get(self.model)
            if bucket is not None:
                bucket['reported_spend_usd'] = str(Decimal(bucket['reported_spend_usd']) + amount)
            self.receipts.append({'request_id': request_id, 'status': bill['status'],
                'billed_cost_usd': str(amount), 'currency': 'USD', 'role': self.role,
                'model': returned_model, 'contender_model': self.model,
                'variant': self.variant, 'case_index': self.case_index})
            if seq is None:   # a repeat of an earlier unpriced call: price that row
                seq = next(i for i, call in enumerate(self.calls) if call['request_id'] == request_id)
            if seq is not None:
                status, cost = bill['status'], str(amount)
                ops.append(lambda job_id: benchmark_calls.update(job_id, seq, receipt_status=status,
                                                                 receipt_model=returned_model, billed_cost_usd=cost))
        self._emit(lambda job_id: [op(job_id) for op in ops])

    def finish_answer(self):
        with self._lock:
            return self._finish_answer()

    def _finish_answer(self):
        from . import benchmark_calls
        bucket = self.models[self.model]
        answer = self._answer()
        for call in self.calls:
            if (call['contender_model'], call['variant'], call['case_index']) == answer:
                call['completed'] = True
        bucket['completed_answers'] += 1
        # Same rule as CI's ledger reconciliation: only fully priced answers sample the projection.
        if answer not in self.unpriced_answers:
            bucket['priced_answers'] += 1
            bucket['completed_spend_usd'] = str(Decimal(bucket['completed_spend_usd']) + self.answer_spend.get(answer, Decimal('0')))
        self._emit(lambda job_id: benchmark_calls.complete_answer(job_id, *answer))

    def snapshot(self, live=False):
        with self._lock:
            return self._snapshot(live)

    def _snapshot(self, live=False):
        """Live snapshots (every call) carry counters only: the poll stays small and CI
        projects cost from them. The final snapshot, written once, carries `calls`
        (for CI's ledger reconciliation) and `receipts`."""
        snap = {'reported_spend_usd': str(self.spend), 'currency': 'USD',
                'calls_with_receipts': len(self.receipts), 'calls_missing_receipts': self.missing + sum(bool(call.get('in_flight')) for call in self.calls),
                'coverage': 'response_receipts',
                'models': {model: dict(bucket) for model, bucket in self.models.items()},
                'billing_context': self.context}
        if not live:
            snap['calls'] = [dict(call) for call in self.calls]
            snap['receipts'] = list(self.receipts)
        return snap


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def text(value, name, limit=100_000):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f'{name} must be non-empty text (up to {limit} characters)')
    return value


def validate(payload):
    """Allowlisted immutable manifest. Runtime credentials never enter the manifest."""
    from prompt_optimizer import p_variables
    variants = {}
    for name in ('baseline', 'optimized'):
        messages = payload.get(name)
        if name == 'optimized' and (messages is None or messages == []):
            continue
        if not isinstance(messages, list) or not 1 <= len(messages) <= 20:
            raise ValueError(f'{name} requires 1–20 messages')
        variants[name] = []
        for message in messages:
            if not isinstance(message, dict) or message.get('role') not in ('system', 'user'):
                raise ValueError('Messages require a system or user role')
            variants[name].append({'role': message['role'], 'content': text(message.get('content'), 'content')})
    models = payload.get('models')
    if not isinstance(models, list) or not 1 <= len(models) <= 3:
        raise ValueError('Choose 1–3 contender models')
    models = list(dict.fromkeys(text(m, 'model', 200) for m in models))
    rows = payload.get('rows')
    if not isinstance(rows, list) or not 1 <= len(rows) <= 50:
        raise ValueError('Provide 1–50 evaluation rows')
    variables = set().union(*(p_variables(m['content']) for ms in variants.values() for m in ms))
    normalized = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get('inputs'), dict):
            raise ValueError('Each row requires an inputs object')
        inputs = row['inputs']
        if set(inputs) != variables or any(not isinstance(v, str) or len(v) > 100_000 for v in inputs.values()):
            raise ValueError(f'Every row must supply exactly these input variables as text: {sorted(variables)}')
        if variables & {'expected_output', '_benchmark_context'}:
            raise ValueError('expected_output and _benchmark_context are reserved variable names')
        reference = row.get('expected_output')
        if isinstance(reference, list):
            if (len(reference) > 1000 or any(not isinstance(item, str) for item in reference)
                    or len(json.dumps(reference, ensure_ascii=False, separators=(',', ':'))) > 100_000):
                raise ValueError('expected_output must be a string or an array of strings (up to 100,000 characters)')
            reference = list(reference)
        elif reference is not None:
            reference = text(reference, 'expected_output')
        normalized.append({'inputs': dict(inputs), 'expected_output': reference})
    # Duplicate inputs make binomial intervals misleading and invite split leakage.
    hashes = [digest(r['inputs']) for r in normalized]
    if len(set(hashes)) != len(hashes):
        raise ValueError('Duplicate test inputs are not allowed; use distinct cases')
    criteria = payload.get('criteria', [])
    if not isinstance(criteria, list) or len(criteria) > 20:
        raise ValueError('Provide at most 20 rubric criteria')
    criteria = [text(c, 'criterion', 1000) for c in criteria]
    # Countable/format criteria are compiled to mandatory code checks. Explicit client
    # checks (legacy `deterministic_checks`) are ignored: the compiled rubric supersedes them.
    from ps_utils.rubric_parser import compile_rubric, VERSION as RUBRIC_COMPILER
    compiled = compile_rubric(criteria)
    evaluation_mode = payload.get('evaluation_mode', 'auto')
    if evaluation_mode not in ('reference', 'rubric', 'both', 'auto'):
        raise ValueError('Unsupported evaluation mode')
    if evaluation_mode in ('reference', 'both'):
        for index, row in enumerate(normalized):
            if row['expected_output'] is None:
                raise ValueError(f'Add expected_output to case {index + 1}')
    if evaluation_mode in ('rubric', 'both') and not criteria:
        raise ValueError('Add at least one quality criterion')
    if evaluation_mode == 'auto' and any(r['expected_output'] is None for r in normalized) and not criteria:
        raise ValueError('Rows without expected outputs require explicit quality criteria')
    scorer = payload.get('label_scorer', 'equivalence')
    if scorer not in ('equivalence', 'correctness'):
        raise ValueError('Unsupported label scorer')
    effort = payload.get('reasoning_effort', 'minimal')
    if effort not in ('minimal', 'low', 'medium', 'high'):
        raise ValueError('Unsupported reasoning effort')
    if type(payload.get('holdout_confirmed')) is not bool:
        raise ValueError('holdout_confirmed must be a boolean')
    held_out = payload['holdout_confirmed']
    price = payload.get('price_snapshot', {})
    if not isinstance(price, dict) or not isinstance(price.get('models', {}), dict):
        raise ValueError('Invalid price snapshot')
    rates = {}
    for model in models:
        rate = price.get('models', {}).get(model, {})
        if not isinstance(rate, dict):
            raise ValueError('Invalid model price')
        rates[model] = {}
        for key in ('input_per_million', 'output_per_million'):
            value = rate.get(key)
            if value is not None and (type(value) not in (float, int) or not math.isfinite(value) or value < 0):
                raise ValueError('Prices must be non-negative finite numbers or null')
            rates[model][key] = value
    judge_model = text(payload.get('judge_model'), 'judge model', 200)
    if judge_model in models:
        # A contender grading its own answers inflates its score (self-preference).
        raise ValueError('The judge model must differ from every contender model')
    manifest = {'schema_version': SCHEMA, 'name': text(payload.get('name', 'Prompt benchmark'), 'name', 100),
                'variants': variants, 'prompt_hashes': {k: digest(v) for k, v in variants.items()},
                'models': models, 'dataset': {'name': text(payload.get('dataset_name'), 'dataset name', 100),
                'digest': digest(normalized), 'rows': normalized, 'split': 'test' if held_out else 'validation',
                'holdout_status': 'user_attested' if held_out else 'not_attested', 'input_hashes': hashes},
                'scoring': {'evaluation_mode': evaluation_mode, 'label_scorer': scorer, 'criteria': criteria,
                            'rubric_compiler': RUBRIC_COMPILER, 'compiled_rubric': compiled,
                            'judge_selection': judge_selection(payload.get('judge_selection')),
                            'judge_model': judge_model,
                            'version': SCHEMA},
                'inference': {'reasoning_effort': effort},
                'price_snapshot': {'captured_at': price.get('captured_at'), 'source': 'ci_catalog',
                                   'currency': 'USD', 'models': rates},
                'cost_method': 'uncached_list_price_v1', 'techniques': [],
                'request_headers': {'x-ci-techniques': 'off', 'x-ci-prompt-cache': 'off'}}
    context = payload.get('billing_context')
    if context is not None:
        if not isinstance(context, dict) or context.get('mode') not in ('development', 'selected_key'):
            raise ValueError('Invalid billing context')
        manifest['billing_context'] = {'mode': context['mode']}
        if context['mode'] == 'selected_key':
            key_id = context.get('api_key_id')
            if not isinstance(key_id, str):
                raise ValueError('Invalid billing key ID')
            manifest['billing_context']['api_key_id'] = str(uuid.UUID(key_id))
    provenance = payload.get('synthetic_provenance')
    if provenance is not None:
        if not isinstance(provenance, dict) or len(json.dumps(provenance, ensure_ascii=False)) > 20000:
            raise ValueError('Invalid synthetic provenance')
        # Informational CI receipt, never an attestation of dataset independence.
        manifest['dataset']['synthetic_provenance'] = {**provenance, 'holdout_status': 'unverified'}
        manifest['dataset']['split'] = 'validation'
        manifest['dataset']['holdout_status'] = 'not_attested'
    return manifest


def judge_selection(value):
    """CI's judge health-check record (informational); bounded and shape-checked."""
    if not isinstance(value, dict) or len(json.dumps(value, ensure_ascii=False)) > 5000:
        return None
    tried = value.get('tried')
    if not isinstance(tried, list) or not all(isinstance(t, dict) and isinstance(t.get('model'), str) for t in tried):
        return None
    return {'method': str(value.get('method', ''))[:100],
            'tried': [{'model': t['model'][:200], 'ok': t.get('ok') is True,
                       **({'reason': str(t['reason'])[:100]} if t.get('reason') else {}),
                       **({'ms': t['ms']} if isinstance(t.get('ms'), (int, float)) else {}),
                       **({'calls': t['calls']} if type(t.get('calls')) is int else {})} for t in tried[:10]]}


def interval(passed, n):
    if not n:
        return None
    z = 1.959963984540054
    p = passed / n
    d = 1 + z*z/n
    center = (p + z*z/(2*n)) / d
    radius = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return [max(0, center-radius), min(1, center+radius)]


def bootstrap_mean_interval(scores, resamples=2000, seed=0):
    """Percentile bootstrap 95% interval of the mean case score. A fixed seed makes
    re-running the same report reproduce the same interval. None below 2 cases."""
    import random
    if len(scores) < 2:
        return None
    rng, n = random.Random(seed), len(scores)
    means = sorted(sum(rng.choice(scores) for _ in range(n)) / n for _ in range(resamples))
    return [means[int(.025 * resamples)], means[int(.975 * resamples) - 1]]


def criterion_rates(cases):
    """Pass rate per rubric criterion across cases (criterion order preserved).
    A criterion counts when it has a usable verdict, even if another criterion of the
    same case errored; errors are counted separately, never as failures."""
    rows = {}
    for case in cases:
        for index, detail in enumerate((case.get('rubric_evaluation') or {}).get('criteria', [])):
            row = rows.setdefault(index, {'criterion': detail.get('criterion'),
                                          'method': detail.get('compiled_method') or detail.get('method'),
                                          'n': 0, 'evaluated': 0, 'passed': 0})
            row['n'] += 1
            if detail.get('passed') is not None:
                row['evaluated'] += 1
                row['passed'] += detail['passed'] is True
    result = []
    for index in sorted(rows):
        row = rows[index]
        row.update(evaluation_errors=row['n'] - row['evaluated'],
                   pass_rate=row['passed'] / row['evaluated'] if row['evaluated'] else None,
                   interval_95=interval(row['passed'], row['evaluated']))
        result.append(row)
    return result


def record_strict_fail(case, evaluation, exc):
    """A judge error on an answer that already failed a criterion cannot change the
    all-criteria verdict: keep the case as an evaluated strict fail. Its partial credit
    is unknown, so the score stays None and the case is left out of the mean."""
    if case.get('output') is None or (case.get('rubric_evaluation') or {}).get('overall_pass') is not False:
        return False
    evaluation.update(score=None, error=None, overall_pass=False, judge_error=type(exc).__name__)
    return True


def is_task_failure(case, exc):
    """The model call itself failed (timeout, connection, 5xx, rate limit, other provider
    error) and returned no answer. That is the configuration's failure, not the
    benchmark's: a real user would get no answer either."""
    import openai
    return case.get('output') is None and isinstance(exc, openai.APIError)


def record_task_failure(case, kinds, compiled, exc):
    """Count a failed task call as an evaluated Fail with no partial credit."""
    name = type(exc).__name__
    for kind in kinds:
        case['evaluations'][kind] = {'score': 0.0, 'error': None, 'overall_pass': False, 'task_error': name}
    if len(kinds) == 1:
        case['score'] = 0.0
    case.update(error=None, task_error=name,
                error_detail=f'The model returned no answer ({name}); counted as a failed answer.')
    if 'rubric' in kinds and compiled:
        note = f'No answer: the model call failed ({name}).'
        case['rubric_evaluation'] = {
            'criteria': [{'criterion': entry['criterion'], 'passed': False, 'status': 'fail', 'method': 'deterministic',
                          'compiled_method': entry['method'], 'explanation': note, 'evidence': None} for entry in compiled],
            'passed': 0, 'total': len(compiled), 'aggregation': 'all_criteria', 'overall_pass': False,
            'evaluation_errors': 0, 'task_error': name}


def summarize(cases, rate):
    metrics = {}
    for kind in ('reference', 'rubric'):
        group = [c['evaluations'][kind] if 'evaluations' in c else c for c in cases
                 if kind in c.get('evaluations', {}) or ('evaluations' not in c and c['graded_by'] == kind)]
        scored = [c for c in group if not c.get('error') and c.get('score') is not None]
        # Strict rate also counts known fails whose partial credit is unknown.
        evaluated = scored + [c for c in group if not c.get('error') and c.get('score') is None and c.get('overall_pass') is False]
        passed = sum(c.get('score') == 1 for c in scored)
        metrics[kind] = {'n': len(group), 'passed': passed,
                         'evaluated': len(evaluated), 'evaluation_errors': len(group)-len(evaluated), 'failed': len(evaluated)-passed,
                         'pass_rate': passed/len(evaluated) if evaluated else None,
                         'interval_95': interval(passed, len(evaluated)),
                         'mean_cases': len(scored),
                         'mean_score': sum(c['score'] for c in scored)/len(scored) if scored else None,
                         'mean_interval_95': bootstrap_mean_interval([c['score'] for c in scored])}
    metrics['rubric']['criteria'] = criterion_rates(cases)
    # Cost per answered call: a call that returned no usage (a timeout) is left out of the
    # average and shown through usage.measured_cases, instead of making cost unavailable.
    measured = [c for c in cases if c.get('usage') is not None]
    complete = bool(measured)
    avg_in = sum(c['usage']['input'] for c in measured)/len(measured) if complete else None
    avg_out = sum(c['usage']['output'] for c in measured)/len(measured) if complete else None
    cost = None
    if complete and all(rate.get(k) is not None for k in ('input_per_million', 'output_per_million')):
        cost = (avg_in*rate['input_per_million'] + avg_out*rate['output_per_million']) / 1_000_000
    latencies = sorted(c['latency_s'] for c in cases if c.get('latency_s') is not None)
    return {'quality': metrics, 'errors': sum(c.get('error') is not None for c in cases),
            'usage': {'measured_cases': len(measured), 'total_cases': len(cases),
                      'avg_input': avg_in, 'avg_output': avg_out}, 'estimated_cost_per_call': cost,
            'latency_p95_s': latencies[math.ceil(.95*len(latencies))-1] if latencies else None}


class Tracker:
    """Run identifiers for a benchmark report. Nothing is written to MLflow: no one read the runs back
    (CI, the views and the cost/timing commands all read GP's own Run/BenchmarkCall rows), and the
    store only grew. CI's report schema still requires `mlflow.run_id`, `mlflow.experiment_id` and
    each arm's `mlflow_run_id` (React keys, frontier ids), so GP generates them; reports written
    before this change keep their real MLflow ids."""
    def __init__(self, owner, benchmark_id, manifest):
        self.experiment_id = 'ci-benchmarks-' + digest(owner)[:24]
        self.run_id = uuid.uuid4().hex
        manifest['runtime'] = {'benchmark_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                              'evaluator_sha256': hashlib.sha256((Path(__file__).resolve().parents[2] / 'prompt_optimizer.py').read_bytes()).hexdigest()}

    def arm(self, model, variant, manifest):
        return uuid.uuid4().hex

    def finish_arm(self, run_id, arm):
        pass

    def finish(self, report):
        pass

    def fail(self, status='FAILED'):
        pass


def benchmark_concurrency() -> int:
    """GP_BENCHMARK_CONCURRENCY: benchmark cases (across all arms) run at once (default 1, range 1-8)."""
    try:
        return min(8, max(1, int(os.environ.get('GP_BENCHMARK_CONCURRENCY', '1'))))
    except ValueError:
        return 1


def _run_cases(items, run_case, concurrency):
    """Run work items (`run_case(*item)`), up to `concurrency` at once; results in item order.

    Each case runs in a copy of the caller's context and closes its own database
    connections. A fatal error (cancellation, lost lease, an unexpected bug) stops new
    cases from starting; cases already in flight finish, then the first error is raised.
    """
    if concurrency <= 1 or len(items) <= 1:
        return [run_case(*item) for item in items]
    import contextvars
    import threading
    from concurrent.futures import ThreadPoolExecutor
    stop = threading.Event()

    def task(item):
        try:
            if stop.is_set():
                return None
            return run_case(*item)
        except BaseException:
            stop.set()
            raise
        finally:
            try:
                from django.db import connections
                connections.close_all()   # this worker thread's connections only
            except Exception:  # noqa: BLE001
                pass

    with ThreadPoolExecutor(max_workers=min(concurrency, len(items))) as pool:
        futures = [pool.submit(contextvars.copy_context().run, task, item) for item in items]
        results, fatal = [], None
        for future in futures:
            try:
                results.append(future.result())
            except BaseException as error:  # noqa: BLE001 - re-raised below, first one wins
                fatal = fatal or error
    if fatal is not None:
        raise fatal
    return results


def execute(manifest, endpoint, tracker, stage=lambda message: None, billing=None):
    """Use PS's existing scorer. Separate evaluation coverage from answer quality."""
    from .durable import JobCancelled, LeaseLost
    from prompt_optimizer import OptimizationConfig, Mode, Prompt, PromptOptimizer, UsageMeter, _usage_fields, _scorer_value
    import threading
    arms = []
    completed = 0
    completed_lock = threading.Lock()
    concurrency = benchmark_concurrency()
    total = len(manifest['models']) * len(manifest['variants']) * len(manifest['dataset']['rows'])
    def progress(detail):
        stage(f'Benchmarking · {completed} of {total} answers evaluated · {detail}')
    progress('Starting')
    # One pool for every arm, queued case by case (case 1 of every model and variant, then
    # case 2, ...): each contender gets a priced answer early, so CI can project the total
    # within the first answers; no worker idles between arms; and a case's variants run
    # close together in time.
    arm_specs = [(model, variant, messages) for model in manifest['models']
                 for variant, messages in manifest['variants'].items()]
    run_ids = [tracker.arm(model, variant, manifest) for model, variant, _ in arm_specs]
    def run_case(arm_index, index, row):
        nonlocal completed
        model, variant, messages = arm_specs[arm_index]
        if billing is not None:
            billing.model, billing.variant, billing.case_index, billing.role = model, variant, index + 1, 'task'
        variant_label = 'Original' if variant == 'baseline' and len(manifest['variants']) == 1 else variant
        context_label = f'{variant_label} · {model} · Case {index+1} of {len(manifest["dataset"]["rows"])}'
        progress(f'{context_label} · Generating answer')
        mode = manifest['scoring'].get('evaluation_mode', 'auto')
        reference = mode in ('reference', 'both') or (mode == 'auto' and row['expected_output'] is not None)
        both = mode == 'both'
        flat = dict(row['inputs'])
        if reference:
            flat['expected_output'] = (json.dumps(row['expected_output'], ensure_ascii=False)
                                       if isinstance(row['expected_output'], list) else row['expected_output'])
        # Common task context: use baseline instructions for BOTH variants.
        else:
            from prompt_optimizer import fill_template
            flat['_benchmark_context'] = '\n'.join(fill_template(m['content'], row['inputs'])
                                                  for m in manifest['variants']['baseline'])
        prompts = [Prompt(role=m['role'], template=m['content']) for m in messages]
        cfg = OptimizationConfig(name='benchmark', prompts=prompts,
            mode=Mode.TRAINING_DATA if reference else Mode.FEW_SHOT,
            task_model=model, judge_model=manifest['scoring']['judge_model'],
            label_scorer=manifest['scoring']['label_scorer'],
            quality_criteria=manifest['scoring']['criteria'],
            task_reasoning_effort=manifest['inference']['reasoning_effort'])
        case = {'id': manifest['dataset']['input_hashes'][index], 'inputs': row['inputs'],
                'expected_output': row['expected_output'], 'graded_by': 'reference' if reference else 'rubric',
                'score': None, 'output': None, 'error': None, 'usage': None, 'latency_s': None}
        kinds = ['reference', 'rubric'] if both else ['reference' if reference else 'rubric']
        case['evaluations'] = {kind: {'score': None, 'error': None} for kind in kinds}
        if both:
            case['graded_by'] = 'both'
        def observe(response, duration):
            if billing is not None:
                billing.role = 'judge'
            progress(f'{context_label} · Checking reference' if reference else f'{context_label} · Checking quality criteria')
            case['usage'] = _usage_fields(getattr(response, 'usage', None))
            case['latency_s'] = duration
            case['output'] = response.choices[0].message.content or ''
            case['returned_model'] = getattr(response, 'model', None)
            receipt = ci_receipt(response)
            if receipt:
                receipt_billing = receipt.get('billing')
                receipt_billing = receipt_billing if isinstance(receipt_billing, dict) else {}
                case['billing_receipt'] = {
                    'request_id': receipt.get('request_id'),
                    'status': receipt_billing.get('status'),
                    'billed_cost_usd': receipt_billing.get('billed_cost_usd'),
                    'currency': receipt_billing.get('currency'),
                }
        opt = PromptOptimizer(endpoint=endpoint)   # per case: its own usage meter
        opt._usage = UsageMeter()
        try:
            result = opt.evaluate_prompts(cfg, prompts, [flat], on_prediction=observe,
                                          on_judgment=lambda detail: case.update(rubric_evaluation=detail),
                                          compiled_rubric=manifest["scoring"].get("compiled_rubric"),
                                          strict_labels=True)
            score = result['scores'][0]
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError('Invalid score')
            case['evaluations'][kinds[0]]['score'] = score
            if not both:
                case['score'] = score
        except (BenchmarkCancelled, JobCancelled, LeaseLost):
            raise
        except Exception as exc:
            # Provider error messages may echo request credentials or private URLs.
            from ps_utils.rubric_checks import RubricEvaluationError
            if is_task_failure(case, exc):
                record_task_failure(case, kinds, manifest['scoring'].get('compiled_rubric'), exc)
            elif not (kinds[0] == 'rubric' and record_strict_fail(case, case['evaluations']['rubric'], exc)):
                case['error_detail'] = str(exc) if isinstance(exc, RubricEvaluationError) else 'Evaluation could not complete; no quality verdict is available.'
                case['error'] = type(exc).__name__
                case['evaluations'][kinds[0]]['error'] = type(exc).__name__
        if both and not case.get('task_error'):
            if case['output'] is None:
                case['evaluations']['rubric']['error'] = case['error'] or 'NoOutput'
            else:
                try:
                    from prompt_optimizer import fill_template
                    context = dict(row['inputs'])
                    context['_benchmark_context'] = '\n'.join(fill_template(m['content'], row['inputs']) for m in manifest['variants']['baseline'])
                    progress(f'{context_label} · Checking quality criteria')
                    judge = opt._build_guidelines_scorer(cfg, cfg.quality_criteria,
                        on_judgment=lambda detail: case.update(rubric_evaluation=detail),
                                          compiled_rubric=manifest["scoring"].get("compiled_rubric"))
                    rubric_score = _scorer_value(judge(inputs=context, outputs=case['output']))
                    if not math.isfinite(rubric_score) or not 0 <= rubric_score <= 1:
                        raise ValueError('Invalid score')
                    case['evaluations']['rubric']['score'] = rubric_score
                except (BenchmarkCancelled, JobCancelled, LeaseLost):
                    raise
                except Exception as exc:
                    from ps_utils.rubric_checks import RubricEvaluationError
                    if not record_strict_fail(case, case['evaluations']['rubric'], exc):
                        case['error_detail'] = str(exc) if isinstance(exc, RubricEvaluationError) else 'Evaluation could not complete; no quality verdict is available.'
                        case['evaluations']['rubric']['error'] = type(exc).__name__
                        case['error'] = case['error'] or type(exc).__name__
        case['judge_usage'] = opt._usage.as_dict()['by_role'].get('judge', {'calls':0, 'input':0, 'output':0, 'reasoning':0, 'total':0})
        if billing is not None:
            billing.finish_answer()
        with completed_lock:
            completed += 1
            done = completed
        progress('Saving results' if done == total else 'Continuing')
        return case
    work = [(arm_index, index, row) for index, row in enumerate(manifest['dataset']['rows'])
            for arm_index in range(len(arm_specs))]
    results = _run_cases(work, run_case, concurrency)
    for arm_index, (model, variant, _) in enumerate(arm_specs):
        cases = [case for (owner, _, _), case in zip(work, results) if owner == arm_index]
        arm = {'model': model, 'variant': variant, 'mlflow_run_id': run_ids[arm_index], 'cases': cases,
               **summarize(cases, manifest['price_snapshot']['models'][model])}
        tracker.finish_arm(run_ids[arm_index], arm)
        arms.append(arm)
    return {'schema_version': SCHEMA, 'manifest': manifest, 'arms': arms,
            **({'billing': billing.snapshot()} if billing is not None else {}),
            'mlflow': {'run_id': tracker.run_id, 'experiment_id': tracker.experiment_id},
            'completed_at': datetime.now(timezone.utc).isoformat()}


def run(job, run_id, manifest, credentials):
    from .models import JobState, Run
    from .durable import assert_lease, lease_write, finish_job, JobCancelled, LeaseLost
    from .storage import bounded_json_size, StorageBudgetExceeded
    tracker = None
    def persist_billing(op, snapshot):
        # One call's row change and the live counters commit together. An in-flight
        # response may arrive after cancellation: only the current unexpired worker
        # may record it, atomically in the row, the job and the history JSON.
        result = {'billing': snapshot}
        bounded_json_size(result, 'result')
        with lease_write(job.id, job.lease_token, allow_cancelled=True):
            if op is not None:
                op(job.id)
            JobState.objects.filter(pk=job.id).update(result=result)
            Run.objects.filter(pk=run_id, job_id=job.id, owner=job.owner).update(result=result)
        object.__setattr__(job, 'result', result)
    billing = None
    def finish_terminal(status, stage, error=''):
        # Live snapshots carry counters only. A cancelled or failed run commits the full
        # billing snapshot with its terminal state, atomically in job and history.
        if billing is not None:
            try:
                finish_job(job.id, job.lease_token, status, owner=job.owner, stage=stage, error=error,
                           result={'billing': billing.snapshot()})
                return
            except StorageBudgetExceeded:
                # Too large to store: still end the run, keeping the last live snapshot.
                import traceback
                traceback.print_exc()
        finish_job(job.id, job.lease_token, status, owner=job.owner, stage=stage, error=error)
    def check_cancelled():
        assert_lease(job.id, job.lease_token)
    def set_stage(message):
        # Stage only: `job.stage = …` would save the whole job snapshot (Job.__setattr__),
        # and a case worker could then overwrite newer billing counters with a stale result.
        with lease_write(job.id, job.lease_token):
            JobState.objects.filter(pk=job.id, lease_token=job.lease_token).update(stage=message)
        object.__setattr__(job, 'stage', message)
    def update_stage(message):
        check_cancelled()
        set_stage(message)
    try:
        check_cancelled()
        billing = BillingCollector(manifest, persist=persist_billing)
        persist_billing(None, billing.snapshot(live=True))
        set_stage('Starting benchmark')
        tracker = Tracker(credentials['_owner'], job.id, manifest)
        endpoint = services._endpoint_for(credentials)
        object.__setattr__(endpoint, '_client', CancellationClient(endpoint.client.with_options(
            default_headers=manifest['request_headers'], max_retries=0), check_cancelled, billing))
        report = execute(manifest, endpoint, tracker, stage=update_stage, billing=billing)
        report['run_id'] = run_id
        check_cancelled()
        tracker.finish(report)
        finish_job(job.id, job.lease_token, 'done', owner=job.owner,
                   stage='Benchmark complete', result=report, error='')
    except LeaseLost:
        # Do not let a dead/stale worker overwrite recovered state (no fenced job writes).
        if tracker:
            try:
                tracker.fail()
            except Exception:  # noqa: BLE001 - cleanup must not mask the lease loss
                pass
        raise
    except (BenchmarkCancelled, JobCancelled):
        if tracker:
            try:
                tracker.fail(status='KILLED')
            except Exception:
                pass
        finish_terminal('cancelled', 'Cancelled')
    except Exception:
        if tracker:
            try:
                tracker.fail()
            except Exception:
                pass
        finish_terminal('error', 'Error', 'Benchmark failed; check service logs')
        import traceback
        traceback.print_exc()
        sentry_sdk.capture_exception()


def run_queued(job, payload):
    from .models import Run
    stored = Run.objects.get(job_id=job.id, kind='benchmark', owner=job.owner)
    run(job, str(stored.id), payload['manifest'], payload)


def start(payload):
    from django.db import transaction
    from .models import Run
    from .durable import enqueue
    manifest = validate(payload)
    if not payload.get('_api_key') or not payload.get('_base_url'):
        raise ValueError('An inference key and base URL are required')
    # Admission and its history entry commit together, before the worker can
    # claim the job. Only the queue stores the encrypted credential.
    queued = {k: payload[k] for k in ('_owner', '_api_key', '_base_url')}
    queued['manifest'] = manifest
    with transaction.atomic():
        job = enqueue('benchmark', queued, payload.get('_idempotency_key'))
        Run.objects.get_or_create(job_id=job.id, kind='benchmark', owner=payload['_owner'],
            defaults={'status': 'running', 'label': manifest['name'], 'config': manifest})
    return job
