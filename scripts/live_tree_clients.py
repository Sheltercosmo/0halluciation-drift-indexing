"""Budgeted, resumable Gemini generation for the live factorial experiment."""
import json
import os
from pathlib import Path
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from scripts.bounded_clients import Audit, Budget, Codex, Embeddings, path_lock, save, signature, validate_answers, validate_rankings


class LiveEmbeddings(Embeddings):
    """Same ordered 64-input requests, with at most eight independent batches in flight."""
    def __init__(self, output, budget):
        super().__init__(output, budget)
        self.pool = ThreadPoolExecutor(max_workers=8)

    def embed(self, texts, stage):
        import numpy as np
        paths = {s: self.cache / (signature({'model': self.model, 'dimensions': 768, 'text': s}) + '.json') for s in texts}
        # Lock individual keys in stable order. Concurrent questions sharing one document
        # cannot send duplicate embeddings, while unrelated documents can make progress.
        with ExitStack() as stack:
            for path in sorted(set(paths.values())):
                stack.enter_context(path_lock(path))
            missing = [s for s, path in paths.items() if not path.exists()]
            jobs = [self.pool.submit(Embeddings.embed.__wrapped__, self, missing[i:i+64], stage)
                    for i in range(0, len(missing), 64)]
            failures = []
            for job in jobs:
                try:
                    job.result()
                except Exception as error:
                    failures.append(error)
            if failures:
                raise failures[0]
            return np.asarray([json.loads(paths[s].read_text(encoding='utf-8')) for s in texts])


class LiveBudget(Budget):
    def total(self, value):
        return (value['previous_work_reserved_usd'] + value['embedding_tokens_reserved'] *
                value['embedding_usd_per_million'] / 1e6 + value.get('generation_reserved_usd', 0))

    def reserve(self, name, amount=1, questions=0, tokens=0):
        with self.lock:
            value = dict(self.value)
            key = name + '_reserved'
            if amount < 0 or value[key] + amount > value[name + '_limit']:
                raise RuntimeError(name + ' budget exhausted')
            value[key] += amount
            if name == 'embedding_inputs':
                if type(tokens) is not int or tokens <= 0:
                    raise ValueError('Provider token count required')
                value['embedding_tokens_reserved'] += tokens
            if name == 'jev_calls':
                value['jev_questions_reserved'] += questions
                if value['jev_questions_reserved'] > value['jev_questions_limit']:
                    raise RuntimeError('Jev question budget exhausted')
            self.persist(value)

    def generation(self, input_tokens, output_tokens):
        if type(input_tokens) is not int or input_tokens <= 0 or type(output_tokens) is not int or output_tokens <= 0:
            raise ValueError('Positive native input count and output cap required')
        with self.lock:
            value = dict(self.value)
            value['generation_reserved_usd'] = value.get('generation_reserved_usd', 0) + (input_tokens * .30 + output_tokens * 2.50) / 1e6
            value['generation_calls_reserved'] = value.get('generation_calls_reserved', 0) + 1
            self.persist(value)

    def persist(self, value):
        value['gemini_reserved_upper_bound_usd'] = self.total(value)
        if value['gemini_reserved_upper_bound_usd'] > value['gemini_cap_usd']:
            raise RuntimeError('Gemini dollar cap exhausted')
        save(self.path, value)
        self.value = value


class Gemini(Codex):
    """Same isolated reader/ranker instructions; no tools and no thinking tokens."""
    model = 'gemini-2.5-flash'

    def __init__(self, output, budget):
        self.output, self.budget = Path(output), budget
        self.cache = self.output / 'gemini-cache'
        self.cache.mkdir(exist_ok=True)
        self.responses = self.output / 'gemini-responses'
        self.responses.mkdir(exist_ok=True)
        self.audit = Audit(self.output / 'gemini-audit.jsonl')

    def http(self, operation, body):
        req = Request('https://generativelanguage.googleapis.com/v1beta/models/' + self.model + ':' + operation,
                      json.dumps(body, ensure_ascii=False).encode(), headers={
                          'Content-Type': 'application/json', 'x-goog-api-key': os.environ['GEMINI_API_KEY']})
        with urlopen(req, timeout=120) as response:
            return json.load(response)

    def structured(self, instruction, payload, schema, validator, stage):
        prompt = instruction + '\nINPUT:\n' + json.dumps(payload, ensure_ascii=False)
        return self.generate(prompt, schema, validator, stage)

    def _execute(self, cases, mode, schema, prompt, key, cached):
        validator = (lambda v: validate_answers(v, [c['id'] for c in cases])) if mode == 'reader' else (
            lambda v: validate_rankings(v, {c['id']: len(c['candidates']) for c in cases}))
        return self.generate(prompt, schema, validator, mode)

    def generate(self, prompt, schema, validator, stage):
        body = {'contents': [{'role': 'user', 'parts': [{'text': prompt}]}], 'generationConfig': {
            'temperature': 0, 'maxOutputTokens': 512, 'thinkingConfig': {'thinkingBudget': 0},
            'responseMimeType': 'application/json', 'responseJsonSchema': schema}}
        key = signature({'model': self.model, 'body': body})
        path = self.cache / (key + '.json')
        with path_lock(path):
            if path.exists():
                value = json.loads(path.read_text(encoding='utf-8')); validator(value); return value
            # Include the schema plus a conservative wrapper allowance in the native token count.
            count = self.http('countTokens', {'contents': [{'parts': [{'text': prompt + '\n' + json.dumps(schema)}]}]})['totalTokens'] + 256
            prior = len(list(self.responses.glob(key + '-*.json')))
            if prior >= 3:
                raise RuntimeError('Persistent generation retry limit exhausted')
            for attempt in range(prior, 3):
                self.budget.generation(count, 512)
                row = {'stage': stage, 'model': self.model, 'request_sha256': key, 'attempt': attempt + 1,
                       'status': 'failed', 'reserved_input_tokens': count, 'reserved_output_tokens': 512}
                started = time.perf_counter()
                response_path = self.responses / (key + '-' + str(attempt + 1) + '.json')
                try:
                    result = self.http('generateContent', body)
                    save(response_path, result)
                    row['usage'] = result.get('usageMetadata', {})
                    row['response_model'] = result.get('modelVersion')
                    if row['usage'].get('promptTokenCount', 0) > count or row['usage'].get('candidatesTokenCount', 0) + row['usage'].get('thoughtsTokenCount', 0) > 512:
                        raise RuntimeError('Provider exceeded reserved token allowance')
                    candidate = result['candidates'][0]
                    if candidate.get('finishReason') != 'STOP':
                        raise ValueError('Generation did not complete')
                    raw = ''.join(p.get('text', '') for p in candidate['content']['parts'] if not p.get('thought'))
                    value = json.loads(raw)
                    validator(value)
                    row['status'] = 'ok'; save(path, value); return value
                except HTTPError as error:
                    row['http_status'] = error.code
                    save(response_path, {'error_type': 'HTTPError', 'http_status': error.code})
                    if error.code != 429 and error.code < 500:
                        raise RuntimeError('Gemini HTTP ' + str(error.code)) from None
                except (ValueError, KeyError, IndexError) as error:
                    row['error_type'] = type(error).__name__
                finally:
                    if not response_path.exists():
                        save(response_path, {'status': 'unknown_outcome'})
                    row['seconds'] = time.perf_counter() - started
                    self.audit.append(row)
                if attempt < 2:
                    time.sleep(2 ** (attempt + 1))
            raise RuntimeError('Gemini structural/transient retry limit exhausted')
