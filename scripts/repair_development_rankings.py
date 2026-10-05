"""Registered development-only recovery of omitted ranking IDs, without a model.

Keep every returned unique valid ID in its original order. Append omitted IDs
in the original hybrid retrieval order. Reject duplicates or out-of-range IDs.
Never inspect gold labels. The append-only repair record preserves raw output.
"""
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.bounded_clients import save, signature
from scripts.bounded_eval import context_for, read_json, tokenizer
from scripts.development_iteration import OLD, verify


def complete_order(order, expected, fallback):
    if not isinstance(order, list) or any(type(x) is not int or x < 0 or x >= expected for x in order):
        raise ValueError('Cannot repair invalid candidate IDs')
    if len(set(order)) != len(order) or sorted(fallback) != list(range(expected)):
        raise ValueError('Cannot repair duplicates or invalid fallback')
    return order + [i for i in fallback if i not in set(order)]


def repair(output):
    verify(output)
    if any((output / 'predictions').glob('*/*.json')):
        raise ValueError('This amendment is only registered before reader predictions')
    rows = [json.loads(s) for s in (output / 'codex-audit.jsonl').read_text(encoding='utf-8').splitlines()]
    docs = read_json(OLD / 'documents.json')
    enc, seen, records = tokenizer(), set(), []
    for row in reversed(rows):
        key = row['request_sha256']
        if row['stage'] != 'reranker' or row['status'] != 'failed' or key in seen:
            continue
        seen.add(key)
        answer_file = output / 'codex-isolated' / (key + '-answer.json')
        answer = read_json(answer_file)
        if set(x['id'] for x in answer['rankings']) != set(row['case_ids']) or len(answer['rankings']) != len(row['case_ids']):
            raise ValueError('Wrong response identities')
        for ranking in answer['rankings']:
            path = output / 'retrieval' / (signature(ranking['id']) + '.json')
            retrieval = read_json(path)
            if 'codex_pool_8192' in retrieval['methods']:
                continue
            original = retrieval['pools']['recursive']['candidates']
            shuffled = original[:]
            random.Random(signature({'seed': 20261005, 'id': ranking['id']})).shuffle(shuffled)
            positions = {c['id']: i for i, c in enumerate(shuffled)}
            fallback = [positions[c['id']] for c in original]
            order = complete_order(ranking['order'], len(shuffled), fallback)
            chunks = [shuffled[i] for i in order]
            context, spans, tokens = context_for(docs[retrieval['doc_id']], chunks, enc)
            retrieval['methods']['codex_pool_8192'] = {'context': context, 'spans': spans, 'context_tokens': tokens,
                'ranking': [c['id'] for c in chunks], 'format_repair': True}
            records.append({'id': ranking['id'], 'request_sha256': key, 'raw_order': ranking['order'],
                'expected_candidates': len(shuffled), 'completed_order': order,
                'omitted_ids_appended': [i for i in order if i not in ranking['order']]})
            save(path, retrieval)
    save(output / 'ranking-repair.json', {'status': 'development_amendment_before_reading',
        'rule': 'Preserve valid unique returned IDs; append omitted IDs in original hybrid retrieval order.',
        'model_calls': 0, 'records': records})
    print(json.dumps({'repaired_case_records': len(records), 'omitted_ids_appended': sum(len(r['omitted_ids_appended']) for r in records)}))


if __name__ == '__main__':
    repair(ROOT / 'output/development-pool-v1')
