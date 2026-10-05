"""Development-only QuALITY control for answering without retrieved source text."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.bounded_clients import Codex, save, signature, validate_answers
from scripts.bounded_eval import batches, parallel_progress, query_text, read_json, sha
from scripts.development_iteration import OLD, research_budget


def prepare(output):
    if (output / 'manifest.json').exists():
        raise ValueError('Control is already frozen')
    screen = set(read_json(ROOT / 'evals/iterations/v1/manifest.json')['development_screening_ids'])
    cases = [c for c in read_json(OLD / 'cases.json') if c['id'] in screen and c['dataset'] == 'quality']
    save(output / 'cases.json', cases)
    save(output / 'manifest.json', {'status': 'registered_before_inference', 'partition': 'exposed_development',
        'purpose': 'Measure how informative this QuALITY screen is about retrieval.', 'questions': len(cases),
        'source_context': 'No document source has been supplied.',
        'cases_sha256': sha(output / 'cases.json'),
        'source_hashes': {name: sha(ROOT / name) for name in
            ('scripts/closed_book_probe.py', 'scripts/bounded_clients.py', 'scripts/bounded_eval.py')}})
    print(json.dumps({'questions': len(cases), 'expected_calls': len(list(batches(cases)))}))


def verify(output):
    manifest = read_json(output / 'manifest.json')
    if sha(output / 'cases.json') != manifest['cases_sha256']:
        raise ValueError('Changed control cases')
    for name, expected in manifest['source_hashes'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Changed control implementation: ' + name)
    return manifest


def run(output):
    manifest = verify(output)
    # Research runs execute sequentially so they share one authoritative ledger.
    client = Codex(output, research_budget())
    cases = read_json(output / 'cases.json')
    def read(batch):
        folder = output / 'predictions'
        if all((folder / (signature(c['id']) + '.json')).exists() for c in batch):
            return
        request = [{'id': c['id'], 'question': query_text(c), 'context': manifest['source_context']} for c in batch]
        answers = validate_answers(client.call(request, 'reader'), [c['id'] for c in batch])
        for case in batch:
            answer = answers[case['id']].strip()
            save(folder / (signature(case['id']) + '.json'), {'id': case['id'], 'answer': answer,
                'status': 'ok' if answer in ('A', 'B', 'C', 'D') else 'failed', 'request_sha256': signature(request)})
    parallel_progress(read, list(batches(cases)), 2, 'closed-book-development')


def score(output):
    verify(output)
    gold, rows = read_json(OLD / 'gold.json'), []
    for case in read_json(output / 'cases.json'):
        prediction = read_json(output / 'predictions' / (signature(case['id']) + '.json'))
        rows.append({**prediction, 'doc_id': case['doc_id'], 'correct': int(
            prediction['status'] == 'ok' and prediction['answer'] == gold[case['id']]['label'])})
    result = {'status': 'development_control_only', 'questions': len(rows), 'correct': sum(r['correct'] for r in rows),
              'accuracy': sum(r['correct'] for r in rows)/len(rows), 'invalid': sum(r['status'] != 'ok' for r in rows)}
    save(output / 'scores.json', rows)
    save(output / 'summary.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'score'])
    parser.add_argument('--output', type=Path, default=ROOT / 'output/closed-book-v1')
    args = parser.parse_args()
    {'prepare': prepare, 'run': run, 'score': score}[args.command](args.output)
