"""Register validation once, score every prediction, and freeze its selected method.

This is an auditable workflow guard, not an access-control boundary around files.
The test gate recomputes validation selection; it never trusts a status label or
a caller-supplied aggregate score. Only validation labels are read here.
"""
import argparse
import importlib.util
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.bounded_clients import signature
from scripts.bounded_eval import read_json, sha

SELECTION = 'mean_primary_gain_then_query_tokens_then_config_hash_v1'
BASELINE_KINDS = {'hybrid_recursive', 'hybrid_semantic', 'generative_reranker', 'published_algorithm', 'original_jev'}
GATE_SOURCES = {'scripts/validation_gate.py', 'scripts/iteration_splits.py',
                'scripts/bounded_clients.py', 'scripts/bounded_eval.py'}


def checked_path(root, name):
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError('Expected a relative artifact path')
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Artifact path escapes its directory: ' + name)
    return path


def check_sources(config):
    sources = config.get('source_hashes', {})
    if not GATE_SOURCES <= sources.keys():
        raise ValueError('Registration must freeze both validation and test gate code')
    for name, expected in sources.items():
        if sha(checked_path(ROOT, name)) != expected:
            raise ValueError('Frozen implementation mismatch: ' + name)


def check_data(output):
    manifest = read_json(output / 'manifest.json')
    registry = read_json(output / 'registry.json')
    if signature(registry) != manifest.get('registry_sha256'):
        raise ValueError('Split registry changed')
    for name, expected in manifest['data_hashes'].items():
        if sha(checked_path(output, name)) != expected:
            raise ValueError('Prepared split changed: ' + name)
    return manifest, registry


def check_registration(config, manifest):
    if config.get('status') != 'registered_before_validation':
        raise ValueError('Expected a validation registration')
    if config.get('registry_sha256') != manifest['registry_sha256']:
        raise ValueError('Registration targets a different split')
    candidates, baselines = config.get('candidates', {}), config.get('baselines', {})
    if not isinstance(candidates, dict) or not 1 <= len(candidates) <= 3:
        raise ValueError('Register one to three candidates before validation')
    if not isinstance(baselines, dict) or candidates.keys() & baselines.keys():
        raise ValueError('Candidates and baselines need distinct method IDs')
    if any(not isinstance(v, dict) or not v for v in [*candidates.values(), *baselines.values()]):
        raise ValueError('Each method needs a nonempty frozen configuration')
    if not BASELINE_KINDS <= {v.get('kind') for v in baselines.values()}:
        raise ValueError('Missing a required comparison or original-Jev regression control')
    if len({signature(v) for v in candidates.values()}) != len(candidates):
        raise ValueError('Duplicate candidate configurations')
    for method, value in baselines.items():
        if value.get('kind') == 'published_algorithm':
            if not all(value.get(k) for k in ('upstream_url', 'upstream_commit', 'adapter_source')):
                raise ValueError('Published comparator needs a pinned upstream and adapter')
            if value['adapter_source'] not in config.get('source_hashes', {}):
                raise ValueError('Published comparator adapter must be frozen')
    if config.get('selection_rule') != SELECTION:
        raise ValueError('Unknown selection rule')
    if type(config.get('reader_replicates')) is not int or config['reader_replicates'] < 1:
        raise ValueError('Fix the number of reader replicates before validation')
    margin = config.get('evidence_recall_noninferiority_margin')
    if type(margin) not in (int, float) or not math.isfinite(margin) or not 0 <= margin < 1:
        raise ValueError('Fix the evidence-recall margin before validation')
    common = config.get('common', {})
    if not all(common.get(k) for k in ('reader_model', 'reader_prompt_sha256', 'context_token_limit')):
        raise ValueError('Freeze the common reader and context budget')
    if type(common['context_token_limit']) is not int or common['context_token_limit'] <= 0:
        raise ValueError('Invalid context budget')
    check_sources(config)


def write_once(path, value):
    """An existing marker must match exactly, including on concurrent attempts."""
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    except FileExistsError:
        if read_json(path) != value:
            raise ValueError('This partition is already bound to a different registration')


def open_validation(output, registration):
    if (output / 'test-opened.json').exists():
        raise ValueError('Test has already been exposed; start a new research allocation')
    manifest, _ = check_data(output)
    config = read_json(registration)
    check_registration(config, manifest)
    record = {'status': 'validation_opened_no_new_candidates',
              'registration': config, 'registration_sha256': signature(config),
              'registry_sha256': manifest['registry_sha256']}
    write_once(output / 'validation-opened.json', record)
    return output / 'validation'


def validation_record(output):
    path = output / 'validation-opened.json'
    if not path.exists():
        raise ValueError('Validation was not registered before opening')
    record = read_json(path)
    if record.get('status') != 'validation_opened_no_new_candidates':
        raise ValueError('Invalid validation opening marker')
    config = record['registration']
    manifest, registry = check_data(output)
    check_registration(config, manifest)
    if record.get('registration_sha256') != signature(config) or record.get('registry_sha256') != manifest['registry_sha256']:
        raise ValueError('Validation registration marker changed')
    return record, config, registry


def score_validation(output, predictions):
    record, config, registry = validation_record(output)
    artifact = read_json(predictions)
    if artifact.get('registration_sha256') != record['registration_sha256']:
        raise ValueError('Predictions came from a different validation registration')
    methods = {**config['candidates'], **config['baselines']}
    cases = {r['id']: r for r in registry if r['partition'] == 'validation'}
    if {r['dataset'] for r in cases.values()} != {'qasper', 'quality'}:
        raise ValueError('Validation must cover both registered benchmarks')
    expected = {(qid, method, replicate) for qid in cases for method in methods
                for replicate in range(config['reader_replicates'])}
    rows = artifact.get('predictions', [])
    if any(type(r.get('replicate')) is not int for r in rows):
        raise ValueError('Reader replicate IDs must be integers')
    keys = [(r.get('id'), r.get('method'), r.get('replicate')) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('Validation prediction coverage is incomplete, duplicated or unexpected')
    # All ID/coverage checks happen before accessing validation labels.
    gold = read_json(output / 'validation/gold.json')
    qasper_path = output / 'validation/qasper_evaluator.py'
    spec = importlib.util.spec_from_file_location('qasper_validation_gate', qasper_path)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    refs = official.get_answers_and_evidence({'p': {'qas': [
        {'question_id': qid, 'answers': [{'answer': a} for a in gold[qid]['answers']]}
        for qid, case in cases.items() if case['dataset'] == 'qasper']}}, False)
    scores = {m: {'qasper': [], 'quality': [], 'query_tokens': []} for m in methods}
    for row in rows:
        dataset = cases[row['id']]['dataset']
        if row.get('status') not in ('ok', 'failed') or not isinstance(row.get('answer'), str):
            raise ValueError('Every prediction must record its answer and failure status')
        usage = row.get('query_tokens')
        context_tokens = row.get('context_tokens')
        if type(usage) is not int or usage < 0:
            raise ValueError('Record total query model tokens, including failed attempts')
        if type(context_tokens) is not int or not 0 <= context_tokens <= config['common']['context_token_limit']:
            raise ValueError('Prediction exceeds the registered source context budget')
        answer = row['answer'].strip()
        value = 0.0
        if row['status'] == 'ok':
            if dataset == 'quality':
                value = float(answer in ('A', 'B', 'C', 'D') and answer == gold[row['id']]['label'])
            else:
                value = max(official.token_f1_score(answer, r['answer']) for r in refs[row['id']])
        scores[row['method']][dataset].append(value)
        scores[row['method']]['query_tokens'].append(usage)
    aggregate = {m: {k: statistics.mean(v) for k, v in values.items()} for m, values in scores.items()}
    strongest = {d: max(aggregate[m][d] for m in config['baselines']) for d in ('qasper', 'quality')}
    rankings = []
    for method, candidate in config['candidates'].items():
        a = aggregate[method]
        gain = statistics.mean(a[d] - strongest[d] for d in ('qasper', 'quality'))
        rankings.append({'method': method, 'mean_primary_gain': gain,
                         'query_tokens': a['query_tokens'], 'configuration_sha256': signature(candidate)})
    rankings.sort(key=lambda r: (-r['mean_primary_gain'], r['query_tokens'], r['configuration_sha256']))
    return {'aggregate': aggregate, 'candidate_ranking': rankings, 'selected_method': rankings[0]['method'],
            'questions': len(cases), 'predictions': len(rows), 'registration_sha256': record['registration_sha256']}


def freeze_winner(output, predictions):
    if (output / 'test-opened.json').exists():
        raise ValueError('Cannot select a new method after test exposure')
    # Store the original predictions intact; never substitute caller aggregates.
    result = score_validation(output, predictions)
    artifact = read_json(predictions)
    record, config, _ = validation_record(output)
    write_once(output / 'validation-predictions.json', artifact)
    write_once(output / 'validation-selection.json', result)
    frozen = {'status': 'selected_after_registered_validation',
              'registration_sha256': record['registration_sha256'],
              'validation_predictions_sha256': sha(output / 'validation-predictions.json'),
              'validation_selection_sha256': sha(output / 'validation-selection.json'),
              'selected_method': result['selected_method'],
              'selected_configuration': config['candidates'][result['selected_method']],
              'baselines': config['baselines'], 'source_hashes': config['source_hashes']}
    write_once(output / 'frozen-winner.json', frozen)
    return output / 'frozen-winner.json'


def verify_winner(output, frozen):
    config = read_json(frozen)
    if config.get('status') != 'selected_after_registered_validation':
        raise ValueError('Test requires a completed registered validation selection')
    required = ('validation-opened.json', 'validation-predictions.json', 'validation-selection.json', 'frozen-winner.json')
    if any(not (output / name).exists() for name in required):
        raise ValueError('Test requires validation registration, complete predictions and selection artifacts')
    if sha(frozen) != sha(output / 'frozen-winner.json'):
        raise ValueError('Frozen winner bytes changed')
    if config.get('validation_predictions_sha256') != sha(output / 'validation-predictions.json') or config.get('validation_selection_sha256') != sha(output / 'validation-selection.json'):
        raise ValueError('Validation artifacts changed after selection')
    recomputed = score_validation(output, output / 'validation-predictions.json')
    record, registration, _ = validation_record(output)
    if recomputed != read_json(output / 'validation-selection.json'):
        raise ValueError('Validation selection does not reproduce from predictions')
    if (config.get('registration_sha256') != record['registration_sha256'] or
        config.get('selected_method') != recomputed['selected_method'] or
        config.get('selected_configuration') != registration['candidates'][recomputed['selected_method']] or
        config.get('baselines') != registration['baselines'] or
        config.get('source_hashes') != registration['source_hashes']):
        raise ValueError('Frozen winner differs from the registered validation selection')
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['open-validation', 'freeze-winner'])
    parser.add_argument('--output', type=Path, default=ROOT / 'output/improvement-v1')
    parser.add_argument('--registration', type=Path)
    parser.add_argument('--predictions', type=Path)
    args = parser.parse_args()
    if args.command == 'open-validation':
        if args.registration is None:
            parser.error('--registration is required')
        print(open_validation(args.output, args.registration))
    else:
        if args.predictions is None:
            parser.error('--predictions is required')
        print(freeze_winner(args.output, args.predictions))
