"""A test opening requires a replayable, complete registered validation run."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.bounded_clients import save, signature
from scripts.bounded_eval import sha
from scripts.iteration_splits import open_test
from scripts.validation_gate import (ROOT, BASELINE_KINDS, GATE_SOURCES, SELECTION,
    freeze_winner, open_validation, score_validation)


class ValidationGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        registry = [{'id': 'q1', 'dataset': 'qasper', 'partition': 'validation', 'doc_id': 'paper'},
                    {'id': 'q2', 'dataset': 'quality', 'partition': 'validation', 'doc_id': 'article'},
                    {'id': 't1', 'dataset': 'quality', 'partition': 'test', 'doc_id': 'unseen'}]
        save(self.root / 'registry.json', registry)
        save(self.root / 'validation/gold.json', {'q1': {'answers': ['yes']}, 'q2': {'label': 'A'}})
        # Small scorer fixture tests the gate workflow; benchmark metric fidelity
        # is checked independently against the archived official evaluator.
        evaluator = self.root / 'validation/qasper_evaluator.py'
        evaluator.write_text('def get_answers_and_evidence(data, _):\n'
            '    return {q["question_id"]: q["answers"] for q in data["p"]["qas"]}\n'
            'def token_f1_score(a, b):\n    return float(a == b)\n', encoding='utf-8')
        save(self.root / 'test/gold.json', {'t1': {'label': 'D'}})
        names = ('validation/gold.json', 'validation/qasper_evaluator.py', 'test/gold.json')
        manifest = {'registry_sha256': signature(registry), 'data_hashes': {n: sha(self.root / n) for n in names}}
        save(self.root / 'manifest.json', manifest)
        self.config = {'status': 'registered_before_validation', 'registry_sha256': signature(registry),
            'candidates': {'good': {'policy': 'good'}, 'bad': {'policy': 'bad'}},
            'baselines': {k: {'kind': k} for k in BASELINE_KINDS},
            'selection_rule': SELECTION, 'reader_replicates': 1,
            'evidence_recall_noninferiority_margin': .02,
            'common': {'reader_model': 'fixture', 'reader_prompt_sha256': 'f'*64, 'context_token_limit': 2048},
            'source_hashes': {n: sha(ROOT / n) for n in GATE_SOURCES}}
        self.config['baselines']['published_algorithm'].update(upstream_url='https://example.org/paper',
            upstream_commit='a'*40, adapter_source='scripts/validation_gate.py')
        self.registration = self.root / 'registration.json'
        save(self.registration, self.config)
        self.predictions = self.root / 'input-predictions.json'

    def predictions_for(self, config=None):
        config = self.config if config is None else config
        rows = []
        for method in [*config['candidates'], *config['baselines']]:
            for qid, correct in [('q1', 'yes'), ('q2', 'A')]:
                rows.append({'id': qid, 'method': method, 'replicate': 0,
                    'answer': correct if method == 'good' else 'wrong',
                    'status': 'ok', 'query_tokens': 100, 'context_tokens': 200})
        return {'registration_sha256': signature(config), 'predictions': rows}

    def opened(self):
        self.assertEqual(open_validation(self.root, self.registration), self.root / 'validation')
        artifact = self.predictions_for()
        save(self.predictions, artifact)
        return artifact

    def test_claimed_status_without_validation_cannot_open_test(self):
        fake = self.root / 'fake.json'
        save(fake, {'status': 'selected_after_registered_validation',
                    'source_hashes': self.config['source_hashes'], 'baselines': self.config['baselines']})
        with self.assertRaisesRegex(ValueError, 'complete predictions'):
            open_test(self.root, fake)
        self.assertFalse((self.root / 'test-opened.json').exists())

    def test_replays_winner_from_answers_ignoring_claimed_scores_and_resumes_same_test(self):
        artifact = self.opened()
        for row in artifact['predictions']:
            row['answer_score'] = int(row['method'] == 'bad')
        save(self.predictions, artifact)
        frozen = freeze_winner(self.root, self.predictions)
        self.assertEqual(json.loads(frozen.read_text())['selected_method'], 'good')
        self.assertEqual(open_test(self.root, frozen), self.root / 'test')
        self.assertEqual(open_test(self.root, frozen), self.root / 'test')
        with self.assertRaisesRegex(ValueError, 'already been exposed'):
            open_validation(self.root, self.registration)

    def test_missing_duplicate_or_unregistered_predictions_cannot_select(self):
        artifact = self.opened()
        variants = [artifact['predictions'][:-1], artifact['predictions']+[artifact['predictions'][0]]]
        extra = copy.deepcopy(artifact['predictions'])
        extra[0]['method'] = 'unregistered'
        variants.append(extra)
        for rows in variants:
            save(self.predictions, {**artifact, 'predictions': rows})
            with self.assertRaisesRegex(ValueError, 'coverage'):
                freeze_winner(self.root, self.predictions)
        self.assertFalse((self.root / 'frozen-winner.json').exists())

    def test_validation_cannot_be_reopened_with_an_added_candidate(self):
        self.opened()
        config = copy.deepcopy(self.config)
        config['candidates']['new'] = {'policy': 'new'}
        save(self.registration, config)
        with self.assertRaisesRegex(ValueError, 'different registration'):
            open_validation(self.root, self.registration)

    def test_changed_predictions_or_source_prevent_test_access(self):
        self.opened()
        frozen = freeze_winner(self.root, self.predictions)
        path = self.root / 'validation-predictions.json'
        original = path.read_bytes()
        value = json.loads(original)
        value['predictions'][0]['answer'] = 'changed'
        save(path, value)
        with self.assertRaisesRegex(ValueError, 'artifacts changed'):
            open_test(self.root, frozen)
        path.write_bytes(original)
        source_record = self.root / 'validation-opened.json'
        value = json.loads(source_record.read_text())
        value['registration']['source_hashes']['scripts/validation_gate.py'] = '0'*64
        save(source_record, value)
        with self.assertRaisesRegex(ValueError, 'implementation mismatch'):
            open_test(self.root, frozen)
        self.assertFalse((self.root / 'test-opened.json').exists())

    def test_failed_predictions_count_as_zero_without_dropping_denominator(self):
        artifact = self.opened()
        artifact['predictions'][0]['status'] = 'failed'
        save(self.predictions, artifact)
        result = score_validation(self.root, self.predictions)
        self.assertEqual(result['aggregate']['good']['qasper'], 0)
        self.assertEqual(result['predictions'], 14)

    def test_tie_uses_query_usage_then_configuration_hash(self):
        artifact = self.opened()
        for row in artifact['predictions']:
            row['answer'] = 'yes' if row['id'] == 'q1' else 'A'
            row['query_tokens'] = 90 if row['method'] == 'bad' else 100
        save(self.predictions, artifact)
        self.assertEqual(score_validation(self.root, self.predictions)['selected_method'], 'bad')
        for row in artifact['predictions']:
            row['query_tokens'] = 100
        save(self.predictions, artifact)
        expected = min(self.config['candidates'], key=lambda m: signature(self.config['candidates'][m]))
        self.assertEqual(score_validation(self.root, self.predictions)['selected_method'], expected)

    def test_context_budget_and_reader_replicate_contract(self):
        artifact = self.opened()
        artifact['predictions'][0]['context_tokens'] = 2049
        save(self.predictions, artifact)
        with self.assertRaisesRegex(ValueError, 'context budget'):
            score_validation(self.root, self.predictions)
        artifact['predictions'][0]['context_tokens'] = 200
        artifact['predictions'][0]['replicate'] = False
        save(self.predictions, artifact)
        with self.assertRaisesRegex(ValueError, 'replicate IDs'):
            score_validation(self.root, self.predictions)


if __name__ == '__main__':
    unittest.main()
