import tempfile
import unittest
import json
import io
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
from pathlib import Path

from scripts.bounded_clients import save
from scripts.live_tree_clients import LiveBudget, LiveEmbeddings
from scripts.live_tree_eval import tree
from zero_index.index import reselect_representatives
from zero_index.embeddings import CentroidRepresentatives


class LiveEvaluationTests(unittest.TestCase):
    def test_concurrent_embedding_requests_deduplicate_shared_texts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            budget = LiveBudget(root / 'budget.json')
            client = LiveEmbeddings(root, budget)
            # The transport concurrency check bypasses the production quota pacer.
            from scripts.bounded_clients import Embeddings
            client.batch = lambda texts, stage: Embeddings.embed.__wrapped__(client, texts, stage)
            seen, active, peak = [], 0, 0
            lock = threading.Lock()
            def respond(request, timeout):
                nonlocal active, peak
                body = json.loads(request.data)
                texts = [r['content']['parts'][0]['text'] for r in body['requests']]
                with lock:
                    active += 1; peak = max(peak, active); seen.extend(texts)
                time.sleep(.02)
                with lock: active -= 1
                return io.BytesIO(json.dumps({'embeddings': [{'values': [1.] + [0.] * 767} for _ in texts]}).encode())
            texts = [f'text {i}' for i in range(128)]
            with patch.dict('os.environ', {'GEMINI_API_KEY': 'test'}), patch.object(client, 'count_tokens', return_value=128), patch('scripts.bounded_clients.urlopen', side_effect=respond):
                with ThreadPoolExecutor(max_workers=2) as pool:
                    jobs = [pool.submit(client.embed, texts, 'fixture') for _ in range(2)]
                    values = [job.result() for job in jobs]
            client.pool.shutdown()
            self.assertEqual(sorted(seen), sorted(texts))
            self.assertEqual(values[0].shape, (128, 768))
            self.assertTrue((values[0] == values[1]).all())
            self.assertGreater(peak, 1)

    def test_generation_and_embedding_share_one_hard_dollar_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'budget.json'
            save(path, {'gemini_cap_usd': 1, 'previous_work_reserved_usd': .7,
                        'embedding_tokens_reserved': 0, 'embedding_usd_per_million': .2,
                        'embedding_inputs_reserved': 0, 'embedding_inputs_limit': 100})
            budget = LiveBudget(path)
            budget.generation(100000, 40000)  # $0.13
            budget.reserve('embedding_inputs', 1, tokens=500000)  # $0.10
            self.assertAlmostEqual(budget.value['gemini_reserved_upper_bound_usd'], .93)
            before = path.read_bytes()
            with self.assertRaises(RuntimeError):
                budget.generation(100000, 40000)
            self.assertEqual(path.read_bytes(), before)

    def test_native_paragraphs_survive_tree_assembly_without_reparsing(self):
        source = 'First topic. More detail.\n\nSecond topic.'
        units = [{'start': 0, 'end': 25, 'heading': 'Intro', 'section': 0, 'paragraph': 0},
                 {'start': 27, 'end': len(source), 'heading': 'Results', 'section': 1, 'paragraph': 1}]
        index = tree({'id': 'test', 'title': 'Paper', 'text': source, 'units': units}, [[units[0]], [units[1]]], 'J')
        self.assertEqual([(n.start, n.end) for n in index.root.walk() if n.kind == 'paragraph'], [(0, 25), (27, len(source))])
        self.assertTrue(all(source[n.start:n.end] == n.title for n in index.root.walk() if n.kind == 'sentence'))
        selected = reselect_representatives(index, CentroidRepresentatives(lambda text: [1., 2.], model_name='fixture'), sentence_budget=8)
        self.assertEqual(selected.metadata['config']['sentence_budget'], 8)
        self.assertTrue(all(n.central is not None for n in selected.root.walk() if n.kind == 'paragraph'))


if __name__ == '__main__':
    unittest.main()
