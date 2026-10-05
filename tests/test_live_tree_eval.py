import tempfile
import unittest
from pathlib import Path

from scripts.bounded_clients import save
from scripts.live_tree_clients import LiveBudget
from scripts.live_tree_eval import tree


class LiveEvaluationTests(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
