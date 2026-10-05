import unittest

from scripts.bounded_eval import build_document
from scripts.live_tree_eval import tree
from scripts.live_tree_followup import bottom_up_context
from scripts.live_tree_ancestor import ancestor_context


class BottomUpContextTests(unittest.TestCase):
    def fixture(self):
        doc = build_document('test', 'Doc', [('Heading', ['Alpha fact.', 'Beta evidence.', 'Gamma detail.'])])
        index = tree(doc, [doc['units']], 'J')
        leaf = index._node('p1s0')
        search = {'source_sha256': index.metadata['source_sha256'], 'needs': ['Evidence'],
                  'leaves': [{'node_id': leaf.node_id, 'start': leaf.start, 'end': leaf.end, 'need_scores': {0: .8}}]}
        return index, search

    def test_bottom_up_expansion_recovers_whole_topic_when_it_fits(self):
        index, search = self.fixture()
        result = bottom_up_context(index, search, len, budget=200)
        for text in ['Alpha fact.', 'Beta evidence.', 'Gamma detail.']:
            self.assertIn(text, result['context'])
        for a, b in result['spans']:
            self.assertIn(index.source[a:b], result['context'])

    def test_budget_never_displaces_core_for_larger_ancestor(self):
        index, search = self.fixture()
        result = bottom_up_context(index, search, len, budget=38)
        self.assertIn('Beta evidence.', result['context'])
        self.assertLessEqual(result['context_tokens'], 38)
        self.assertNotIn('Gamma detail.', result['context'])

    def test_successive_ancestors_reach_other_headings_and_preserve_core(self):
        doc = build_document('test', 'Doc', [('One', ['Alpha fact.']), ('Two', ['Beta evidence.']), ('Three', ['Gamma detail.'])])
        index = tree(doc, [[u] for u in doc['units']], 'J')
        leaf = index._node('p1s0')
        search = {'source_sha256': index.metadata['source_sha256'], 'needs': ['Evidence'],
                  'leaves': [{'node_id': leaf.node_id, 'start': leaf.start, 'end': leaf.end, 'need_scores': {0: .8}}]}
        initial = bottom_up_context(index, search, len, budget=38)
        result = ancestor_context(index, search, initial, len, budget=200)
        for text in ['Alpha fact.', 'Beta evidence.', 'Gamma detail.']: self.assertIn(text, result['context'])
        limited = ancestor_context(index, search, initial, len, budget=38)
        self.assertIn('Beta evidence.', limited['context'])
        self.assertLessEqual(limited['context_tokens'], 38)


if __name__ == '__main__':
    unittest.main()
