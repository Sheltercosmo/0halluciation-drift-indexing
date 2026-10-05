import unittest
from scripts.bounded_eval import build_document, make_chunks
from zero_index.context_merge import merge_context, topic_parents


class ContextMergeTests(unittest.TestCase):
    def setUp(self):
        self.doc = build_document('doc', 'T', [('H', ['a'*30, 'b'*30, 'c'*30]), ('H2', ['d'*20])])
        self.chunks = [{'id': str(i), **u, 'text': self.doc['text'][u['start']:u['end']]}
                       for i, u in enumerate(self.doc['units'])]
        self.parents = topic_parents(self.doc, [])

    def test_two_children_expand_to_exact_parent_without_duplicate_source(self):
        result = merge_context(self.doc, self.chunks[:2], self.parents, len, budget=130)
        self.assertEqual(result['spans'], [(0, 94)])
        self.assertEqual(len(result['merges']), 1)
        self.assertLessEqual(result['context_tokens'], 130)
        self.assertIn('c'*30, result['context'])
        self.assertEqual(result['context'].count('a'*30), 1)

    def test_parent_expansion_cannot_exceed_budget(self):
        result = merge_context(self.doc, self.chunks[:2], self.parents, len, budget=90)
        self.assertEqual(len(result['spans']), 2)
        self.assertEqual(result['merges'], [])
        self.assertLessEqual(result['context_tokens'], 90)

    def test_topic_cuts_and_native_headings_bound_parents(self):
        parents = topic_parents(self.doc, [{'candidate': self.chunks[1]['start'], 'cut': True}])
        self.assertEqual([(p['start'],p['end']) for p in parents], [(0,30), (32,94), (96,116)])
        result = merge_context(self.doc, self.chunks[:2], parents, len, budget=200)
        self.assertEqual(result['merges'], [])

    def test_wrong_source_text_is_rejected(self):
        with self.assertRaises(ValueError):
            merge_context(self.doc, [{**self.chunks[0], 'text':'invented'}], self.parents, len)


if __name__ == '__main__':
    unittest.main()
