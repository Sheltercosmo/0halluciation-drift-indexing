"""Regression checks for isolation, candidate exposure and irreversible test access."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts.iteration_splits import load_jsonl, normalized, open_test, resolve_groups
from zero_index.context import candidate_pool


class IterationIntegrityTests(unittest.TestCase):
    def test_json_lines_preserve_unicode_line_separators_inside_text(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'data.jsonl'
            path.write_text('{"article":"one\u2028two"}\n{"article":"three"}\n', encoding='utf-8')
            self.assertEqual(load_jsonl(path), [{'article': 'one\u2028two'}, {'article': 'three'}])

    def test_duplicate_document_cannot_remain_in_test(self):
        docs = {k: {'title': title, 'text': body} for k, title, body in [
            ('a', 'An exposed scientific document', 'ABC  def'),
            ('b', 'A differently titled document', 'abc def'),
            ('c', 'A differently titled document', 'Different body'),
            ('d', 'Unrelated scientific document', 'Other body')]}
        result, moved = resolve_groups(docs, [], {'a': 'development', 'b': 'test', 'c': 'validation', 'd': 'test'})
        self.assertEqual(result, {'a': 'development', 'b': 'development', 'c': 'development', 'd': 'test'})
        self.assertEqual(len(moved), 2)

    def test_pool_counts_tokens_not_chunk_count_and_never_truncates(self):
        small = [{'id': str(i), 'text': 'a' * 10} for i in range(20)]
        large = [{'id': str(i), 'text': 'a' * 20} for i in range(10)]
        a = candidate_pool(small, len, max_source_tokens=150)
        b = candidate_pool(large, len, max_source_tokens=150)
        self.assertEqual((a['candidate_count'], a['source_tokens']), (15, 150))
        self.assertEqual((b['candidate_count'], b['source_tokens']), (7, 140))
        self.assertEqual(a['candidates'], small[:15])

    def test_pool_deduplicates_and_fills_after_oversize_chunk(self):
        chunks = [{'id': 'a', 'text': '123'}, {'id': 'a', 'text': '123'},
                  {'id': 'b', 'text': 'x'*20}, {'id': 'c', 'text': '456'}]
        result = candidate_pool(chunks, len, max_source_tokens=6)
        self.assertEqual([c['id'] for c in result['candidates']], ['a', 'c'])
        self.assertEqual(result['source_tokens'], 6)
        with self.assertRaises(ValueError):
            candidate_pool(chunks, len, max_source_tokens=True)

    def test_unregistered_candidate_cannot_open_test(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'manifest.json').write_text('{}', encoding='utf-8')
            candidate = root / 'candidate.json'
            candidate.write_text('{"status":"development"}', encoding='utf-8')
            with self.assertRaises(ValueError):
                open_test(root, candidate)
            self.assertFalse((root / 'test-opened.json').exists())


if __name__ == '__main__':
    unittest.main()
