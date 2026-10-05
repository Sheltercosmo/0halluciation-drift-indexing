"""Evaluation integrity checks; no paid calls or model outputs are mocked as results."""
import json
from pathlib import Path
import tempfile
import unittest
from scripts.bounded_clients import Budget, validate_answers, validate_rankings
from scripts.bounded_eval import batches, build_document, covered_paragraphs, context_for, paired_interval, split_span


class ByteTokenizer:
    def encode(self, text, **kwargs):
        return list(text.encode("utf-8"))

    def decode_single_token_bytes(self, token):
        return bytes([token])


class BoundedIntegrityTests(unittest.TestCase):
    def test_budget_persists_reservations_and_rejects_over_cap_before_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "budget.json"
            budget = Budget(path)
            budget.reserve("embedding_inputs", 2, tokens=1000)
            restored = Budget(path)
            self.assertEqual(restored.value["embedding_tokens_reserved"], 1000)
            old = path.read_bytes()
            with self.assertRaises(RuntimeError):
                restored.reserve("embedding_inputs", 1, tokens=130_000_000)
            self.assertEqual(path.read_bytes(), old)
            with self.assertRaises(ValueError):
                restored.reserve("embedding_inputs")
            self.assertEqual(path.read_bytes(), old)

    def test_utf8_split_preserves_every_source_character(self):
        doc = {"text": "界🙂 café " * 200}
        spans = split_span(doc, 0, len(doc["text"]), ByteTokenizer(), maximum=31)
        self.assertEqual("".join(doc["text"][a:b] for a, b in spans), doc["text"])
        self.assertTrue(all(len(doc["text"][a:b].encode()) <= 31 for a, b in spans))
        self.assertEqual(spans[0][0], 0)
        self.assertEqual(spans[-1][1], len(doc["text"]))

    def test_partial_paragraph_is_not_evidence_and_adjacent_spans_can_complete_it(self):
        doc = build_document("d", "title", [("h", ["abcdefgh", "second"])])
        self.assertEqual(covered_paragraphs(doc, [(0, 4)]), [])
        self.assertEqual(covered_paragraphs(doc, [(0, 4), (4, 8)]), ["abcdefgh"])

    def test_context_wrappers_count_towards_budget(self):
        doc = build_document("d", "title", [("head", ["x" * 80, "y" * 80])])
        chunks = [{"id": str(i), **u, "text": doc["text"][u["start"]:u["end"]]} for i, u in enumerate(doc["units"])]
        context, spans, tokens = context_for(doc, chunks, ByteTokenizer(), budget=130)
        self.assertLessEqual(tokens, 130)
        self.assertEqual(len(spans), 1)
        self.assertEqual(tokens, len(context.encode()))

    def test_reader_batch_never_shares_document(self):
        cases = [{"id": str(i), "doc_id": str(i // 2)} for i in range(19)]
        groups = list(batches(cases, 8))
        self.assertEqual(sorted(x["id"] for g in groups for x in g), sorted(x["id"] for x in cases))
        self.assertTrue(all(len({x["doc_id"] for x in g}) == len(g) <= 8 for g in groups))

    def test_schema_checks_coverage_and_permutations(self):
        with self.assertRaises(ValueError):
            validate_answers({"answers": [{"id": "a", "answer": "A"}, {"id": "a", "answer": "B"}]}, ["a", "b"])
        with self.assertRaises(ValueError):
            validate_rankings({"rankings": [{"id": "a", "order": [False, 1]}]}, {"a": 2})
        self.assertEqual(validate_rankings({"rankings": [{"id": "a", "order": [1, 0]}]}, {"a": 2}), {"a": [1, 0]})

    def test_paired_interval_counts_documents_not_questions(self):
        try:
            import numpy
        except ImportError:
            self.skipTest("Optional bounded-eval NumPy is not installed")
        rows = [{"id": str(i), "doc_id": str(i // 2), "method": method, "score": value}
                for i in range(6) for method, value in (("jev_blocking_rerank", .75), ("baseline", .5))]
        result = paired_interval(rows, "score", "baseline", replicates=100)
        self.assertEqual(result["questions"], 6)
        self.assertEqual(result["documents"], 3)
        self.assertEqual(result["difference"], .25)
        self.assertEqual(result["ci95"], [.25, .25])


if __name__ == "__main__":
    unittest.main()
