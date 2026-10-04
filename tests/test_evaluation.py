import io
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from evals.run import boundary_summary, precision_recall_f1
from scripts.scifact_pilot import make_index, metrics
from zero_index import JevScorer
from zero_index.parse import parse_blocks, sentence_spans


class EvaluationTests(unittest.TestCase):
    def test_public_data_preserves_declared_sentences_and_evidence_sets(self):
        papers = [{"doc_id": 8, "title": "Paper", "abstract": ["Value 1.5 is measured.", "Another sentence."]}]
        index, lookup = make_index(papers)
        leaves = [n for n in index.root.walk() if n.kind == "sentence"]
        self.assertEqual([index.read(n.node_id)["text"] for n in leaves], papers[0]["abstract"])
        self.assertEqual(lookup[leaves[1].node_id], (8, 1))
        scored = metrics([(8, 0), (8, 1), (9, 0)], {(8, 1), (9, 1)})
        self.assertEqual(scored, {"hit_at_1": 0, "hit_at_3": 1, "recall_at_3": .5, "mrr": .5})

    def test_boundary_counts_include_both_false_splits_and_missed_cuts(self):
        counts = precision_recall_f1([2, 4], [1, 4])
        self.assertEqual(counts, {"tp": 1, "fp": 1, "fn": 1})
        self.assertEqual(boundary_summary([counts])["f1"], .5)
        self.assertEqual(boundary_summary([precision_recall_f1([2], [])])["f1"], 0)

    def test_corpus_labels_address_real_sentences_and_balanced_positions(self):
        corpus = json.loads(Path("evals/pilot.json").read_text())
        positions = []
        for doc in corpus["documents"]:
            source = "\n\n".join(" ".join(p["sentences"]) for p in doc["paragraphs"])
            blocks = parse_blocks(source)
            self.assertEqual(len(blocks), 6)
            for paragraph, block in zip(doc["paragraphs"], blocks):
                self.assertEqual([source[a:b] for a, b in sentence_spans(source, block)], paragraph["sentences"])
                positions.append(paragraph["central"])
            for query in doc["queries"]:
                self.assertTrue(query["gold"])
                for p, s in query["gold"]:
                    self.assertTrue(doc["paragraphs"][p]["sentences"][s])
        self.assertEqual([positions.count(i) for i in range(3)], [12, 12, 12])

    @patch("zero_index.jev.urlopen")
    def test_native_provider_and_usage_do_not_expose_credentials(self, transport):
        transport.return_value = io.BytesIO(json.dumps({"model": "jev-1.13.0", "usage": {"input_tokens": 90, "output_tokens": 20},
            "answers": {"same_topic": {"type": "noul", "noul": .8}}}).encode())
        scorer = JevScorer(api_key="private-test-key", provider="typesafe")
        self.assertEqual(scorer.score("left", "right"), .8)
        request = transport.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.typesafe.ai/v1/systemone")
        self.assertEqual(json.loads(request.data)["model"], "jev-1.13.0")
        metadata = scorer.metadata()
        self.assertEqual(metadata["request_metrics"][0]["input_tokens"], 90)
        self.assertNotIn("private-test-key", json.dumps(metadata))

    def test_provider_model_mismatch_is_rejected_before_network(self):
        with self.assertRaises(ValueError):
            JevScorer(api_key="test", provider="typesafe", model="typesafe/jev-1.13")

    def test_ranking_is_reproducible_across_python_hash_seeds(self):
        command = """
import json
from zero_index import build_index, find
index = build_index('Stars and planets cross a dark sky. The sky has stars. Planets reflect light.')
print(json.dumps(find(index, 'stars sky planets light', 'sky stars and light')))
"""
        outputs = [subprocess.check_output([sys.executable, "-c", command],
                    env={**os.environ, "PYTHONHASHSEED": str(seed)}) for seed in (1, 41)]
        self.assertEqual(outputs[0], outputs[1])


if __name__ == "__main__":
    unittest.main()
