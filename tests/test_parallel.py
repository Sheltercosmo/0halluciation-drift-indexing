import io
import json
import unittest
from unittest.mock import patch

from zero_index import Config, JevScorer, build_index
from zero_index.parse import parse_blocks, sentence_spans
from zero_index.segment import central_sentences


class WaveScorer:
    name = "wave-test"
    score_kind = "probability"

    def __init__(self):
        self.waves = []

    def score(self, left, right):
        return .95

    def representatives(self, pairs):
        self.waves.append(pairs)
        return [.9 if sentence == "Core." else .1 for sentence, _ in pairs]


def groups_for(text):
    return [sentence_spans(text, block) for block in parse_blocks(text)]


class ParallelSearchTests(unittest.TestCase):
    def test_outer_pairs_from_every_paragraph_share_a_wave(self):
        text = "A. B. Core. D. E.\n\nF. G. H.\n\nI. J."
        scorer = WaveScorer()
        results = central_sentences(text, groups_for(text), scorer)
        self.assertEqual([[sentence for sentence, _ in wave] for wave in scorer.waves], [
            ["A.", "E.", "F.", "H.", "I.", "J."],
            ["B.", "D.", "G."],
            ["Core."],
        ])
        self.assertEqual(results[0]["text"], "Core.")
        self.assertEqual([r["candidate_indices"] for r in results], [[0, 4, 1, 3, 2], [0, 2, 1], [0, 1]])
        self.assertTrue(all(result["exhaustive"] for result in results))

    def test_section_and_paragraph_questions_share_waves(self):
        text = "A. B. C.\n\nD. Core. F."
        scorer = WaveScorer()
        results = central_sentences(text, groups_for(text), scorer, include_section=True)
        self.assertEqual([len(wave) for wave in scorer.waves], [8, 4])
        self.assertEqual(results[0]["candidate_indices"], [0, 2, 3, 5, 1, 4])
        self.assertEqual(results[0]["text"], "Core.")
        self.assertEqual(results[2]["text"], "Core.")
        self.assertEqual(len(set(context for _, context in scorer.waves[0])), 3)

    def test_explicit_budget_caps_each_target_without_completing_one_paragraph_first(self):
        text = "A. B. C.\n\nD. E. F."
        scorer = WaveScorer()
        results = central_sentences(text, groups_for(text), scorer, budget=2)
        self.assertEqual(len(scorer.waves), 1)
        self.assertEqual([sentence for sentence, _ in scorer.waves[0]], ["A.", "C.", "D.", "F."])
        self.assertFalse(any(result["exhaustive"] for result in results))

    def test_single_sentence_and_empty_targets_need_no_requests(self):
        scorer = WaveScorer()
        results = central_sentences("Single.", [[], [(0, 7)]], scorer)
        self.assertIsNone(results[0])
        self.assertEqual(results[1]["text"], "Single.")
        self.assertEqual(scorer.waves, [])

    def test_build_uses_exhaustive_parallel_search_by_default(self):
        scorer = WaveScorer()
        text = "A. B. C. D. E.\n\nF. G. Core. H. I."
        index = build_index(text, scorer=scorer)
        self.assertIsNone(Config().sentence_budget)
        self.assertEqual([len(wave) for wave in scorer.waves], [8, 8, 4])
        self.assertEqual(index.root.children[0].central["text"], "Core.")
        self.assertTrue(all(node.central["exhaustive"] for node in index.root.walk() if node.central))

    def test_partial_batch_result_is_rejected(self):
        class Partial:
            def representatives(self, pairs):
                return [.5]
        with self.assertRaises(ValueError):
            central_sentences("A. B.", groups_for("A. B."), Partial())


class JevBatchTests(unittest.TestCase):
    @patch("zero_index.jev.urlopen")
    def test_multiquestion_requests_share_context_and_map_by_id(self, transport):
        payloads = []
        def respond(request, timeout):
            payload = json.loads(request.data)
            payloads.append(payload)
            answers = {key: {"type": "noul", "noul": (int(key[1:]) + 1) / 10}
                       for key in reversed(payload["questions"])}
            return io.BytesIO(json.dumps({"model": "build-test", "answers": answers}).encode())
        transport.side_effect = respond
        scorer = JevScorer(api_key="test-only", batch_size=3)
        pairs = [("A", "ABC"), ("C", "ABC"), ("D", "DEF"), ("F", "DEF"), ("A", "ABC")]
        self.assertEqual(scorer.representatives(pairs), [.1, .2, .3, .1, .1])
        self.assertEqual([len(p["questions"]) for p in payloads], [3, 1])
        self.assertEqual(payloads[0]["state"]["contexts"], {"c0": "ABC", "c1": "DEF"})
        self.assertIn("`contexts.c0`", payloads[0]["questions"]["q1"]["instructions"])
        self.assertIn("`sentences.s1`", payloads[0]["questions"]["q1"]["instructions"])
        self.assertEqual(scorer.metadata()["questions_answered"], 4)
        scorer.representatives(list(reversed(pairs)))
        self.assertEqual(transport.call_count, 2)

    @patch("zero_index.jev.urlopen")
    def test_one_missing_answer_does_not_cache_a_partial_batch(self, transport):
        transport.return_value = io.BytesIO(json.dumps({"answers": {"q0": {"type": "noul", "noul": .5}}}).encode())
        scorer = JevScorer(api_key="test-only")
        with self.assertRaises(ValueError):
            scorer.representatives([("A", "AB"), ("B", "AB")])
        self.assertEqual(scorer._cache, {})
        self.assertEqual(scorer.questions_answered, 0)

    @patch("zero_index.jev.urlopen")
    def test_state_guard_splits_batches_without_truncation(self, transport):
        payloads = []
        def respond(request, timeout):
            payload = json.loads(request.data)
            payloads.append(payload)
            return io.BytesIO(json.dumps({"answers": {key: {"type": "noul", "noul": .5}
                                                      for key in payload["questions"]}}).encode())
        transport.side_effect = respond
        pairs = [("A", "x" * 50), ("B", "y" * 50)]
        scorer = JevScorer(api_key="test-only", max_state_chars=130)
        scorer.representatives(pairs)
        self.assertEqual(len(payloads), 2)
        self.assertEqual([list(p["state"]["contexts"].values())[0] for p in payloads], ["x" * 50, "y" * 50])
        with self.assertRaises(ValueError):
            scorer.representatives([("C", "z" * 200)])
        self.assertEqual(transport.call_count, 2)

    @patch("zero_index.jev.urlopen")
    def test_call_budget_counts_requests_not_questions(self, transport):
        transport.return_value = io.BytesIO(json.dumps({"answers": {"q0": {"type": "noul", "noul": .5},
                                                                 "q1": {"type": "noul", "noul": .5}}}).encode())
        scorer = JevScorer(api_key="test-only", max_calls=1)
        self.assertEqual(scorer.representatives([("A", "AB"), ("B", "AB")]), [.5, .5])
        with self.assertRaises(RuntimeError):
            scorer.representatives([("C", "CD")])
        self.assertEqual(transport.call_count, 1)


if __name__ == "__main__":
    unittest.main()
