import io
import json
from dataclasses import asdict
import hashlib
from threading import Event, Lock
import unittest
from unittest.mock import patch

from zero_index import Config, JevScorer, build_index
from zero_index.parse import parse_blocks, sentence_spans
from zero_index.segment import central_sentences, segment, segment_runs


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
    def test_outside_in_order_survives_coalescing_waves(self):
        text = "A. B. Core. D. E.\n\nF. G. H.\n\nI. J."
        scorer = WaveScorer()
        results = central_sentences(text, groups_for(text), scorer)
        self.assertEqual([[sentence for sentence, _ in wave] for wave in scorer.waves], [
            ["A.", "E.", "F.", "H.", "I.", "J.", "B.", "D.", "G.", "Core."],
        ])
        self.assertEqual(results[0]["text"], "Core.")
        self.assertEqual([r["candidate_indices"] for r in results], [[0, 4, 1, 3, 2], [0, 2, 1], [0, 1]])
        self.assertTrue(all(result["exhaustive"] for result in results))

    def test_section_and_paragraph_questions_share_waves(self):
        text = "A. B. C.\n\nD. Core. F."
        scorer = WaveScorer()
        results = central_sentences(text, groups_for(text), scorer, include_section=True)
        self.assertEqual([len(wave) for wave in scorer.waves], [12])
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
        self.assertEqual([len(wave) for wave in scorer.waves], [20])
        self.assertEqual(index.root.children[0].central["text"], "Core.")
        self.assertTrue(all(node.central["exhaustive"] for node in index.root.walk() if node.central))

    def test_partial_batch_result_is_rejected(self):
        class Partial:
            def representatives(self, pairs):
                return [.5]
        with self.assertRaises(ValueError):
            central_sentences("A. B.", groups_for("A. B."), Partial())

    def test_sections_share_one_plan_and_ties_keep_outside_in_priority(self):
        scorer = WaveScorer()
        index = build_index("# One\n\nA. B. C.\n\n# Two\n\nD. E. F.", scorer=scorer)
        self.assertEqual(len(scorer.waves), 1)
        sections = [node for node in index.root.walk() if node.kind == "section"]
        self.assertEqual([node.central["text"] for node in sections], ["A.", "D."])
        self.assertEqual([node.central["candidate_indices"] for node in sections], [[0, 2, 1]] * 2)

    def test_section_budget_still_counts_candidates_before_deduplication(self):
        scorer = WaveScorer()
        text = "A. A. B.\n\nC. D. E."
        section, *paragraphs = central_sentences(text, groups_for(text), scorer, 2, include_section=True)
        self.assertEqual(section["candidate_indices"], [0, 2])
        self.assertEqual([p["candidate_indices"] for p in paragraphs], [[0, 2], [0, 2]])


class TopicFrontierTests(unittest.TestCase):
    def test_independent_runs_share_rounds_and_only_advance_after_a_cut(self):
        class Scalar:
            score_kind = "probability"
            def score(self, left, right):
                return .05 if (left, right) == ("A", "C") else .95
        class Batched(Scalar):
            def __init__(self):
                self.rounds = []
            def score_many(self, pairs):
                self.rounds.append(pairs)
                return [self.score(*pair) for pair in pairs]
        text = "A\n\nB\n\nC\n\nD\n\nE\n\nF"
        blocks = parse_blocks(text)
        runs = [[], blocks[:4], blocks[4:], []]
        scorer = Batched()
        result = segment_runs(text, runs, scorer, Config())
        expected = [segment(text, run, Scalar(), Config()) for run in runs]
        self.assertEqual(result, expected)
        self.assertEqual(scorer.rounds, [[("A", "B"), ("E", "F")], [("A", "C")], [("C", "D")]])

    def test_invalid_or_incomplete_topic_batch_fails(self):
        for values in ([.5], [.5, float("nan")]):
            class Bad:
                def score_many(self, pairs):
                    return values
            text = "A\n\nB\n\nC\n\nD"
            blocks = parse_blocks(text)
            with self.subTest(values=values), self.assertRaises(ValueError):
                segment_runs(text, [blocks[:2], blocks[2:]], Bad(), Config())


class SchedulingCompatibilityTests(unittest.TestCase):
    def test_trees_and_decisions_match_archived_method_for_fixed_scores(self):
        from scripts.benchmark_parallel import legacy_package
        legacy = legacy_package()

        class Fixed:
            name = "fixed-test"
            score_kind = "probability"
            def score(self, left, right):
                return .05 if "switch" in right.lower() else .95
            def score_many(self, pairs):
                return [self.score(*pair) for pair in pairs]
            def representatives(self, pairs):
                return [int.from_bytes(hashlib.sha256(json.dumps(pair).encode()).digest()[:2], "big") / 65535
                        for pair in pairs]

        sources = ["", "# Empty", "Single.", "A. B. Core. D. E.\n\nF. G. H.",
                   "# One\r\n\r\nRepeat. Repeat.\r\n\r\nSwitch. New detail.\r\n\r\n# Two\r\n\r\n你好。 More.",
                   "# Contents\n\n- [One](#one)\n- [Two](#two)\n\n# One\n\nA. B.\n\n"
                   "## Nested\n\n```python\n# code\n```\n\n# Two\n\nC. D. E."]
        for source in sources:
            for budget in (None, 2, 3):
                for scorer in (None, Fixed()):
                    with self.subTest(source=source, budget=budget, semantic=scorer is not None):
                        previous = legacy.build_index(source, scorer=scorer,
                                                      config=legacy.Config(sentence_budget=budget))
                        current = build_index(source, scorer=scorer, config=Config(sentence_budget=budget))
                        self.assertEqual(asdict(current.root), asdict(previous.root))
                        self.assertEqual(current.decisions, previous.decisions)

    def test_concurrent_end_to_end_build_matches_legacy_with_fewer_requests(self):
        from scripts.benchmark_parallel import run
        result = run(delay=0, repeats=1, headings=2, paragraphs=4, sentences=5, batch_size=8, workers=3)
        self.assertTrue(result["fixed_score_tree_and_decisions_equal"])
        rows = {name: value["runs"][0] for name, value in result["results"].items()}
        self.assertEqual(len({row["questions"] for row in rows.values()}), 1)
        self.assertEqual(rows["coalesced_serial"]["requests"], rows["coalesced_concurrent"]["requests"])
        self.assertLess(rows["coalesced_serial"]["requests"], rows["legacy_waves_serial"]["requests"])


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


def response_for(body, value=.5):
    return io.BytesIO(json.dumps({"model": "test", "answers": {
        key: {"type": "noul", "noul": value} for key in reversed(body["questions"])
    }}).encode())


class ConcurrentRequestTests(unittest.TestCase):
    @patch("zero_index.jev.urlopen")
    def test_slow_first_request_does_not_block_refill_and_results_keep_input_order(self, transport):
        third_started = Event()
        seen, finished = [], []
        lock = Lock()
        def respond(request, timeout):
            body = json.loads(request.data)
            sentence = body["state"]["sentences"]["s0"]
            with lock:
                seen.append(sentence)
            if sentence == "A":
                if not third_started.wait(3):
                    raise TimeoutError("A slow first request blocked replenishment")
            elif sentence == "C":
                third_started.set()
            with lock:
                finished.append(sentence)
            return response_for(body, {"A": .1, "B": .2, "C": .3}[sentence])
        transport.side_effect = respond
        scorer = JevScorer(api_key="test", batch_size=1, max_concurrency=2)
        pairs = [("A", "context"), ("B", "context"), ("C", "context"), ("A", "context")]
        self.assertEqual(scorer.representatives(pairs), [.1, .2, .3, .1])
        self.assertCountEqual(seen, ["A", "B", "C"])
        self.assertLess(finished.index("C"), finished.index("A"))
        self.assertEqual(scorer.peak_in_flight, 2)
        self.assertEqual(scorer.questions_answered, 3)
        self.assertEqual(sorted(m["request_number"] for m in scorer.request_metrics), [1, 2, 3])
        scorer.representatives(pairs)
        self.assertEqual(transport.call_count, 3)

    @patch("zero_index.jev.urlopen")
    def test_concurrent_call_budget_never_overruns(self, transport):
        two_started = Event()
        lock = Lock()
        attempts = []
        def respond(request, timeout):
            with lock:
                attempts.append(request)
                if len(attempts) == 2:
                    two_started.set()
            if not two_started.wait(3):
                raise TimeoutError
            return response_for(json.loads(request.data))
        transport.side_effect = respond
        scorer = JevScorer(api_key="test", batch_size=1, max_concurrency=4, max_calls=2)
        with self.assertRaisesRegex(RuntimeError, "call budget"):
            scorer.representatives([(str(i), "context") for i in range(10)])
        self.assertEqual(transport.call_count, 2)
        self.assertEqual(scorer.calls, 2)
        self.assertEqual(scorer.questions_answered, 2)
        self.assertEqual(scorer._in_flight, 0)

    @patch("zero_index.jev.urlopen")
    def test_failure_stops_refill_joins_inflight_and_does_not_cache_bad_batch(self, transport):
        good_started, bad_returned = Event(), Event()
        sent = []
        def respond(request, timeout):
            body = json.loads(request.data)
            sentence = body["state"]["sentences"]["s0"]
            sent.append(sentence)
            if sentence == "bad":
                if not good_started.wait(3):
                    raise TimeoutError
                bad_returned.set()
                return io.BytesIO(b'{"answers": {}}')
            good_started.set()
            if not bad_returned.wait(3):
                raise TimeoutError
            return response_for(body)
        transport.side_effect = respond
        scorer = JevScorer(api_key="test", batch_size=1, max_concurrency=2)
        with self.assertRaisesRegex(ValueError, "missing or invalid"):
            scorer.representatives([("bad", "c"), ("good", "c")])
        self.assertCountEqual(sent, ["bad", "good"])
        self.assertNotIn(("representative", "bad", "c"), scorer._cache)
        self.assertEqual(scorer._in_flight, 0)

    @patch("zero_index.jev.urlopen")
    def test_reranking_and_topic_batches_use_the_same_concurrency_controls(self, transport):
        for operation in ("rerank", "score_many"):
            with self.subTest(operation=operation):
                both_started = Event()
                lock = Lock()
                requests = []
                def respond(request, timeout):
                    with lock:
                        requests.append(json.loads(request.data))
                        if len(requests) == 2:
                            both_started.set()
                    if not both_started.wait(3):
                        raise TimeoutError
                    return response_for(json.loads(request.data), .8)
                transport.side_effect = respond
                scorer = JevScorer(api_key="test", batch_size=1, max_concurrency=2)
                if operation == "rerank":
                    cards = [{"node_id": str(i), "text": str(i), "heading_path": [],
                              "paragraph_representative": ""} for i in range(2)]
                    result = scorer.rerank("question", "need", cards)
                else:
                    result = scorer.score_many([("A", "B"), ("C", "D")])
                self.assertEqual(result, [.8, .8])
                self.assertEqual(scorer.peak_in_flight, 2)

    @patch("zero_index.jev.urlopen")
    def test_empty_cached_and_oversized_work_never_launches_requests(self, transport):
        scorer = JevScorer(api_key="test", max_concurrency=4, max_state_chars=10)
        self.assertEqual(scorer.representatives([]), [])
        self.assertEqual(scorer.score_many([]), [])
        scorer._cache[("representative", "A", "B")] = .8
        self.assertEqual(scorer.representatives([("A", "B")]), [.8])
        with self.assertRaisesRegex(ValueError, "max_state_chars"):
            scorer.representatives([("huge", "context" * 100)])
        transport.assert_not_called()

    def test_concurrency_configuration_is_explicit_and_validated(self):
        self.assertEqual(JevScorer(api_key="test").max_concurrency, 1)
        for value in (0, -1, 1.5, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                JevScorer(api_key="test", max_concurrency=value)


if __name__ == "__main__":
    unittest.main()
