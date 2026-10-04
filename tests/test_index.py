import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from zero_index import Config, DocumentIndex, EmbeddingScorer, JevScorer, build_index, navigate
from zero_index.model import adjust_probability, posterior_same
from zero_index.parse import parse_blocks, sentence_spans
from zero_index.segment import central_sentence, outside_in, segment


class RecordingScorer:
    name = "test-probabilities"
    score_kind = "probability"

    def __init__(self, values):
        self.values = iter(values)
        self.pairs = []

    def score(self, left, right):
        self.pairs.append((left, right))
        return next(self.values)


class ProbabilityTests(unittest.TestCase):
    def test_neutral_evidence_recovers_prior(self):
        self.assertAlmostEqual(posterior_same(0.5, Config()), 0.7)

    def test_probability_adjustment_removes_reference_prior(self):
        config = Config(same_topic_prior=0.8, probability_reference_prior=0.2)
        self.assertAlmostEqual(adjust_probability(0.2, config), 0.8)
        for value in (0, 0.01, 0.3, 0.9, 1):
            self.assertAlmostEqual(adjust_probability(value, Config(same_topic_prior=0.5)), value)

    def test_scores_are_monotone_finite_at_endpoints(self):
        values = [posterior_same(score, Config()) for score in (0, .1, .5, .9, 1)]
        self.assertEqual(values, sorted(values))
        self.assertTrue(all(math.isfinite(value) for value in values))
        self.assertGreater(values[-1], .999)

    def test_invalid_configuration(self):
        for kwargs in ({"same_topic_prior": 1}, {"minimum_drop": -1}, {"same_alpha": 0},
                       {"probability_reference_prior": float("nan")}, {"sentence_budget": 1},
                       {"sentence_budget": 2.5}, {"posterior_cutoff": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Config(**kwargs)


class SegmentationTests(unittest.TestCase):
    def test_anchor_stays_fixed_then_resets_at_cut(self):
        text = "A\n\nB\n\nC\n\nD"
        scorer = RecordingScorer([.95, .05, .95])
        groups, trace = segment(text, parse_blocks(text), scorer, Config(same_topic_prior=.5))
        self.assertEqual(scorer.pairs, [("A", "B"), ("A", "C"), ("C", "D")])
        self.assertEqual([[text[b.start:b.end] for b in group] for group in groups], [["A", "B"], ["C", "D"]])
        self.assertEqual([item["cut"] for item in trace], [False, True, False])
        self.assertEqual(trace[-1]["previous_probability"], .5)

    def test_low_probability_alone_is_not_a_sudden_drop(self):
        config = Config(same_topic_prior=.5, minimum_drop=.2)
        groups, trace = segment("A\n\nB\n\nC", parse_blocks("A\n\nB\n\nC"), RecordingScorer([.4, .3]), config)
        self.assertEqual(len(groups), 1)
        self.assertFalse(any(item["cut"] for item in trace))

    def test_immediate_shift_cuts_against_prior(self):
        groups, _ = segment("A\n\nB", parse_blocks("A\n\nB"), RecordingScorer([.01]), Config())
        self.assertEqual(len(groups), 2)

    def test_invalid_scorer_rejected(self):
        for score in (float("nan"), float("inf"), -1, 1.1):
            with self.subTest(score=score), self.assertRaises(ValueError):
                segment("A\n\nB", parse_blocks("A\n\nB"), RecordingScorer([score]), Config())


class CentralSentenceTests(unittest.TestCase):
    def test_outside_in_without_repeated_middle(self):
        self.assertEqual(list(outside_in(5)), [0, 4, 1, 3, 2])
        self.assertEqual(list(outside_in(4)), [0, 3, 1, 2])
        self.assertEqual(list(outside_in(0)), [])

    def test_budget_can_miss_middle_and_exhaustive_recovers_it(self):
        class Representative:
            def representative(self, sentence, context):
                return .9 if sentence == "Core." else .1
        text = "First. Second. Core. Fourth. Last."
        spans = sentence_spans(text, parse_blocks(text)[0])
        bounded = central_sentence(text, spans, Representative(), 2)
        exhaustive = central_sentence(text, spans, Representative(), None)
        self.assertEqual(bounded["candidate_indices"], [0, 4])
        self.assertEqual(bounded["comparisons"], 2)
        self.assertFalse(bounded["exhaustive"])
        self.assertEqual(exhaustive["text"], "Core.")
        self.assertTrue(exhaustive["exhaustive"])


class TreeTests(unittest.TestCase):
    def test_titles_nest_and_never_compare_across_headings(self):
        scorer = RecordingScorer([])
        index = build_index("Preamble.\n\n# A\n\nAlpha.\n\n### B\n\nBeta.\n\n# C\n\nGamma.", scorer=scorer)
        self.assertEqual([n.kind for n in index.root.children], ["section", "heading", "heading"])
        first_heading = index.root.children[1]
        self.assertEqual(first_heading.children[1].title, "B")
        self.assertNotIn("# C", index.read(first_heading.node_id)["text"])
        self.assertEqual(scorer.pairs, [])

    def test_fenced_code_keeps_hashes_and_blank_lines_as_content(self):
        source = "# Real\n\n```python\n# not heading\n\nx = 3.14\n```\n\nText."
        blocks = parse_blocks(source)
        self.assertEqual([block.kind for block in blocks], ["heading", "paragraph", "paragraph"])
        self.assertTrue(blocks[1].verbatim)
        self.assertEqual(len(sentence_spans(source, blocks[1])), 1)

    def test_setext_heading(self):
        blocks = parse_blocks("Title\n=====\n\nParagraph.")
        self.assertEqual((blocks[0].kind, blocks[0].title, blocks[0].level), ("heading", "Title", 1))

    def test_unicode_crlf_and_repeated_sentences_keep_exact_spans(self):
        source = "# Café\r\n\r\nSame sentence. Same sentence.\r\n\r\n你好。 More words!"
        index = build_index(source)
        sentences = [node for node in index.root.walk() if node.kind == "sentence"]
        self.assertEqual(sentences[0].title, sentences[1].title)
        self.assertNotEqual(sentences[0].start, sentences[1].start)
        for node in index.root.walk():
            if node.central:
                self.assertEqual(node.central["text"], source[node.central["start"]:node.central["end"]])
            read = index.read(node.node_id)
            self.assertEqual(read["text"], source[node.start:node.end])
        self.assertEqual(index.read(sentences[0].node_id)["citation"]["start_line"], 3)
        self.assertEqual(index.read(sentences[-1].node_id)["citation"]["start_line"], 5)

    def test_roundtrip_empty_and_single_sentence(self):
        for text in ("", "  \n", "# Heading only", "One sentence."):
            index = build_index(text)
            loaded = DocumentIndex.from_dict(json.loads(json.dumps(index.to_dict())))
            self.assertEqual(loaded.read("root")["text"], text)
            self.assertEqual(loaded.to_dict(), index.to_dict())

    def test_tampered_source_and_representatives_rejected(self):
        data = build_index("One sentence.").to_dict()
        data["source"] = "Other text."
        with self.assertRaises(ValueError):
            DocumentIndex.from_dict(data)
        data = build_index("One sentence.").to_dict()
        data["root"]["children"][0]["central"]["text"] = "Invented."
        with self.assertRaises(ValueError):
            DocumentIndex.from_dict(data)

    def test_demo_has_two_topics_under_first_heading(self):
        source = Path("examples/sample.md").read_text(encoding="utf-8")
        index = build_index(source)
        heading = index.root.children[0].children[0]
        self.assertEqual(heading.title, "Telescope maintenance")
        self.assertEqual(len(heading.children), 2)
        self.assertTrue(any(item["cut"] for item in index.decisions))


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self):
        return json.dumps(self.payload).encode()


class JevTests(unittest.TestCase):
    @patch("zero_index.jev.urlopen")
    def test_wire_format_cache_and_probability_not_confidence(self, transport):
        transport.return_value = FakeResponse({"model": "test-build", "answers": {
            "same_topic": {"type": "noul", "noul": .81, "confidence": .12}}})
        scorer = JevScorer(api_key="test-only", max_calls=1)
        self.assertEqual(scorer.score("A", "B"), .81)
        self.assertEqual(scorer.score("A", "B"), .81)
        self.assertEqual(transport.call_count, 1)
        request = transport.call_args.args[0]
        self.assertEqual(request.full_url, "https://openrouter.ai/api/alpha/decisions")
        payload = json.loads(request.data)
        self.assertEqual(payload["state"], {"anchor": "A", "candidate": "B"})
        self.assertEqual(payload["questions"]["same_topic"]["type"], "noul")
        self.assertEqual(scorer.metadata()["response_models"], ["test-build"])
        self.assertNotIn("test-only", json.dumps(scorer.metadata()))
        with self.assertRaises(RuntimeError):
            scorer.score("B", "A")

    @patch("zero_index.jev.urlopen")
    def test_representative_has_separate_question(self, transport):
        transport.return_value = FakeResponse({"answers": {"representative": {"type": "noul", "noul": .6}}})
        scorer = JevScorer(api_key="test-only")
        self.assertEqual(scorer.representative("A", "A B"), .6)
        payload = json.loads(transport.call_args.args[0].data)
        self.assertEqual(payload["state"], {"sentence": "A", "context": "A B"})

    @patch("zero_index.jev.urlopen")
    def test_bad_responses_fail_without_silent_lexical_fallback(self, transport):
        for answer in ({"type": "choice", "noul": .5}, {"type": "noul", "noul": True},
                       {"type": "noul", "noul": "0.9"}, {"type": "noul", "noul": -1},
                       {"type": "noul", "noul": float("nan")}, {}):
            transport.return_value = FakeResponse({"answers": {"same_topic": answer}})
            with self.subTest(answer=answer), self.assertRaises(ValueError):
                JevScorer(api_key="test-only").score("A", "B")

    @patch.dict("os.environ", {}, clear=True)
    def test_key_required(self):
        with self.assertRaises(ValueError):
            JevScorer()


class RetrievalTests(unittest.TestCase):
    def test_navigation_can_backtrack_and_returns_exact_read_evidence(self):
        index = build_index("# A\n\nWrong fact.\n\n# B\n\nDesired fact.")
        a, b = index.children()
        instructions = iter([
            {"action": "read", "node_id": a["node_id"]},
            {"action": "children", "node_id": "root"},
            {"action": "read", "node_id": b["node_id"]},
            {"action": "finish", "node_ids": [b["node_id"]]},
        ])
        result = navigate(index, "Desired?", lambda state: next(instructions))
        self.assertEqual(len(result["evidence"]), 1)
        self.assertIn("Desired fact.", result["evidence"][0]["text"])
        self.assertEqual(result["steps"], 4)

    def test_cannot_finish_with_unread_nodes(self):
        with self.assertRaises(ValueError):
            navigate(build_index("Text."), "?", lambda state: {"action": "finish", "node_ids": ["root"]})

    def test_step_budget_and_no_evidence(self):
        index = build_index("Text.")
        with self.assertRaises(RuntimeError):
            navigate(index, "?", lambda state: {"action": "children", "node_id": "root"}, max_steps=2)
        self.assertEqual(navigate(index, "?", lambda state: {"action": "finish", "node_ids": []})["evidence"], [])


class CliTests(unittest.TestCase):
    def test_build_load_and_read(self):
        temporary_root = Path("output").resolve()
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as directory:
            self.assertTrue(Path(directory).resolve().is_relative_to(temporary_root))
            output = Path(directory) / "index.json"
            build = subprocess.run([sys.executable, "-m", "zero_index", "build", "examples/sample.md", "-o", str(output)], capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stderr)
            read = subprocess.run([sys.executable, "-m", "zero_index", "read", str(output), "root"], capture_output=True, text=True)
            self.assertEqual(read.returncode, 0, read.stderr)
            self.assertIn("Observatory", json.loads(read.stdout)["text"])


class EmbeddingTests(unittest.TestCase):
    def test_cosine_cache_and_zero_vector(self):
        calls = []
        vectors = {"a": [1, 0], "b": [0, 1], "zero": [0, 0]}
        def embed(text):
            calls.append(text)
            return vectors[text]
        scorer = EmbeddingScorer(embed, model_name="test")
        self.assertEqual(scorer.score("a", "a"), 1)
        self.assertEqual(scorer.score("a", "b"), 0)
        self.assertEqual(scorer.score("a", "zero"), 0)
        self.assertEqual(calls, ["a", "b", "zero"])

    def test_bad_vectors_fail(self):
        for vectors in ({"a": [], "b": []}, {"a": [1], "b": [1, 2]},
                        {"a": [float("nan")], "b": [1]}):
            with self.subTest(vectors=vectors), self.assertRaises(ValueError):
                EmbeddingScorer(vectors.__getitem__, model_name="test").score("a", "b")


if __name__ == "__main__":
    unittest.main()
