from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from scripts.bounded_eval import build_document
from scripts.jev_evidence_comparison import pairwise_rerank
from scripts.retrieval_standard import RetrievalAdapter, retrieve_standard
from scripts.standard_index import build_standard_index
from zero_index.configuration import IndexingConfig, RetrievalConfig, SearchConfig, MEASURED_JJJ_CONFIG
from zero_index.index import DocumentIndex


def document(n=18):
    return build_document("doc", "Title", [("Methods ::: Evidence", [
        f"Opening {i}. Supporting evidence {i}. Final sentence {i}." for i in range(n)])])


class JevStub:
    name = "typesafe-jev"
    score_kind = "probability"

    def __init__(self):
        self.topic_calls = self.central_calls = 0

    def score(self, left, right):
        self.topic_calls += 1
        return .2

    def representatives(self, pairs):
        self.central_calls += len(pairs)
        return [.95] * len(pairs)


class ConfigurationTests(unittest.TestCase):
    def test_default_and_frozen_measured_configuration_are_distinct(self):
        self.assertEqual(RetrievalConfig().variant, "EEJ")
        self.assertEqual(MEASURED_JJJ_CONFIG.variant, "JJJ")
        self.assertEqual(replace(MEASURED_JJJ_CONFIG, variant="EEJ"), RetrievalConfig())

    def test_json_round_trip_and_checked_in_defaults(self):
        config = replace(RetrievalConfig.for_effort("low"), pairwise_threshold=.65).with_search(acceptance=.3)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            config.save(path)
            self.assertEqual(RetrievalConfig.load(path), config)
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(RetrievalConfig.load(root / "configs/retrieval-standard.json"), RetrievalConfig())
        self.assertEqual(RetrievalConfig.load(root / "configs/retrieval-jjj-measured.json"), MEASURED_JJJ_CONFIG)

    def test_rejects_unknown_invalid_and_inconsistent_settings(self):
        bad = [{"variant": "EEE"}, {"varient": "JJJ"}, {"schema_version": True},
               {"search": {"acceptence": .2}}, {"search": {"acceptance": .9, "refine_below": .8}},
               {"search": {"beam": True}}, {"search": {"acceptance": float("nan")}},
               {"search": {"acceptance": True}}, {"pairwise_threshold": .4},
               {"deferred_search_fraction": float("inf")}, {"dense_candidates": 0},
               {"output_paragraphs": 13}, {"shared_targets": 31},
               {"indexing": {"embedding_split_quantile": 2}},
               {"indexing": {"sentence_budget": 1}},
               {"indexing": {"sentence_stop_threshold": float("nan")}}]
        for value in bad:
            with self.subTest(value=value), self.assertRaises(ValueError):
                RetrievalConfig.from_dict(value)

    def test_effort_presets_preserve_output_and_have_explicit_bounds(self):
        values = [RetrievalConfig.for_effort(name) for name in ("low", "standard", "high")]
        self.assertEqual([v.output_paragraphs for v in values], [5, 5, 5])
        self.assertEqual([v.decision_ceiling() for v in values], [1280, 4990, 9784])
        self.assertEqual(RetrievalConfig.for_effort("standard"), RetrievalConfig())
        with self.assertRaises(ValueError):
            RetrievalConfig.for_effort("fastest")


class IndexAdapterTests(unittest.TestCase):
    def test_eej_index_makes_only_embedding_calls_and_retains_exact_sources(self):
        doc = document(4)
        calls = []
        jev = JevStub()
        def embed(texts, purpose):
            calls.append((purpose, list(texts)))
            return [[2, 1] for _ in texts]
        index = build_standard_index(doc, embed=embed, embedding_model="test", jev=jev)
        self.assertEqual([p for p, _ in calls], ["splitting", "central-sentences"])
        self.assertEqual((jev.topic_calls, jev.central_calls), (0, 0))
        self.assertTrue(calls[0][1][0].startswith("task: sentence similarity | query: "))
        self.assertEqual(index.metadata["standard_index"]["factors"], "EE")
        restored = DocumentIndex.from_dict(index.to_dict())
        for i, unit in enumerate(doc["units"]):
            node = restored._node(f"p{i}")
            self.assertEqual((node.start, node.end), (unit["start"], unit["end"]))
            self.assertEqual(node.central["text"], doc["text"][node.central["start"]:node.central["end"]])
        self.assertEqual(restored.root.children[0].title, "Methods")

    def test_factor_switches_call_only_the_required_providers(self):
        for variant in ("EEJ", "EJJ", "JEJ", "JJJ"):
            with self.subTest(variant=variant):
                jev = JevStub()
                purposes = []
                def embed(texts, purpose):
                    purposes.append(purpose)
                    return [[1, 0] for _ in texts]
                index = build_standard_index(document(3), config=RetrievalConfig(variant=variant),
                    embed=embed, embedding_model="test", jev=jev)
                self.assertEqual(jev.topic_calls > 0, variant[0] == "J")
                self.assertEqual(jev.central_calls > 0, variant[1] == "J")
                self.assertEqual("splitting" in purposes, variant[0] == "E")
                self.assertEqual("central-sentences" in purposes, variant[1] == "E")
                self.assertEqual(index.metadata["standard_index"]["factors"], variant[:2])

    def test_embedding_quantile_changes_real_boundaries_and_records_them(self):
        def embed(texts, purpose):
            return [[1, 0], [1, 0], [0, 1], [-1, 0]] if purpose == "splitting" else [[1, 0]] * len(texts)
        sizes = []
        for quantile in (0, 1):
            config = RetrievalConfig(indexing=IndexingConfig(embedding_split_quantile=quantile))
            index = build_standard_index(document(4), config=config, embed=embed, embedding_model="test")
            sizes.append(len([n for n in index.root.walk() if n.kind == "section"]))
            self.assertEqual(sum(row["cut"] for row in index.decisions), sizes[-1] - 1)
        self.assertEqual(sizes, [3, 1])

    def test_jev_drop_boundary_and_early_stop_are_wired(self):
        for minimum_drop, expected_groups in ((.2, 3), (.9, 1)):
            config = RetrievalConfig(variant="JJJ", indexing=IndexingConfig(
                minimum_drop=minimum_drop, sentence_budget=8, sentence_stop_threshold=.9))
            index = build_standard_index(document(3), config=config, jev=JevStub())
            self.assertEqual(len([n for n in index.root.walk() if n.kind == "section"]), expected_groups)
            for node in index.root.walk():
                if node.central:
                    wave_size = 2 if node.kind == "paragraph" else 2 * len(node.children)
                    self.assertLessEqual(len(node.central["candidate_indices"]), wave_size)
                    self.assertLess(len(node.central["candidate_indices"]), 8)
                    self.assertEqual(node.central["stop_reason"], "threshold")

    def test_invalid_providers_documents_and_vectors_fail_explicitly(self):
        with self.assertRaises(ValueError):
            build_standard_index(document())
        with self.assertRaises(ValueError):
            build_standard_index(document(), config=MEASURED_JJJ_CONFIG)
        bad = document()
        bad["units"][1]["paragraph"] = 99
        with self.assertRaises(ValueError):
            build_standard_index(bad, embed=lambda *a: self.fail("No provider call expected"), embedding_model="test")
        for vectors in ([], [[0, 0]] * 2, [[float("nan"), 1]] * 2, [[1], [1, 2]]):
            with self.subTest(vectors=vectors), self.assertRaises(ValueError):
                build_standard_index(document(2), embed=lambda *a: vectors, embedding_model="test")


class RetrievalAdapterTests(unittest.TestCase):
    def adapter(self, config=None, route=None):
        def compare(q, pairs):
            return [.9 if int(a["node_id"][1:]) > int(b["node_id"][1:]) else .1 for a, b in pairs]
        return RetrievalAdapter(route=route or (lambda q, n, cards: [.95] * len(cards)),
            compare=compare, select=lambda q, p, ids: [int(i[1:]) / 100 for i in ids],
            embed=lambda texts, purpose: [[1, 0]] * len(texts), embedding_model="test",
            config=config or RetrievalConfig())

    def test_limits_reach_actual_pairwise_and_shared_calls_and_whole_paragraph_output(self):
        config = RetrievalConfig(pairwise_candidates=6, shared_targets=4, output_paragraphs=2)
        adapter = self.adapter(config)
        doc = document()
        index = adapter.build_index(doc)
        result = adapter.retrieve(doc, index, "original question")
        self.assertEqual(result["variant"], "EEJ")
        self.assertEqual(result["pairwise"]["decisions"], 6 * 5)
        self.assertEqual(result["selection"]["result"]["decisions"], 2 * 4)
        self.assertEqual(len(result["selected"]), 2)
        self.assertEqual(result["selection"]["packet"]["selection_size"], 2)
        self.assertEqual(result["configuration"], config.to_dict())
        for passage in result["selection"]["packet"]["passages"]:
            self.assertEqual(passage["text"], doc["text"][passage["start"]:passage["end"]])

    def test_acceptance_and_total_routing_cap_change_exploration(self):
        doc = document(3)
        def route(q, n, cards):
            return [.5] * len(cards)
        low = self.adapter(RetrievalConfig().with_search(acceptance=.4), route)
        index = low.build_index(doc)
        self.assertTrue(low.retrieve(doc, index, "q")["selected"])
        high = self.adapter(RetrievalConfig().with_search(acceptance=.6), route)
        self.assertFalse(high.retrieve(doc, index, "q")["selected"])
        limited = self.adapter(RetrievalConfig().with_search(max_decisions=2), route)
        result = limited.retrieve(doc, index, "q")
        self.assertLessEqual(result["retrieval"]["decisions"], 2)
        self.assertEqual(result["retrieval"]["status"], "budget_exhausted")

    def test_hybrid_uses_independent_dense_candidates_with_same_selector(self):
        config = RetrievalConfig(dense_candidates=2)
        adapter = self.adapter(config, lambda q, n, cards: [0.] * len(cards))
        doc = document()
        result = adapter.retrieve(doc, adapter.build_index(doc), "q", dense_ranking=["p17", "p16", "p15"])
        self.assertEqual(result["mode"], "hybrid")
        self.assertEqual(set(result["pairwise"]["candidate_pool"]), {"p17", "p16"})
        self.assertEqual(set(result["selected"]), {"p17", "p16"})

    def test_beam_and_deferred_allowance_control_real_branch_expansion(self):
        doc = build_document("doc", "Title", [(name, [f"Evidence in {name}."]) for name in ("A", "B", "C")])
        config = RetrievalConfig(deferred_search_fraction=0).with_search(beam=1)
        narrow = self.adapter(config)
        index = narrow.build_index(doc)
        first = narrow.retrieve(doc, index, "q")
        backtrack = self.adapter(replace(config, deferred_search_fraction=1)).retrieve(doc, index, "q")
        wide = self.adapter(config.with_search(beam=3)).retrieve(doc, index, "q")
        self.assertEqual(len(first["retrieval"]["ranking"]), 1)
        self.assertEqual(len(backtrack["retrieval"]["ranking"]), 3)
        self.assertEqual(len(wide["initial_retrieval"]["ranking"]), 3)
        self.assertEqual(first["retrieval"]["extra_decisions"], 0)
        self.assertLessEqual(backtrack["retrieval"]["extra_decisions"], first["initial_retrieval"]["decisions"])

    def test_mismatched_index_or_dense_ids_fail_before_inference(self):
        doc = document()
        adapter = self.adapter()
        index = adapter.build_index(doc)
        no_call = lambda *a: self.fail("Invalid input must not incur inference")
        for changes in ({"config": MEASURED_JJJ_CONFIG},
                        {"config": RetrievalConfig(indexing=IndexingConfig(sentence_budget=2))},
                        {"dense_ranking": ["p100"]}, {"dense_ranking": ["p1", "p1"]}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                retrieve_standard(doc, index, "q", [], no_call, no_call, no_call, **changes)

    def test_pairwise_confidence_threshold_changes_win_into_tie(self):
        doc = document(2)
        values = lambda *a: [.6, .4]
        default = pairwise_rerank(doc, "q", ["p0", "p1"], values)
        conservative = pairwise_rerank(doc, "q", ["p0", "p1"], values, threshold=.7)
        self.assertEqual(default["scores"], [1, 0])
        self.assertEqual(conservative["scores"], [.5, .5])
        with self.assertRaises(ValueError):
            pairwise_rerank(doc, "q", [], values, threshold=.4)


if __name__ == "__main__":
    unittest.main()
