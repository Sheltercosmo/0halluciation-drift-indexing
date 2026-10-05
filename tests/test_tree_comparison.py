import json
import math
import unittest
from unittest.mock import patch

from zero_index import (CentroidRepresentatives, Config, DocumentIndex, EmbeddingTreeRouter,
                        JevScorer, TreeSearchConfig, build_index,
                        pack_tree_context, propose_needs, reselect_representatives, search_tree)
from zero_index.parse import parse_blocks, sentence_spans
from zero_index.segment import central_sentence


class FixedRepresentatives:
    name = "fixture-representatives"
    def __init__(self, preferred):
        self.preferred = preferred
    def representatives(self, pairs):
        return [.9 if sentence == self.preferred else .1 for sentence, _ in pairs]
    def score(self, *args):
        raise AssertionError("Representative swap must not call the splitter")


class FixtureRouter:
    name = "fixture-router"
    def __init__(self):
        self.calls = []
    def route(self, question, need, cards):
        self.calls.append(cards)
        return [.9 if ("Biology" if need == "plants" else "Finance") in
                " ".join(card["heading_path"]) else .1 for card in cards]


SOURCE = "# Biology\n\nLeaves use sunlight. Roots take water.\n\n# Finance\n\nBanks offer credit. Loans accrue interest."


class RepresentativeControls(unittest.TestCase):
    def test_centroid_matches_all_pairs_and_embeds_each_sentence_once(self):
        vectors = {"A.": (1, 0), "B.": (0, 1), "C.": (1, 1), "D.": (-1, 0)}
        calls = []
        def embed(text):
            calls.append(text)
            return vectors[text]
        scorer = CentroidRepresentatives(embed, model_name="fixture")
        names = list(vectors)
        values = scorer.score_representative_groups([names], [[0, 3, 1, 2]])[0]
        normalized = {k: tuple(x/math.hypot(*v) for x in v) for k,v in vectors.items()}
        for index, value in zip([0, 3, 1, 2], values):
            expected = sum(sum(a*b for a,b in zip(normalized[names[index]], normalized[name]))
                           for j,name in enumerate(names) if j != index) / 3
            self.assertAlmostEqual(value, (expected+1)/2)
        scorer.score_representative_groups([names], [[0, 1]])
        self.assertEqual(calls, names)

    def test_matched_outside_in_budget_keeps_full_reference_context(self):
        vectors = {"A.": (1,0), "B.": (0,1), "C.": (0,1), "D.": (0,1)}
        calls = []
        def embed(text):
            calls.append(text)
            return vectors[text]
        source = "A. B. C. D."
        central = central_sentence(source, sentence_spans(source, parse_blocks(source)[0]),
                                   CentroidRepresentatives(embed, model_name="fixture"), 2)
        self.assertEqual(central["candidate_indices"], [0,3])
        self.assertEqual(central["text"], "D.")
        self.assertEqual(len(calls), 4)
        self.assertEqual(central["method"], "leave-one-out-cosine-centroid")

    def test_swap_preserves_splits_ids_source_and_original_index(self):
        index = build_index(SOURCE)
        original = index.to_dict()
        swapped = reselect_representatives(index, FixedRepresentatives("Roots take water."))
        signature = lambda tree: [(n.node_id,n.kind,n.start,n.end,[c.node_id for c in n.children])
                                  for n in tree.root.walk()]
        self.assertEqual(signature(index), signature(swapped))
        self.assertEqual(index.decisions, swapped.decisions)
        self.assertEqual(index.to_dict(), original)
        self.assertEqual(swapped.root.children[0].children[0].central["text"], "Roots take water.")
        self.assertEqual(DocumentIndex.from_dict(swapped.to_dict()).to_dict(), swapped.to_dict())

    def test_build_accepts_independent_representative_model(self):
        class Splitter:
            name = "fixture-splitter"
            score_kind = "probability"
            def score(self, left, right):
                return .01
        index = build_index("A. B.\n\nC. D.", scorer=Splitter(),
                            representative_scorer=FixedRepresentatives("B."))
        self.assertEqual(len(index.root.children), 2)
        self.assertEqual(index.root.children[0].central["text"], "B.")
        self.assertEqual(index.metadata["scorer"], "fixture-splitter")
        self.assertEqual(index.metadata["representative_scorer"], "fixture-representatives")

    def test_invalid_centroid_vectors_fail(self):
        for vectors in ({"a": [0,0], "b": [1,0]}, {"a": [1,0], "b": [1]},
                        {"a": [float("nan")], "b": [1]}):
            with self.subTest(vectors=vectors), self.assertRaises(ValueError):
                CentroidRepresentatives(vectors.__getitem__, model_name="fixture").score_representative_groups(
                    [["a", "b"]], [[0,1]])


class TreeControls(unittest.TestCase):
    def test_root_to_leaf_trace_cannot_jump_to_an_unselected_branch(self):
        index = build_index(SOURCE)
        router = FixtureRouter()
        result = search_tree(index, "How do plants grow?", ["plants"], router, token_count=len,
                             config=TreeSearchConfig(beam_width=1))
        allowed = {"root"}
        for step in result["trace"]:
            self.assertEqual(set(step["parents"]), allowed)
            for card in step["candidates"]:
                self.assertIn(index.parent(card["node_id"])["node_id"], allowed)
                self.assertNotIn("children", card)
            allowed = set(step["selected_ids"])
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(result["leaves"]), 1)
        leaf = index.read(result["leaves"][0]["node_id"])
        self.assertEqual(leaf["text"], "Leaves use sunlight.")
        self.assertEqual([s["depth"] for s in result["trace"]], [1,2,3,4])

    def test_budget_checks_whole_round_before_any_calls(self):
        index = build_index(SOURCE)
        for config in (TreeSearchConfig(max_node_scores=3), TreeSearchConfig(max_preview_tokens=1)):
            router = FixtureRouter()
            result = search_tree(index, "?", ["plants", "credit"], router, token_count=len, config=config)
            self.assertEqual(result["status"], "routing_budget_exhausted")
            self.assertEqual(result["leaves"], [])
            self.assertEqual(router.calls, [])
            self.assertEqual(result["node_scores"], 0)

    def test_depth_exhaustion_does_not_turn_preview_into_evidence(self):
        index = build_index(SOURCE)
        result = search_tree(index, "?", ["plants"], FixtureRouter(), token_count=len,
                             config=TreeSearchConfig(max_depth=2))
        self.assertEqual(result["status"], "depth_budget_exhausted")
        packed = pack_tree_context(index, result, token_count=len)
        self.assertEqual(packed["context"], "")
        self.assertEqual(packed["evidence"], [])

    def test_multiple_needs_and_source_packing_are_bounded_and_deduplicated(self):
        index = build_index(SOURCE)
        result = search_tree(index, "?", ["plants", "credit"], FixtureRouter(), token_count=len,
                             config=TreeSearchConfig(beam_width=1))
        # JSON roundtrip turns integer need keys into strings; replay remains valid.
        packed = pack_tree_context(index, json.loads(json.dumps(result)), token_count=len, budget=2048)
        self.assertEqual(len(packed["evidence"]), 2)
        self.assertIn("Roots take water.", packed["context"])
        self.assertIn("Loans accrue interest.", packed["context"])
        self.assertEqual(packed["context_tokens"], len(packed["context"]))
        short = pack_tree_context(index, result, token_count=len, budget=10)
        self.assertEqual(short["context"], "")
        self.assertEqual(len(short["skipped_paragraphs"]), 2)

    def test_planner_sees_titles_and_question_without_representatives_or_gold(self):
        index = build_index(SOURCE)
        states = []
        def planner(state):
            states.append(state)
            return ["plants"]
        self.assertEqual(propose_needs(index, "?", planner), ["plants"])
        self.assertEqual(len(states), 1)
        self.assertNotIn("Leaves use sunlight", json.dumps(states))
        self.assertNotIn("gold", states[0])
        for bad in ([], [""], ["A", "a"], ["x"]*4, ["x"*301]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                propose_needs(index, "?", lambda state: bad)

    def test_embedding_query_and_document_roles_receive_same_preview(self):
        seen = {"query": [], "document": []}
        def query(text):
            seen["query"].append(text)
            return [1,0]
        def document(text):
            seen["document"].append(text)
            return [1,0] if "Biology" in text else [-1,0]
        router = EmbeddingTreeRouter(document, embed_query=query, model_name="fixture")
        cards = [{"text":"Leaves.","heading_path":["Biology"]},
                 {"text":"Loans.","heading_path":["Finance"]}]
        self.assertEqual(router.route("question", "need", cards), [1,0])
        self.assertEqual(seen["document"], ["Biology\nLeaves.", "Finance\nLoans."])
        self.assertEqual(seen["query"], ["Question: question\nEvidence need: need"])

    def test_jev_route_uses_dedicated_routing_judgment_and_cache(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self):
                return json.dumps({"answers":{"q0":{"type":"noul","noul":.8}}}).encode()
        jev = JevScorer(api_key="test-only", max_calls=1)
        cards = [{"node_id":"n1","kind":"heading","text":"Biology","heading_path":["Biology"]}]
        with patch("zero_index.jev.urlopen", return_value=Response()) as transport:
            self.assertEqual(jev.route("?", "plants", cards), [.8])
            self.assertEqual(jev.route("?", "plants", cards), [.8])
            self.assertEqual(transport.call_count, 1)
            body = json.loads(transport.call_args.args[0].data)
        self.assertEqual(body["state"]["previews"]["c0"], cards[0])
        self.assertIn("unseen descendants", body["questions"]["q0"]["instructions"])

    def test_bad_scores_and_empty_document(self):
        class Bad(FixtureRouter):
            def route(self, *args): return [float("nan")]
        with self.assertRaises(ValueError):
            search_tree(build_index("One."), "?", ["need"], Bad(), token_count=len)
        router = FixtureRouter()
        result = search_tree(build_index("# Empty"), "?", ["need"], router, token_count=len)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["leaves"], [])
        self.assertEqual(router.calls, [])


if __name__ == "__main__":
    unittest.main()
