import io
import json
import unittest
from unittest.mock import patch

from zero_index import JevScorer, build_index, find, retrieve
from zero_index.ranking import candidates


class EvidenceRanker:
    name = "test-relevance"

    def __init__(self):
        self.calls = []

    def rerank(self, question, need, cards):
        self.calls.append((question, need, cards))
        return [.95 if "sunset" in card["text"].casefold() else .05 for card in cards]


class RankingTests(unittest.TestCase):
    def test_jev_can_reorder_candidates_against_lexical_order(self):
        index = build_index("# Schedule\n\nThe schedule mentions opening. Actual access begins at sunset.")
        ranker = EvidenceRanker()
        result = find(index, "Opening schedule?", "opening", reranker=ranker, candidate_limit=None)
        self.assertIn("sunset", result["matches"][0]["text"])
        self.assertEqual(result["matches"][0]["relevance"], .95)
        self.assertEqual(ranker.calls[0][:2], ("Opening schedule?", "opening"))
        self.assertFalse(result["prefiltered"])

    def test_prefilter_is_explicit_and_can_be_disabled(self):
        index = build_index("Alpha. Beta. Gamma.")
        self.assertTrue(find(index, "Alpha", "Alpha", candidate_limit=1)["prefiltered"])
        self.assertEqual(find(index, "x", "y", candidate_limit=None)["candidate_count"], 3)
        self.assertFalse(find(index, "x", "y")["lexical_overlap"])

    def test_contents_entry_scopes_real_body_not_toc_text(self):
        source = "# Contents\n\n- [A](#a)\n- [B](#b)\n\n# A\n\nFirst fact.\n\n# B\n\nSecond fact."
        index = build_index(source)
        entry = index.outline()["contents"][1]
        pool = candidates(index, "fact", "fact", scope_id=entry["node_id"])
        self.assertEqual([c["text"] for c in pool["candidates"]], ["Second fact."])
        self.assertTrue(all(c["kind"] == "sentence" for c in pool["candidates"]))
        unresolved = build_index("# Contents\n\n- [Missing](#missing)")
        with self.assertRaises(ValueError):
            find(unresolved, "?", "?", scope_id=unresolved.outline()["contents"][0]["node_id"])

    def test_relevance_validation_and_empty_pool(self):
        class Bad:
            name = "bad"
            def rerank(self, question, need, cards):
                return [float("nan")] * len(cards)
        with self.assertRaises(ValueError):
            find(build_index("Text."), "?", "?", reranker=Bad())
        ranker = EvidenceRanker()
        self.assertEqual(find(build_index("# Empty"), "?", "?", reranker=ranker)["matches"], [])
        self.assertEqual(ranker.calls, [])


class BottomUpTests(unittest.TestCase):
    def test_propose_rerank_read_then_expand_and_deduplicate_evidence(self):
        index = build_index("# Observatory\n\n## Hours\n\nGates open at sunset. They close at dawn.")
        seen = []
        leaf = paragraph = None
        def choose(state):
            nonlocal leaf, paragraph
            history = state["history"]
            seen.append(state)
            if not history:
                return {"action": "find", "need": "the time when access begins"}
            if history[-1]["action"] == "find":
                leaf = history[-1]["result"]["matches"][0]["node_id"]
                return {"action": "read", "node_id": leaf}
            if history[-1]["action"] == "read":
                return {"action": "up", "node_id": leaf}
            paragraph = history[-1]["result"]["node_id"]
            return {"action": "finish", "node_ids": [leaf, paragraph]}
        result = retrieve(index, "When can I enter?", choose, reranker=EvidenceRanker())
        self.assertEqual(result["steps"], 4)
        self.assertEqual(len(result["evidence"]), 1)
        self.assertEqual(result["evidence"][0]["kind"], "paragraph")
        self.assertEqual(result["evidence"][0]["text"], "Gates open at sunset. They close at dawn.")
        self.assertEqual(result["evidence"][0]["node_id"], paragraph)
        self.assertIn("outline", seen[0])

    def test_cannot_skip_to_root_or_finish_with_a_preview(self):
        index = build_index("A fact.")
        with self.assertRaises(ValueError):
            retrieve(index, "?", lambda state: {"action": "read", "node_id": "root"})
        with self.assertRaises(ValueError):
            retrieve(index, "?", lambda state: {"action": "up", "node_id": "root"})
        def finish_preview(state):
            if not state["history"]:
                return {"action": "find", "need": "fact"}
            return {"action": "finish", "node_ids": [state["history"][-1]["result"]["matches"][0]["node_id"]]}
        with self.assertRaises(ValueError):
            retrieve(index, "?", finish_preview)

    def test_large_parent_does_not_silently_truncate_or_become_evidence(self):
        index = build_index("A. " + "B" * 100 + ".")
        leaf = next(n for n in index.root.walk() if n.kind == "sentence")
        steps = iter([{"action": "find", "need": "A"}, {"action": "read", "node_id": leaf.node_id},
                      {"action": "up", "node_id": leaf.node_id}, {"action": "finish", "node_ids": [leaf.node_id]}])
        result = retrieve(index, "A", lambda state: next(steps), max_read_chars=20)
        self.assertEqual(result["history"][-1]["result"]["status"], "read_budget_exceeded")
        self.assertEqual(result["evidence"][0]["text"], "A.")

    def test_ranking_budget_prevents_provider_calls(self):
        ranker = EvidenceRanker()
        steps = iter([{"action": "find", "need": "all"}, {"action": "finish", "node_ids": []}])
        result = retrieve(build_index("A. B."), "?", lambda state: next(steps),
                          reranker=ranker, max_ranked_candidates=1)
        self.assertEqual(result["history"][0]["result"]["status"], "ranking_budget_exceeded")
        self.assertEqual(ranker.calls, [])

    def test_another_need_can_read_another_branch(self):
        index = build_index("# A\n\nAlpha fact.\n\n# B\n\nBeta fact.")
        leaves = [n.node_id for n in index.root.walk() if n.kind == "sentence"]
        steps = iter([{"action": "find", "need": "Alpha"}, {"action": "read", "node_id": leaves[0]},
                      {"action": "find", "need": "Beta"}, {"action": "read", "node_id": leaves[1]},
                      {"action": "finish", "node_ids": leaves}])
        result = retrieve(index, "Facts?", lambda state: next(steps))
        self.assertEqual(len(result["evidence"]), 2)

    def test_steps_and_empty_evidence(self):
        index = build_index("Text.")
        with self.assertRaises(RuntimeError):
            retrieve(index, "?", lambda state: {"action": "find", "need": "?"}, max_steps=1)
        self.assertEqual(retrieve(index, "?", lambda state: {"action": "finish", "node_ids": []})["evidence"], [])


class JevRelevanceTests(unittest.TestCase):
    @patch("zero_index.jev.urlopen")
    def test_relevance_is_batched_uses_original_question_and_preserves_mapping(self, transport):
        requests = []
        def respond(request, timeout):
            body = json.loads(request.data)
            requests.append(body)
            return io.BytesIO(json.dumps({"answers": {key: {"type": "noul", "noul": .8}
                                                      for key in reversed(body["questions"])}}).encode())
        transport.side_effect = respond
        cards = candidates(build_index("A. B. C."), "A", "A")["candidates"]
        scorer = JevScorer(api_key="test", batch_size=2)
        self.assertEqual(scorer.rerank("Original?", "Desired content", cards), [.8] * 3)
        self.assertEqual([len(body["questions"]) for body in requests], [2, 1])
        self.assertEqual(requests[0]["state"]["question"], "Original?")
        self.assertEqual(requests[0]["state"]["requested_content"], "Desired content")
        self.assertIn("contradicts", requests[0]["questions"]["q0"]["criteria"]["true"])
        self.assertIn("candidates.c1.text", requests[0]["questions"]["q1"]["instructions"])
        scorer.rerank("Original?", "Desired content", cards)
        self.assertEqual(len(requests), 2)
        scorer.rerank("Different?", "Desired content", cards)
        self.assertEqual(len(requests), 4)

    @patch("zero_index.jev.urlopen")
    def test_malformed_batch_is_not_cached(self, transport):
        transport.return_value = io.BytesIO(b'{"answers": {"q0": {"type": "noul", "noul": 0.8}}}')
        cards = candidates(build_index("A. B."), "A", "A")["candidates"]
        scorer = JevScorer(api_key="test")
        with self.assertRaises(ValueError):
            scorer.rerank("?", "needed", cards)
        self.assertEqual(scorer._cache, {})


if __name__ == "__main__":
    unittest.main()
