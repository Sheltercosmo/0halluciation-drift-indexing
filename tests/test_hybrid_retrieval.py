import unittest

from zero_index import (EmbeddingPassageRetriever, TreeSearchConfig, build_index,
                        fuse_retrieval_paths, search_tree, tree_passages)


class HybridRetrievalTests(unittest.TestCase):
    def test_direct_path_can_recover_evidence_outside_tree_route(self):
        source = "# Plants\n\nPlants grow in light.\n\n# Credit\n\nLoans need repayment."
        index = build_index(source)
        class WrongRouter:
            name = "fixture-wrong-route"
            def route(self, question, need, cards):
                return [.9 if "Credit" in card["heading_path"] else .1 for card in cards]
        tree = search_tree(index, "How do plants grow?", ["growth"], WrongRouter(), token_count=len,
                           config=TreeSearchConfig(beam_width=1))
        tree_ranked = tree_passages(index, tree)
        self.assertEqual([p["text"] for p in tree_ranked], ["Loans need repayment."])
        passages = [{"start": n.start, "end": n.end, "text": source[n.start:n.end]}
                    for n in index.root.walk() if n.kind == "paragraph"]
        direct = EmbeddingPassageRetriever(lambda text: [1,0] if "Plants" in text else [0,1],
                                            model_name="fixture", embed_query=lambda text: [1,0])
        dense_ranked = direct.retrieve(source, "How do plants grow?", "growth", passages, limit=1)
        self.assertEqual(dense_ranked[0]["text"], "Plants grow in light.")
        fused = fuse_retrieval_paths(source, tree_ranked, dense_ranked, token_count=len)
        self.assertIn("Plants grow in light.", fused["context"])
        self.assertIn("Loans need repayment.", fused["context"])
        self.assertEqual(len(fused["spans"]), 2)

    def test_overlap_deduplicates_without_filling_unretrieved_gaps(self):
        source = "abcdefghijklmnopqrst"
        passage = lambda a,b: {"start":a,"end":b,"text":source[a:b]}
        result = fuse_retrieval_paths(source, [passage(0,8),passage(16,20)],
                                      [passage(4,12)], token_count=len)
        self.assertEqual(result["spans"], [[0,12],[16,20]])
        self.assertNotIn("mnop", result["context"])
        self.assertEqual(result["context"].count("efgh"), 1)

    def test_duplicate_span_gets_two_path_support_and_budget_is_common(self):
        source = "Alpha. Beta. Gamma."
        a={"start":0,"end":6,"text":"Alpha."}
        b={"start":7,"end":12,"text":"Beta."}
        result=fuse_retrieval_paths(source,[a,b],[b,a],token_count=len,budget=29)
        self.assertEqual(len(result["ranking"]),2)
        self.assertEqual(set(result["ranking"][0]["ranks"]),{"jev_tree","direct_embedding"})
        self.assertLessEqual(len(result["context"]),29)
        self.assertEqual(len(result["selected"]),1)
        with self.assertRaises(ValueError):
            fuse_retrieval_paths(source,[a,a],[b],token_count=len)

    def test_invalid_source_and_empty_results(self):
        with self.assertRaises(ValueError):
            fuse_retrieval_paths("abc",[{"start":0,"end":3,"text":"fake"}],[],token_count=len)
        result=fuse_retrieval_paths("abc",[],[],token_count=len)
        self.assertEqual(result["context_tokens"],0)
        self.assertEqual(result["spans"],[])


if __name__ == "__main__":
    unittest.main()
