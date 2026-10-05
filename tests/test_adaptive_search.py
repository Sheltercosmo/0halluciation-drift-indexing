import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from zero_index import (CentroidRepresentatives, Config, TreeSearchConfig, build_index,
                        pack_tree_context, reselect_representatives, search_tree)
from zero_index.cli import main
from zero_index.parse import parse_blocks, sentence_spans
from zero_index.segment import central_sentences


def groups(text):
    return [sentence_spans(text,b) for b in parse_blocks(text)]


class AdaptiveRepresentatives(unittest.TestCase):
    def test_stops_targets_independently_after_parallel_wave(self):
        source = "A. B. C. D. E.\n\nF. G. H. I. J."
        class Scorer:
            def __init__(self): self.calls=[]
            def representatives(self,pairs):
                self.calls.append(pairs)
                return [.9 if sentence == "A." else .95 if sentence == "H." else .1
                        for sentence,context in pairs]
        scorer=Scorer()
        first,second=central_sentences(source,groups(source),scorer,stop_threshold=.9)
        self.assertEqual([[s for s,_ in wave] for wave in scorer.calls],
                         [["A.","E.","F.","J."],["G.","I."],["H."]])
        self.assertEqual(first["candidate_indices"],[0,4])
        self.assertEqual(first["comparisons"],2)
        self.assertTrue(first["early_stopped"])
        self.assertEqual(first["stop_reason"],"threshold")
        self.assertEqual(second["text"],"H.")
        self.assertTrue(second["exhaustive"])
        self.assertFalse(second["early_stopped"])

    def test_section_stop_does_not_stop_paragraphs_and_counts_inflight_candidates(self):
        source="A. B. C.\n\nD. E. F."
        class Scorer:
            def __init__(self): self.calls=[]
            def representatives(self,pairs):
                self.calls.append(pairs)
                return [.95 if "\nD." in context else .1 for sentence,context in pairs]
        scorer=Scorer()
        section,*paragraphs=central_sentences(source,groups(source),scorer,include_section=True,stop_threshold=.9)
        self.assertEqual([len(wave) for wave in scorer.calls],[8,2])
        self.assertEqual(section["candidate_indices"],[0,2,3,5])
        self.assertEqual(section["comparisons"],4)
        self.assertTrue(section["early_stopped"])
        self.assertTrue(all(p["exhaustive"] for p in paragraphs))

    def test_threshold_is_inclusive_and_wave_winner_retains_outside_in_tie_order(self):
        class Scorer:
            def representatives(self,pairs): return [.9]*len(pairs)
        text="A. B. C. D."
        result=central_sentences(text,groups(text),Scorer(),stop_threshold=.9)[0]
        self.assertEqual(result["text"],"A.")
        self.assertEqual(result["candidate_indices"],[0,3])

    def test_no_threshold_preserves_coalescing_and_budget_still_limits_enabled_search(self):
        class Scorer:
            def __init__(self): self.calls=[]
            def representatives(self,pairs):
                self.calls.append(pairs)
                return [.1]*len(pairs)
        text="A. B. C. D. E."
        scorer=Scorer()
        result=central_sentences(text,groups(text),scorer)[0]
        self.assertEqual(len(scorer.calls),1)
        self.assertNotIn("early_stopped",result)
        result=central_sentences(text,groups(text),Scorer(),budget=3,stop_threshold=.9)[0]
        self.assertEqual(result["candidate_indices"],[0,4,1])
        self.assertFalse(result["early_stopped"])
        self.assertEqual(result["stop_reason"],"candidate_budget")

    def test_singleton_and_empty_targets_make_no_calls(self):
        class Scorer:
            def representatives(self,pairs): raise AssertionError("Unexpected provider call")
        result=central_sentences("A.",[[],[(0,2)]],Scorer(),stop_threshold=.9)
        self.assertIsNone(result[0])
        self.assertFalse(result[1]["early_stopped"])
        self.assertEqual(result[1]["comparisons"],0)

    def test_grouped_embeddings_use_full_context_but_only_eligible_candidates(self):
        vectors={"A.":[1,0],"B.":[1,0],"C.":[1,0],"D.":[1,0]}
        calls=[]
        def embed(text):
            calls.append(text)
            return vectors[text]
        text="A. B. C. D."
        result=central_sentences(text,groups(text),CentroidRepresentatives(embed,model_name="fixture"),
                                 stop_threshold=.9)[0]
        self.assertEqual(result["candidate_indices"],[0,3])
        self.assertEqual(len(calls),4) # Stopping cannot erase required centroid context embeddings.
        self.assertTrue(result["early_stopped"])

    def test_invalid_threshold_or_partial_later_wave_fails(self):
        for threshold in (-1,1.1,float("nan"),float("inf"),True,"0.9"):
            with self.subTest(threshold=threshold),self.assertRaises(ValueError):
                Config(sentence_stop_threshold=threshold)
            with self.subTest(threshold=threshold),self.assertRaises(ValueError):
                central_sentences("A. B.",groups("A. B."),object(),stop_threshold=threshold)
        class Partial:
            def __init__(self): self.calls=0
            def representatives(self,pairs):
                self.calls+=1
                return [.1]*len(pairs) if self.calls==1 else []
        with self.assertRaises(ValueError):
            central_sentences("A. B. C.",groups("A. B. C."),Partial(),stop_threshold=.9)

    def test_build_reselect_and_cli_expose_option(self):
        class Scorer:
            name="fixture"
            def representatives(self,pairs): return [.95]*len(pairs)
            def score(self,left,right): return .9
        tree=build_index("A. B. C. D.",scorer=Scorer(),config=Config(sentence_stop_threshold=.9))
        self.assertTrue(tree.root.children[0].central["early_stopped"])
        full=reselect_representatives(tree,Scorer())
        self.assertTrue(full.root.children[0].central["exhaustive"])
        self.assertIsNone(full.metadata["config"]["sentence_stop_threshold"])
        with tempfile.TemporaryDirectory() as directory:
            source,output=Path(directory)/'input.md',Path(directory)/'index.json'
            source.write_text("Same. Same. Same. Same.")
            with patch('sys.argv',['zero-index','build',str(source),'-o',str(output),'--sentence-stop-threshold','0.9']),patch('sys.stdout',new_callable=io.StringIO):
                main()
            saved=json.loads(output.read_text())
            self.assertEqual(saved['metadata']['config']['sentence_stop_threshold'],.9)
            self.assertTrue(saved['root']['children'][0]['central']['early_stopped'])


class RouteAcceptance(unittest.TestCase):
    def test_threshold_rejection_is_explicit_and_never_falls_back_to_best_bad_branch(self):
        class Low:
            name="fixture"
            def route(self,question,need,cards): return [.4]*len(cards)
        index=build_index("# A\n\nOne fact.\n\n# B\n\nAnother fact.")
        result=search_tree(index,"?",["fact"],Low(),token_count=len,
                           config=TreeSearchConfig(acceptance_threshold=.5))
        self.assertEqual(result['status'],'no_accepted_branches')
        self.assertEqual(len(result['trace'][0]['below_threshold_ids']),2)
        self.assertEqual(result['leaves'],[])
        self.assertEqual(pack_tree_context(index,result,token_count=len)['context'],'')

    def test_acceptance_is_inclusive_and_opt_in(self):
        class Medium:
            name="fixture"
            def route(self,question,need,cards): return [.5]*len(cards)
        index=build_index("A. B.")
        for threshold in (None,.5):
            result=search_tree(index,"?",["fact"],Medium(),token_count=len,
                               config=TreeSearchConfig(acceptance_threshold=threshold))
            self.assertTrue(result['leaves'])
        for threshold in (False,-.1,1.1,float('nan')):
            with self.assertRaises(ValueError): TreeSearchConfig(acceptance_threshold=threshold)


if __name__ == '__main__':
    unittest.main()
