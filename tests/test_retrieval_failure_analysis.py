import unittest
from scripts.retrieval_v3_failure_analysis import best_reference,tree_loss


class FailureLocations(unittest.TestCase):
    def test_alternative_reference_and_tie_are_not_unioned(self):
        self.assertEqual(best_reference(['p1'],[['p1','p2'],['p1']]),['p1'])
        self.assertEqual(best_reference([], [['p1'],['p2']]),['p1'])

    def test_deepest_progress_across_requests_controls_location(self):
        result={'ranking':[], 'need_rankings':[[],[]], 'status':'complete',
            'config':{'acceptance':.2},'trace':[
                {'need_index':0,'candidates':['h'],'scores':[.1],'selected':[]},
                {'need_index':1,'candidates':['h'],'scores':[.9],'selected':['h']},
                {'need_index':1,'candidates':['t'],'scores':[.7],'selected':['t']},
                {'need_index':1,'candidates':['p'],'scores':[.19],'selected':[]}]}
        self.assertEqual(tree_loss('p',{'p':'t','t':'h','h':'r'},
            {'p':'paragraph','t':'topic','h':'heading'},'r',result),'paragraph:below_threshold')

    def test_beam_loss_differs_from_threshold_and_final_rank_loss(self):
        result={'ranking':['p0','p1','p2','p3','p4','p5'],'need_rankings':[[]],
            'status':'complete','config':{'acceptance':.2},'trace':[
                {'need_index':0,'candidates':['h','other'],'scores':[.2,.9],'selected':['other']}]}
        self.assertEqual(tree_loss('p5',{}, {},'r',result),'reached_but_ranked_below_5')
        self.assertEqual(tree_loss('missing',{'missing':'h','h':'r'},
            {'h':'heading','missing':'paragraph'},'r',result),'heading:outside_beam')
        with self.assertRaises(ValueError):tree_loss('p0',{}, {},'r',result)

    def test_budget_stop_is_distinct_from_unexplained_missing_trace(self):
        result={'ranking':[],'need_rankings':[[]],'status':'truncated','trace':[],
                'config':{'acceptance':.2}}
        self.assertEqual(tree_loss('p',{'p':'r'},{'p':'paragraph'},'r',result),'decision_budget')
        result['status']='complete'
        with self.assertRaises(ValueError):tree_loss('p',{'p':'r'},{'p':'paragraph'},'r',result)


if __name__=='__main__':unittest.main()
