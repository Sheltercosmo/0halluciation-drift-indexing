import unittest
from scripts.retrieval_v3_statistics import holm,cluster_summary,contrast


class StatisticsTests(unittest.TestCase):
    def test_holm_preserves_familywise_monotonicity(self):
        self.assertEqual(holm([.04,.01,.03]),[.06,.03,.06])

    def test_cluster_mean_keeps_questions_weighted(self):
        rows=[]
        for doc,q,v in [('a','1',1),('a','2',1),('b','3',0)]:
            for m in ['E','J']:rows.append({'doc_id':doc,'id':q,'method':m,'recall':v})
        means,boot,sums,counts=cluster_summary(rows,['E','J'],'recall',draws=100)
        self.assertAlmostEqual(means[0],2/3)
        r=contrast(sums,counts,boot,[-1,1],draws=100)
        self.assertEqual(r['delta'],0);self.assertEqual(r['ci95'],[0,0]);self.assertEqual(r['p_two_sided'],1)

    def test_missing_results_cannot_change_denominator(self):
        rows=[{'doc_id':'a','id':'1','method':'E','recall':.5}]
        with self.assertRaisesRegex(ValueError,'coverage'):cluster_summary(rows,['E','J'],'recall')


if __name__=='__main__':unittest.main()
