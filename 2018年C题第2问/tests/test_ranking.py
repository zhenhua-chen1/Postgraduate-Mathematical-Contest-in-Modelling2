"""表2边界回归测试：不使用真实案件，也不训练模型。"""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from main import rank_queries


class RankingTests(unittest.TestCase):
    def test_rank_direction_missing_candidates_and_minimum_support(self):
        queries = pd.DataFrame({'eventid':[201701010001,201701010002], 'country_txt':['A','B']})
        target = pd.DataFrame({'eventid':range(100,115), 'country_txt':['A']*15})
        labels = np.repeat(np.arange(5),3)
        top = pd.DataFrame({'cluster':range(5),'嫌疑人代号':range(1,6)})

        def fake_candidates(model, queries, ref, top_k):
            # 3号比1号更相似；5号只有一条高分参照；2/4号没有候选。
            group = (int(ref.eventid.iloc[0])-100)//3
            scores = {0:[.92,.91,.82],2:[.99,.98,.96],4:[1.]}.get(group,[])
            return pd.DataFrame([{'query_index':0,'reference_index':i,'score':s}
                                 for i,s in enumerate(scores)],
                                columns=['query_index','reference_index','score'])

        with patch('main.score_candidates',side_effect=fake_candidates):
            ranks,support,evidence,sensitivity = rank_queries(None,queries,target,labels,top,.85,.8)
        self.assertEqual(ranks.loc[0,'3号嫌疑人'],1)
        self.assertEqual(ranks.loc[0,'1号嫌疑人'],2)
        self.assertTrue(pd.isna(ranks.loc[0,'5号嫌疑人']))
        self.assertTrue(ranks.iloc[1,1:].isna().all())
        self.assertTrue(pd.isna(support.loc[0,'2号支持度']))
        self.assertEqual(support.loc[0,'5号支持度'],1)
        self.assertEqual(support.loc[0,'5号高分参照数'],1)
        self.assertIn('没有符合地理条件',support.loc[1,'解释'])
        self.assertEqual(len(evidence),7)
        self.assertTrue(sensitivity['有候选事件数'].eq(1).all())


if __name__=='__main__':
    unittest.main()
