"""案件关联模块的边界测试；只用小型内存数据，不训练任何模型。

项目目录内运行：
    PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
"""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


# 支持从项目目录或其父目录发现测试，不依赖 Spyder 的当前工作目录。
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
from association import (  # noqa: E402
    FEATURE_NAMES, candidate_pairs, cluster_graph, clustering_metrics, pair_features,
)


def small_events():
    """类别编号故意不连续，日期、地理位置及部分未知值覆盖常见边界。"""
    return pd.DataFrame({
        "eventid": ["201501010001", "201502020002", "201603030003", "201604040004"],
        "iyear": [2015, 2015, 2016, 2016], "imonth": [1, 2, 3, 4], "iday": [1, 2, 3, 4],
        "country": [11, 11, 22, 22], "provstate": ["North", "North", "East", None],
        "attacktype1": [1, 2, 1, 2], "targtype1": [2, 2, 3, 3],
        "targsubtype1": [11, 11, 22, 22], "weaptype1": [2, 2, 3, 3],
        "weapsubtype1": [5, 5, 6, 6], "suicide": [0, 1, 0, 0],
        "ishostkid": [0, 0, 1, -9], "multiple": [0, 1, 1, 0], "success": [1, 1, 0, 1],
        "latitude": [10.0, 10.1, 20.0, np.nan], "longitude": [30.0, 30.1, 40.0, np.nan],
    })


class PairFeaturesTests(unittest.TestCase):
    def test_swapping_pair_endpoints_is_symmetric(self):
        data = small_events()
        left, right = data.iloc[[0, 1, 2]], data.iloc[[3, 0, 1]]
        forward = pair_features(left, right)
        backward = pair_features(right, left)
        self.assertEqual(forward.shape, (3, len(FEATURE_NAMES)))
        np.testing.assert_allclose(forward, backward, rtol=0, atol=1e-7)

    def test_nominal_recoding_preserves_all_pair_features(self):
        original = small_events()
        recoded = original.copy()
        # 一一重编号保持“是否相同”关系；编号大小/编号间差值不能改变相似度输入。
        mappings = {
            "country": {11: 201, 22: 7}, "provstate": {"North": "region-z", "East": "region-a"},
            "attacktype1": {1: 4, 2: 5}, "targtype1": {2: 7, 3: 8},
            "targsubtype1": {11: 105, 22: 4}, "weaptype1": {2: 5, 3: 6},
            "weapsubtype1": {5: 105, 6: 4},
        }
        for column, mapping in mappings.items():
            recoded[column] = recoded[column].map(mapping)
        left = [0, 0, 1, 2]
        right = [1, 2, 3, 3]
        np.testing.assert_array_equal(
            pair_features(original.iloc[left], original.iloc[right]),
            pair_features(recoded.iloc[left], recoded.iloc[right]),
        )

    def test_missing_values_do_not_count_as_matching_categories(self):
        left = pd.DataFrame({"country": [np.nan, np.nan], "attacktype1": [9, 1], "suicide": [0, 0]})
        right = pd.DataFrame({"country": [np.nan, 11], "attacktype1": [9, 9], "suicide": [0, 0]})
        result = pair_features(left, right)
        index = FEATURE_NAMES.index
        np.testing.assert_array_equal(result[:, index("country_equal")], [0, 0])
        np.testing.assert_array_equal(result[:, index("country_both_missing")], [1, 0])
        np.testing.assert_array_equal(result[:, index("country_one_missing")], [0, 1])
        # GTD 中 attacktype1=9 是未知，两个未知不能作为相同袭击方式的证据。
        np.testing.assert_array_equal(result[:, index("attacktype1_equal")], [0, 0])
        np.testing.assert_array_equal(result[:, index("attacktype1_both_missing")], [1, 0])
        # 二元变量中的 0 是“否”，不应被误处理为缺失。
        np.testing.assert_array_equal(result[:, index("suicide_equal")], [1, 1])
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_array_equal(result[:, index("geography_missing")], [1, 1])
        np.testing.assert_array_equal(result[:, index("date_missing")], [1, 1])

    def test_pair_lengths_must_match(self):
        data = small_events()
        with self.assertRaises(ValueError):
            pair_features(data.iloc[:1], data.iloc[:2])


class CandidateGraphTests(unittest.TestCase):
    def test_candidates_have_no_self_edges_or_duplicates(self):
        data = small_events()
        # 三个同国事件使用相同经纬度，检验地理近邻中“自身不一定排第一”的情况。
        data.loc[:2, "country"] = 11
        data.loc[:2, "latitude"] = 10.0
        data.loc[:2, "longitude"] = 30.0
        pairs = candidate_pairs(data, neighbors=2)
        self.assertGreater(len(pairs), 0)
        self.assertTrue(np.all(pairs[:, 0] < pairs[:, 1]))
        self.assertEqual(len(pairs), len(set(map(tuple, pairs))))
        self.assertTrue(np.all((pairs >= 0) & (pairs < len(data))))
        np.testing.assert_array_equal(pairs, candidate_pairs(data, neighbors=2))
        # 最后一个案件无经纬度且在另一个国家，应保持没有候选的状态。
        self.assertNotIn(3, pairs.ravel())

    def test_isolated_nodes_are_preserved_and_weak_edges_are_removed(self):
        edges = pd.DataFrame({"i": [0, 1], "j": [1, 2], "score": [0.95, 0.30]})
        labels = cluster_graph(5, edges, threshold=0.9)
        self.assertEqual(len(labels), 5)
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(len(set(labels)), 4)
        self.assertEqual(len(set(labels[2:])), 3)
        self.assertNotIn(labels[0], labels[2:])

    def test_empty_graph_keeps_every_event_as_a_singleton(self):
        empty = pd.DataFrame(columns=["i", "j", "score"])
        np.testing.assert_array_equal(cluster_graph(4, empty, threshold=0.9), np.arange(4))
        self.assertEqual(cluster_graph(0, empty, threshold=0.9).shape, (0,))
        self.assertEqual(candidate_pairs(small_events().iloc[:0]).shape, (0, 2))


class ClusteringMetricsTests(unittest.TestCase):
    def test_pair_precision_and_recall_use_actual_pair_counts(self):
        # 真实同组对有 (0,1)、(2,3) 两对；预测同组对为前三个事件的三对。
        # 交集只有 (0,1)，因此 precision=1/3，recall=1/2。
        metrics = clustering_metrics(["A", "A", "B", "B"], [0, 0, 0, 1])
        self.assertAlmostEqual(metrics["pair_precision"], 1 / 3)
        self.assertAlmostEqual(metrics["pair_recall"], 1 / 2)
        self.assertEqual(metrics["clusters"], 2)
        self.assertEqual(metrics["singleton_events"], 1)

    def test_exact_partition_and_singleton_boundary(self):
        exact = clustering_metrics(["A", "A", "B", "B"], [7, 7, 3, 3])
        self.assertEqual(exact["pair_precision"], 1.0)
        self.assertEqual(exact["pair_recall"], 1.0)
        self.assertEqual(exact["ari"], 1.0)
        self.assertEqual(exact["nmi"], 1.0)
        singletons = clustering_metrics(["A", "A", "B", "B"], [0, 1, 2, 3])
        self.assertEqual(singletons["pair_precision"], 0.0)
        self.assertEqual(singletons["pair_recall"], 0.0)
        self.assertEqual(singletons["singleton_events"], 4)


if __name__ == "__main__":
    unittest.main()
