"""第二问的案件关联模型：独立组织验证、有限候选图与社区划分。

分数表示模型在特定抽样案件对上的相似程度，不是法律意义上的归责概率。
组织名称只用作训练/验证标签；伤亡、财产损失、摘要和组织名称均不进入特征。
"""

from __future__ import annotations

from itertools import combinations
from typing import Callable

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    adjusted_rand_score, average_precision_score, normalized_mutual_info_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.neighbors import BallTree


CATEGORY_COLUMNS = (
    "country", "provstate", "attacktype1", "targtype1", "targsubtype1",
    "weaptype1", "weapsubtype1", "suicide", "ishostkid", "multiple", "success",
)
MISSING = "__missing__"
EARTH_KM = 6371.0088
FEATURE_NAMES = [name for column in CATEGORY_COLUMNS for name in
                 (f"{column}_equal", f"{column}_one_missing", f"{column}_both_missing")]
FEATURE_NAMES += ["log_distance_km", "geography_missing", "log_days_apart", "date_missing"]


def _series(data: pd.DataFrame, name: str) -> pd.Series:
    """缺列也返回同长度数据，避免某个附件没有某列时中断。"""
    return data[name] if name in data else pd.Series(np.nan, index=data.index)


def prepare_features(data: pd.DataFrame) -> pd.DataFrame:
    """统一未知值及日期；不拟合全数据统计量，因此不会产生预处理泄漏。"""
    if data.attrs.get("association_prepared"):
        return data
    prepared = pd.DataFrame(index=np.arange(len(data)))
    binary = {"suicide", "ishostkid", "multiple", "success"}
    unknown_codes = {"attacktype1": {9}, "targtype1": {20}, "weaptype1": {13}}
    for column in CATEGORY_COLUMNS:
        source = _series(data, column).reset_index(drop=True)
        numeric = pd.to_numeric(source, errors="coerce")
        text = source.astype("string").str.strip().str.lower().str.replace(r"\.0$", "", regex=True)
        missing = source.isna() | text.isin(["", "nan", "none", "unknown", "-9", "-99"])
        if column != "provstate":
            missing |= numeric.lt(0)
            if column not in binary:
                missing |= numeric.eq(0)
            missing |= numeric.isin(unknown_codes.get(column, set()))
        prepared[column] = text.mask(missing, MISSING).fillna(MISSING).astype(str)
    for column in ("latitude", "longitude"):
        prepared[column] = pd.to_numeric(_series(data, column), errors="coerce").to_numpy()
    bad_geo = (~prepared.latitude.between(-90, 90) | ~prepared.longitude.between(-180, 180)
               | ((prepared.latitude == 0) & (prepared.longitude == 0)))
    prepared.loc[bad_geo, ["latitude", "longitude"]] = np.nan
    dates = pd.to_datetime(pd.DataFrame({
        "year": pd.to_numeric(_series(data, "iyear"), errors="coerce").to_numpy(),
        "month": pd.to_numeric(_series(data, "imonth"), errors="coerce").to_numpy(),
        "day": pd.to_numeric(_series(data, "iday"), errors="coerce").to_numpy(),
    }), errors="coerce")
    prepared["date_day"] = (dates - pd.Timestamp("1970-01-01")).dt.total_seconds() / 86400
    prepared["eventid"] = _series(data, "eventid").reset_index(drop=True).astype("string").fillna("")
    prepared.attrs["association_prepared"] = True
    return prepared


def pair_features(leftdf: pd.DataFrame, rightdf: pd.DataFrame) -> np.ndarray:
    """逐行配对，只比较类别是否相同，绝不把类别编号当作连续大小。"""
    if len(leftdf) != len(rightdf):
        raise ValueError("左右案件对数量必须一致")
    left, right = prepare_features(leftdf), prepare_features(rightdf)
    features = []
    for column in CATEGORY_COLUMNS:
        a, b = left[column].to_numpy(), right[column].to_numpy()
        am, bm = a == MISSING, b == MISSING
        features.extend([(a == b) & ~(am | bm), am ^ bm, am & bm])
    lat1, lat2 = np.radians(left.latitude.to_numpy()), np.radians(right.latitude.to_numpy())
    lon1, lon2 = np.radians(left.longitude.to_numpy()), np.radians(right.longitude.to_numpy())
    h = np.sin((lat1 - lat2) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon1 - lon2) / 2) ** 2
    distance = 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
    gap = np.abs(left.date_day.to_numpy() - right.date_day.to_numpy())
    features.extend([np.nan_to_num(np.log1p(distance)), np.isnan(distance),
                     np.nan_to_num(np.log1p(gap)), np.isnan(gap)])
    return np.column_stack(features).astype(np.float32)


def _predict_pairs(model, prepared, left, right, batch_size=20000):
    """分批构造特征，限制万级事件候选图的峰值内存。"""
    left, right = np.asarray(left, dtype=int), np.asarray(right, dtype=int)
    output = np.empty(len(left), dtype=float)
    for start in range(0, len(left), batch_size):
        end = start + batch_size
        output[start:end] = model.predict_proba(pair_features(
            prepared.iloc[left[start:end]], prepared.iloc[right[start:end]]))[:, 1]
    return output


def _sample_pairs(data, rng, max_positive_per_group=240):
    """平衡抽样；70%负例尝试来自同国，其中一半还要求相同袭击/武器。

    样本对不是总体全部案件对的随机样本，故精确率、AP、分数均依赖本抽样方案。
    同一事件允许出现在同一集合的多对中，但绝不跨训练、验证及对应测试集合。
    """
    groups = data.groupby("gname", sort=True).indices
    positive = set()
    for indices in groups.values():
        count = min(max_positive_per_group, len(indices) * (len(indices) - 1) // 2)
        if len(indices) <= 23:
            possible = list(combinations(indices.tolist(), 2))
            if len(possible) > count:
                possible = [possible[k] for k in rng.choice(len(possible), count, replace=False)]
            positive.update(possible)
        else:
            selected = set()
            while len(selected) < count:
                i, j = rng.choice(indices, 2, replace=False)
                selected.add((min(i, j), max(i, j)))
            positive.update(selected)
    positive = sorted(positive)
    if not positive or len(groups) < 2:
        raise ValueError("独立验证需要至少两个组织，且每个集合需要有重复案件的组织")
    prepared = prepare_features(data)
    country = prepared.country.to_numpy()
    names = data.gname.astype(str).to_numpy()
    by_country = {k: np.flatnonzero(country == k) for k in np.unique(country)}
    pattern = (prepared.country + "|" + prepared.attacktype1 + "|" + prepared.weaptype1).to_numpy()
    by_pattern = {k: np.flatnonzero(pattern == k) for k in np.unique(pattern)}
    negatives = set()
    target = len(positive)
    # 避免只在负例中使用另一种来源或缺失模式；锚点从正例事件中抽取。
    anchors = np.asarray(positive).ravel()
    attempts = 0
    while len(negatives) < target and attempts < target * 100:
        i = int(rng.choice(anchors))
        mode = attempts % 10
        candidates = (by_pattern[pattern[i]] if mode < 4 else
                      by_country[country[i]] if mode < 7 else np.arange(len(data)))
        j = int(rng.choice(candidates))
        attempts += 1
        if i != j and names[i] != names[j]:
            negatives.add((min(i, j), max(i, j)))
    if not negatives:
        raise ValueError("无法形成不同组织的负例")
    pairs = np.asarray(positive + sorted(negatives), dtype=int)
    labels = np.r_[np.ones(len(positive), dtype=int), np.zeros(len(negatives), dtype=int)]
    return pairs, labels


def _pair_metrics(labels, scores, threshold):
    predicted = scores >= threshold
    return {
        "threshold": float(threshold), "pairs": len(labels), "true_positive_pairs": int(labels.sum()),
        "positive_fraction": float(np.mean(labels)),
        "accepted_pairs": int(predicted.sum()),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else np.nan,
        "average_precision": float(average_precision_score(labels, scores)),
    }


def _select_threshold(table, minimum_accepted=20):
    """优先满足精确率目标，再取召回率最高阈值；不满足时如实返回 False。"""
    sufficient = table[table.accepted_pairs >= minimum_accepted]
    qualified = sufficient[sufficient.precision >= .90]
    if len(qualified):
        chosen = qualified.sort_values(["recall", "precision"], ascending=False).iloc[0]
    else:
        pool = sufficient if len(sufficient) else table
        chosen = pool.sort_values(["precision", "recall"], ascending=False).iloc[0]
    return chosen, len(qualified) > 0


def _group_benchmark_scores(model, data, pair_threshold, seed):
    """在真正留出组织内部，2015 年作为参考、2016 年作为查询，验证组级规则。

    每组织最多取 40 起参考、3 起查询，限制计算且避免最大组织淹没结果。
    推理特征不含标签；标签只在分数算完后用于评价。参考组至少有两起事件。
    """
    references, queries = [], []
    for _, group in data.groupby("gname", sort=True):
        ref = group[pd.to_numeric(group.iyear, errors="coerce").eq(2015)]
        query = group[pd.to_numeric(group.iyear, errors="coerce").eq(2016)]
        if len(ref) >= 2 and len(query):
            references.append(ref.sample(min(40, len(ref)), random_state=seed))
            queries.append(query.sample(min(3, len(query)), random_state=seed))
    if len(references) < 2:
        return pd.DataFrame(columns=["query_eventid", "reference_group", "same_group", "score", "support_count"])
    reference = pd.concat(references, ignore_index=True)
    query = pd.concat(queries, ignore_index=True)
    reference_p, query_p = prepare_features(reference), prepare_features(query)
    joined = pd.concat([query_p, reference_p], ignore_index=True)
    joined.attrs["association_prepared"] = True
    # 全组评估上限约数万对，不使用输出社区的信息选择阈值。
    left = np.repeat(np.arange(len(query)), len(reference))
    right = np.tile(np.arange(len(reference)), len(query)) + len(query)
    features = pair_features(joined.iloc[left], joined.iloc[right])
    geography_ok = (features[:, 0] > 0) | ((features[:, -4] <= np.log1p(300)) & (features[:, -3] == 0))
    scores = np.full(len(left), -1.0)
    valid = np.flatnonzero(geography_ok)
    for start in range(0, len(valid), 20000):
        positions = valid[start:start + 20000]
        scores[positions] = model.predict_proba(features[positions])[:, 1]
    scores = scores.reshape(len(query), len(reference))
    records = []
    for q in range(len(query)):
        for group, indices in reference.groupby("gname", sort=True).indices.items():
            top = np.sort(scores[q, indices])[::-1][:3]
            top = top[top >= 0]
            support_count = int((top >= pair_threshold).sum())
            records.append({"query_eventid": str(query.iloc[q].eventid), "reference_group": group,
                "same_group": int(query.iloc[q].gname == group),
                "score": float(top.mean()) if support_count >= 2 else -1.0,
                "support_count": support_count})
    return pd.DataFrame(records)


def clustering_metrics(truth, labels):
    """使用列联表计数计算成对指标，无需建立 n×n 矩阵。"""
    contingency = pd.crosstab(pd.Series(np.asarray(truth)), pd.Series(np.asarray(labels)))
    choose2 = lambda x: np.sum(np.asarray(x) * (np.asarray(x) - 1) / 2)
    true_positive = choose2(contingency.to_numpy())
    predicted_pairs = choose2(contingency.sum(axis=0).to_numpy())
    true_pairs = choose2(contingency.sum(axis=1).to_numpy())
    return {
        "ari": float(adjusted_rand_score(truth, labels)),
        "nmi": float(normalized_mutual_info_score(truth, labels)),
        "pair_precision": float(true_positive / predicted_pairs) if predicted_pairs else 0.0,
        "pair_recall": float(true_positive / true_pairs) if true_pairs else 0.0,
        "clusters": int(len(np.unique(labels))),
        "singleton_events": int(sum(pd.Series(labels).value_counts() == 1)),
    }


def candidate_pairs(data, neighbors=20):
    """地理近邻 + 同国时间近邻 + 同国同作案特征时间近邻，避免全部两两比较。

    跨国候选限于 300 km 地理近邻；同国时间候选不限地理距离。
    这是计算与召回率之间的取舍，远距离跨国组织可能被拆分。
    """
    prepared = prepare_features(data)
    n = len(prepared)
    pairs = set()
    def add(i, j):
        if i != j:
            pairs.add((min(int(i), int(j)), max(int(i), int(j))))
    valid = np.flatnonzero(prepared.latitude.notna().to_numpy())
    if len(valid) > 1:
        coordinates = np.radians(prepared.loc[valid, ["latitude", "longitude"]].to_numpy())
        distances, indices = BallTree(coordinates, metric="haversine").query(
            coordinates, k=min(neighbors + 1, len(valid)))
        countries = prepared.country.to_numpy()
        for local_i, (ds, js) in enumerate(zip(distances, indices)):
            i = valid[local_i]
            for d, local_j in zip(ds, js):
                j = valid[local_j]
                if countries[i] == countries[j] != MISSING or d * EARTH_KM <= 300:
                    add(i, j)
    # 多种 blocking 的并集：同省、同袭击/武器组合可以连接时间更远的同类事件。
    windows = [( ["country"], max(2, neighbors // 2)),
               (["country", "provstate"], max(2, neighbors // 4)),
               (["country", "attacktype1", "weaptype1"], max(2, neighbors // 4))]
    order_date = prepared.date_day.fillna(prepared.date_day.median()).fillna(0).to_numpy()
    for columns, width in windows:
        for key, index in prepared.groupby(columns, sort=True).indices.items():
            country_key = key[0] if isinstance(key, tuple) else key
            if country_key == MISSING:
                continue
            ordered = index[np.argsort(order_date[index], kind="stable")]
            for offset in range(1, width + 1):
                for i, j in zip(ordered[:-offset], ordered[offset:]):
                    add(i, j)
    return np.asarray(sorted(pairs), dtype=int).reshape(-1, 2)


def build_graph(model, unknown_df, threshold, neighbors=20, seed=2025):
    """返回位置编号 i、j 和高于阈值的模型分数；不改变原表索引。"""
    del seed  # 候选构造本身是确定性的；保留参数使调用接口统一。
    prepared = prepare_features(unknown_df)
    pairs = candidate_pairs(prepared, neighbors)
    if not len(pairs):
        return pd.DataFrame(columns=["i", "j", "score"])
    scores = _predict_pairs(model, prepared, pairs[:, 0], pairs[:, 1])
    keep = scores >= threshold
    return pd.DataFrame({"i": pairs[keep, 0], "j": pairs[keep, 1], "score": scores[keep]})


def cluster_graph(n, edges, threshold, seed=2025, resolution=1.2):
    """加权 Louvain 社区划分；区别于连通分量，单条链边不会强制合并整个图。

    孤立事件各保留一个社区。社区只是潜在案件关联集合，不能直接等同真实组织。
    resolution 固定为 1.2，不利用未知事件的结果反向调参。
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    selected = edges.loc[edges["score"] >= threshold]
    graph.add_weighted_edges_from(selected[["i", "j", "score"]].itertuples(index=False, name=None))
    if graph.number_of_edges() == 0:
        return np.arange(n, dtype=int)
    communities = nx.community.louvain_communities(graph, weight="weight", resolution=resolution, seed=seed)
    communities = sorted(communities, key=lambda nodes: min(nodes))
    labels = np.empty(n, dtype=int)
    for label, nodes in enumerate(communities):
        labels[list(nodes)] = label
    return labels


def score_candidates(model, query_df, reference_df, top_k=5):
    """各查询返回候选参考案件最高 top_k 分，用于报告可追溯的相似案例。

    候选须同国或距离不超过 300 km，排除相同 eventid。
    每个参考组较小时穷举组内候选；大组最多取同国的 600 个时间近邻及
    300 km 范围的 200 个地理近邻。这些分数不归一化成嫌疑人概率。
    """
    query, reference = prepare_features(query_df), prepare_features(reference_df)
    results = []
    for q in range(len(query)):
        row = query.iloc[q]
        countries = reference.country.to_numpy()
        same_country = np.flatnonzero((countries == row.country) & (countries != MISSING))
        if len(same_country) > 600:
            days = np.abs(reference.date_day.to_numpy()[same_country] - row.date_day)
            same_country = same_country[np.argsort(np.nan_to_num(days, nan=np.inf), kind="stable")[:600]]
        candidates = set(same_country.tolist())
        if np.isfinite(row.latitude) and np.isfinite(row.longitude):
            lat = np.radians(reference.latitude.to_numpy())
            lon = np.radians(reference.longitude.to_numpy())
            h = np.sin((lat - np.radians(row.latitude)) / 2) ** 2 + np.cos(lat) * np.cos(np.radians(row.latitude)) * np.sin((lon - np.radians(row.longitude)) / 2) ** 2
            distance = 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
            nearest = np.argsort(np.nan_to_num(distance, nan=np.inf), kind="stable")[:200]
            candidates.update(nearest[distance[nearest] <= 300].tolist())
        positions = np.asarray(sorted(candidates), dtype=int)
        if row.eventid and len(positions):
            positions = positions[reference.eventid.to_numpy()[positions] != row.eventid]
        if not len(positions):
            continue
        repeated = query.iloc[np.full(len(positions), q)]
        scores = model.predict_proba(pair_features(repeated, reference.iloc[positions]))[:, 1]
        best = np.argsort(-scores, kind="stable")[:top_k]
        results.extend({"query_index": q, "reference_index": int(positions[k]), "score": float(scores[k])} for k in best)
    return pd.DataFrame(results, columns=["query_index", "reference_index", "score"])


def train_and_validate(known_df, seed=2025, progress: Callable[[str], None] | None = None):
    """训练一次并冻结；验证组织选阈值，两份测试集只报告指标、不调参。"""
    report = progress or (lambda message: None)
    data = known_df.copy().reset_index(drop=True)
    if "gname" not in data or "iyear" not in data:
        raise ValueError("已知组织训练数据必须包含 gname 和 iyear")
    original_n = len(data)
    name = data.gname.astype("string").str.strip()
    clean = name.notna() & ~name.str.lower().isin(["unknown", "", "individual", "unaffiliated individual(s)"])
    for column in ("gname2", "gname3"):
        if column in data:
            second = data[column].astype("string").str.strip().str.lower()
            clean &= second.isna() | second.isin(["", "unknown", "nan"])
    if "guncertain1" in data:
        clean &= pd.to_numeric(data.guncertain1, errors="coerce").eq(0)
    if "individual" in data:
        clean &= ~pd.to_numeric(data.individual, errors="coerce").eq(1)
    data = data.loc[clean].copy()
    data["gname"] = data.gname.astype(str).str.strip()
    data = data[pd.to_numeric(data.iyear, errors="coerce").isin([2015, 2016])]
    if "eventid" in data:
        data = data.drop_duplicates("eventid")
    counts = data.gname.value_counts()
    eligible = sorted(counts[counts >= 4].index)
    if len(eligible) < 12:
        raise ValueError("2015—2016 年至少需要 12 个有 4 起以上明确案件的已知组织进行整组验证")
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(eligible)[rng.permutation(len(eligible))]
    split1, split2 = int(len(shuffled) * .60), int(len(shuffled) * .80)
    train_groups, val_groups, test_groups = shuffled[:split1], shuffled[split1:split2], shuffled[split2:]
    year = pd.to_numeric(data.iyear, errors="coerce")
    subsets = {
        "train_2015": data[data.gname.isin(train_groups) & year.eq(2015)].reset_index(drop=True),
        "validation_unseen_organizations": data[data.gname.isin(val_groups)].reset_index(drop=True),
        "test_unseen_organizations": data[data.gname.isin(test_groups)].reset_index(drop=True),
        "test_temporal_2016": data[data.gname.isin(train_groups) & year.eq(2016)].reset_index(drop=True),
    }
    observed_training_groups = set(subsets["train_2015"].gname)
    subsets["test_temporal_2016"] = subsets["test_temporal_2016"].loc[
        subsets["test_temporal_2016"].gname.isin(observed_training_groups)].reset_index(drop=True)
    # 在构造案件对前检查事件集合隔离；同一事件不能以另一案件对身份流入测试。
    if "eventid" in data:
        identifiers = {split: set(subset.eventid.astype(str)) for split, subset in subsets.items()}
        for left_split, right_split in combinations(identifiers, 2):
            if identifiers[left_split] & identifiers[right_split]:
                raise ValueError(f"事件划分发生重叠：{left_split} / {right_split}")
    pairs_by_split = {}
    audit = []
    for split, subset in subsets.items():
        pairs, targets = _sample_pairs(subset, rng)
        pairs_by_split[split] = pairs, targets
        feature_data = prepare_features(subset)
        negative_pairs = pairs[targets == 0]
        c = feature_data.country.to_numpy()
        attack, weapon = feature_data.attacktype1.to_numpy(), feature_data.weaptype1.to_numpy()
        a, b = negative_pairs[:, 0], negative_pairs[:, 1]
        same_country = (c[a] == c[b]) & (c[a] != MISSING)
        same_method = same_country & (attack[a] == attack[b]) & (weapon[a] == weapon[b])
        audit.append({"split": split, "events": len(subset), "organizations": subset.gname.nunique(),
                      "positive_pairs": int(targets.sum()), "negative_pairs": int((targets == 0).sum()),
                      "negative_same_country_fraction": float(same_country.mean()),
                      "negative_same_country_method_fraction": float(same_method.mean()),
                      "years": ",".join(map(str, sorted(subset.iyear.astype(int).unique())))})
    report(f"独立划分：训练 {len(subsets['train_2015']):,} 起；组织验证 {len(subsets['validation_unseen_organizations']):,} 起")
    training = prepare_features(subsets["train_2015"])
    pairs, targets = pairs_by_split["train_2015"]
    model = HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=15, learning_rate=.08,
        min_samples_leaf=35, l2_regularization=2, early_stopping=False, random_state=seed)
    training_features = pair_features(training.iloc[pairs[:, 0]], training.iloc[pairs[:, 1]])
    model.fit(training_features, targets)
    # 地理基线接受相同事件、相同抽样和相同独立划分，只能看国家/省份/距离。
    geography_columns = [0, 3, len(FEATURE_NAMES) - 4, len(FEATURE_NAMES) - 3]
    geography_model = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=7,
        min_samples_leaf=35, l2_regularization=2, early_stopping=False, random_state=seed)
    geography_model.fit(training_features[:, geography_columns], targets)
    validation = prepare_features(subsets["validation_unseen_organizations"])
    pairs, targets = pairs_by_split["validation_unseen_organizations"]
    val_scores = _predict_pairs(model, validation, pairs[:, 0], pairs[:, 1])
    thresholds = np.unique(np.r_[np.linspace(.50, .95, 10), .975, .99, .995])
    threshold_table = pd.DataFrame([_pair_metrics(targets, val_scores, t) for t in thresholds])
    # 要求至少 20 条被接受验证边，避免单个正确案例造成“100% 精确率”。
    chosen, target_met = _select_threshold(threshold_table)
    threshold = float(chosen.threshold)
    threshold_table["selected"] = threshold_table.threshold.eq(threshold)
    validation_features = pair_features(validation.iloc[pairs[:, 0]], validation.iloc[pairs[:, 1]])
    geography_val_scores = geography_model.predict_proba(validation_features[:, geography_columns])[:, 1]
    geography_thresholds = pd.DataFrame([_pair_metrics(targets, geography_val_scores, t) for t in thresholds])
    geography_chosen, _ = _select_threshold(geography_thresholds)
    metric_rows, score_rows, hard_rows = [], [], []
    for split in ("validation_unseen_organizations", "test_unseen_organizations", "test_temporal_2016"):
        subset = subsets[split]
        pairs, targets = pairs_by_split[split]
        prepared = prepare_features(subset)
        features = pair_features(prepared.iloc[pairs[:, 0]], prepared.iloc[pairs[:, 1]])
        scores = model.predict_proba(features)[:, 1]
        geo_scores = geography_model.predict_proba(features[:, geography_columns])[:, 1]
        for model_name, prediction, cutoff in (("full_pair_model", scores, threshold),
                ("geography_only", geo_scores, float(geography_chosen.threshold))):
            metric_rows.append({"split": split, "model": model_name,
                                **_pair_metrics(targets, prediction, cutoff)})
            same_country = features[:, 0] == 1
            if same_country.any():
                hard_rows.append({"split": split, "model": model_name,
                    "subset": "same_country_pairs", **_pair_metrics(targets[same_country], prediction[same_country], cutoff)})
        # 输出可靠性分箱描述实际分数，不声称经过后验概率校准。
        for lower, upper in zip(np.arange(0, 1, .1), np.arange(.1, 1.1, .1)):
            mask = (scores >= lower) & (scores < upper if upper < 1 else scores <= upper)
            if mask.any():
                score_rows.append({"split": split, "score_lower": lower, "score_upper": min(upper, 1),
                    "pairs": int(mask.sum()), "mean_score": float(scores[mask].mean()),
                    "observed_same_group_fraction": float(targets[mask].mean())})
    report(f"验证集选择相似度阈值 {threshold:.3f}；精确率目标是否达到：{'是' if target_met else '否'}")
    group_validation_scores = _group_benchmark_scores(model, subsets["validation_unseen_organizations"], threshold, seed)
    group_test_scores = _group_benchmark_scores(model, subsets["test_unseen_organizations"], threshold, seed)
    group_rows = []
    if len(group_validation_scores) and len(np.unique(group_validation_scores.same_group)) == 2:
        group_thresholds = pd.DataFrame([_pair_metrics(group_validation_scores.same_group.to_numpy(),
             group_validation_scores.score.to_numpy(), t) for t in thresholds])
        group_chosen, group_target_met = _select_threshold(group_thresholds, minimum_accepted=10)
        group_threshold = float(group_chosen.threshold)
        group_thresholds["selected"] = group_thresholds.threshold.eq(group_threshold)
        for split, records in (("validation_unseen_organizations_2016_vs_2015", group_validation_scores),
                               ("test_unseen_organizations_2016_vs_2015", group_test_scores)):
            if len(records):
                group_rows.append({"split": split, **_pair_metrics(records.same_group.to_numpy(),
                    records.score.to_numpy(), group_threshold)})
    else:
        group_threshold, group_target_met = threshold, False
        group_thresholds = pd.DataFrame()
    report(f"组级 top3 平均分阈值 {group_threshold:.3f}；验证精确率目标是否达到：{'是' if group_target_met else '否'}")
    # 固定组织测试子集做多阈值图聚类，所有行仅作测试报告，不据此再选阈值。
    test_data = subsets["test_unseen_organizations"]
    # 显式 concat 兼容旧版 pandas，并且保留组织名称。
    benchmark = pd.concat([group.sample(min(30, len(group)), random_state=seed)
                           for _, group in test_data.groupby("gname", sort=True)], ignore_index=True)
    graph_thresholds = sorted(set([.50, .70, .85, .95, threshold]))
    graph_edges = build_graph(model, benchmark, min(graph_thresholds), neighbors=20)
    graph_rows = []
    for t in graph_thresholds:
        labels = cluster_graph(len(benchmark), graph_edges, t, seed=seed)
        graph_rows.append({"split": "test_unseen_organizations_fixed_subsample", "threshold": t,
            "selected_on_validation": t == threshold, "events": len(benchmark),
            **clustering_metrics(benchmark.gname.to_numpy(), labels)})
    event_audit = pd.concat([pd.DataFrame({"split": split, "eventid": _series(subset, "eventid").astype(str),
                                         "gname": subset.gname}) for split, subset in subsets.items()], ignore_index=True)
    config = {
        "seed": seed, "threshold": threshold, "precision_target": .90, "precision_target_met": target_met,
        "group_threshold": group_threshold, "group_precision_target_met": group_target_met,
        "group_top_k": 3, "group_min_support": 2,
        "group_score_rule": "同国或300km内候选；top3平均相似度，至少2条top3案例超过案件对阈值",
        "group_validation_limitations": "独立已知组织的2015参考/2016查询代理验证；未知社区可能存在分裂合并，阈值迁移不保证真实归属精确率",
        "validation_precision": float(chosen.precision), "validation_recall": float(chosen.recall),
        "input_known_events": original_n, "clean_known_events_2015_2016": len(data),
        "eligible_organizations": len(eligible), "feature_names": FEATURE_NAMES,
        "model": "HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=15)",
        "production_refit": False, "clustering": "weighted Louvain, resolution=1.2, fixed seed",
        "pair_sampling": "最多240正对/组织；约1:1正负；70%负例抽样尝试同国，40%同时要求相同袭击/武器；最终比例另计",
        "validation_design": "组织按60/20/20分组；只用训练组织2015年训练；验证组织选阈值；独立组织及训练组织2016年测试",
        "score_interpretation": "抽样案件对的模型相似度；未校准为真实作案者后验概率",
        "feature_exclusions": "gname仅作为标签；summary可能直接包含组织名，故排除；伤亡和财产损失只用于危害评估",
        "candidate_limitations": "跨国候选限300km近邻；时间/地理blocking及有限邻居可能漏掉远距离同组织案件",
    }
    tables = {"pair_validation": pd.DataFrame(metric_rows), "threshold_validation": threshold_table,
              "clustering_validation": pd.DataFrame(graph_rows), "score_reliability": pd.DataFrame(score_rows),
              "split_audit": pd.DataFrame(audit), "split_events": event_audit,
              "hard_negative_validation": pd.DataFrame(hard_rows),
              "group_validation": pd.DataFrame(group_rows), "group_threshold_validation": group_thresholds}
    return model, tables, config
