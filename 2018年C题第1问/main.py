#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2018 年研究生数学建模竞赛 C 题第一问：客观赋权危害分级模型。

方法：语义清洗 -> 留一平滑类别风险编码 -> 两层 CRITIC -> 一维 KMeans
有序分级 -> 熵权/PCA/Bootstrap 稳定性验证。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


RANDOM_SEED = 2025
DIRECT_IMPACT_WEIGHT = 0.70
HIGH_RISK_EVENTS = [
    200108110012,
    200511180002,
    200901170021,
    201402110015,
    201405010071,
    201411070002,
    201412160041,
    201508010015,
    201705080012,
]

USE_COLUMNS = [
    "eventid", "iyear", "imonth", "iday", "country_txt", "region_txt",
    "city", "latitude", "longitude", "summary", "extended", "success",
    "suicide", "multiple", "attacktype1", "targtype1", "weaptype1",
    "nkill", "nkillter", "nwound", "nwoundte", "nhostkid", "property",
    "propextent", "propvalue",
]


def progress(message: str) -> None:
    """立即输出运行进度，确保脚本和 Jupyter 公屏都能实时看到。"""
    print(message, flush=True)


def numeric(series: pd.Series) -> pd.Series:
    """把异常文本安全转为缺失值，避免单个脏数据中断计算。"""
    return pd.to_numeric(series, errors="coerce")


def percentile_indicator(series: pd.Series, log_transform: bool = False) -> pd.Series:
    """把正计数映射为百分位；真实的零仍保持为零。"""
    values = numeric(series).replace([np.inf, -np.inf], np.nan)
    observed = values.dropna()
    fill = float(observed.median()) if not observed.empty else 0.0
    values = values.fillna(fill).clip(lower=0)
    if log_transform:
        values = np.log1p(values)
    result = pd.Series(np.zeros(len(values)), index=values.index, dtype=float)
    positive = values > 0
    if positive.any():
        result.loc[positive] = values.loc[positive].rank(method="average", pct=True)
    return result


def robust_unit(series: pd.Series) -> pd.Series:
    """按 1%—99% 分位稳健归一化到 [0, 1]，降低极端值影响。"""
    values = numeric(series).replace([np.inf, -np.inf], np.nan)
    fill = float(values.median()) if values.notna().any() else 0.0
    values = values.fillna(fill)
    low, high = values.quantile([0.01, 0.99])
    if not np.isfinite(high - low) or high <= low:
        return pd.Series(np.zeros(len(values)), index=values.index, dtype=float)
    return ((values - low) / (high - low)).clip(0, 1).astype(float)


def leave_one_out_risk(category: pd.Series, anchor: pd.Series, smoothing: float = 50.0) -> pd.Series:
    """用留一平滑风险编码处理类别变量，避免把类别编号误当连续数值。"""
    key = category.fillna("__UNKNOWN__").astype(str)
    table = pd.DataFrame({"key": key, "anchor": anchor})
    sums = table.groupby("key", observed=True)["anchor"].transform("sum")
    counts = table.groupby("key", observed=True)["anchor"].transform("count")
    global_mean = float(anchor.mean())
    encoded = (sums - anchor + smoothing * global_mean) / (counts - 1 + smoothing)
    return robust_unit(encoded)


def critic_weights(frame: pd.DataFrame) -> pd.Series:
    """计算 CRITIC 权重：指标差异越大、与其他指标重复越少，权重越高。"""
    x = frame.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if x.shape[1] == 1:
        return pd.Series([1.0], index=x.columns, dtype=float)
    std = x.std(axis=0, ddof=0)
    corr = x.corr().fillna(0.0).clip(-1, 1)
    conflict = (1.0 - corr).sum(axis=1)
    information = std * conflict
    if not np.isfinite(information.sum()) or information.sum() <= 1e-12:
        return pd.Series(1.0 / x.shape[1], index=x.columns, dtype=float)
    return information / information.sum()


def entropy_weights(frame: pd.DataFrame) -> pd.Series:
    """计算熵权，仅用于和主模型进行稳健性对照。"""
    x = frame.astype(float).clip(lower=0).to_numpy()
    n, m = x.shape
    if m == 1:
        return pd.Series([1.0], index=frame.columns, dtype=float)
    col_sums = x.sum(axis=0)
    p = np.divide(x, col_sums, out=np.full_like(x, 1.0 / n), where=col_sums > 0)
    entropy = -(p * np.log(np.clip(p, 1e-15, None))).sum(axis=0) / np.log(n)
    diversity = np.maximum(1.0 - entropy, 0.0)
    if diversity.sum() <= 1e-12:
        diversity[:] = 1.0
    return pd.Series(diversity / diversity.sum(), index=frame.columns, dtype=float)


def prepare_indicators(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]], pd.DataFrame]:
    """清洗原始字段，构造危害指标、指标分组和数据完整性标签。"""
    # 总伤亡中包含袭击者伤亡；危害对象应聚焦受害者，因此先予以扣除。
    nkill = numeric(data["nkill"])
    nkillter = numeric(data["nkillter"]).fillna(0)
    nwound = numeric(data["nwound"])
    nwoundte = numeric(data["nwoundte"]).fillna(0)

    victim_killed = (nkill - nkillter).clip(lower=0)
    victim_wounded = (nwound - nwoundte).clip(lower=0)
    hostages = numeric(data["nhostkid"]).clip(lower=0)

    # GTD 中 -9 表示未知，不等价于“无财产损失”。
    property_flag = numeric(data["property"]).replace(-9, np.nan).map({0.0: 0.0, 1.0: 1.0})
    extent = numeric(data["propextent"]).map({1.0: 1.0, 2.0: 2 / 3, 3.0: 1 / 3})
    extent = extent.where(property_flag.ne(0), 0.0)
    property_score = extent.fillna(property_flag.map({1.0: 0.5, 0.0: 0.0})).fillna(0.0)

    # 对长尾计数先取对数、再做百分位转换，避免少数极端事件支配模型。
    indicators = pd.DataFrame(index=data.index)
    indicators["victim_fatalities"] = percentile_indicator(victim_killed, log_transform=True)
    indicators["victim_injuries"] = percentile_indicator(victim_wounded, log_transform=True)
    indicators["hostages_kidnapped"] = percentile_indicator(hostages, log_transform=True)
    indicators["property_damage"] = property_flag.fillna(property_flag.median()).fillna(0.0)
    indicators["property_extent"] = property_score
    indicators["attack_success"] = numeric(data["success"]).fillna(0).clip(0, 1)
    indicators["suicide_attack"] = numeric(data["suicide"]).fillna(0).clip(0, 1)
    indicators["extended_event"] = numeric(data["extended"]).fillna(0).clip(0, 1)
    indicators["multiple_incident"] = numeric(data["multiple"]).fillna(0).clip(0, 1)

    # 类别风险编码的锚点仅使用直接后果，避免类别编号本身制造虚假距离。
    direct_anchor = indicators[
        ["victim_fatalities", "victim_injuries", "hostages_kidnapped", "property_extent"]
    ].mean(axis=1)
    indicators["target_type_risk"] = leave_one_out_risk(data["targtype1"], direct_anchor)
    indicators["attack_type_risk"] = leave_one_out_risk(data["attacktype1"], direct_anchor)
    indicators["weapon_type_risk"] = leave_one_out_risk(data["weaptype1"], direct_anchor)
    indicators["region_risk"] = leave_one_out_risk(data["region_txt"], direct_anchor)

    groups = {
        "casualty_impact": ["victim_fatalities", "victim_injuries", "hostages_kidnapped"],
        "property_impact": ["property_damage", "property_extent"],
        "operational_complexity": ["attack_success", "suicide_attack", "extended_event", "multiple_incident"],
        "contextual_risk": ["target_type_risk", "attack_type_risk", "weapon_type_risk", "region_risk"],
    }

    # 完整性分数不直接改变危害等级，但随结果输出，便于判断结论可信度。
    quality = pd.DataFrame(index=data.index)
    quality["known_fatalities"] = nkill.notna().astype(int)
    quality["known_injuries"] = nwound.notna().astype(int)
    quality["known_hostages"] = hostages.notna().astype(int)
    quality["known_property"] = property_flag.notna().astype(int)
    quality["known_property_extent"] = extent.notna().astype(int)
    quality["data_completeness"] = quality.mean(axis=1)
    quality["data_quality"] = pd.cut(
        quality["data_completeness"],
        bins=[-np.inf, 0.4, 0.8, np.inf],
        labels=["low", "medium", "high"],
    ).astype(str)
    return indicators.astype(float), groups, quality


def fit_hierarchical_weights(
    indicators: pd.DataFrame,
    groups: dict[str, list[str]],
    method: str = "critic",
    fit_index: np.ndarray | None = None,
) -> tuple[pd.Series, pd.DataFrame, pd.Series, dict[str, pd.Series]]:
    """分层计算指标权重，并合成为每起事件的连续危害得分。"""
    fit = indicators if fit_index is None else indicators.iloc[fit_index]
    weight_function = critic_weights if method == "critic" else entropy_weights
    within: dict[str, pd.Series] = {}
    group_scores = pd.DataFrame(index=indicators.index)
    for group, columns in groups.items():
        weights = weight_function(fit[columns])
        within[group] = weights
        group_scores[group] = indicators[columns].mul(weights, axis=1).sum(axis=1)

    # 人员伤亡和财产损失属于“任一严重即可构成重大危害”的非补偿关系。
    # 取二者较高值，避免无财产损失的重大伤亡事件被线性加权抵消。
    group_scores["direct_impact"] = group_scores[["casualty_impact", "property_impact"]].max(axis=1)
    top_columns = ["direct_impact", "operational_complexity", "contextual_risk"]
    fit_groups = group_scores[top_columns] if fit_index is None else group_scores[top_columns].iloc[fit_index]
    raw_group_weights = weight_function(fit_groups)
    # 直接后果是危害等级的基础，固定占 70%；剩余权重由客观方法在修正维度间分配。
    # 这是可解释的结构约束，避免纯离散度赋权让伤亡/财产被背景变量淹没。
    modifier = raw_group_weights[["operational_complexity", "contextual_risk"]]
    modifier = modifier / modifier.sum()
    group_weights = pd.Series({
        "direct_impact": DIRECT_IMPACT_WEIGHT,
        "operational_complexity": float((1 - DIRECT_IMPACT_WEIGHT) * modifier["operational_complexity"]),
        "contextual_risk": float((1 - DIRECT_IMPACT_WEIGHT) * modifier["contextual_risk"]),
    })
    final_score = group_scores[top_columns].mul(group_weights, axis=1).sum(axis=1)
    return final_score, group_scores, group_weights, within


def ordered_levels(score: pd.Series, random_seed: int = RANDOM_SEED) -> tuple[np.ndarray, pd.DataFrame]:
    """对一维得分聚成五类，再按聚类中心从高到低编号为 1—5 级。"""
    model = KMeans(n_clusters=5, n_init=50, random_state=random_seed)
    cluster = model.fit_predict(score.to_numpy().reshape(-1, 1))
    centers = model.cluster_centers_.ravel()
    order = np.argsort(-centers)
    mapping = {int(cluster_id): level + 1 for level, cluster_id in enumerate(order)}
    levels = np.array([mapping[int(value)] for value in cluster], dtype=int)
    rows = []
    for cluster_id in order:
        mask = cluster == cluster_id
        rows.append({
            "level": mapping[int(cluster_id)],
            "score_center": float(centers[cluster_id]),
            "score_min": float(score.to_numpy()[mask].min()),
            "score_max": float(score.to_numpy()[mask].max()),
            "event_count": int(mask.sum()),
        })
    return levels, pd.DataFrame(rows).sort_values("level")


def flatten_weights(group_weights: pd.Series, within: dict[str, pd.Series]) -> pd.DataFrame:
    """把分层权重展开成便于写入 Excel 和论文表格的长表。"""
    rows = []
    for group, weights in within.items():
        top_group = "direct_impact" if group in {"casualty_impact", "property_impact"} else group
        for indicator, local_weight in weights.items():
            rows.append({
                "dimension": group,
                "indicator": indicator,
                "top_dimension": top_group,
                "dimension_weight": float(group_weights[top_group]),
                "within_dimension_weight": float(local_weight),
                "linear_equivalent_weight": (
                    np.nan
                    if top_group == "direct_impact"
                    else float(group_weights[top_group] * local_weight)
                ),
                "aggregation_rule": "max(casualty, property)" if top_group == "direct_impact" else "weighted_sum",
            })
    return pd.DataFrame(rows).sort_values(
        ["dimension_weight", "within_dimension_weight"], ascending=False
    )


def pca_score(group_scores: pd.DataFrame) -> pd.Series:
    """生成 PCA 对照得分，并统一方向为“数值越大，危害越高”。"""
    x = StandardScaler().fit_transform(group_scores)
    values = PCA(n_components=1, random_state=RANDOM_SEED).fit_transform(x).ravel()
    if np.corrcoef(values, group_scores.mean(axis=1))[0, 1] < 0:
        values = -values
    return robust_unit(pd.Series(values, index=group_scores.index))


def validation_analysis(
    indicators: pd.DataFrame,
    groups: dict[str, list[str]],
    critic_score: pd.Series,
    critic_levels: np.ndarray,
    bootstrap_runs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """比较熵权法、PCA，并用 Bootstrap 检验模型稳定性。"""
    entropy_score, _, _, _ = fit_hierarchical_weights(indicators, groups, method="entropy")
    _, group_scores, _, _ = fit_hierarchical_weights(indicators, groups, method="critic")
    pca_values = pca_score(group_scores[["direct_impact", "operational_complexity", "contextual_risk"]])

    method_rows = []
    for name, score in [("entropy", entropy_score), ("pca", pca_values)]:
        rho = float(spearmanr(critic_score, score).statistic)
        levels, _ = ordered_levels(score)
        agreement = float(np.mean(levels == critic_levels))
        top_overlap = len(set(score.nlargest(10).index) & set(critic_score.nlargest(10).index))
        method_rows.append({
            "comparison": f"critic_vs_{name}",
            "spearman_rank_correlation": rho,
            "five_level_agreement": agreement,
            "top10_overlap": top_overlap,
        })

    rng = np.random.default_rng(RANDOM_SEED)
    fit_size = min(30000, len(indicators))
    eval_index = rng.choice(len(indicators), size=min(10000, len(indicators)), replace=False)
    bootstrap_rows = []
    baseline_eval = critic_score.iloc[eval_index]
    baseline_levels = critic_levels[eval_index]
    baseline_top = set(baseline_eval.nlargest(100).index)
    # Bootstrap 最耗时；每完成约 10% 主动打印一次，避免公屏长时间无反馈。
    report_interval = max(1, bootstrap_runs // 10)
    for run in range(bootstrap_runs):
        fit_index = rng.choice(len(indicators), size=fit_size, replace=True)
        score, _, group_weights, _ = fit_hierarchical_weights(
            indicators, groups, method="critic", fit_index=fit_index
        )
        eval_score = score.iloc[eval_index]
        levels, _ = ordered_levels(eval_score, random_seed=RANDOM_SEED + run + 1)
        bootstrap_rows.append({
            "run": run + 1,
            "spearman_rank_correlation": float(spearmanr(baseline_eval, eval_score).statistic),
            "five_level_agreement": float(np.mean(levels == baseline_levels)),
            "top100_overlap": len(baseline_top & set(eval_score.nlargest(100).index)) / 100,
            **{f"weight_{key}": float(value) for key, value in group_weights.items()},
        })
        completed = run + 1
        if completed == 1 or completed % report_interval == 0 or completed == bootstrap_runs:
            progress(f"      Bootstrap：{completed}/{bootstrap_runs}（{completed / bootstrap_runs:.0%}）")
    return pd.DataFrame(method_rows), pd.DataFrame(bootstrap_rows)


def direct_weight_sensitivity(
    group_scores: pd.DataFrame,
    group_weights: pd.Series,
    baseline_score: pd.Series,
    baseline_levels: np.ndarray,
) -> pd.DataFrame:
    """比较直接后果权重取 60%、70%、80% 时的结果变化。"""
    modifier_share = group_weights[["operational_complexity", "contextual_risk"]]
    modifier_share = modifier_share / modifier_share.sum()
    baseline_top = set(baseline_score.nlargest(10).index)
    rows = []
    for direct_weight in [0.60, 0.70, 0.80]:
        weights = pd.Series({
            "direct_impact": direct_weight,
            "operational_complexity": (1 - direct_weight) * modifier_share["operational_complexity"],
            "contextual_risk": (1 - direct_weight) * modifier_share["contextual_risk"],
        })
        score = group_scores[weights.index].mul(weights, axis=1).sum(axis=1)
        levels, _ = ordered_levels(score)
        rows.append({
            "direct_impact_weight": direct_weight,
            "operational_complexity_weight": float(weights["operational_complexity"]),
            "contextual_risk_weight": float(weights["contextual_risk"]),
            "spearman_vs_default": float(spearmanr(baseline_score, score).statistic),
            "level_agreement_vs_default": float(np.mean(levels == baseline_levels)),
            "top10_overlap_vs_default": len(baseline_top & set(score.nlargest(10).index)),
        })
    return pd.DataFrame(rows)


def save_figures(result: pd.DataFrame, weights: pd.DataFrame, output: Path) -> None:
    """仅在指定 --output 时保存分布图和客观权重图。"""
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(9, 5))
    for level in range(1, 6):
        values = result.loc[result["危害等级"] == level, "危害得分"]
        ax.hist(values, bins=35, alpha=0.65, label=f"Level {level}")
    ax.set_xlabel("Hazard score")
    ax.set_ylabel("Event count")
    ax.set_title("Hazard-score distribution by ordered level")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "hazard_score_distribution.png", dpi=180)
    plt.close(fig)

    # 对非线性“直接后果”维度使用等效展示权重，仅用于绘图说明。
    display = weights.assign(
        display_weight=weights["linear_equivalent_weight"].fillna(
            weights["dimension_weight"] * weights["within_dimension_weight"]
        )
    ).sort_values("display_weight")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(display["indicator"], display["display_weight"], color="#2a6f97")
    ax.set_xlabel("Weight within the hierarchical model")
    ax.set_title("Objective indicator weights")
    fig.tight_layout()
    fig.savefig(figure_dir / "critic_weights.png", dpi=180)
    plt.close(fig)


def save_legacy_named_workbooks(
    result: pd.DataFrame,
    level_summary: pd.DataFrame,
    weights: pd.DataFrame,
    top10: pd.DataFrame,
    typical: pd.DataFrame,
    method_validation: pd.DataFrame,
    bootstrap: pd.DataFrame,
    weight_sensitivity: pd.DataFrame,
) -> None:
    """覆盖原项目的两个 Excel 文件名，不额外改变用户既有命名。"""
    # 第一个文件严格保持原结构：无表头，仅保存事件编号和危害等级。
    result[["eventid", "危害等级"]].to_excel(
        "案件等级编号.xlsx", sheet_name="数据", header=False, index=False
    )
    # 第二个文件集中保存模型解释、重点事件和全部稳定性检验结果。
    with pd.ExcelWriter("降维之后各变量均值.xlsx") as writer:
        level_summary.to_excel(writer, sheet_name="五级统计", index=False)
        weights.to_excel(writer, sheet_name="指标权重", index=False)
        top10.to_excel(writer, sheet_name="十大事件", index=False)
        typical.to_excel(writer, sheet_name="典型事件", index=False)
        method_validation.to_excel(writer, sheet_name="方法一致性", index=False)
        weight_sensitivity.to_excel(writer, sheet_name="权重敏感性", index=False)
        bootstrap.to_excel(writer, sheet_name="Bootstrap", index=False)


def main() -> None:
    """组织完整计算流程，并把关键阶段实时显示在公屏。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("附件1.xlsx"))
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="可选：另行保存 CSV、图表和运行摘要的目录",
    )
    parser.add_argument("--bootstrap-runs", type=int, default=100)
    args = parser.parse_args()
    started_at = perf_counter()

    if not args.input.exists():
        raise FileNotFoundError(f"找不到输入文件：{args.input.resolve()}")
    if args.bootstrap_runs < 1:
        raise ValueError("--bootstrap-runs 必须大于或等于 1")

    progress("=" * 60)
    progress("2018 年 C 题第一问：危害分级模型开始运行")
    progress(f"输入文件：{args.input.resolve()}")
    progress(f"Bootstrap 次数：{args.bootstrap_runs}")
    progress("=" * 60)

    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)

    progress("[1/8] 正在读取原始数据……")
    data = pd.read_excel(args.input, usecols=USE_COLUMNS)
    progress(f"      已读取 {len(data):,} 起事件、{len(USE_COLUMNS)} 个原始字段。")

    progress("[2/8] 正在清洗数据并构造评价指标……")
    indicators, groups, quality = prepare_indicators(data)

    progress("[3/8] 正在计算分层 CRITIC 权重与综合危害得分……")
    score, group_scores, group_weights, within = fit_hierarchical_weights(indicators, groups)

    progress("[4/8] 正在进行一维聚类并划分五级危害等级……")
    levels, level_summary = ordered_levels(score)
    weights = flatten_weights(group_weights, within)

    result = pd.DataFrame({
        "eventid": data["eventid"].astype("int64"),
        "危害等级": levels,
        "危害得分": score,
        "人员伤亡得分": group_scores["casualty_impact"],
        "财产损失得分": group_scores["property_impact"],
        "直接后果得分": group_scores["direct_impact"],
        "行动复杂度得分": group_scores["operational_complexity"],
        "情境风险得分": group_scores["contextual_risk"],
        "数据完整率": quality["data_completeness"],
        "数据质量": quality["data_quality"],
    })

    metadata_columns = [
        "eventid", "iyear", "imonth", "iday", "country_txt", "region_txt",
        "city", "nkill", "nkillter", "nwound", "nwoundte", "nhostkid",
        "property", "propextent", "summary",
    ]
    progress("[5/8] 正在筛选十大事件和题目中的典型事件……")
    detailed = result.merge(data[metadata_columns], on="eventid", how="left", validate="one_to_one")
    top10 = detailed.sort_values(["危害得分", "eventid"], ascending=[False, True]).head(10)
    typical = detailed[detailed["eventid"].isin(HIGH_RISK_EVENTS)].sort_values("eventid")
    progress("[6/8] 正在进行熵权法、PCA 与 Bootstrap 稳定性检验……")
    method_validation, bootstrap = validation_analysis(
        indicators, groups, score, levels, max(args.bootstrap_runs, 1)
    )
    progress("[7/8] 正在进行核心权重敏感性分析……")
    weight_sensitivity = direct_weight_sensitivity(
        group_scores, group_weights, score, levels
    )

    # 详细 CSV 和图表属于可选输出；默认只覆盖原来的两个 Excel 文件。
    if args.output is not None:
        progress(f"      正在写入可选详细结果目录：{args.output.resolve()}")
        result.to_csv(args.output / "案件危害分级.csv", index=False, encoding="utf-8-sig")
        top10.to_csv(args.output / "危害程度最高的十大事件.csv", index=False, encoding="utf-8-sig")
        typical.to_csv(args.output / "表1典型事件分级.csv", index=False, encoding="utf-8-sig")
        weights.to_csv(args.output / "CRITIC指标权重.csv", index=False, encoding="utf-8-sig")
        level_summary.to_csv(args.output / "五级划分统计.csv", index=False, encoding="utf-8-sig")
        method_validation.to_csv(args.output / "多方法一致性.csv", index=False, encoding="utf-8-sig")
        bootstrap.to_csv(args.output / "Bootstrap稳定性.csv", index=False, encoding="utf-8-sig")
        weight_sensitivity.to_csv(args.output / "直接后果权重敏感性.csv", index=False, encoding="utf-8-sig")
        save_figures(result, weights, args.output)
    progress("[8/8] 正在覆盖原名称的两个 Excel 结果文件……")
    save_legacy_named_workbooks(
        result,
        level_summary,
        weights,
        top10,
        typical,
        method_validation,
        bootstrap,
        weight_sensitivity,
    )

    summary = {
        "random_seed": RANDOM_SEED,
        "events": int(len(result)),
        "method": "leave-one-out risk encoding + consequence-constrained two-level CRITIC + ordered 1D KMeans",
        "bootstrap_runs": int(len(bootstrap)),
        "direct_impact_weight": DIRECT_IMPACT_WEIGHT,
        "group_weights": {key: float(value) for key, value in group_weights.items()},
        "level_counts": {str(k): int(v) for k, v in result["危害等级"].value_counts().sort_index().items()},
        "top10_eventids": [int(value) for value in top10["eventid"]],
        "median_bootstrap_rank_correlation": float(bootstrap["spearman_rank_correlation"].median()),
        "median_bootstrap_level_agreement": float(bootstrap["five_level_agreement"].median()),
    }
    if args.output is not None:
        (args.output / "run_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    progress("\n运行摘要：")
    progress(json.dumps(summary, ensure_ascii=False, indent=2))
    progress(f"\n运行完成，总耗时 {perf_counter() - started_at:.1f} 秒。")
    progress("已生成：案件等级编号.xlsx、降维之后各变量均值.xlsx")


if __name__ == "__main__":
    main()
