#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Question 5: empirical-domain bi-objective magnetic component optimization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


SEED = 2024
ID = "样本编号"
TEMPERATURE = "温度_oC"
FREQUENCY = "频率_Hz"
MATERIAL = "磁芯材料"
WAVEFORM = "励磁波形"
BM = "磁通密度峰值B_m_T"
ACTUAL_LOSS = "实际磁芯损耗_w每m3"
PREDICTED_LOSS = "第四问OOF预测损耗_w每m3"
ENERGY = "传输磁能指标_f×B_m"
LOSS_COST = "对数损耗归一化代价"
ENERGY_COST = "对数传输磁能归一化代价"
COMPROMISE_SCORE = "等权理想点距离"


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="第五问：磁性元件双目标优化")
    parser.add_argument(
        "--input", type=Path, default=base / "附件一_第五问优化数据.csv"
    )
    parser.add_argument("--output-dir", type=Path, default=base)
    return parser.parse_args()


def configure_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC",
        "Arial Unicode MS",
        "Microsoft YaHei",
        "SimHei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 220


def load_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = [
        ID,
        TEMPERATURE,
        FREQUENCY,
        MATERIAL,
        WAVEFORM,
        BM,
        ACTUAL_LOSS,
        PREDICTED_LOSS,
    ]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"输入数据缺少列：{missing}")
    if data[required].isna().any().any():
        raise ValueError("输入数据存在缺失值。")
    if data[ID].duplicated().any():
        raise ValueError("样本编号不得重复。")
    positive = [FREQUENCY, BM, ACTUAL_LOSS, PREDICTED_LOSS]
    if (data[positive] <= 0).any().any():
        raise ValueError("频率、B_m和损耗必须为正数。")
    data = data.copy()
    data[ENERGY] = data[FREQUENCY] * data[BM]
    return data


def validation_metrics(data: pd.DataFrame) -> pd.DataFrame:
    actual = data[ACTUAL_LOSS].to_numpy(dtype=float)
    predicted = data[PREDICTED_LOSS].to_numpy(dtype=float)
    return pd.DataFrame(
        [
            {
                "验证方式": "第四问5折折外预测",
                "样本数": len(data),
                "R2": float(r2_score(actual, predicted)),
                "RMSE_w每m3": float(np.sqrt(mean_squared_error(actual, predicted))),
                "MAE_w每m3": float(mean_absolute_error(actual, predicted)),
                "MAPE_percent": float(np.mean(np.abs((actual - predicted) / actual)) * 100),
                "RMSLE": float(np.sqrt(np.mean((np.log(actual) - np.log(predicted)) ** 2))),
            }
        ]
    )


def pareto_front(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exact nondominated sorting for min loss and max energy in O(n log n)."""
    ordered = data.sort_values(
        [PREDICTED_LOSS, ENERGY, ID], ascending=[True, False, True]
    ).reset_index(drop=True)
    previous_best_energy = ordered[ENERGY].cummax().shift(fill_value=-np.inf)
    ordered["Pareto最优"] = ordered[ENERGY] > previous_best_energy
    front = ordered[ordered["Pareto最优"]].copy()
    front = front.sort_values(ENERGY, ascending=True).reset_index(drop=True)
    front["Pareto序号"] = np.arange(1, len(front) + 1)

    log_loss = np.log(front[PREDICTED_LOSS].to_numpy(dtype=float))
    log_energy = np.log(front[ENERGY].to_numpy(dtype=float))
    front[LOSS_COST] = (log_loss - log_loss.min()) / (log_loss.max() - log_loss.min())
    front[ENERGY_COST] = (log_energy.max() - log_energy) / (
        log_energy.max() - log_energy.min()
    )
    front[COMPROMISE_SCORE] = np.sqrt(
        0.5 * front[LOSS_COST] ** 2 + 0.5 * front[ENERGY_COST] ** 2
    )

    score_map = front.set_index(ID)[["Pareto序号", LOSS_COST, ENERGY_COST, COMPROMISE_SCORE]]
    all_candidates = ordered.merge(score_map, left_on=ID, right_index=True, how="left")
    all_candidates = all_candidates.sort_values(ID).reset_index(drop=True)
    return all_candidates, front


def representative_solutions(
    data: pd.DataFrame, front: pd.DataFrame
) -> pd.DataFrame:
    min_loss = front.sort_values([PREDICTED_LOSS, ENERGY], ascending=[True, False]).iloc[0]
    max_energy = front.sort_values([ENERGY, PREDICTED_LOSS], ascending=[False, True]).iloc[0]
    compromise = front.sort_values(
        [COMPROMISE_SCORE, PREDICTED_LOSS], ascending=[True, True]
    ).iloc[0]
    labels_and_rows = [
        ("损耗最小方案", min_loss),
        ("传输磁能最大方案", max_energy),
        ("等权综合推荐方案", compromise),
    ]
    rows: list[dict[str, object]] = []
    for label, row in labels_and_rows:
        rows.append(
            {
                "方案": label,
                ID: int(row[ID]),
                TEMPERATURE: int(row[TEMPERATURE]),
                FREQUENCY: float(row[FREQUENCY]),
                MATERIAL: row[MATERIAL],
                WAVEFORM: row[WAVEFORM],
                BM: float(row[BM]),
                PREDICTED_LOSS: float(row[PREDICTED_LOSS]),
                ACTUAL_LOSS: float(row[ACTUAL_LOSS]),
                ENERGY: float(row[ENERGY]),
                "是否Pareto最优": "是",
                COMPROMISE_SCORE: float(row.get(COMPROMISE_SCORE, np.nan)),
            }
        )
    result = pd.DataFrame(rows)
    loss_min = result[PREDICTED_LOSS].min()
    energy_max = result[ENERGY].max()
    result["相对最小损耗_倍"] = result[PREDICTED_LOSS] / loss_min
    result["相对最大传输磁能_percent"] = result[ENERGY] / energy_max * 100
    return result


def weight_sensitivity(front: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for loss_weight in np.round(np.arange(0.10, 0.901, 0.05), 2):
        energy_weight = 1.0 - loss_weight
        scores = np.sqrt(
            loss_weight * front[LOSS_COST] ** 2
            + energy_weight * front[ENERGY_COST] ** 2
        )
        best_index = scores.sort_values(kind="mergesort").index[0]
        best = front.loc[best_index]
        rows.append(
            {
                "损耗权重": float(loss_weight),
                "传输磁能权重": float(energy_weight),
                "加权理想点距离": float(scores.loc[best_index]),
                ID: int(best[ID]),
                TEMPERATURE: int(best[TEMPERATURE]),
                FREQUENCY: float(best[FREQUENCY]),
                MATERIAL: best[MATERIAL],
                WAVEFORM: best[WAVEFORM],
                BM: float(best[BM]),
                PREDICTED_LOSS: float(best[PREDICTED_LOSS]),
                ENERGY: float(best[ENERGY]),
            }
        )
    result = pd.DataFrame(rows)
    result["方案切换"] = np.where(
        result[ID].ne(result[ID].shift()), "是", "否"
    )
    return result


def factor_distribution(front: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column, label in [
        (TEMPERATURE, "温度"),
        (MATERIAL, "磁芯材料"),
        (WAVEFORM, "励磁波形"),
    ]:
        grouped = front.groupby(column, observed=True)
        for value, group in grouped:
            rows.append(
                {
                    "因素": label,
                    "水平": value,
                    "Pareto解数量": len(group),
                    "Pareto解占比_percent": float(len(group) / len(front) * 100),
                    "中位预测损耗_w每m3": float(group[PREDICTED_LOSS].median()),
                    "中位传输磁能指标": float(group[ENERGY].median()),
                }
            )
    return pd.DataFrame(rows)


def plot_pareto(
    candidates: pd.DataFrame, front: pd.DataFrame, representatives: pd.DataFrame, output: Path
) -> None:
    fig, axis = plt.subplots(figsize=(10, 7))
    axis.scatter(
        candidates[PREDICTED_LOSS], candidates[ENERGY],
        s=9, alpha=0.16, color="#94A3B8", label="全部候选工况"
    )
    ordered = front.sort_values(PREDICTED_LOSS)
    axis.plot(
        ordered[PREDICTED_LOSS], ordered[ENERGY], color="#DC2626", linewidth=2.0,
        marker="o", markersize=3, label=f"Pareto前沿（{len(front)}个）"
    )
    colors = ["#2563EB", "#F59E0B", "#16A34A"]
    for (_, row), color in zip(representatives.iterrows(), colors):
        axis.scatter(row[PREDICTED_LOSS], row[ENERGY], s=115, color=color,
                     edgecolor="white", linewidth=1.2, zorder=5, label=row["方案"])
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("第四问预测磁芯损耗 P (W/m³，越小越好)")
    axis.set_ylabel("传输磁能指标 f×B_m（越大越好）")
    axis.set_title("磁芯损耗与传输磁能的 Pareto 前沿")
    axis.text(0.02, 0.97, "理想方向：左上", transform=axis.transAxes,
              va="top", color="#166534", fontweight="bold")
    axis.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_normalized_front(front: pd.DataFrame, representatives: pd.DataFrame, output: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.6, 6.8))
    axis.plot(front[LOSS_COST], front[ENERGY_COST], color="#DC2626", linewidth=1.8)
    axis.scatter(front[LOSS_COST], front[ENERGY_COST], s=22, color="#EF4444", alpha=0.8)
    comp_id = int(representatives.loc[representatives["方案"] == "等权综合推荐方案", ID].iloc[0])
    comp = front.loc[front[ID] == comp_id].iloc[0]
    axis.scatter([0], [0], marker="*", s=180, color="#16A34A", label="不可同时达到的理想点")
    axis.scatter([comp[LOSS_COST]], [comp[ENERGY_COST]], s=120, color="#2563EB",
                 edgecolor="white", linewidth=1.2, label="等权综合推荐")
    axis.plot([0, comp[LOSS_COST]], [0, comp[ENERGY_COST]], "--", color="#2563EB")
    axis.set_xlabel("对数损耗归一化代价（越小越好）")
    axis.set_ylabel("对数传输磁能归一化代价（越小越好）")
    axis.set_title("归一化双目标空间与等权折中解")
    axis.set_xlim(-0.03, 1.03)
    axis.set_ylim(-0.03, 1.03)
    axis.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_representative_comparison(
    front: pd.DataFrame, representatives: pd.DataFrame, output: Path
) -> None:
    work = representatives.copy()
    log_loss = np.log(work[PREDICTED_LOSS])
    log_energy = np.log(work[ENERGY])
    front_log_loss = np.log(front[PREDICTED_LOSS])
    front_log_energy = np.log(front[ENERGY])
    work["损耗效用"] = (
        front_log_loss.max() - log_loss
    ) / (front_log_loss.max() - front_log_loss.min())
    work["传输磁能效用"] = (
        log_energy - front_log_energy.min()
    ) / (front_log_energy.max() - front_log_energy.min())
    long = work.melt(
        id_vars="方案", value_vars=["损耗效用", "传输磁能效用"],
        var_name="目标", value_name="归一化效用"
    )
    fig, axis = plt.subplots(figsize=(10, 5.8))
    sns.barplot(data=long, x="方案", y="归一化效用", hue="目标", ax=axis,
                palette=["#2563EB", "#F59E0B"])
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("归一化效用（越大越好）")
    axis.set_xlabel("")
    axis.set_title("三类代表性方案的双目标表现")
    axis.tick_params(axis="x", rotation=8)
    axis.legend(title="")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_weight_sensitivity(sensitivity: pd.DataFrame, output: Path) -> None:
    fig, axis_left = plt.subplots(figsize=(10, 5.8))
    axis_right = axis_left.twinx()
    axis_left.plot(sensitivity["损耗权重"], sensitivity[PREDICTED_LOSS], marker="o",
                   color="#2563EB", label="预测损耗")
    axis_right.plot(sensitivity["损耗权重"], sensitivity[ENERGY], marker="s",
                    color="#F59E0B", label="传输磁能指标")
    axis_left.set_yscale("log")
    axis_right.set_yscale("log")
    axis_left.axvline(0.5, color="#16A34A", linestyle="--", linewidth=1.5, label="等权")
    axis_left.set_xlabel("损耗权重")
    axis_left.set_ylabel("选中方案预测损耗 (W/m³)", color="#2563EB")
    axis_right.set_ylabel("选中方案 f×B_m", color="#F59E0B")
    axis_left.set_title("目标权重变化下的推荐结果敏感性")
    lines1, labels1 = axis_left.get_legend_handles_labels()
    lines2, labels2 = axis_right.get_legend_handles_labels()
    axis_left.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_factor_distribution(distribution: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for axis, factor in zip(axes, ["温度", "磁芯材料", "励磁波形"]):
        subset = distribution[distribution["因素"] == factor].copy()
        subset["水平"] = subset["水平"].astype(str)
        sns.barplot(data=subset, x="水平", y="Pareto解占比_percent", ax=axis, color="#0F766E")
        axis.set_title(f"{factor}在Pareto解中的分布")
        axis.set_xlabel(factor)
        axis.set_ylabel("占比 (%)")
        axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    configure_plot_style()
    data = load_data(args.input)
    validation = validation_metrics(data)
    candidates, front = pareto_front(data)
    representatives = representative_solutions(data, front)
    sensitivity = weight_sensitivity(front)
    distribution = factor_distribution(front)

    candidates.to_csv(output / "全部候选工况.csv", index=False, encoding="utf-8-sig")
    front.to_csv(output / "Pareto最优解.csv", index=False, encoding="utf-8-sig")
    representatives.to_csv(output / "代表性方案.csv", index=False, encoding="utf-8-sig")
    sensitivity.to_csv(output / "权重敏感性分析.csv", index=False, encoding="utf-8-sig")
    distribution.to_csv(output / "Pareto条件分布.csv", index=False, encoding="utf-8-sig")
    validation.to_csv(output / "第四问模型衔接验证.csv", index=False, encoding="utf-8-sig")

    plot_pareto(candidates, front, representatives, output / "Pareto前沿.png")
    plot_normalized_front(front, representatives, output / "归一化目标空间.png")
    plot_representative_comparison(front, representatives, output / "代表性方案对比.png")
    plot_weight_sensitivity(sensitivity, output / "权重敏感性.png")
    plot_factor_distribution(distribution, output / "Pareto最优条件分布.png")

    compromise = representatives.loc[
        representatives["方案"] == "等权综合推荐方案"
    ].iloc[0]
    min_loss = representatives.loc[representatives["方案"] == "损耗最小方案"].iloc[0]
    max_energy = representatives.loc[
        representatives["方案"] == "传输磁能最大方案"
    ].iloc[0]
    conclusions = {
        "竞赛": "2024 年“华为杯”中国研究生数学建模竞赛",
        "题目": "C 题——数据驱动下磁性元件的磁芯损耗建模",
        "问题": "第五问",
        "可行域": "附件一的12400个实测工况，不做超出实验范围的外推",
        "目标": ["最小化第四问模型预测损耗", "最大化频率×磁通密度峰值"],
        "Pareto解数量": int(len(front)),
        "第四问模型衔接指标": validation.iloc[0].to_dict(),
        "损耗最小方案": min_loss.to_dict(),
        "传输磁能最大方案": max_energy.to_dict(),
        "等权综合推荐方案": compromise.to_dict(),
        "综合方案权衡": {
            "相对最小损耗的损耗倍数": float(compromise[PREDICTED_LOSS] / min_loss[PREDICTED_LOSS]),
            "达到最大传输磁能的比例_percent": float(compromise[ENERGY] / max_energy[ENERGY] * 100),
            "相对最大传输磁能方案的损耗降低_percent": float((1 - compromise[PREDICTED_LOSS] / max_energy[PREDICTED_LOSS]) * 100),
        },
        "解读": [
            "两个目标不能同时取得各自的绝对最优值，因此不存在唯一的无偏好最优解。",
            "工程上应从Pareto前沿中按损耗与传输磁能的重要性选择方案。",
            "等权推荐是在对数归一化目标空间中距离理想点最近的方案。",
        ],
    }
    with (output / "分析结论.json").open("w", encoding="utf-8") as file:
        json.dump(conclusions, file, ensure_ascii=False, indent=2, default=float)

    print(f"Pareto最优解数量：{len(front)}")
    print("等权综合推荐：")
    print(compromise[[TEMPERATURE, FREQUENCY, WAVEFORM, BM, MATERIAL, PREDICTED_LOSS, ENERGY]].to_string())
    print(f"结果已写入：{output}")


if __name__ == "__main__":
    main()
