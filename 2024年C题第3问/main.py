#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2024 年“华为杯”中国研究生数学建模竞赛 C 题第三问。"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm


SEED = 2024
N_SPLITS = 5
TEMPERATURE_LEVELS = ["25", "50", "70", "90"]
WAVEFORM_LEVELS = ["正弦波", "三角波", "梯形波"]
MATERIAL_LEVELS = ["材料1", "材料2", "材料3", "材料4"]

CONTROL_FORMULA = "log_loss ~ log_f_c + log_b_c + log_f_c:log_b_c"
MAIN_EFFECT_FORMULA = (
    CONTROL_FORMULA
    + " + C(temperature, Sum) + C(waveform, Sum) + C(material, Sum)"
)
PAIRWISE_FORMULA = (
    MAIN_EFFECT_FORMULA
    + " + C(temperature, Sum):C(waveform, Sum)"
    + " + C(temperature, Sum):C(material, Sum)"
    + " + C(waveform, Sum):C(material, Sum)"
)

TERM_LABELS = {
    "C(temperature, Sum)": "温度",
    "C(waveform, Sum)": "励磁波形",
    "C(material, Sum)": "磁芯材料",
    "C(temperature, Sum):C(waveform, Sum)": "温度×励磁波形",
    "C(temperature, Sum):C(material, Sum)": "温度×磁芯材料",
    "C(waveform, Sum):C(material, Sum)": "励磁波形×磁芯材料",
    "log_f_c": "对数频率（控制变量）",
    "log_b_c": "对数磁通密度峰值（控制变量）",
    "log_f_c:log_b_c": "频率×磁通密度（控制项）",
}


def configure_plot_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC",
        "Arial Unicode MS",
        "Microsoft YaHei",
        "SimHei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 140


def load_data(path: Path) -> tuple[pd.DataFrame, float, float]:
    raw = pd.read_csv(path)
    required = [
        "样本编号",
        "温度_oC",
        "频率_Hz",
        "磁芯损耗_w每m3",
        "励磁波形",
        "磁芯材料",
        "磁通密度峰值B_m_T",
    ]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"输入数据缺少列：{missing}")
    if raw[required].isna().any().any():
        raise ValueError("输入数据存在缺失值。")
    for column in ["频率_Hz", "磁芯损耗_w每m3", "磁通密度峰值B_m_T"]:
        if (raw[column] <= 0).any():
            raise ValueError(f"{column} 必须为正数。")

    data = pd.DataFrame(
        {
            "sample_id": raw["样本编号"].astype(int),
            "temperature": raw["温度_oC"].astype(int).astype(str),
            "frequency": raw["频率_Hz"].astype(float),
            "loss": raw["磁芯损耗_w每m3"].astype(float),
            "waveform": raw["励磁波形"].astype(str),
            "material": raw["磁芯材料"].astype(str),
            "flux_peak": raw["磁通密度峰值B_m_T"].astype(float),
        }
    )
    data["temperature"] = pd.Categorical(
        data["temperature"], categories=TEMPERATURE_LEVELS, ordered=True
    )
    data["waveform"] = pd.Categorical(
        data["waveform"], categories=WAVEFORM_LEVELS, ordered=True
    )
    data["material"] = pd.Categorical(
        data["material"], categories=MATERIAL_LEVELS, ordered=True
    )
    if data[["temperature", "waveform", "material"]].isna().any().any():
        raise ValueError("数据中出现未知的温度、波形或材料类别。")

    log_f = np.log(data["frequency"].to_numpy())
    log_b = np.log(data["flux_peak"].to_numpy())
    f_reference = float(np.exp(log_f.mean()))
    b_reference = float(np.exp(log_b.mean()))
    data["log_loss"] = np.log(data["loss"])
    data["log_f_c"] = log_f - log_f.mean()
    data["log_b_c"] = log_b - log_b.mean()
    return data, f_reference, b_reference


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "R2": float(r2_score(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "MAE": float(mean_absolute_error(actual, predicted)),
        "MAPE_percent": float(np.mean(np.abs((actual - predicted) / actual)) * 100),
        "RMSLE": float(
            np.sqrt(np.mean((np.log(actual) - np.log(predicted)) ** 2))
        ),
    }


def cross_validate_formula(
    data: pd.DataFrame, formula: str
) -> tuple[np.ndarray, dict[str, float], int]:
    fitted = ols(formula, data=data).fit()
    design = np.asarray(fitted.model.exog, dtype=float)
    log_loss = data["log_loss"].to_numpy(dtype=float)
    loss = data["loss"].to_numpy(dtype=float)
    predictions = np.empty(len(data), dtype=float)
    folds = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for training_indices, validation_indices in folds.split(design):
        coefficients, *_ = np.linalg.lstsq(
            design[training_indices], log_loss[training_indices], rcond=None
        )
        training_residual = (
            log_loss[training_indices] - design[training_indices] @ coefficients
        )
        smearing_factor = float(np.mean(np.exp(training_residual)))
        predictions[validation_indices] = (
            np.exp(design[validation_indices] @ coefficients) * smearing_factor
        )
    return predictions, calculate_metrics(loss, predictions), design.shape[1]


def effect_size_label(value: float) -> str:
    if value >= 0.14:
        return "大"
    if value >= 0.06:
        return "中"
    if value >= 0.01:
        return "小"
    return "很小"


def build_anova_table(model) -> pd.DataFrame:
    table = anova_lm(model, typ=3, robust="hc3").reset_index()
    table = table.rename(
        columns={
            "index": "原始项",
            "sum_sq": "平方和",
            "df": "自由度",
            "F": "稳健F值_HC3",
            "PR(>F)": "p值_HC3",
        }
    )
    residual_ss = float(
        table.loc[table["原始项"] == "Residual", "平方和"].iloc[0]
    )
    table = table[~table["原始项"].isin(["Intercept", "Residual"])].copy()
    table["因素或交互项"] = table["原始项"].map(TERM_LABELS).fillna(table["原始项"])
    table["偏eta平方"] = table["平方和"] / (table["平方和"] + residual_ss)
    table["Cohen_f"] = np.sqrt(table["偏eta平方"] / (1.0 - table["偏eta平方"]))
    table["效应量等级"] = table["偏eta平方"].map(effect_size_label)
    table["是否显著_0.05"] = np.where(table["p值_HC3"] < 0.05, "是", "否")
    return table[
        [
            "因素或交互项",
            "平方和",
            "自由度",
            "稳健F值_HC3",
            "p值_HC3",
            "偏eta平方",
            "Cohen_f",
            "效应量等级",
            "是否显著_0.05",
        ]
    ]


def adjusted_effect_tables(
    data: pd.DataFrame, model, smearing_factor: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grid = pd.DataFrame(
        product(TEMPERATURE_LEVELS, WAVEFORM_LEVELS, MATERIAL_LEVELS),
        columns=["temperature", "waveform", "material"],
    )
    grid["log_f_c"] = 0.0
    grid["log_b_c"] = 0.0
    grid["调整预测损耗_w每m3"] = np.exp(model.predict(grid)) * smearing_factor

    raw_summary = (
        data.groupby(["temperature", "waveform", "material"], observed=True)["loss"]
        .agg(
            原始平均损耗_w每m3="mean",
            原始中位损耗_w每m3="median",
            样本数="count",
        )
        .reset_index()
    )
    combinations = grid.merge(
        raw_summary, on=["temperature", "waveform", "material"], how="left"
    )
    combinations["调整损耗排名"] = combinations["调整预测损耗_w每m3"].rank(
        method="min"
    ).astype(int)
    combinations = combinations.sort_values("调整损耗排名").reset_index(drop=True)

    main_rows: list[dict[str, object]] = []
    for factor, levels, label in [
        ("temperature", TEMPERATURE_LEVELS, "温度"),
        ("waveform", WAVEFORM_LEVELS, "励磁波形"),
        ("material", MATERIAL_LEVELS, "磁芯材料"),
    ]:
        for level in levels:
            values = combinations.loc[
                combinations[factor] == level, "调整预测损耗_w每m3"
            ]
            main_rows.append(
                {
                    "因素": label,
                    "水平": f"{level}°C" if factor == "temperature" else level,
                    "调整平均损耗_w每m3": float(values.mean()),
                }
            )
    main_effects = pd.DataFrame(main_rows)
    main_effects["同因素相对最低值"] = main_effects.groupby("因素")[
        "调整平均损耗_w每m3"
    ].transform(lambda values: values / values.min())

    interaction_frames: list[pd.DataFrame] = []
    for first, second, first_label, second_label in [
        ("temperature", "waveform", "温度", "励磁波形"),
        ("temperature", "material", "温度", "磁芯材料"),
        ("waveform", "material", "励磁波形", "磁芯材料"),
    ]:
        grouped = (
            combinations.groupby([first, second], observed=True)[
                "调整预测损耗_w每m3"
            ]
            .mean()
            .reset_index()
            .rename(columns={first: "因素A水平", second: "因素B水平"})
        )
        grouped.insert(0, "因素对", f"{first_label}×{second_label}")
        interaction_frames.append(grouped)
    interactions = pd.concat(interaction_frames, ignore_index=True)
    return main_effects, interactions, combinations


def random_forest_group_importance(data: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    features = data[
        ["log_f_c", "log_b_c", "temperature", "waveform", "material"]
    ].copy()
    target = data["log_loss"].to_numpy(dtype=float)
    strata = (
        data[["temperature", "waveform", "material"]]
        .astype(str)
        .agg("|".join, axis=1)
    )
    train_indices, test_indices = train_test_split(
        np.arange(len(data)),
        test_size=0.2,
        random_state=SEED,
        stratify=strata,
    )
    preprocessing = ColumnTransformer(
        [
            (
                "category",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["temperature", "waveform", "material"],
            )
        ],
        remainder="passthrough",
    )
    model = make_pipeline(
        preprocessing,
        RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=3,
            max_features=0.9,
            n_jobs=-1,
            random_state=SEED,
        ),
    )
    model.fit(features.iloc[train_indices], target[train_indices])
    baseline_prediction = model.predict(features.iloc[test_indices])
    baseline_rmsle = float(
        np.sqrt(np.mean((target[test_indices] - baseline_prediction) ** 2))
    )
    names = {
        "log_f_c": "频率",
        "log_b_c": "磁通密度峰值",
        "temperature": "温度",
        "waveform": "励磁波形",
        "material": "磁芯材料",
    }
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, float | str]] = []
    repeats = 15
    for column in features.columns:
        increases: list[float] = []
        for _ in range(repeats):
            permuted = features.iloc[test_indices].copy()
            permuted[column] = rng.permutation(permuted[column].to_numpy())
            prediction = model.predict(permuted)
            rmsle = float(
                np.sqrt(np.mean((target[test_indices] - prediction) ** 2))
            )
            increases.append(rmsle - baseline_rmsle)
        rows.append(
            {
                "变量": names[column],
                "置换后RMSLE增量": float(np.mean(increases)),
                "RMSLE增量标准差": float(np.std(increases, ddof=1)),
                "相对基准增幅_percent": float(
                    np.mean(increases) / baseline_rmsle * 100
                ),
            }
        )
    importance = pd.DataFrame(rows).sort_values(
        "置换后RMSLE增量", ascending=False
    )
    target_mask = importance["变量"].isin(["温度", "励磁波形", "磁芯材料"])
    target_total = importance.loc[target_mask, "置换后RMSLE增量"].sum()
    importance["三因素内部影响占比"] = np.where(
        target_mask,
        importance["置换后RMSLE增量"] / target_total,
        np.nan,
    )
    return importance.reset_index(drop=True), baseline_rmsle


def save_figures(
    output_dir: Path,
    data: pd.DataFrame,
    prediction_table: pd.DataFrame,
    main_effects: pd.DataFrame,
    interactions: pd.DataFrame,
    anova_table: pd.DataFrame,
    importance: pd.DataFrame,
    combinations: pd.DataFrame,
) -> None:
    configure_plot_style()
    palette = ["#2563EB", "#059669", "#F59E0B", "#DC2626"]

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)
    for axis, factor, color in zip(
        axes, ["温度", "励磁波形", "磁芯材料"], ["#2563EB", "#059669", "#7C3AED"]
    ):
        subset = main_effects[main_effects["因素"] == factor]
        bars = axis.bar(
            subset["水平"], subset["调整平均损耗_w每m3"], color=color, alpha=0.86
        )
        axis.bar_label(bars, fmt="%.0f", fontsize=8, padding=2)
        axis.set_title(f"{factor}的调整主效应")
        axis.set_ylabel("调整损耗 / (W·m⁻³)")
        axis.tick_params(axis="x", rotation=15)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("在统一频率和磁通密度下的主效应")
    figure.savefig(output_dir / "主效应调整结果.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    configs = [
        ("温度×励磁波形", TEMPERATURE_LEVELS, WAVEFORM_LEVELS, "温度 / °C"),
        ("温度×磁芯材料", TEMPERATURE_LEVELS, MATERIAL_LEVELS, "温度 / °C"),
        ("励磁波形×磁芯材料", WAVEFORM_LEVELS, MATERIAL_LEVELS, "励磁波形"),
    ]
    for axis, (pair, x_levels, line_levels, x_label) in zip(axes, configs):
        subset = interactions[interactions["因素对"] == pair]
        for index, line_level in enumerate(line_levels):
            line = subset[subset["因素B水平"].astype(str) == str(line_level)].copy()
            line["因素A水平"] = pd.Categorical(
                line["因素A水平"].astype(str), categories=x_levels, ordered=True
            )
            line = line.sort_values("因素A水平")
            axis.plot(
                line["因素A水平"].astype(str),
                line["调整预测损耗_w每m3"],
                marker="o",
                linewidth=2,
                color=palette[index],
                label=line_level,
            )
        axis.set_title(pair)
        axis.set_xlabel(x_label)
        axis.set_ylabel("调整损耗 / (W·m⁻³)")
        axis.tick_params(axis="x", rotation=15)
        axis.legend(frameon=False, fontsize=8)
        axis.grid(alpha=0.2)
    figure.suptitle("两两交互作用（等权平均其他因素）")
    figure.savefig(output_dir / "两两交互作用.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    factor_terms = [
        "温度",
        "励磁波形",
        "磁芯材料",
        "温度×励磁波形",
        "温度×磁芯材料",
        "励磁波形×磁芯材料",
    ]
    effect = anova_table[anova_table["因素或交互项"].isin(factor_terms)].sort_values(
        "偏eta平方"
    )
    target_importance = importance[
        importance["变量"].isin(["温度", "励磁波形", "磁芯材料"])
    ].sort_values("三因素内部影响占比")
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    axes[0].barh(effect["因素或交互项"], effect["偏eta平方"], color="#2563EB")
    axes[0].set_xlabel("偏 η²")
    axes[0].set_title("协方差分析效应量")
    axes[0].grid(axis="x", alpha=0.2)
    bars = axes[1].barh(
        target_importance["变量"],
        target_importance["三因素内部影响占比"] * 100,
        color="#059669",
    )
    axes[1].bar_label(bars, fmt="%.1f%%", fontsize=9, padding=3)
    axes[1].set_xlabel("三因素内部影响占比 / %")
    axes[1].set_title("随机森林置换重要性")
    axes[1].grid(axis="x", alpha=0.2)
    figure.suptitle("因素影响程度的两种量化结果")
    figure.savefig(output_dir / "效应量与影响程度.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    heatmap_data = combinations.pivot_table(
        index=["waveform", "material"],
        columns="temperature",
        values="调整预测损耗_w每m3",
        observed=True,
    ).reindex(
        pd.MultiIndex.from_product(
            [WAVEFORM_LEVELS, MATERIAL_LEVELS], names=["waveform", "material"]
        )
    )
    figure, axis = plt.subplots(figsize=(9, 8), constrained_layout=True)
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".0f",
        cmap="YlGnBu_r",
        cbar_kws={"label": "调整损耗 / (W·m⁻³)"},
        ax=axis,
    )
    axis.set_xlabel("温度 / °C")
    axis.set_ylabel("励磁波形、磁芯材料")
    axis.set_yticklabels([f"{a}·{b}" for a, b in heatmap_data.index], rotation=0)
    axis.set_title("统一频率和磁通密度下的48种组合")
    figure.savefig(output_dir / "组合损耗热力图.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    actual = prediction_table["实际磁芯损耗_w每m3"].to_numpy()
    predicted = prediction_table["两两交互模型_五折预测"].to_numpy()
    limits = [min(actual.min(), predicted.min()) * 0.8, max(actual.max(), predicted.max()) * 1.25]
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    axes[0].scatter(actual, predicted, s=9, alpha=0.35, color="#2563EB")
    axes[0].plot(limits, limits, "--", color="#111827")
    axes[0].set(xscale="log", yscale="log", xlim=limits, ylim=limits)
    axes[0].set_xlabel("实际损耗 / (W·m⁻³)")
    axes[0].set_ylabel("五折预测损耗 / (W·m⁻³)")
    axes[0].set_title("实际值与交叉验证预测值")
    log_residual = np.log(actual) - np.log(predicted)
    axes[1].hist(log_residual, bins=45, color="#7C3AED", alpha=0.85)
    axes[1].axvline(0, linestyle="--", color="#111827")
    axes[1].set_xlabel("对数残差 ln(实际/预测)")
    axes[1].set_ylabel("样本数")
    axes[1].set_title("交叉验证残差分布")
    figure.suptitle("两两交互模型验证")
    figure.savefig(output_dir / "模型验证.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=script_dir / "附件一_第三问分析数据.csv",
        help="第三问紧凑输入数据",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=script_dir, help="结果输出目录"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data, f_reference, b_reference = load_data(args.input)
    formulas = [
        ("仅控制频率与磁通密度", CONTROL_FORMULA),
        ("加入三因素主效应", MAIN_EFFECT_FORMULA),
        ("加入两两交互作用", PAIRWISE_FORMULA),
    ]
    validation_rows: list[dict[str, float | int | str]] = []
    full_cv_prediction: np.ndarray | None = None
    for name, formula in formulas:
        cv_prediction, metrics, parameter_count = cross_validate_formula(data, formula)
        fitted = ols(formula, data=data).fit()
        validation_rows.append(
            {
                "模型": name,
                "参数个数": parameter_count,
                "全样本调整R2": float(fitted.rsquared_adj),
                "AIC": float(fitted.aic),
                **{f"五折_{key}": value for key, value in metrics.items()},
            }
        )
        if name == "加入两两交互作用":
            full_cv_prediction = cv_prediction
    assert full_cv_prediction is not None
    model_validation = pd.DataFrame(validation_rows)

    full_model = ols(PAIRWISE_FORMULA, data=data).fit()
    smearing_factor = float(np.mean(np.exp(full_model.resid)))
    anova_table = build_anova_table(full_model)
    main_effects, interactions, combinations = adjusted_effect_tables(
        data, full_model, smearing_factor
    )
    importance, rf_baseline_rmsle = random_forest_group_importance(data)

    prediction_table = pd.DataFrame(
        {
            "样本编号": data["sample_id"].astype(int),
            "实际磁芯损耗_w每m3": data["loss"],
            "两两交互模型_五折预测": full_cv_prediction,
            "相对误差_percent": (full_cv_prediction - data["loss"]) / data["loss"] * 100,
            "对数残差": np.log(data["loss"]) - np.log(full_cv_prediction),
        }
    )

    factor_effects = anova_table[
        anova_table["因素或交互项"].isin(
            [
                "温度",
                "励磁波形",
                "磁芯材料",
                "温度×励磁波形",
                "温度×磁芯材料",
                "励磁波形×磁芯材料",
            ]
        )
    ].copy()
    factor_effects = factor_effects.sort_values("偏eta平方", ascending=False)
    factor_effects.insert(0, "影响排名", np.arange(1, len(factor_effects) + 1))

    minimum_adjusted = combinations.iloc[0]
    raw_minimum = combinations.sort_values("原始平均损耗_w每m3").iloc[0]
    minimum_conditions = pd.DataFrame(
        [
            {
                "结论类型": "控制频率和磁通密度后的模型结论",
                "温度_oC": int(minimum_adjusted["temperature"]),
                "励磁波形": minimum_adjusted["waveform"],
                "磁芯材料": minimum_adjusted["material"],
                "参考频率_Hz": f_reference,
                "参考磁通密度B_m_T": b_reference,
                "损耗_w每m3": minimum_adjusted["调整预测损耗_w每m3"],
                "说明": "正式结论：在共同参考工况下比较。",
            },
            {
                "结论类型": "未控制工况的原始组均值最小",
                "温度_oC": int(raw_minimum["temperature"]),
                "励磁波形": raw_minimum["waveform"],
                "磁芯材料": raw_minimum["material"],
                "参考频率_Hz": np.nan,
                "参考磁通密度B_m_T": np.nan,
                "损耗_w每m3": raw_minimum["原始平均损耗_w每m3"],
                "说明": "仅作描述，受频率和磁通密度分布影响，不作为正式因果结论。",
            },
        ]
    )

    overview_rows: list[dict[str, object]] = [
        {"指标": "总样本数", "数值": len(data), "单位或说明": "条"},
        {"指标": "温度水平", "数值": 4, "单位或说明": "25、50、70、90°C"},
        {"指标": "励磁波形", "数值": 3, "单位或说明": "正弦波、三角波、梯形波"},
        {"指标": "磁芯材料", "数值": 4, "单位或说明": "材料1—材料4"},
        {"指标": "几何平均参考频率", "数值": f_reference, "单位或说明": "Hz"},
        {"指标": "几何平均参考磁通密度", "数值": b_reference, "单位或说明": "T"},
        {"指标": "随机森林独立验证RMSLE", "数值": rf_baseline_rmsle, "单位或说明": "20% 分层留出集"},
    ]
    data_overview = pd.DataFrame(overview_rows)

    combination_output = combinations.rename(
        columns={
            "temperature": "温度_oC",
            "waveform": "励磁波形",
            "material": "磁芯材料",
            "log_f_c": "中心化对数频率",
            "log_b_c": "中心化对数磁通密度峰值",
        }
    )
    output_tables = {
        "数据概况.csv": data_overview,
        "模型验证.csv": model_validation,
        "方差分析与效应量.csv": anova_table,
        "因素与交互作用排名.csv": factor_effects,
        "随机森林置换重要性.csv": importance,
        "主效应调整均值.csv": main_effects,
        "两两交互调整均值.csv": interactions,
        "48种组合调整损耗.csv": combination_output,
        "最低损耗条件.csv": minimum_conditions,
        "模型五折预测.csv": prediction_table,
    }
    for filename, table in output_tables.items():
        table.to_csv(args.output_dir / filename, index=False, encoding="utf-8-sig")

    target_importance = importance[
        importance["变量"].isin(["温度", "励磁波形", "磁芯材料"])
    ].sort_values("三因素内部影响占比", ascending=False)
    factor_rank = factor_effects[
        factor_effects["因素或交互项"].isin(["温度", "励磁波形", "磁芯材料"])
    ].sort_values("偏eta平方", ascending=False)
    conclusions = {
        "sample_count": int(len(data)),
        "reference_condition": {
            "frequency_Hz": f_reference,
            "flux_peak_T": b_reference,
        },
        "ancova_main_effect_rank": factor_rank["因素或交互项"].tolist(),
        "random_forest_factor_rank": target_importance["变量"].tolist(),
        "minimum_adjusted_condition": {
            "temperature_oC": int(minimum_adjusted["temperature"]),
            "waveform": minimum_adjusted["waveform"],
            "material": minimum_adjusted["material"],
            "predicted_loss_w_per_m3": float(
                minimum_adjusted["调整预测损耗_w每m3"]
            ),
        },
        "validation": {
            "folds": N_SPLITS,
            "seed": SEED,
            "pairwise_model": model_validation.iloc[-1].to_dict(),
        },
        "model_formula": PAIRWISE_FORMULA,
        "note": "正式最低损耗条件在共同频率和磁通密度峰值下比较。",
    }
    (args.output_dir / "分析结论.json").write_text(
        json.dumps(conclusions, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    save_figures(
        args.output_dir,
        data,
        prediction_table,
        main_effects,
        interactions,
        anova_table,
        importance,
        combinations,
    )

    best_validation = model_validation.iloc[-1]
    print(f"样本数：{len(data)}")
    print(
        "两两交互模型五折验证："
        f"MAPE={best_validation['五折_MAPE_percent']:.3f}%，"
        f"RMSLE={best_validation['五折_RMSLE']:.4f}"
    )
    print(
        "控制工况后的最低损耗组合："
        f"{int(minimum_adjusted['temperature'])}°C、"
        f"{minimum_adjusted['waveform']}、{minimum_adjusted['material']}"
    )
    print(f"结果已保存至：{args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
