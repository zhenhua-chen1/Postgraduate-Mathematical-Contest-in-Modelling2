#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2024 Huawei Cup postgraduate mathematical modeling contest, Question 4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEED = 2024
N_SPLITS = 5
CATEGORICAL_FEATURES = ["磁芯材料", "励磁波形"]
ID_COLUMN = "样本编号"
TARGET_COLUMN = "磁芯损耗_w每m3"
KEY_RANGE_FEATURES = [
    "温度_oC",
    "频率_Hz",
    "磁通密度峰值B_m_T",
    "磁通密度_RMS_T",
    "THD_2_10",
    "斜率RMS归一化",
    "总变差归一化",
]
HIGHLIGHT_IDS = [16, 76, 98, 126, 168, 230, 271, 338, 348, 379]


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


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="第四问：数据驱动磁芯损耗预测")
    parser.add_argument(
        "--train", type=Path, default=base / "附件一_第四问特征数据.csv"
    )
    parser.add_argument(
        "--test", type=Path, default=base / "附件三_第四问特征数据.csv"
    )
    parser.add_argument("--output-dir", type=Path, default=base)
    return parser.parse_args()


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.clip(np.asarray(predicted, dtype=float), 1e-12, None)
    return {
        "R2": float(r2_score(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "MAE": float(mean_absolute_error(actual, predicted)),
        "MAPE_percent": float(np.mean(np.abs((actual - predicted) / actual)) * 100),
        "RMSLE": float(np.sqrt(np.mean((np.log(actual) - np.log(predicted)) ** 2))),
    }


def make_preprocessor(numeric_features: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("num", StandardScaler(), numeric_features),
        ],
        remainder="drop",
    )


def candidate_models(numeric_features: list[str]) -> dict[str, Pipeline]:
    def pipeline(model: object) -> Pipeline:
        return Pipeline(
            [("preprocessor", make_preprocessor(numeric_features)), ("model", model)]
        )

    candidates: dict[str, Pipeline] = {
        "Ridge": pipeline(Ridge(alpha=3.0)),
        "HistGradientBoosting": pipeline(
            HistGradientBoostingRegressor(
                learning_rate=0.055,
                max_iter=420,
                max_leaf_nodes=31,
                l2_regularization=0.3,
                random_state=SEED,
            )
        ),
        "RandomForest": pipeline(
            RandomForestRegressor(
                n_estimators=360,
                max_features=0.82,
                min_samples_leaf=1,
                n_jobs=-1,
                random_state=SEED,
            )
        ),
        "ExtraTrees": pipeline(
            ExtraTreesRegressor(
                n_estimators=440,
                max_features=0.90,
                min_samples_leaf=1,
                n_jobs=-1,
                random_state=SEED,
            )
        ),
    }
    try:
        from xgboost import XGBRegressor

        candidates["XGBoost"] = pipeline(
            XGBRegressor(
                objective="reg:squarederror",
                n_estimators=700,
                learning_rate=0.045,
                max_depth=8,
                min_child_weight=2,
                subsample=0.88,
                colsample_bytree=0.88,
                reg_alpha=0.02,
                reg_lambda=1.5,
                n_jobs=-1,
                random_state=SEED,
            )
        )
    except ImportError:
        print("未安装 xgboost，将用其他候选模型完成比较。")
    return candidates


def pipeline_predict(model: Pipeline, data: pd.DataFrame) -> np.ndarray:
    """Predict with a compatibility fallback for older XGBoost releases."""
    try:
        return np.asarray(model.predict(data), dtype=float)
    except AttributeError as error:
        if "__sklearn_tags__" not in str(error):
            raise
        transformed = model.named_steps["preprocessor"].transform(data)
        return np.asarray(model.named_steps["model"].predict(transformed), dtype=float)


def load_and_validate(
    train_path: Path, test_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    required_common = [ID_COLUMN, "温度_oC", "频率_Hz", *CATEGORICAL_FEATURES]
    for label, frame in [("训练集", train), ("测试集", test)]:
        missing = [column for column in required_common if column not in frame.columns]
        if missing:
            raise ValueError(f"{label}缺少列：{missing}")
        if frame.isna().any().any():
            raise ValueError(f"{label}存在缺失值。")
    if TARGET_COLUMN not in train.columns:
        raise ValueError(f"训练集缺少目标列 {TARGET_COLUMN}。")
    if (train[TARGET_COLUMN] <= 0).any():
        raise ValueError("训练集磁芯损耗必须为正数。")
    if test[ID_COLUMN].duplicated().any() or set(test[ID_COLUMN]) != set(range(1, 401)):
        raise ValueError("附件三样本编号必须为 1–400 且不重复。")

    feature_columns = [
        column
        for column in train.columns
        if column not in {ID_COLUMN, TARGET_COLUMN}
    ]
    if set(feature_columns) != set(column for column in test.columns if column != ID_COLUMN):
        raise ValueError("训练集与测试集特征列不一致。")
    for column in CATEGORICAL_FEATURES:
        unseen = sorted(set(test[column]) - set(train[column]))
        if unseen:
            raise ValueError(f"测试集 {column} 出现未知类别：{unseen}")
    return train, test, feature_columns


def cross_validate_candidates(
    x: pd.DataFrame,
    y: np.ndarray,
    candidates: dict[str, Pipeline],
    strata: pd.Series,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, float]]:
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_log_predictions: dict[str, np.ndarray] = {}
    correction_factors: dict[str, float] = {}
    rows: list[dict[str, float | str]] = []
    log_y = np.log(y)

    for name, template in candidates.items():
        print(f"正在评估 {name} ...")
        oof_log = np.empty(len(y), dtype=float)
        fold_id = np.empty(len(y), dtype=int)
        for fold, (train_idx, valid_idx) in enumerate(
            splitter.split(x, strata), start=1
        ):
            model = clone(template)
            model.fit(x.iloc[train_idx], log_y[train_idx])
            oof_log[valid_idx] = pipeline_predict(model, x.iloc[valid_idx])
            fold_id[valid_idx] = fold
        correction = float(np.exp(np.mean(log_y - oof_log)))
        predicted = np.exp(oof_log) * correction
        result = metrics(y, predicted)
        rows.append(
            {
                "模型": name,
                "验证方式": "5折温度×材料×波形分层交叉验证",
                "调参说明": "固定候选参数，不使用附件三调参",
                "对数偏差修正因子": correction,
                **result,
            }
        )
        oof_log_predictions[name] = oof_log
        correction_factors[name] = correction

    comparison = pd.DataFrame(rows).sort_values(
        ["RMSLE", "MAPE_percent", "RMSE"], ascending=True
    )
    comparison["综合排名"] = np.arange(1, len(comparison) + 1)
    return comparison, oof_log_predictions, correction_factors


def choose_model(
    comparison: pd.DataFrame,
    oof_log: dict[str, np.ndarray],
    y: np.ndarray,
) -> tuple[dict[str, object], np.ndarray, pd.DataFrame, pd.DataFrame]:
    ordered = comparison["模型"].tolist()
    best_name = ordered[0]
    best_factor = float(
        comparison.loc[comparison["模型"] == best_name, "对数偏差修正因子"].iloc[0]
    )
    best_prediction = np.exp(oof_log[best_name]) * best_factor
    choice: dict[str, object] = {
        "type": "single",
        "models": [best_name],
        "weights": [1.0],
        "bias_factor": best_factor,
        "display_name": best_name,
    }

    if len(ordered) >= 2:
        first, second = ordered[:2]
        blend_rows: list[dict[str, float]] = []
        best_blend_prediction: np.ndarray | None = None
        best_blend_score = float("inf")
        best_weight = 1.0
        best_blend_factor = 1.0
        log_y = np.log(y)
        for weight in np.linspace(0.0, 1.0, 21):
            blended_log = weight * oof_log[first] + (1.0 - weight) * oof_log[second]
            factor = float(np.exp(np.mean(log_y - blended_log)))
            predicted = np.exp(blended_log) * factor
            row = {"模型1权重": float(weight), "偏差修正因子": factor, **metrics(y, predicted)}
            blend_rows.append(row)
            if row["RMSLE"] < best_blend_score:
                best_blend_score = row["RMSLE"]
                best_weight = float(weight)
                best_blend_factor = factor
                best_blend_prediction = predicted

        best_single_score = metrics(y, best_prediction)["RMSLE"]
        if best_blend_prediction is not None and best_blend_score < best_single_score:
            display_name = f"加权集成({first}+{second})"
            choice = {
                "type": "blend",
                "models": [first, second],
                "weights": [best_weight, 1.0 - best_weight],
                "bias_factor": best_blend_factor,
                "display_name": display_name,
            }
            best_prediction = best_blend_prediction
            ensemble_row = {
                "模型": display_name,
                "验证方式": "前两名OOF对数预测加权集成",
                "调参说明": "21个权重候选，仅使用OOF预测选择",
                "对数偏差修正因子": best_blend_factor,
                **metrics(y, best_prediction),
                "综合排名": 0,
            }
            comparison = pd.concat([pd.DataFrame([ensemble_row]), comparison], ignore_index=True)
        blend_search = pd.DataFrame(blend_rows)
        blend_search.insert(0, "模型1", first)
        blend_search.insert(1, "模型2", second)
    else:
        blend_search = pd.DataFrame()

    comparison = comparison.sort_values(["RMSLE", "MAPE_percent", "RMSE"]).reset_index(drop=True)
    comparison["综合排名"] = np.arange(1, len(comparison) + 1)
    return choice, best_prediction, blend_search, comparison


def fit_selected_models(
    choice: dict[str, object],
    candidates: dict[str, Pipeline],
    x_train: pd.DataFrame,
    y: np.ndarray,
    x_test: pd.DataFrame,
) -> tuple[np.ndarray, list[tuple[str, float, Pipeline]]]:
    log_y = np.log(y)
    selected: list[tuple[str, float, Pipeline]] = []
    log_prediction = np.zeros(len(x_test), dtype=float)
    for name, weight in zip(choice["models"], choice["weights"]):
        fitted = clone(candidates[str(name)])
        fitted.fit(x_train, log_y)
        log_prediction += float(weight) * pipeline_predict(fitted, x_test)
        selected.append((str(name), float(weight), fitted))
    prediction = np.exp(log_prediction) * float(choice["bias_factor"])
    return prediction, selected


def group_validation_table(
    train: pd.DataFrame, actual: np.ndarray, predicted: np.ndarray
) -> pd.DataFrame:
    work = train[["温度_oC", "磁芯材料", "励磁波形"]].copy()
    work["实际损耗"] = actual
    work["预测损耗"] = predicted
    rows: list[dict[str, object]] = []
    for column, label in [
        ("温度_oC", "温度"),
        ("磁芯材料", "磁芯材料"),
        ("励磁波形", "励磁波形"),
    ]:
        for value, group in work.groupby(column, observed=True):
            result = metrics(group["实际损耗"], group["预测损耗"])
            rows.append(
                {"分组类型": label, "分组值": value, "样本数": len(group), **result}
            )
    return pd.DataFrame(rows)


def coverage_tables(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray]:
    rows: list[dict[str, object]] = []
    outside_matrix = np.zeros((len(test), len(KEY_RANGE_FEATURES)), dtype=bool)
    for index, feature in enumerate(KEY_RANGE_FEATURES):
        train_min = float(train[feature].min())
        train_max = float(train[feature].max())
        outside = (test[feature] < train_min) | (test[feature] > train_max)
        outside_matrix[:, index] = outside
        rows.append(
            {
                "特征": feature,
                "训练集最小值": train_min,
                "训练集最大值": train_max,
                "测试集最小值": float(test[feature].min()),
                "测试集最大值": float(test[feature].max()),
                "测试集超范围样本数": int(outside.sum()),
                "测试集超范围比例_percent": float(outside.mean() * 100),
            }
        )
    return pd.DataFrame(rows), outside_matrix.sum(axis=1)


def extract_feature_importance(
    fitted_models: list[tuple[str, float, Pipeline]],
) -> pd.DataFrame:
    combined: dict[str, float] = {}
    used_weight = 0.0
    for name, weight, fitted in fitted_models:
        model = fitted.named_steps["model"]
        preprocessor = fitted.named_steps["preprocessor"]
        names = preprocessor.get_feature_names_out()
        if hasattr(model, "feature_importances_"):
            values = np.asarray(model.feature_importances_, dtype=float)
        elif hasattr(model, "coef_"):
            values = np.abs(np.ravel(model.coef_).astype(float))
        else:
            continue
        total = values.sum()
        if total <= 0:
            continue
        values = values / total
        for feature, value in zip(names, values):
            clean = feature.replace("cat__", "").replace("num__", "")
            combined[clean] = combined.get(clean, 0.0) + weight * float(value)
        used_weight += weight
    if not combined:
        return pd.DataFrame(columns=["特征", "归一化重要性", "排名"])
    result = pd.DataFrame(
        {"特征": list(combined), "归一化重要性": list(combined.values())}
    )
    result["归一化重要性"] /= max(used_weight, 1e-12)
    result = result.sort_values("归一化重要性", ascending=False).reset_index(drop=True)
    result["排名"] = np.arange(1, len(result) + 1)
    return result


def plot_model_comparison(comparison: pd.DataFrame, output: Path) -> None:
    ordered = comparison.sort_values("RMSLE", ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.barplot(data=ordered, x="RMSLE", y="模型", ax=axes[0], color="#4C78A8")
    axes[0].set_title("五折交叉验证 RMSLE（越小越好）")
    sns.barplot(data=ordered, x="MAPE_percent", y="模型", ax=axes[1], color="#F58518")
    axes[1].set_title("五折交叉验证 MAPE（%，越小越好）")
    axes[1].set_ylabel("")
    fig.suptitle("第四问候选模型比较", fontsize=15)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_validation(actual: np.ndarray, predicted: np.ndarray, output: Path) -> None:
    log_actual = np.log10(actual)
    log_predicted = np.log10(predicted)
    residual = np.log(predicted) - np.log(actual)
    lo = min(log_actual.min(), log_predicted.min())
    hi = max(log_actual.max(), log_predicted.max())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hexbin(log_actual, log_predicted, gridsize=55, mincnt=1, cmap="viridis")
    axes[0].plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="1:1")
    axes[0].set_xlabel("log10(实际损耗)")
    axes[0].set_ylabel("log10(预测损耗)")
    axes[0].set_title("五折交叉验证：实际值 vs 预测值")
    axes[0].legend()
    sns.histplot(residual, bins=45, kde=True, ax=axes[1], color="#4C78A8")
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1.5)
    axes[1].set_xlabel("ln(预测值) - ln(实际值)")
    axes[1].set_title("对数残差分布")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_group_errors(grouped: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for axis, group_type in zip(axes, ["温度", "磁芯材料", "励磁波形"]):
        subset = grouped[grouped["分组类型"] == group_type].copy()
        subset["分组值"] = subset["分组值"].astype(str)
        sns.barplot(data=subset, x="分组值", y="MAPE_percent", ax=axis, color="#54A24B")
        axis.set_title(f"{group_type}分组 MAPE")
        axis.set_xlabel(group_type)
        axis.set_ylabel("MAPE (%)")
        axis.tick_params(axis="x", rotation=20)
    fig.suptitle("不同工况下的泛化误差", fontsize=15)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(importance: pd.DataFrame, output: Path) -> None:
    top = importance.head(18).sort_values("归一化重要性", ascending=True)
    fig, axis = plt.subplots(figsize=(9, 7))
    sns.barplot(data=top, x="归一化重要性", y="特征", ax=axis, color="#B279A2")
    axis.set_title("最终模型特征重要性（Top 18）")
    axis.set_xlabel("归一化重要性")
    axis.set_ylabel("")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_test_predictions(predictions: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(
        data=predictions,
        x="预测磁芯损耗_w每m3",
        hue="磁芯材料",
        bins=35,
        element="step",
        log_scale=True,
        ax=axes[0],
    )
    axes[0].set_title("附件三预测损耗分布")
    axes[0].set_xlabel("预测磁芯损耗 (W/m³，对数坐标)")
    sns.scatterplot(
        data=predictions,
        x="样本编号",
        y="预测磁芯损耗_w每m3",
        hue="磁芯材料",
        style="励磁波形",
        s=35,
        ax=axes[1],
    )
    axes[1].set_yscale("log")
    axes[1].set_title("400 个待预测样本")
    axes[1].set_ylabel("预测磁芯损耗 (W/m³)")
    axes[1].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    configure_plot_style()

    train, test, feature_columns = load_and_validate(args.train, args.test)
    numeric_features = [column for column in feature_columns if column not in CATEGORICAL_FEATURES]
    x_train = train[feature_columns]
    x_test = test[feature_columns]
    y = train[TARGET_COLUMN].to_numpy(dtype=float)
    strata = (
        train["温度_oC"].astype(str)
        + "|"
        + train["磁芯材料"].astype(str)
        + "|"
        + train["励磁波形"].astype(str)
    )

    candidates = candidate_models(numeric_features)
    comparison, oof_log, correction_factors = cross_validate_candidates(
        x_train, y, candidates, strata
    )
    choice, oof_prediction, blend_search, comparison = choose_model(
        comparison, oof_log, y
    )
    test_prediction, fitted_models = fit_selected_models(
        choice, candidates, x_train, y, x_test
    )

    grouped = group_validation_table(train, y, oof_prediction)
    coverage, outside_count = coverage_tables(train, test)
    importance = extract_feature_importance(fitted_models)
    validation = train[
        [ID_COLUMN, "温度_oC", "频率_Hz", "磁芯材料", "励磁波形", TARGET_COLUMN]
    ].copy()
    validation["五折预测磁芯损耗_w每m3"] = oof_prediction
    validation["相对误差_percent"] = (
        (oof_prediction - y) / y * 100
    )
    validation["绝对百分比误差"] = np.abs(validation["相对误差_percent"])

    predictions = test[
        [ID_COLUMN, "温度_oC", "频率_Hz", "磁芯材料", "励磁波形", "磁通密度峰值B_m_T"]
    ].copy()
    predictions["预测磁芯损耗_w每m3"] = test_prediction
    predictions["附件四填写值_保留1位小数"] = np.round(test_prediction, 1)
    predictions["超出训练集关键特征范围数"] = outside_count
    predictions["预测范围标记"] = np.where(
        outside_count == 0, "范围内", "需谨慎（存在特征外推）"
    )
    predictions = predictions.sort_values(ID_COLUMN).reset_index(drop=True)
    highlights = predictions[predictions[ID_COLUMN].isin(HIGHLIGHT_IDS)].copy()
    highlights["题目指定顺序"] = pd.Categorical(
        highlights[ID_COLUMN], categories=HIGHLIGHT_IDS, ordered=True
    )
    highlights = highlights.sort_values("题目指定顺序").drop(columns="题目指定顺序")

    comparison.to_csv(output / "模型比较.csv", index=False, encoding="utf-8-sig")
    grouped.to_csv(output / "分组验证指标.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(output / "训练测试分布检查.csv", index=False, encoding="utf-8-sig")
    importance.to_csv(output / "特征重要性.csv", index=False, encoding="utf-8-sig")
    validation.to_csv(output / "模型五折预测.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(output / "附件三预测结果.csv", index=False, encoding="utf-8-sig")
    highlights.to_csv(output / "指定样本预测结果.csv", index=False, encoding="utf-8-sig")
    if not blend_search.empty:
        blend_search.to_csv(output / "集成权重搜索.csv", index=False, encoding="utf-8-sig")

    plot_model_comparison(comparison, output / "候选模型比较.png")
    plot_validation(y, oof_prediction, output / "五折验证实际值与预测值.png")
    plot_group_errors(grouped, output / "不同工况泛化误差.png")
    if not importance.empty:
        plot_feature_importance(importance, output / "特征重要性.png")
    plot_test_predictions(predictions, output / "附件三预测分布.png")

    overall = metrics(y, oof_prediction)
    conclusions = {
        "竞赛": "2024 年“华为杯”中国研究生数学建模竞赛",
        "题目": "C 题——数据驱动下磁性元件的磁芯损耗建模",
        "问题": "第四问",
        "训练样本数": int(len(train)),
        "测试样本数": int(len(test)),
        "最终模型": choice,
        "五折验证总体指标": overall,
        "测试集关键特征全部在训练范围内的样本数": int((outside_count == 0).sum()),
        "工业应用建议": [
            "模型适用于附件一覆盖的4种材料、3种波形、4个温度档位及相近频率与磁通密度范围。",
            "实际使用时先检查温度、频率、B_m和波形谐波特征是否超出训练范围。",
            "对于未见材料或大幅外推工况，应增加实测样本后重新训练，不宜直接用本结果替代实验标定。",
            "设计选型时建议使用交叉验证的分组误差作为安全余量参考。",
        ],
    }
    with (output / "分析结论.json").open("w", encoding="utf-8") as file:
        json.dump(conclusions, file, ensure_ascii=False, indent=2)

    print("\n最终模型：", choice["display_name"])
    print("五折验证指标：", json.dumps(overall, ensure_ascii=False, indent=2))
    print("结果已写入：", output)


if __name__ == "__main__":
    main()
