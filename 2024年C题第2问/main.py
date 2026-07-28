#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2024 年“华为杯”中国研究生数学建模竞赛 C 题第二问。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TEMPERATURE_COLUMN = "温度，oC"
FREQUENCY_COLUMN = "频率，Hz"
LOSS_COLUMN = "磁芯损耗，w/m3"
WAVEFORM_COLUMN = "励磁波形"
SEED = 2024
N_SPLITS = 5
T_REFERENCE = 25.0
T_SCALE = 65.0


def configure_plot_style() -> None:
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC",
        "Arial Unicode MS",
        "Microsoft YaHei",
        "SimHei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 140


def load_data(path: Path) -> pd.DataFrame:
    data = pd.read_excel(path)
    required = [TEMPERATURE_COLUMN, FREQUENCY_COLUMN, LOSS_COLUMN, WAVEFORM_COLUMN]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"输入文件缺少必要列：{missing}")
    if data.isna().any().any():
        raise ValueError("输入数据存在缺失值，请先清洗。")
    if (data[FREQUENCY_COLUMN] <= 0).any() or (data[LOSS_COLUMN] <= 0).any():
        raise ValueError("频率和磁芯损耗必须为正数。")
    if not data[WAVEFORM_COLUMN].astype(str).str.contains("正弦").all():
        raise ValueError("第二问只应使用正弦波样本。")

    flux = data.iloc[:, 4:].to_numpy(dtype=float)
    # 半峰峰值比直接取最大绝对值更能抵消很小的直流偏置。
    data = data.iloc[:, :4].copy()
    data["磁通密度峰值B_m，T"] = (flux.max(axis=1) - flux.min(axis=1)) / 2.0
    if (data["磁通密度峰值B_m，T"] <= 0).any():
        raise ValueError("磁通密度峰值必须为正数。")
    return data


def build_design(
    data: pd.DataFrame,
    indices: np.ndarray,
    corrected: bool,
    f_reference: float,
    b_reference: float,
) -> np.ndarray:
    frequency = data[FREQUENCY_COLUMN].to_numpy(dtype=float)[indices]
    flux_peak = data["磁通密度峰值B_m，T"].to_numpy(dtype=float)[indices]
    temperature = data[TEMPERATURE_COLUMN].to_numpy(dtype=float)[indices]
    log_f = np.log(frequency / f_reference)
    log_b = np.log(flux_peak / b_reference)
    if not corrected:
        return np.column_stack([np.ones(len(indices)), log_f, log_b])
    tau = (temperature - T_REFERENCE) / T_SCALE
    return np.column_stack(
        [
            np.ones(len(indices)),
            log_f,
            log_b,
            tau,
            tau**2,
            tau * log_f,
            tau * log_b,
        ]
    )


def fit_log_linear(design: np.ndarray, loss: np.ndarray) -> np.ndarray:
    coefficients, *_ = np.linalg.lstsq(design, np.log(loss), rcond=None)
    return coefficients


def predict(design: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return np.exp(design @ coefficients)


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = actual - predicted
    ss_total = np.sum((actual - actual.mean()) ** 2)
    return {
        "R2": float(1.0 - np.sum(residual**2) / ss_total),
        "RMSE": float(np.sqrt(np.mean(residual**2))),
        "MAE": float(np.mean(np.abs(residual))),
        "MAPE_percent": float(np.mean(np.abs(residual / actual)) * 100.0),
        "RMSLE": float(np.sqrt(np.mean((np.log(actual) - np.log(predicted)) ** 2))),
    }


def make_folds(sample_count: int) -> list[np.ndarray]:
    rng = np.random.default_rng(SEED)
    shuffled = rng.permutation(sample_count)
    return [fold.astype(int) for fold in np.array_split(shuffled, N_SPLITS)]


def cross_validated_predictions(
    data: pd.DataFrame,
    corrected: bool,
    f_reference: float,
    b_reference: float,
) -> np.ndarray:
    loss = data[LOSS_COLUMN].to_numpy(dtype=float)
    predictions = np.empty(len(data), dtype=float)
    all_indices = np.arange(len(data))
    for validation_indices in make_folds(len(data)):
        training_indices = np.setdiff1d(all_indices, validation_indices, assume_unique=True)
        train_design = build_design(
            data, training_indices, corrected, f_reference, b_reference
        )
        coefficients = fit_log_linear(train_design, loss[training_indices])
        validation_design = build_design(
            data, validation_indices, corrected, f_reference, b_reference
        )
        predictions[validation_indices] = predict(validation_design, coefficients)
    return predictions


def leave_one_temperature_out(
    data: pd.DataFrame,
    corrected: bool,
    f_reference: float,
    b_reference: float,
) -> pd.DataFrame:
    temperature = data[TEMPERATURE_COLUMN].to_numpy(dtype=float)
    loss = data[LOSS_COLUMN].to_numpy(dtype=float)
    rows: list[dict[str, float | str]] = []
    for held_out in sorted(np.unique(temperature)):
        validation_indices = np.flatnonzero(temperature == held_out)
        training_indices = np.flatnonzero(temperature != held_out)
        coefficients = fit_log_linear(
            build_design(data, training_indices, corrected, f_reference, b_reference),
            loss[training_indices],
        )
        predicted = predict(
            build_design(data, validation_indices, corrected, f_reference, b_reference),
            coefficients,
        )
        rows.append(
            {
                "模型": "温度修正模型" if corrected else "传统模型",
                "留出温度_oC": float(held_out),
                "样本数": int(len(validation_indices)),
                **calculate_metrics(loss[validation_indices], predicted),
            }
        )
    return pd.DataFrame(rows)


def save_figures(
    output_dir: Path,
    result: pd.DataFrame,
    temperature_summary: pd.DataFrame,
    corrected_coefficients: np.ndarray,
) -> None:
    configure_plot_style()
    colors = {25: "#3B82F6", 50: "#10B981", 70: "#F59E0B", 90: "#EF4444"}
    actual = result["实际磁芯损耗_w每m3"].to_numpy()
    limits = [actual.min() * 0.8, actual.max() * 1.25]

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, prediction_column, title in [
        (axes[0], "传统模型_五折预测", "传统 Steinmetz 模型"),
        (axes[1], "温度修正模型_五折预测", "温度修正模型"),
    ]:
        for temperature, group in result.groupby("温度_oC"):
            axis.scatter(
                group["实际磁芯损耗_w每m3"],
                group[prediction_column],
                s=16,
                alpha=0.65,
                color=colors[int(temperature)],
                label=f"{int(temperature)}°C",
            )
        axis.plot(limits, limits, "--", color="#111827", linewidth=1.2)
        axis.set(xscale="log", yscale="log", xlim=limits, ylim=limits)
        axis.set_xlabel("实际磁芯损耗 / (W·m⁻³)")
        axis.set_ylabel("五折预测磁芯损耗 / (W·m⁻³)")
        axis.set_title(title)
        axis.grid(alpha=0.2)
    axes[1].legend(title="温度", frameon=False)
    figure.suptitle("实际值与交叉验证预测值对比")
    figure.savefig(output_dir / "实际值与预测值对比.png", bbox_inches="tight")
    plt.close(figure)

    pivot = temperature_summary.pivot(
        index="温度_oC", columns="模型", values="MAPE_percent"
    )
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    pivot[["传统模型", "温度修正模型"]].plot(
        kind="bar", ax=axis, color=["#94A3B8", "#2563EB"], width=0.72
    )
    axis.set_xlabel("温度 / °C")
    axis.set_ylabel("五折交叉验证 MAPE / %")
    axis.set_title("不同温度下的预测误差")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(output_dir / "温度误差对比.png", bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, prediction_column, title, color in [
        (axes[0], "传统模型_五折预测", "传统模型", "#64748B"),
        (axes[1], "温度修正模型_五折预测", "温度修正模型", "#2563EB"),
    ]:
        log_residual = np.log(actual) - np.log(result[prediction_column].to_numpy())
        axis.scatter(result[prediction_column], log_residual, s=14, alpha=0.55, color=color)
        axis.axhline(0, linestyle="--", linewidth=1.2, color="#111827")
        axis.set_xscale("log")
        axis.set_xlabel("五折预测磁芯损耗 / (W·m⁻³)")
        axis.set_ylabel("对数残差 ln(实际/预测)")
        axis.set_title(title)
        axis.grid(alpha=0.2)
    figure.suptitle("交叉验证残差诊断")
    figure.savefig(output_dir / "残差诊断.png", bbox_inches="tight")
    plt.close(figure)

    tau = np.linspace(0.0, 1.0, 200)
    temperature = T_REFERENCE + T_SCALE * tau
    scale_factor = np.exp(
        corrected_coefficients[3] * tau + corrected_coefficients[4] * tau**2
    )
    alpha = corrected_coefficients[1] + corrected_coefficients[5] * tau
    beta = corrected_coefficients[2] + corrected_coefficients[6] * tau
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    series = [
        (scale_factor, "损耗尺度修正因子", "#7C3AED"),
        (alpha, "频率指数 α(T)", "#059669"),
        (beta, "磁通密度指数 β(T)", "#DC2626"),
    ]
    for axis, (values, title, color) in zip(axes, series):
        axis.plot(temperature, values, color=color, linewidth=2.2)
        axis.scatter([25, 50, 70, 90], np.interp([25, 50, 70, 90], temperature, values),
                     color=color, s=28)
        axis.set_xlabel("温度 / °C")
        axis.set_title(title)
        axis.grid(alpha=0.25)
    figure.suptitle("温度修正模型参数随温度的变化")
    figure.savefig(output_dir / "温度修正参数.png", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=script_dir / "材料1正弦波.xlsx",
        help="材料1正弦波数据文件",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=script_dir, help="结果输出目录"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(args.input)
    frequency = data[FREQUENCY_COLUMN].to_numpy(dtype=float)
    flux_peak = data["磁通密度峰值B_m，T"].to_numpy(dtype=float)
    loss = data[LOSS_COLUMN].to_numpy(dtype=float)
    f_reference = float(np.exp(np.mean(np.log(frequency))))
    b_reference = float(np.exp(np.mean(np.log(flux_peak))))
    all_indices = np.arange(len(data))

    base_design = build_design(data, all_indices, False, f_reference, b_reference)
    corrected_design = build_design(data, all_indices, True, f_reference, b_reference)
    base_coefficients = fit_log_linear(base_design, loss)
    corrected_coefficients = fit_log_linear(corrected_design, loss)
    base_fitted = predict(base_design, base_coefficients)
    corrected_fitted = predict(corrected_design, corrected_coefficients)
    base_cv = cross_validated_predictions(data, False, f_reference, b_reference)
    corrected_cv = cross_validated_predictions(data, True, f_reference, b_reference)

    model_summary = pd.DataFrame(
        [
            {
                "模型": "传统模型",
                "参数个数": 3,
                **{f"五折_{k}": v for k, v in calculate_metrics(loss, base_cv).items()},
                **{f"全样本_{k}": v for k, v in calculate_metrics(loss, base_fitted).items()},
            },
            {
                "模型": "温度修正模型",
                "参数个数": 7,
                **{
                    f"五折_{k}": v
                    for k, v in calculate_metrics(loss, corrected_cv).items()
                },
                **{
                    f"全样本_{k}": v
                    for k, v in calculate_metrics(loss, corrected_fitted).items()
                },
            },
        ]
    )

    result = pd.DataFrame(
        {
            "样本编号": np.arange(1, len(data) + 1),
            "温度_oC": data[TEMPERATURE_COLUMN].astype(int),
            "频率_Hz": frequency,
            "磁通密度峰值B_m_T": flux_peak,
            "实际磁芯损耗_w每m3": loss,
            "传统模型_五折预测": base_cv,
            "温度修正模型_五折预测": corrected_cv,
            "传统模型_相对误差_percent": (base_cv - loss) / loss * 100.0,
            "温度修正模型_相对误差_percent": (corrected_cv - loss) / loss * 100.0,
            "传统模型_全样本拟合": base_fitted,
            "温度修正模型_全样本拟合": corrected_fitted,
        }
    )

    temperature_rows: list[dict[str, float | str]] = []
    for temperature, group in result.groupby("温度_oC"):
        group_actual = group["实际磁芯损耗_w每m3"].to_numpy()
        for model, column in [
            ("传统模型", "传统模型_五折预测"),
            ("温度修正模型", "温度修正模型_五折预测"),
        ]:
            temperature_rows.append(
                {
                    "温度_oC": int(temperature),
                    "模型": model,
                    "样本数": int(len(group)),
                    **calculate_metrics(group_actual, group[column].to_numpy()),
                }
            )
    temperature_summary = pd.DataFrame(temperature_rows)

    leave_temperature_out = pd.concat(
        [
            leave_one_temperature_out(data, False, f_reference, b_reference),
            leave_one_temperature_out(data, True, f_reference, b_reference),
        ],
        ignore_index=True,
    )

    parameter_rows = [
        {"模型": "传统模型", "参数": "ln(P0)", "数值": base_coefficients[0]},
        {"模型": "传统模型", "参数": "alpha", "数值": base_coefficients[1]},
        {"模型": "传统模型", "参数": "beta", "数值": base_coefficients[2]},
    ]
    corrected_names = [
        "ln(P0)", "alpha0", "beta0", "c1", "c2", "alpha_T", "beta_T"
    ]
    parameter_rows.extend(
        {
            "模型": "温度修正模型",
            "参数": name,
            "数值": float(value),
        }
        for name, value in zip(corrected_names, corrected_coefficients)
    )
    parameter_table = pd.DataFrame(parameter_rows)

    output_tables = {
        "模型比较.csv": model_summary,
        "逐样本预测.csv": result,
        "温度误差统计.csv": temperature_summary,
        "留一温度验证.csv": leave_temperature_out,
        "模型参数.csv": parameter_table,
    }
    for filename, table in output_tables.items():
        table.to_csv(args.output_dir / filename, index=False, encoding="utf-8-sig")

    metadata = {
        "sample_count": int(len(data)),
        "temperature_levels_oC": sorted(
            int(value) for value in data[TEMPERATURE_COLUMN].unique()
        ),
        "f_reference_Hz": f_reference,
        "b_reference_T": b_reference,
        "temperature_reference_oC": T_REFERENCE,
        "temperature_scale_oC": T_SCALE,
        "traditional_coefficients": {
            "ln_P0": float(base_coefficients[0]),
            "P0": float(np.exp(base_coefficients[0])),
            "alpha": float(base_coefficients[1]),
            "beta": float(base_coefficients[2]),
        },
        "corrected_coefficients": {
            name: float(value)
            for name, value in zip(corrected_names, corrected_coefficients)
        },
        "validation": {
            "method": f"{N_SPLITS}-fold shuffled cross-validation",
            "seed": SEED,
        },
    }
    (args.output_dir / "模型参数.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    save_figures(args.output_dir, result, temperature_summary, corrected_coefficients)

    base_mape = model_summary.loc[
        model_summary["模型"] == "传统模型", "五折_MAPE_percent"
    ].iloc[0]
    corrected_mape = model_summary.loc[
        model_summary["模型"] == "温度修正模型", "五折_MAPE_percent"
    ].iloc[0]
    print(f"样本数：{len(data)}")
    print(f"传统模型五折 MAPE：{base_mape:.3f}%")
    print(f"温度修正模型五折 MAPE：{corrected_mape:.3f}%")
    print(f"MAPE 相对降低：{(base_mape - corrected_mape) / base_mape * 100:.2f}%")
    print(f"结果已保存至：{args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
