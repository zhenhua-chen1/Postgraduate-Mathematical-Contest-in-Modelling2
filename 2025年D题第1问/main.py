#!/usr/bin/env python3
"""2025 年中国研究生数学建模竞赛 D 题第一问的端到端实现。

模型 A 融合风廓线雷达与微波辐射计，计算梯度理查逊数；模型 B 只使用
风廓线雷达变量学习模型 A 的诊断结果。交叉验证时以完整的“站点—时次”
垂直廓线为分组单位，避免相邻高度层同时出现在训练集和验证集造成数据泄漏。
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".mplconfig").resolve()))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ------------------------------
# 全局物理常数与建模参数
# ------------------------------
GRAVITY = 9.80665  # 重力加速度，单位 m/s²
DRY_AIR_GAS_CONSTANT = 287.04
KAPPA = 0.286  # 干空气泊松常数 R_d / c_p
RI_TARGET_MIN = 0.0
RI_TARGET_MAX = 20.0  # 限制极稳定层的大 Ri 对回归模型的支配
RANDOM_SEED = 2025
STATION_NAMES = {"station_a": "a", "station_b": "b"}


# ============================================================
# 1. 风廓线雷达数据解析
# ============================================================
def parse_robs_file(path: Path, station_name: str) -> pd.DataFrame:
    """解析单个风廓线雷达 ROBS 文本产品。

    ROBS 段每行依次包含高度、风向、水平风速、垂直风速、两个质量标志和
    折射率结构常数 Cn²。带有斜杠的记录表示缺测，在这里直接跳过。
    """
    lines = path.read_text(encoding="ascii", errors="replace").splitlines()
    if len(lines) < 4 or not lines[0].startswith("WNDROBS"):
        raise ValueError(f"Not a supported WNDROBS file: {path}")

    header = lines[1].split()
    if len(header) < 6:
        raise ValueError(f"Invalid WNDROBS header: {path}")
    station_id, longitude, latitude, site_altitude, timezone, timestamp = header[:6]
    observed_at = pd.to_datetime(timestamp, format="%Y%m%d%H%M%S")

    # “ROBS”标记之前是设备和站点元数据，之后才是逐高度观测记录。
    try:
        start = lines.index("ROBS") + 1
    except ValueError as exc:
        raise ValueError(f"ROBS section not found: {path}") from exc

    rows: list[dict[str, object]] = []
    for line in lines[start:]:
        if line.strip() == "NNNN":
            break
        fields = line.split()
        if len(fields) != 7 or any("/" in value for value in fields):
            continue
        try:
            height, direction, speed, vertical_speed = map(float, fields[:4])
            flag_1, flag_2 = map(int, fields[4:6])
            cn2 = float(fields[6])
        except ValueError:
            continue
        rows.append({
            "station_name": station_name,
            "station_id": station_id,
            "time": observed_at,
            "timezone": timezone,
            "longitude_deg": float(longitude),
            "latitude_deg": float(latitude),
            "site_altitude_m": float(site_altitude),
            "height_m": height,
            "wind_dir_deg": direction,
            "wind_speed_mps": speed,
            "vertical_speed_mps": vertical_speed,
            "quality_flag_1": flag_1,
            "quality_flag_2": flag_2,
            "cn2_m_neg_2_3": cn2,
            "profiler_source": path.name,
        })
    if not rows:
        raise ValueError(f"No valid ROBS rows found: {path}")
    return pd.DataFrame(rows)


def load_wind_profiler(raw_dir: Path) -> pd.DataFrame:
    """读取原始数据目录下 a、b 两站的全部有效风廓线记录。"""
    tables: list[pd.DataFrame] = []
    for directory, station_name in STATION_NAMES.items():
        files = sorted((raw_dir / directory / "wind_profiler").glob("*_P_WPRD_LC_ROBS.TXT"))
        if not files:
            raise FileNotFoundError(f"No ROBS files under {raw_dir / directory}")
        tables.extend(parse_robs_file(path, station_name) for path in files)
    data = pd.concat(tables, ignore_index=True)
    data["profile_id"] = data["station_name"] + "__" + data["time"].astype(str)
    return data.sort_values(["station_name", "time", "height_m"]).reset_index(drop=True)


# ============================================================
# 2. 微波辐射计数据解析与时间匹配
# ============================================================
def read_radiometer_table(path: Path) -> pd.DataFrame:
    """兼容读取 UTF-8 和旧式单字节编码的微波辐射计文件。"""
    last_error = None
    for encoding in ("utf-8", "latin1"):
        try:
            return pd.read_csv(path, sep="\t", skiprows=1, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise ValueError(f"Unable to decode {path}") from last_error


def parse_radiometer_file(path: Path, station_name: str) -> pd.DataFrame:
    """提取数据类型 11 的温度廓线及对应地面气压。"""
    raw = read_radiometer_table(path)
    if raw.shape[1] < 12:
        raise ValueError(f"Unexpected radiometer layout: {path}")

    type_column = raw.columns[2]
    pressure_columns = [column for column in raw.columns if str(column).startswith("SurPre")]
    if not pressure_columns:
        raise ValueError(f"Surface-pressure column not found: {path}")
    pressure_column = pressure_columns[0]

    height_columns: list[tuple[str, float]] = []
    for column in raw.columns:
        match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\(km\)", str(column))
        if match:
            height_columns.append((column, float(match.group(1)) * 1000.0))
    if not height_columns:
        raise ValueError(f"Temperature-height columns not found: {path}")

    temperatures = raw[pd.to_numeric(raw[type_column], errors="coerce") == 11].copy()
    temperatures["time"] = pd.to_datetime(temperatures["DateTime"])
    records: list[dict[str, object]] = []
    for _, row in temperatures.iterrows():
        for column, height_m in height_columns:
            value = pd.to_numeric(row[column], errors="coerce")
            if pd.notna(value):
                records.append({
                    "station_name": station_name,
                    "time": row["time"],
                    "height_m": height_m,
                    "temperature_c": float(value),
                    "surface_pressure_hpa": float(row[pressure_column]),
                    "radiometer_source": path.name,
                })
    profile = pd.DataFrame(records)
    # 官方 b 站文件存在重复时次。先按相同时间和高度求均值，再做时间插值，
    # 可以保证预处理结果唯一、可复现。
    return (
        profile.groupby(["station_name", "time", "height_m"], as_index=False)
        .agg(
            temperature_c=("temperature_c", "mean"),
            surface_pressure_hpa=("surface_pressure_hpa", "mean"),
            radiometer_source=("radiometer_source", "first"),
        )
        .sort_values(["time", "height_m"])
    )


def load_radiometers(raw_dir: Path) -> pd.DataFrame:
    tables = []
    for directory, station_name in STATION_NAMES.items():
        path = raw_dir / directory / "microwave_radiometer.txt"
        if not path.exists():
            raise FileNotFoundError(path)
        tables.append(parse_radiometer_file(path, station_name))
    return pd.concat(tables, ignore_index=True)


def radiometer_at_profiler_times(
    radiometer: pd.DataFrame, target_times: pd.DatetimeIndex
) -> tuple[pd.DataFrame, pd.Series]:
    """把辐射计温度和气压插值到风廓线雷达的观测时次。

    两类设备时间分辨率不同：风廓线为 6 分钟，辐射计约为 2 分钟。内部时次
    使用线性时间插值；序列边界缺测则采用最近的有效廓线。
    """
    temperature = radiometer.pivot_table(
        index="time", columns="height_m", values="temperature_c", aggfunc="mean"
    ).sort_index()
    pressure = radiometer.groupby("time")["surface_pressure_hpa"].mean().sort_index()
    combined_index = temperature.index.union(target_times).sort_values()
    temperature = (
        temperature.reindex(combined_index)
        .interpolate(method="time", limit_direction="both")
        .loc[target_times]
    )
    pressure = (
        pressure.reindex(combined_index)
        .interpolate(method="time", limit_direction="both")
        .loc[target_times]
    )
    return temperature, pressure


# ============================================================
# 3. 模型 A：双设备梯度理查逊数
# ============================================================
def wind_components(direction_deg: pd.Series, speed_mps: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """将气象风向和风速转换为东向 u、北向 v 两个水平分量。"""
    direction = np.deg2rad(direction_deg.to_numpy(float))
    speed = speed_mps.to_numpy(float)
    # 气象风向表示“风从哪个方向吹来”，因此转换到速度矢量时需要负号。
    return -speed * np.sin(direction), -speed * np.cos(direction)


def hydrostatic_pressure(
    heights_m: np.ndarray,
    temperature_k: np.ndarray,
    surface_temperature_k: float,
    surface_pressure_hpa: float,
) -> np.ndarray:
    """从地面向上积分静力平衡方程，估算各观测高度的气压。"""
    z = np.concatenate(([0.0], heights_m))
    temperature = np.concatenate(([surface_temperature_k], temperature_k))
    inverse_temperature = 1.0 / temperature
    # 梯形积分计算 ∫(1/T)dz，再代入 p(z)=p0*exp[-g/Rd*∫(1/T)dz]。
    increments = np.diff(z) * 0.5 * (inverse_temperature[:-1] + inverse_temperature[1:])
    integral = np.concatenate(([0.0], np.cumsum(increments)))
    pressure = surface_pressure_hpa * np.exp(-(GRAVITY / DRY_AIR_GAS_CONSTANT) * integral)
    return pressure[1:]


def compute_model_a(profiler: pd.DataFrame, radiometer: pd.DataFrame) -> pd.DataFrame:
    """融合两类设备，计算模型 A 的理查逊数垂直廓线。"""
    outputs: list[pd.DataFrame] = []
    for station_name, station_profiler in profiler.groupby("station_name", sort=True):
        station_mwr = radiometer[radiometer["station_name"] == station_name]
        target_times = pd.DatetimeIndex(sorted(station_profiler["time"].unique()))
        temperatures, pressures = radiometer_at_profiler_times(station_mwr, target_times)
        source_times = pd.DatetimeIndex(sorted(station_mwr["time"].unique()))

        for time, profile in station_profiler.groupby("time", sort=True):
            profile = profile.sort_values("height_m").copy()
            heights = profile["height_m"].to_numpy(float)
            mwr_heights = temperatures.columns.to_numpy(float)
            # 辐射计和风廓线的垂直层不一致，需要把温度插值到雷达高度层。
            temperature_c = np.interp(heights, mwr_heights, temperatures.loc[time].to_numpy(float))
            surface_temperature_k = float(temperatures.loc[time].iloc[0]) + 273.15
            temperature_k = temperature_c + 273.15
            pressure_hpa = hydrostatic_pressure(
                heights, temperature_k, surface_temperature_k, float(pressures.loc[time])
            )
            potential_temperature = temperature_k * (1000.0 / pressure_hpa) ** KAPPA

            u_ms, v_ms = wind_components(profile["wind_dir_deg"], profile["wind_speed_mps"])
            # np.gradient 支持非等距高度网格，边界使用单侧差分，内部使用中心差分。
            dtheta_dz = np.gradient(potential_temperature, heights)
            du_dz = np.gradient(u_ms, heights)
            dv_dz = np.gradient(v_ms, heights)
            n2 = (GRAVITY / potential_temperature) * dtheta_dz
            shear2 = du_dz**2 + dv_dz**2
            # Ri=N²/S²；当风切变平方过小时不进行除法，避免数值爆炸。
            ri = np.divide(n2, shear2, out=np.full_like(n2, np.nan), where=shear2 > 1e-10)

            nearest_gap = min(abs((time - source_time).total_seconds()) for source_time in source_times) / 60
            profile["temperature_c"] = temperature_c
            profile["pressure_hpa"] = pressure_hpa
            profile["potential_temperature_k"] = potential_temperature
            profile["u_ms"] = u_ms
            profile["v_ms"] = v_ms
            profile["dtheta_dz_k_per_m"] = dtheta_dz
            profile["du_dz_per_s"] = du_dz
            profile["dv_dz_per_s"] = dv_dz
            profile["n2_per_s2"] = n2
            profile["shear2_per_s2"] = shear2
            profile["ri_model_a"] = ri
            profile["ri_target"] = np.clip(ri, RI_TARGET_MIN, RI_TARGET_MAX)
            # turbulence_index 仅用于直观展示：数值越大表示 Ri 越小、风险越高。
            profile["turbulence_index"] = 1.0 / (1.0 + profile["ri_target"])
            profile["nearest_radiometer_gap_min"] = nearest_gap
            outputs.append(profile)
    return pd.concat(outputs, ignore_index=True)


# ============================================================
# 4. 模型 B：仅风廓线雷达的空间与时间特征
# ============================================================
def add_model_b_shear(model_a: pd.DataFrame) -> pd.DataFrame:
    """仅利用风廓线雷达，计算平滑后的垂直风切变特征。"""
    outputs = []
    for _, profile in model_a.groupby("profile_id", sort=True):
        profile = profile.sort_values("height_m").copy()
        heights = profile["height_m"].to_numpy(float)
        # 垂直梯度容易放大单层噪声，先进行居中的五层滑动平均。
        u_smooth = profile["u_ms"].rolling(5, center=True, min_periods=1).mean().to_numpy(float)
        v_smooth = profile["v_ms"].rolling(5, center=True, min_periods=1).mean().to_numpy(float)
        profile["model_b_shear_per_s"] = np.hypot(
            np.gradient(u_smooth, heights), np.gradient(v_smooth, heights)
        )
        outputs.append(profile)
    return pd.concat(outputs, ignore_index=True)


def add_temporal_features(table: pd.DataFrame) -> pd.DataFrame:
    """计算同一站点、同一高度在相邻时次之间的风场变化特征。

    先对时间序列做居中三时次滑动平均，再按照真实时间坐标求导。仅出现一次
    的个别高空层无法计算时间导数，将变化率设为 0，并保留时间支持数，避免
    为了统一网格而删除官方原始记录。
    """
    result = table.copy()
    result["horizontal_wind_tendency_mps2"] = 0.0
    result["vertical_wind_tendency_mps2"] = 0.0
    result["shear_tendency_per_s2"] = 0.0
    result["temporal_support_count"] = 1

    for _, level in result.groupby(["station_name", "height_m"], sort=True):
        ordered = level.sort_values("time")
        indices = ordered.index
        support = len(ordered)
        result.loc[indices, "temporal_support_count"] = support
        if support < 2:
            continue
        seconds = (ordered["time"] - ordered["time"].min()).dt.total_seconds().to_numpy(float)
        if np.unique(seconds).size != support:
            continue

        def tendency(column: str) -> np.ndarray:
            """对单个变量进行时间平滑，并计算每秒变化率。"""
            smoothed = (
                ordered[column].rolling(3, center=True, min_periods=1).mean().to_numpy(float)
            )
            return np.gradient(smoothed, seconds)

        du_dt = tendency("u_ms")
        dv_dt = tendency("v_ms")
        dw_dt = tendency("vertical_speed_mps")
        ds_dt = tendency("model_b_shear_per_s")
        result.loc[indices, "horizontal_wind_tendency_mps2"] = np.hypot(du_dt, dv_dt)
        result.loc[indices, "vertical_wind_tendency_mps2"] = np.abs(dw_dt)
        result.loc[indices, "shear_tendency_per_s2"] = np.abs(ds_dt)
    return result


# ============================================================
# 5. 贝叶斯岭回归与分组交叉验证
# ============================================================
def make_model() -> Pipeline:
    """建立“标准化 + 贝叶斯岭回归”流水线。"""
    return Pipeline([
        ("scale", StandardScaler()),
        ("regression", BayesianRidge(fit_intercept=True)),
    ])


def predict_with_std(model: Pipeline, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """同时返回贝叶斯岭回归的预测均值和标准差。"""
    transformed = model.named_steps["scale"].transform(x)
    return model.named_steps["regression"].predict(transformed, return_std=True)


def grouped_cross_validation(
    table: pd.DataFrame, features: list[str], folds: int
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """按完整站点—时次廓线进行分组交叉验证。

    不能逐行随机划分，因为同一廓线相邻高度层高度相关；若被拆到训练集和
    验证集两侧，会产生明显的数据泄漏并高估模型性能。
    """
    x = table[features].to_numpy(float)
    y = table["ri_target"].to_numpy(float)
    groups = table["profile_id"].to_numpy()
    unique_groups = np.unique(groups)
    if not 2 <= folds <= len(unique_groups):
        raise ValueError(f"folds must be between 2 and {len(unique_groups)}")

    prediction = np.full(len(table), np.nan)
    uncertainty = np.full(len(table), np.nan)
    for train_index, test_index in GroupKFold(n_splits=folds).split(x, y, groups):
        fitted = clone(make_model()).fit(x[train_index], y[train_index])
        prediction[test_index], uncertainty[test_index] = predict_with_std(fitted, x[test_index])
    # 回归模型本身没有输出边界，因此按训练目标的物理区间统一裁剪。
    prediction = np.clip(prediction, RI_TARGET_MIN, RI_TARGET_MAX)
    metrics = {
        "mae": float(mean_absolute_error(y, prediction)),
        "rmse": float(mean_squared_error(y, prediction) ** 0.5),
        "r2": float(r2_score(y, prediction)),
    }
    return prediction, uncertainty, metrics


def raw_formula(model: Pipeline, features: list[str]) -> dict[str, object]:
    """把标准化空间中的回归系数还原到原始物理量纲。"""
    scaler: StandardScaler = model.named_steps["scale"]
    regressor: BayesianRidge = model.named_steps["regression"]
    coefficients = regressor.coef_ / scaler.scale_
    intercept = regressor.intercept_ - np.sum(regressor.coef_ * scaler.mean_ / scaler.scale_)
    return {
        "intercept": float(intercept),
        "coefficients": {name: float(value) for name, value in zip(features, coefficients)},
    }


def contiguous_risk_bands(table: pd.DataFrame, threshold: float = 1.0) -> pd.DataFrame:
    """提取模型 A 中 Ri 小于阈值的连续风险高度带。"""
    bands: list[dict[str, object]] = []
    for (station, time), profile in table.groupby(["station_name", "time"], sort=True):
        profile = profile.sort_values("height_m")
        risky = profile[profile["ri_model_a"] < threshold]
        if risky.empty:
            continue
        heights = risky["height_m"].to_numpy(float)
        split_points = np.where(np.diff(heights) > 1.5 * np.median(np.diff(profile["height_m"])))[0] + 1
        for segment in np.split(heights, split_points):
            bands.append({
                "station_name": station,
                "time": time,
                "threshold": threshold,
                "bottom_m": float(segment.min()),
                "top_m": float(segment.max()),
            })
    return pd.DataFrame(bands)


# ============================================================
# 6. 结果可视化
# ============================================================
def save_figures(results: pd.DataFrame, output_dir: Path, selected_model: str) -> None:
    """绘制时高分布、垂直廓线和折外预测对比图。"""
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=160, constrained_layout=True)
    for ax, station in zip(axes, ["a", "b"]):
        subset = results[results["station_name"] == station]
        minutes = (subset["time"] - subset["time"].min()).dt.total_seconds() / 60
        scatter = ax.scatter(
            minutes, subset["height_m"], c=subset["ri_target"], cmap="viridis_r", s=22,
            vmin=0, vmax=RI_TARGET_MAX,
        )
        ax.set(title=f"Station {station.upper()} — Model A", xlabel="Minutes", ylabel="Height (m)")
        fig.colorbar(scatter, ax=ax, label="Ri (display clipped to 0–20)")
    fig.savefig(output_dir / "model_a_time_height.png")
    plt.close(fig)

    for station in ["a", "b"]:
        subset = results[results["station_name"] == station]
        times = sorted(subset["time"].unique())
        fig, axes = plt.subplots(2, 3, figsize=(11, 8), dpi=160, sharex=True, sharey=True)
        for ax, time in zip(axes.flat, times):
            profile = subset[subset["time"] == time].sort_values("height_m")
            ax.plot(profile["ri_target"], profile["height_m"], label="Model A", linewidth=1.5)
            ax.plot(
                profile["model_b_final"], profile["height_m"],
                label=f"Model B ({selected_model})", linewidth=1.3,
            )
            ax.fill_betweenx(
                profile["height_m"],
                np.clip(
                    profile["model_b_final"] - 1.96 * profile["model_b_final_std"],
                    RI_TARGET_MIN,
                    RI_TARGET_MAX,
                ),
                np.clip(
                    profile["model_b_final"] + 1.96 * profile["model_b_final_std"],
                    RI_TARGET_MIN,
                    RI_TARGET_MAX,
                ),
                alpha=0.15,
            )
            ax.set_title(pd.Timestamp(time).strftime("%H:%M"))
            ax.set_xlabel("Ri")
            ax.set_ylabel("Height (m)")
        axes.flat[0].legend()
        fig.suptitle(f"Station {station.upper()} vertical profiles", fontsize=14)
        fig.tight_layout()
        fig.savefig(output_dir / f"vertical_profiles_station_{station}.png")
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), dpi=160)
    for ax, column, title in zip(
        axes,
        ["baseline_oof", "model_b_spatial_oof", "model_b_spatiotemporal_oof"],
        ["Raw-wind baseline", "Spatial-shear Model B", "Spatiotemporal Model B"],
    ):
        ax.scatter(results["ri_target"], results[column], alpha=0.7, s=17)
        ax.plot([0, RI_TARGET_MAX], [0, RI_TARGET_MAX], "--", color="black", linewidth=1)
        ax.set(title=title, xlabel="Model-A Ri target", ylabel="Out-of-fold prediction")
    fig.tight_layout()
    fig.savefig(output_dir / "model_b_oof_parity.png")
    plt.close(fig)


# ============================================================
# 7. 主流程：读取数据、训练验证、保存结果
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--folds", type=int, default=6)
    args = parser.parse_args()

    # 第一步：直接从官方第一题原始文件建立统一的逐层观测表。
    profiler = load_wind_profiler(args.raw_data)
    radiometer = load_radiometers(args.raw_data)
    # 第二步：计算模型 A，并补充模型 B 所需的空间和时间特征。
    model_a = add_temporal_features(add_model_b_shear(compute_model_a(profiler, radiometer)))
    model_a = model_a.replace([np.inf, -np.inf], np.nan)

    baseline_features = ["height_m", "wind_dir_deg", "wind_speed_mps", "vertical_speed_mps"]
    spatial_features = ["height_m", "vertical_speed_mps", "model_b_shear_per_s"]
    spatiotemporal_features = spatial_features + [
        "horizontal_wind_tendency_mps2",
        "vertical_wind_tendency_mps2",
        "shear_tendency_per_s2",
    ]
    required = baseline_features + spatiotemporal_features + ["ri_target"]
    modeling = model_a.dropna(subset=required).reset_index(drop=True)

    # 第三步：在完全相同的分组划分下比较原始基线、空间版和时空版。
    baseline_oof, baseline_std, baseline_metrics = grouped_cross_validation(
        modeling, baseline_features, args.folds
    )
    spatial_oof, spatial_std, spatial_metrics = grouped_cross_validation(
        modeling, spatial_features, args.folds
    )
    spatiotemporal_oof, spatiotemporal_std, spatiotemporal_metrics = grouped_cross_validation(
        modeling, spatiotemporal_features, args.folds
    )
    spatial_model = make_model().fit(
        modeling[spatial_features].to_numpy(float), modeling["ri_target"].to_numpy(float)
    )
    spatiotemporal_model = make_model().fit(
        modeling[spatiotemporal_features].to_numpy(float), modeling["ri_target"].to_numpy(float)
    )
    spatial_final, spatial_final_std = predict_with_std(
        spatial_model, modeling[spatial_features].to_numpy(float)
    )
    spatiotemporal_final, spatiotemporal_final_std = predict_with_std(
        spatiotemporal_model, modeling[spatiotemporal_features].to_numpy(float)
    )
    spatial_final = np.clip(spatial_final, RI_TARGET_MIN, RI_TARGET_MAX)
    spatiotemporal_final = np.clip(spatiotemporal_final, RI_TARGET_MIN, RI_TARGET_MAX)

    # 只有当时空版的严格折外 R² 不低于空间版时，才将其设为最终模型。
    if spatiotemporal_metrics["r2"] >= spatial_metrics["r2"]:
        selected_name = "spatiotemporal"
        selected_model = spatiotemporal_model
        selected_features = spatiotemporal_features
        final_prediction, final_std = spatiotemporal_final, spatiotemporal_final_std
    else:
        selected_name = "spatial"
        selected_model = spatial_model
        selected_features = spatial_features
        final_prediction, final_std = spatial_final, spatial_final_std

    modeling["baseline_oof"] = baseline_oof
    modeling["baseline_oof_std"] = baseline_std
    modeling["model_b_spatial_oof"] = spatial_oof
    modeling["model_b_spatial_oof_std"] = spatial_std
    modeling["model_b_spatiotemporal_oof"] = spatiotemporal_oof
    modeling["model_b_spatiotemporal_oof_std"] = spatiotemporal_std
    modeling["model_b_spatial_final"] = spatial_final
    modeling["model_b_spatial_final_std"] = spatial_final_std
    modeling["model_b_spatiotemporal_final"] = spatiotemporal_final
    modeling["model_b_spatiotemporal_final_std"] = spatiotemporal_final_std
    modeling["model_b_final"] = final_prediction
    modeling["model_b_final_std"] = final_std

    # 第四步：保存逐层计算结果、指标、风险带、模型文件和图像。
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    model_a.to_csv(output_dir / "model_a_profiles.csv", index=False)
    modeling.to_csv(output_dir / "model_b_predictions.csv", index=False)
    risk_bands = contiguous_risk_bands(model_a)
    risk_bands.to_csv(output_dir / "high_risk_bands.csv", index=False)

    metrics = pd.DataFrame([
        {"model": "baseline_raw_wind", **baseline_metrics},
        {"model": "model_b_spatial_shear", **spatial_metrics},
        {"model": "model_b_spatiotemporal", **spatiotemporal_metrics},
    ])
    metrics.to_csv(output_dir / "validation_metrics.csv", index=False)
    joblib.dump(spatial_model, output_dir / "model_b_spatial.joblib")
    joblib.dump(spatiotemporal_model, output_dir / "model_b_spatiotemporal.joblib")
    joblib.dump(selected_model, output_dir / "model_b_selected.joblib")

    summary = {
        "random_seed": RANDOM_SEED,
        "profiler_files": int(profiler["profiler_source"].nunique()),
        "profiler_rows": int(len(profiler)),
        "radiometer_temperature_rows_after_deduplication": int(len(radiometer)),
        "station_time_profiles": int(modeling["profile_id"].nunique()),
        "modeling_rows": int(len(modeling)),
        "model_a": "gradient Richardson number from profiler wind and radiometer temperature",
        "spatial_features": spatial_features,
        "spatiotemporal_features": spatiotemporal_features,
        "validation": f"{args.folds}-fold GroupKFold by complete station-time profile",
        "baseline_metrics": baseline_metrics,
        "spatial_metrics": spatial_metrics,
        "spatiotemporal_metrics": spatiotemporal_metrics,
        "selected_model": selected_name,
        "selected_model_formula_original_units": raw_formula(selected_model, selected_features),
        "ri_training_target": {"minimum": RI_TARGET_MIN, "maximum": RI_TARGET_MAX},
        "maximum_nearest_radiometer_gap_minutes": float(modeling["nearest_radiometer_gap_min"].max()),
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_figures(modeling, output_dir, selected_name)

    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nParsed {summary['profiler_files']} profiler files and {summary['modeling_rows']} levels.")
    print(f"Results saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
