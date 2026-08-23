#!/usr/bin/env python3
"""2025 D 题第二问：02:00 多源低空湍流三维融合。

本程序融合四类信息：

1. S/X 波段天气雷达匹配表中的谱宽 SW；
2. 三部风廓线雷达 RAD 文件中的五波束谱宽；
3. 第一问方法得到的风矢量、平滑垂直切变及模型 B 域外诊断；
4. 02:00 地面自动站的风速和邻域风矢量差异。

不同观测量不会直接相加。每类传感器先转换为 0--1 的相对湍流指数，再按
时间、距离、覆盖度和不确定度融合。输出同时包含湍流指数、覆盖度和不确定度，
无观测支持的网格保持 NaN。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr
import tifffile


# ========================= 0. 全局配置 =========================

# 题目要求分析的目标时刻。所有雷达扫描都会根据与该时刻的时间差降权。
TARGET_TIME = pd.Timestamp("2025-07-31 02:00:00")

# 全部路径以 main.py 所在目录为基准，从 GitHub 下载后不需改绝对路径。
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_Q1_PROJECT = DATA_DIR

# 各天气雷达的基础可信度：S 波段雷达覆盖更稳定，设为 1.00。
RADAR_BASE_RELIABILITY = {"S9250": 1.00, "X205": 0.85, "ZM206": 0.85, "ZM207": 0.85}

# 距离衰减尺度（km）：距雷达越远，观测权重越小。
RADAR_RANGE_SCALE_KM = {"S9250": 90.0, "X205": 55.0, "ZM206": 55.0, "ZM207": 55.0}

# 站点垂直剖面融合时使用的雷达最低不确定度。
RADAR_NOISE_FLOOR = {"S9250": 0.12, "X205": 0.15, "ZM206": 0.15, "ZM207": 0.15}


@dataclass(frozen=True)
class GridSpec:
    """三维计算网格的统一描述。

    xs_m/ys_m 是相对局地原点的水平米制坐标，zs_m 是离地高度。
    将经纬度暂时换成米制坐标，是为了方便计算距离和做 IDW 插值。
    """

    # 局地米制坐标的经纬度原点。
    lat0: float
    lon0: float
    # 东向、北向和垂直方向的网格坐标轴，单位为 m。
    xs_m: np.ndarray
    ys_m: np.ndarray
    zs_m: np.ndarray
    # 水平和垂直网格分辨率，单位为 m。
    dx_m: float
    dz_m: float

    @property
    def nx(self) -> int:
        return int(self.xs_m.size)

    @property
    def ny(self) -> int:
        return int(self.ys_m.size)

    @property
    def nz(self) -> int:
        return int(self.zs_m.size)


# ========================= 1. 通用数学工具 =========================


def robust_bounds(values: Iterable[float], low_q: float = 0.10, high_q: float = 0.90) -> tuple[float, float]:
    """返回稳健标定上下界；退化分布自动扩展，避免除零。"""
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        raise ValueError("Cannot calibrate an empty variable")
    # 使用分位数而不是最小/最大值，避免少量离群点压缩大部分数据。
    low, high = np.quantile(array, [low_q, high_q])
    if high - low < 1e-6:
        # 所有值几乎相同时人为扩展区间，否则归一化会除以零。
        pad = max(abs(float(low)) * 0.1, 0.5)
        low, high = float(low) - pad, float(high) + pad
    return float(low), float(high)


def robust_unit(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """按给定稳健上下界线性映射到 [0, 1]。"""
    values = np.asarray(values, dtype=float)
    # 低于下界记为 0，高于上界记为 1，中间线性映射。
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def wind_components(direction_deg: np.ndarray, speed_mps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """气象风向转换为东向 u、北向 v。"""
    # 气象风向表示“风从哪里来”，因此两个分量前带负号。
    angle = np.deg2rad(np.asarray(direction_deg, dtype=float))
    speed = np.asarray(speed_mps, dtype=float)
    return -speed * np.sin(angle), -speed * np.cos(angle)


def lonlat_to_xy(
    latitude: np.ndarray, longitude: np.ndarray, lat0: float, lon0: float
) -> tuple[np.ndarray, np.ndarray]:
    """在研究区小范围内，将经纬度近似换算为局地东北向米制坐标。"""
    # 1 纬度约为 111.32 km；经度方向还需乘以 cos(平均纬度)。
    metres_per_degree_lat = 111_320.0
    metres_per_degree_lon = metres_per_degree_lat * math.cos(math.radians(lat0))
    x = (np.asarray(longitude, dtype=float) - lon0) * metres_per_degree_lon
    y = (np.asarray(latitude, dtype=float) - lat0) * metres_per_degree_lat
    return x, y


def xy_to_lonlat(x: np.ndarray, y: np.ndarray, lat0: float, lon0: float) -> tuple[np.ndarray, np.ndarray]:
    """将局地米制坐标转回经纬度，主要用于画图和 GIS 导出。"""
    metres_per_degree_lat = 111_320.0
    metres_per_degree_lon = metres_per_degree_lat * math.cos(math.radians(lat0))
    longitude = lon0 + np.asarray(x, dtype=float) / metres_per_degree_lon
    latitude = lat0 + np.asarray(y, dtype=float) / metres_per_degree_lat
    return longitude, latitude


# ========================= 2. 风廓线雷达 =========================


def parse_robs_file(path: Path, station_name: str) -> pd.DataFrame:
    """解析第一问与第二问共用的 WNDROBS 产品。"""
    # ROBS 是纯文本定长产品，先整体读入，再定位 ROBS...NNNN 数据段。
    lines = path.read_text(encoding="ascii", errors="replace").splitlines()
    if len(lines) < 4 or not lines[0].startswith("WNDROBS"):
        raise ValueError(f"Unsupported ROBS file: {path}")
    header = lines[1].split()
    station_id, longitude, latitude, site_altitude, timezone, timestamp = header[:6]
    try:
        start = lines.index("ROBS") + 1
    except ValueError as exc:
        raise ValueError(f"ROBS section not found: {path}") from exc

    rows: list[dict[str, object]] = []
    for line in lines[start:]:
        if line.strip() == "NNNN":
            break
        fields = line.split()
        # 官方文件用斜杠表示缺测；列数不对或含缺测符号的行直接跳过。
        if len(fields) != 7 or any("/" in value for value in fields):
            continue
        try:
            height, direction, speed, vertical_speed = map(float, fields[:4])
            quality_1, quality_2 = map(int, fields[4:6])
            cn2 = float(fields[6])
        except ValueError:
            continue
        rows.append(
            {
                "station_name": station_name,
                "station_id": station_id,
                "longitude_deg": float(longitude),
                "latitude_deg": float(latitude),
                "site_altitude_m": float(site_altitude),
                "timezone": timezone,
                "time": pd.to_datetime(timestamp, format="%Y%m%d%H%M%S"),
                "height_m": height,
                "wind_dir_deg": direction,
                "wind_speed_mps": speed,
                "vertical_speed_mps": vertical_speed,
                "quality_flag_1": quality_1,
                "quality_flag_2": quality_2,
                "cn2_m_neg_2_3": cn2,
            }
        )
    if not rows:
        raise ValueError(f"No valid ROBS rows in {path}")
    # 后续要沿高度求导，因此必须先按高度升序排列。
    profile = pd.DataFrame(rows).sort_values("height_m").reset_index(drop=True)
    u_ms, v_ms = wind_components(profile["wind_dir_deg"], profile["wind_speed_mps"])
    profile["u_ms"] = u_ms
    profile["v_ms"] = v_ms
    heights = profile["height_m"].to_numpy(float)
    # 五层滑动平均抑制单层噪声，再使用非等距高度求垂直梯度。
    u_smooth = profile["u_ms"].rolling(5, center=True, min_periods=1).mean().to_numpy(float)
    v_smooth = profile["v_ms"].rolling(5, center=True, min_periods=1).mean().to_numpy(float)
    profile["model_b_shear_per_s"] = np.hypot(
        np.gradient(u_smooth, heights), np.gradient(v_smooth, heights)
    )
    return profile


def parse_wpr_rad_low_mode(path: Path, station_name: str) -> pd.DataFrame:
    """解析 WNDRAD 第一种低空模式的五个波束。

    每层三项观测依次为径向速度、信噪比和谱宽。文件后续还有中高空模式；
    第二问只要求 0--2 km，因此只使用首个模式的五波束，避免重复高度混合。
    """
    lines = path.read_text(encoding="ascii", errors="replace").splitlines()
    if len(lines) < 6 or not lines[0].startswith("WNDRAD"):
        raise ValueError(f"Unsupported RAD file: {path}")
    header = lines[1].split()
    station_id, longitude, latitude, site_altitude, timezone = header[:5]
    # 前 5 个 RAD 块对应低空模式的 5 个波束；后续块属于其他量程。
    block_count = 0
    beam = ""
    rows: list[dict[str, object]] = []
    for line in lines[2:]:
        stripped = line.strip()
        if stripped.startswith("RAD "):
            block_count += 1
            if block_count > 5:
                break
            beam = stripped.split(maxsplit=1)[1]
            continue
        if not beam or stripped == "NNNN":
            continue
        fields = stripped.split()
        if len(fields) != 4 or any("/" in field for field in fields):
            continue
        try:
            height, radial_velocity, snr, spectrum_width = map(float, fields)
        except ValueError:
            continue
        # 设置宽松的物理范围，剔除解析错位或显然异常值。
        if not (0.0 <= height <= 4000.0):
            continue
        if not (abs(radial_velocity) < 100 and 0.0 <= snr < 100 and 0.0 <= spectrum_width < 20):
            continue
        rows.append(
            {
                "station_name": station_name,
                "station_id": station_id,
                "longitude_deg": float(longitude),
                "latitude_deg": float(latitude),
                "site_altitude_m": float(site_altitude),
                "timezone": timezone,
                "beam": beam,
                "height_m": height,
                "radial_velocity_mps": radial_velocity,
                "snr_db": snr,
                "spectrum_width_mps": spectrum_width,
            }
        )
    raw = pd.DataFrame(rows)
    if raw.empty:
        raise ValueError(f"No valid low-mode RAD rows in {path}")

    def median_absolute_deviation(series: pd.Series) -> float:
        """计算五波束谱宽的 MAD，作为波束间不一致性指标。"""
        values = series.to_numpy(float)
        median = np.median(values)
        return float(np.median(np.abs(values - median)))

    # 同一高度的 5 个波束聚合成一行：中位数抗离群，MAD 表示不确定性。
    return (
        raw.groupby(
            [
                "station_name",
                "station_id",
                "longitude_deg",
                "latitude_deg",
                "site_altitude_m",
                "timezone",
                "height_m",
            ],
            as_index=False,
        )
        .agg(
            wpr_spectrum_width_mps=("spectrum_width_mps", "median"),
            wpr_spectrum_width_mad=("spectrum_width_mps", median_absolute_deviation),
            valid_beams=("beam", "nunique"),
            median_snr_db=("snr_db", "median"),
        )
        .sort_values("height_m")
        .reset_index(drop=True)
    )


def apply_q1_spatial_diagnostic(profile: pd.DataFrame, q1_project: Path | None) -> pd.DataFrame:
    """应用第一问空间模型，仅作为诊断，不进入第二问融合权重。"""
    # 先建立默认列；即使第一问模型缺失，第二问也能继续运行。
    result = profile.copy()
    result["q1_ri_prediction"] = np.nan
    result["q1_ri_prediction_std"] = np.nan
    result["q1_out_of_domain"] = True
    if q1_project is None:
        return result
    model_path = q1_project / "model_b_spatial.joblib"
    training_path = q1_project / "model_b_predictions.csv"
    if not model_path.exists() or not training_path.exists():
        # 也兼容第一问原工程的 results/ 目录结构。
        model_path = q1_project / "results/model_b_spatial.joblib"
        training_path = q1_project / "results/model_b_predictions.csv"
    if not model_path.exists() or not training_path.exists():
        return result
    try:
        import joblib

        # 这三个特征与第一问空间版模型保持一致。
        features = ["height_m", "vertical_speed_mps", "model_b_shear_per_s"]
        training = pd.read_csv(training_path, usecols=features)
        model = joblib.load(model_path)
        x = result[features].to_numpy(float)
        transformed = model.named_steps["scale"].transform(x)
        prediction, uncertainty = model.named_steps["regression"].predict(
            transformed, return_std=True
        )
        # 逐特征检查第二问样本是否超出第一问训练范围。
        # 域外预测只能帮助诊断，不能当作新区域的可靠真值。
        outside = np.zeros(len(result), dtype=bool)
        for feature in features:
            outside |= result[feature].to_numpy(float) < float(training[feature].min())
            outside |= result[feature].to_numpy(float) > float(training[feature].max())
        result["q1_ri_prediction"] = np.clip(prediction, 0.0, 20.0)
        result["q1_ri_prediction_std"] = uncertainty
        result["q1_out_of_domain"] = outside
    except Exception:
        # 第一问模型文件或依赖不可用时，第二问仍可依靠原始物理量运行。
        pass
    return result


def discover_wpr_files(data_dir: Path) -> list[tuple[str, Path, Path]]:
    """发现 data/ 中的三部风廓线雷达文件。"""
    files: list[tuple[str, Path, Path]] = []
    for station_name in ("a", "b", "c"):
        robs = data_dir / f"wpr_{station_name}_robs.txt"
        rad = data_dir / f"wpr_{station_name}_rad.txt"
        if not robs.exists() or not rad.exists():
            raise FileNotFoundError(
                f"Missing WPR files for station {station_name.upper()} under {data_dir}"
            )
        files.append((station_name, robs, rad))
    return files


def prepare_wpr_profiles(
    data_dir: Path, zs_m: np.ndarray, q1_project: Path | None
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """构建三站 WPR 相对湍流剖面及目标高度矩阵。"""
    tables: list[pd.DataFrame] = []
    for station_name, robs_path, rad_path in discover_wpr_files(data_dir):
        # ROBS 提供风向/风速/垂直速度，RAD 提供五波束谱宽。
        robs = parse_robs_file(robs_path, station_name)
        robs = apply_q1_spatial_diagnostic(robs, q1_project)
        rad = parse_wpr_rad_low_mode(rad_path, station_name)
        merged = pd.merge(
            robs,
            rad[
                [
                    "height_m",
                    "wpr_spectrum_width_mps",
                    "wpr_spectrum_width_mad",
                    "valid_beams",
                    "median_snr_db",
                ]
            ],
            on="height_m",
            how="inner",
        )
        # 动力代理量同时考虑风切变和垂直运动；0.01 s^-1 与 3 m/s 是尺度参数。
        merged["dynamic_proxy_raw"] = np.hypot(
            merged["model_b_shear_per_s"] / 0.01,
            np.abs(merged["vertical_speed_mps"]) / 3.0,
        )
        tables.append(merged)
    profiles = pd.concat(tables, ignore_index=True)
    # 归一化界限只用题目需要的 0--zmax 高度，避免更高层改变低空量纲。
    calibration_subset = profiles[profiles["height_m"].between(0.0, float(zs_m.max()))]
    sw_low, sw_high = robust_bounds(calibration_subset["wpr_spectrum_width_mps"], 0.10, 0.90)
    dyn_low, dyn_high = robust_bounds(calibration_subset["dynamic_proxy_raw"], 0.10, 0.90)
    profiles["wpr_spectrum_ti"] = robust_unit(
        profiles["wpr_spectrum_width_mps"].to_numpy(float), sw_low, sw_high
    )
    profiles["wpr_dynamic_ti"] = robust_unit(
        profiles["dynamic_proxy_raw"].to_numpy(float), dyn_low, dyn_high
    )
    # 原始 RAD 谱宽为主，第一问启发的风切变/垂直运动为辅。
    profiles["wpr_turbulence_index"] = (
        0.75 * profiles["wpr_spectrum_ti"] + 0.25 * profiles["wpr_dynamic_ti"]
    )
    # 不确定度由波束间差异、缺失波束和第一问域外惩罚共同构成。
    scale = max(sw_high - sw_low, 0.5)
    ood_penalty = profiles["q1_out_of_domain"].astype(float)
    profiles["wpr_uncertainty"] = np.clip(
        0.10
        + 0.30 * profiles["wpr_spectrum_width_mad"] / scale
        + 0.15 * (5 - profiles["valid_beams"]) / 5
        + 0.15 * ood_penalty,
        0.08,
        0.70,
    )

    # 下面将每站的不规则高度剖面线性插值到统一高度轴 zs_m。
    station_rows = (
        profiles[
            ["station_name", "latitude_deg", "longitude_deg", "site_altitude_m"]
        ]
        .drop_duplicates("station_name")
        .sort_values("station_name")
        .reset_index(drop=True)
    )
    n_station, nz = len(station_rows), len(zs_m)
    ti = np.full((n_station, nz), np.nan, dtype=float)
    uncertainty = np.full_like(ti, np.nan)
    confidence = np.zeros_like(ti)
    for i, station_name in enumerate(station_rows["station_name"]):
        profile = profiles[profiles["station_name"] == station_name].sort_values("height_m")
        heights = profile["height_m"].to_numpy(float)
        inside = (zs_m >= heights.min()) & (zs_m <= heights.max())
        ti[i, inside] = np.interp(
            zs_m[inside], heights, profile["wpr_turbulence_index"].to_numpy(float)
        )
        uncertainty[i, inside] = np.interp(
            zs_m[inside], heights, profile["wpr_uncertainty"].to_numpy(float)
        )
        beam_support = np.interp(
            zs_m[inside], heights, profile["valid_beams"].to_numpy(float) / 5.0
        )
        # 目标高度离真实观测层越远，垂直置信度越低。
        nearest_gap = np.min(np.abs(zs_m[inside, None] - heights[None, :]), axis=1)
        confidence[i, inside] = beam_support * np.exp(-0.5 * (nearest_gap / 90.0) ** 2)
    calibration = pd.DataFrame(
        [
            {"source": "WPR_spectrum_width", "low": sw_low, "high": sw_high, "unit": "m/s"},
            {"source": "WPR_dynamic_proxy", "low": dyn_low, "high": dyn_high, "unit": "1"},
        ]
    )
    return profiles, calibration, station_rows, ti, uncertainty, confidence


# ========================= 3. 天气雷达 =========================


def add_radar_velocity_gradient_correction(table: pd.DataFrame) -> pd.DataFrame:
    """用局地径向速度梯度估计非湍流谱展宽并订正 SW。

    在同一雷达、扫描时刻和仰角内，对每个站点的邻近径向速度做局地平面拟合。
    将波束直径内由速度梯度产生的谱展宽与仪器噪声底从观测谱宽平方中扣除：

        sigma_t^2 = max(sigma_obs^2 - sigma_shear^2 - sigma_noise^2, 0).

    这仍是基于已有匹配点的近似订正，因此最终量称为相对湍流指数而不是 EDR。
    """
    result = table.copy()
    result["radial_velocity_gradient_per_s"] = np.nan
    if "VEL_mps" not in result.columns:
        result["VEL_mps"] = np.nan
    result["VEL_mps"] = pd.to_numeric(result["VEL_mps"], errors="coerce")

    # 只在“同雷达＋同时刻＋同仰角”内拟合，避免混合不同扫描几何。
    group_columns = ["radar", "time", "elev_code"]
    for _, group in result.groupby(group_columns, sort=False):
        valid = group["VEL_mps"].notna() & group["station_lat"].notna() & group["station_lon"].notna()
        selected = group.loc[valid]
        if len(selected) < 4:
            continue
        lat0 = float(selected["station_lat"].mean())
        lon0 = float(selected["station_lon"].mean())
        x, y = lonlat_to_xy(selected["station_lat"], selected["station_lon"], lat0, lon0)
        points = np.column_stack([x, y])
        velocity = selected["VEL_mps"].to_numpy(float)
        # KDTree 用于快速找到每个匹配点附近最多 8 个邻点。
        tree = cKDTree(points)
        k_eff = min(9, len(selected))
        distances, neighbours = tree.query(points, k=k_eff)
        if k_eff == 1:
            distances, neighbours = distances[:, None], neighbours[:, None]
        gradients = np.full(len(selected), np.nan)
        for i in range(len(selected)):
            neighbour = neighbours[i, 1:]
            distance = distances[i, 1:]
            dx = points[neighbour, 0] - points[i, 0]
            dy = points[neighbour, 1] - points[i, 1]
            dv = velocity[neighbour] - velocity[i]
            # 只使用 30 km 内且径向速度差不超过 20 m/s 的邻点。
            usable = np.isfinite(dv) & (distance <= 30_000.0) & (np.abs(dv) <= 20.0)
            if usable.sum() < 3:
                continue
            design = np.column_stack([dx[usable], dy[usable]])
            spatial_weight = np.exp(-0.5 * (distance[usable] / 12_000.0) ** 2)
            weighted_design = design * np.sqrt(spatial_weight)[:, None]
            weighted_delta = dv[usable] * np.sqrt(spatial_weight)
            # 加权最小二乘拟合 dVr/dx 和 dVr/dy，两者的模即局地速度梯度。
            coefficient, *_ = np.linalg.lstsq(weighted_design, weighted_delta, rcond=None)
            gradients[i] = min(float(np.hypot(coefficient[0], coefficient[1])), 0.01)
        result.loc[selected.index, "radial_velocity_gradient_per_s"] = gradients

    result["radial_velocity_gradient_per_s"] = result[
        "radial_velocity_gradient_per_s"
    ].fillna(0.0)
    # 雷达波束随距离变宽：波束直径 ≈ 距离 × 1°（弧度）。
    beamwidth_rad = math.radians(1.0)
    beam_diameter_m = result["range_km"].to_numpy(float) * 1000.0 * beamwidth_rad
    result["shear_broadening_mps"] = (
        result["radial_velocity_gradient_per_s"].to_numpy(float)
        * beam_diameter_m
        / math.sqrt(12.0)
    )
    result["instrument_noise_floor_mps"] = 0.0
    # 每部雷达的噪声底独立估计，不假设 S/X 波段具有相同噪声。
    for radar, group in result.groupby("radar"):
        positive = group.loc[group["SW_mps"] > 0, "SW_mps"]
        median_positive = float(positive.median()) if not positive.empty else 0.5
        noise_floor = float(np.clip(0.25 * median_positive, 0.10, 0.50))
        result.loc[group.index, "instrument_noise_floor_mps"] = noise_floor
    # 谱宽在方差层面做减法，负值截为 0 后再开方。
    variance = (
        result["SW_mps"].to_numpy(float) ** 2
        - result["shear_broadening_mps"].to_numpy(float) ** 2
        - result["instrument_noise_floor_mps"].to_numpy(float) ** 2
    )
    result["SW_corrected_mps"] = np.sqrt(np.maximum(variance, 0.0))
    return result


def read_radar_matches(path: Path, sheet: str = "merged") -> pd.DataFrame:
    """读取雷达—站点匹配表，完成字段检查、类型转换和基本质控。"""
    # 这些字段是后续时间加权、距离加权和垂直插值不可缺少的。
    required = {
        "radar",
        "time",
        "station_id",
        "station_lat",
        "station_lon",
        "station_alt_m",
        "range_km",
        "SW_mps",
        "beam_h_agl_station_m",
    }
    table = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Radar match workbook is missing columns: {sorted(missing)}")
    table = table.copy()
    table["observed_at"] = pd.to_datetime(
        table["time"].astype("Int64").astype(str), format="%Y%m%d%H%M%S", errors="coerce"
    )
    for column in ["SW_mps", "range_km", "beam_h_agl_station_m", "station_lat", "station_lon"]:
        table[column] = pd.to_numeric(table[column], errors="coerce")
    # 仅保留时间有效、距离有效、0--2500 m 且 SW 在物理合理范围的记录。
    table = table[
        table["observed_at"].notna()
        & table["SW_mps"].notna()
        & table["range_km"].notna()
        & table["beam_h_agl_station_m"].between(0.0, 2500.0)
        & table["SW_mps"].between(0.0, 15.0)
    ].copy()
    if table.empty:
        raise ValueError("No valid radar spectrum-width rows")
    table["station_id"] = table["station_id"].astype(str)
    return add_radar_velocity_gradient_correction(table)


def prepare_radar_profiles(
    matches: pd.DataFrame, zs_m: np.ndarray
) -> tuple[
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    list[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """把各雷达独立标定后，在站点处形成垂直剖面并跨雷达融合。"""
    # 一个 station_id 可能同时被多部雷达覆盖，先建立全部唯一站点的索引。
    stations = (
        matches[["station_id", "station_lat", "station_lon", "station_alt_m"]]
        .drop_duplicates("station_id")
        .sort_values("station_id")
        .reset_index(drop=True)
    )
    station_index = {sid: i for i, sid in enumerate(stations["station_id"])}
    radars = sorted(matches["radar"].dropna().astype(str).unique())
    n_radar, n_station, nz = len(radars), len(stations), len(zs_m)
    # 三维数组维度顺序：[雷达, 站点, 目标高度]。
    source_ti = np.full((n_radar, n_station, nz), np.nan, dtype=float)
    source_unc = np.full_like(source_ti, np.nan)
    source_cov = np.zeros_like(source_ti)
    diagnostics: list[dict[str, object]] = []

    calibrated = matches.copy()
    # 每部雷达分别用自身 Q10--Q90 标定，保留相对强弱而不直接混合不同量程。
    for radar in radars:
        selected = calibrated["radar"].astype(str) == radar
        low, high = robust_bounds(calibrated.loc[selected, "SW_corrected_mps"], 0.10, 0.90)
        calibrated.loc[selected, "radar_ti"] = robust_unit(
            calibrated.loc[selected, "SW_corrected_mps"].to_numpy(float), low, high
        )
        radar_rows = calibrated.loc[selected]
        diagnostics.append(
            {
                "source": radar,
                "kind": "weather_radar",
                "valid_rows": int(len(radar_rows)),
                "stations": int(radar_rows["station_id"].nunique()),
                "scan_times": int(radar_rows["observed_at"].nunique()),
                "calibration_low": low,
                "calibration_high": high,
                "calibration_unit": "m/s",
                "raw_sw_median_mps": float(radar_rows["SW_mps"].median()),
                "corrected_sw_median_mps": float(radar_rows["SW_corrected_mps"].median()),
                "median_shear_broadening_mps": float(radar_rows["shear_broadening_mps"].median()),
                "instrument_noise_floor_mps": float(
                    radar_rows["instrument_noise_floor_mps"].median()
                ),
                "corrected_zero_fraction": float((radar_rows["SW_corrected_mps"] == 0).mean()),
            }
        )

    # 将各雷达的散点波束高度映射到统一目标高度轴。
    for radar_i, radar in enumerate(radars):
        radar_rows = calibrated[calibrated["radar"].astype(str) == radar]
        time_delta_s = (radar_rows["observed_at"] - TARGET_TIME).dt.total_seconds().abs().to_numpy()
        # 离 02:00 越近权重越高；360 s 是时间高斯衰减尺度。
        time_weight = np.exp(-0.5 * (time_delta_s / 360.0) ** 2)
        range_scale = RADAR_RANGE_SCALE_KM.get(radar, 55.0)
        range_weight = np.exp(-0.5 * (radar_rows["range_km"].to_numpy(float) / range_scale) ** 2)
        radar_rows = radar_rows.assign(base_weight=time_weight * range_weight)
        noise = RADAR_NOISE_FLOOR.get(radar, 0.15)
        for station_id, group in radar_rows.groupby("station_id", sort=False):
            station_i = station_index.get(str(station_id))
            if station_i is None:
                continue
            height = group["beam_h_agl_station_m"].to_numpy(float)
            value = group["radar_ti"].to_numpy(float)
            base_weight = group["base_weight"].to_numpy(float)
            # 目标高度与实际波束高度的差使用 180 m 高斯权重，超过 540 m 不贡献。
            vertical_weight = np.exp(-0.5 * ((zs_m[:, None] - height[None, :]) / 180.0) ** 2)
            vertical_weight[np.abs(zs_m[:, None] - height[None, :]) > 540.0] = 0.0
            weight = vertical_weight * base_weight[None, :]
            weight_sum = weight.sum(axis=1)
            supported = weight_sum > 0.05
            if not supported.any():
                continue
            mean = np.divide(
                (weight * value[None, :]).sum(axis=1),
                weight_sum,
                out=np.full(nz, np.nan),
                where=weight_sum > 0,
            )
            # 同一目标高附近观测的加权方差表示局地不一致性。
            variance = np.divide(
                (weight * (value[None, :] - mean[:, None]) ** 2).sum(axis=1),
                weight_sum,
                out=np.full(nz, np.nan),
                where=weight_sum > 0,
            )
            source_ti[radar_i, station_i, supported] = mean[supported]
            source_unc[radar_i, station_i, supported] = np.sqrt(variance[supported] + noise**2)
            source_cov[radar_i, station_i, supported] = 1.0 - np.exp(-weight_sum[supported] / 2.0)

    # 在站点剖面层面跨雷达融合：高覆盖、低不确定度的雷达权重更大。
    combine_weight = np.zeros_like(source_ti)
    for radar_i, radar in enumerate(radars):
        finite = np.isfinite(source_ti[radar_i]) & np.isfinite(source_unc[radar_i])
        combine_weight[radar_i, finite] = (
            RADAR_BASE_RELIABILITY.get(radar, 0.8)
            * source_cov[radar_i, finite]
            / (source_unc[radar_i, finite] ** 2 + 0.08**2)
        )
    weight_sum = combine_weight.sum(axis=0)
    ti = np.divide(
        np.nansum(combine_weight * source_ti, axis=0),
        weight_sum,
        out=np.full((n_station, nz), np.nan),
        where=weight_sum > 0,
    )
    # 多部雷达彼此差异越大，disagreement 越大，最终不确定度也越大。
    disagreement = np.divide(
        np.nansum(combine_weight * (source_ti - ti[None, :, :]) ** 2, axis=0),
        weight_sum,
        out=np.full((n_station, nz), np.nan),
        where=weight_sum > 0,
    )
    uncertainty = np.sqrt(
        np.divide(1.0, weight_sum, out=np.full_like(weight_sum, np.nan), where=weight_sum > 0)
        + disagreement
    )
    coverage = 1.0 - np.prod(1.0 - np.clip(source_cov, 0.0, 1.0), axis=0)
    source_count = np.sum(np.isfinite(source_ti), axis=0).astype(np.int16)
    return (
        stations,
        ti,
        uncertainty,
        coverage,
        source_count,
        pd.DataFrame(diagnostics),
        radars,
        source_ti,
        source_unc,
        source_cov,
    )


# ========================= 4. 地面自动气象站 =========================


def read_surface_stations(path: Path) -> pd.DataFrame:
    """读取地面站风场，用风速和邻域风矢量差构造近地层湍流代理指标。"""
    table = pd.read_excel(path, sheet_name=0, engine="openpyxl").copy()
    required = {"Lat", "Lon", "Alti", "WIN_D_Avg_2mi", "WIN_S_Avg_2mi"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Surface workbook is missing columns: {sorted(missing)}")
    for column in required:
        table[column] = pd.to_numeric(table[column], errors="coerce")
    # 同时检查经纬度和风场物理范围，缺测值在 to_numeric 后会成为 NaN。
    valid = (
        table["Lat"].between(20.0, 45.0)
        & table["Lon"].between(100.0, 130.0)
        & table["WIN_D_Avg_2mi"].between(0.0, 360.0)
        & table["WIN_S_Avg_2mi"].between(0.0, 40.0)
    )
    table = table.loc[valid].reset_index(drop=True)
    table["surface_id"] = [f"SURF_{i + 1:04d}" for i in range(len(table))]
    u_ms, v_ms = wind_components(table["WIN_D_Avg_2mi"], table["WIN_S_Avg_2mi"])
    table["u_ms"] = u_ms
    table["v_ms"] = v_ms

    lat0, lon0 = float(table["Lat"].mean()), float(table["Lon"].mean())
    x, y = lonlat_to_xy(table["Lat"], table["Lon"], lat0, lon0)
    points = np.column_stack([x, y])
    # 对每个站找自身外最近的 6 站，估计局地风矢量空间变化。
    tree = cKDTree(points)
    distances, indices = tree.query(points, k=min(7, len(points)))
    if distances.ndim == 1:
        distances, indices = distances[:, None], indices[:, None]
    neighbour_distance = distances[:, 1:]
    neighbour_index = indices[:, 1:]
    # 加入 5 km 平滑距离，防止两站过近时出现过大权重。
    weight = 1.0 / (neighbour_distance**2 + 5000.0**2)
    weight[neighbour_distance > 30_000.0] = 0.0
    du = table["u_ms"].to_numpy(float)[neighbour_index] - table["u_ms"].to_numpy(float)[:, None]
    dv = table["v_ms"].to_numpy(float)[neighbour_index] - table["v_ms"].to_numpy(float)[:, None]
    weight_sum = weight.sum(axis=1)
    variability = np.sqrt(
        np.divide(
            (weight * (du**2 + dv**2)).sum(axis=1),
            weight_sum,
            out=np.zeros(len(table)),
            where=weight_sum > 0,
        )
    )
    table["local_wind_vector_variability_mps"] = variability
    # 本站风速越大、周围风矢量差异越大，代理指标越高。
    table["surface_proxy_raw"] = np.hypot(
        table["WIN_S_Avg_2mi"].to_numpy(float) / 5.0, variability / 3.0
    )
    low, high = robust_bounds(table["surface_proxy_raw"], 0.10, 0.90)
    table["surface_turbulence_index"] = robust_unit(
        table["surface_proxy_raw"].to_numpy(float), low, high
    )
    # 有效邻站越少，地面代理指标的不确定度越大。
    neighbour_count = (weight > 0).sum(axis=1)
    table["surface_uncertainty"] = np.clip(
        0.12 + 0.20 / np.sqrt(np.maximum(neighbour_count, 1)), 0.12, 0.35
    )
    table.attrs["calibration_low"] = low
    table.attrs["calibration_high"] = high
    return table


def coordinate_key(latitude: float, longitude: float) -> tuple[float, float]:
    """经纬度保留 4 位小数生成坐标键，用于识别雷达匹配点与地面站重合。"""
    return round(float(latitude), 4), round(float(longitude), 4)


def build_master_points(surface: pd.DataFrame, radar_stations: pd.DataFrame) -> pd.DataFrame:
    """合并地面站与雷达剖面位置，避免同一经纬度在空间插值中重复。"""
    rows: list[dict[str, object]] = []
    seen: set[tuple[float, float]] = set()
    for _, row in surface.iterrows():
        key = coordinate_key(row["Lat"], row["Lon"])
        seen.add(key)
        rows.append(
            {
                "point_key": f"{key[0]:.4f}_{key[1]:.4f}",
                "latitude_deg": float(row["Lat"]),
                "longitude_deg": float(row["Lon"]),
                "surface_row": int(row.name),
                "radar_row": -1,
            }
        )
    # lookup 记录“坐标键 -> 统一点行号”。
    lookup = {coordinate_key(row["latitude_deg"], row["longitude_deg"]): i for i, row in enumerate(rows)}
    for radar_i, row in radar_stations.iterrows():
        key = coordinate_key(row["station_lat"], row["station_lon"])
        if key in lookup:
            rows[lookup[key]]["radar_row"] = int(radar_i)
        else:
            lookup[key] = len(rows)
            rows.append(
                {
                    "point_key": f"{key[0]:.4f}_{key[1]:.4f}",
                    "latitude_deg": float(row["station_lat"]),
                    "longitude_deg": float(row["station_lon"]),
                    "surface_row": -1,
                    "radar_row": int(radar_i),
                }
            )
    return pd.DataFrame(rows)


# ========================= 5. 三维网格与空间插值 =========================


def make_grid(
    master: pd.DataFrame,
    wpr_stations: pd.DataFrame,
    dx_m: float,
    dz_m: float,
    zmax_m: float,
) -> GridSpec:
    """以全部观测点的包络范围创建规则三维网格。"""
    latitude = np.concatenate(
        [master["latitude_deg"].to_numpy(float), wpr_stations["latitude_deg"].to_numpy(float)]
    )
    longitude = np.concatenate(
        [master["longitude_deg"].to_numpy(float), wpr_stations["longitude_deg"].to_numpy(float)]
    )
    # 使用所有观测点的平均经纬度作为局地投影原点。
    lat0, lon0 = float(latitude.mean()), float(longitude.mean())
    x, y = lonlat_to_xy(latitude, longitude, lat0, lon0)
    # 向外取整到 dx 的整数倍，确保所有观测点均在网格内。
    xmin = math.floor(float(x.min()) / dx_m) * dx_m
    xmax = math.ceil(float(x.max()) / dx_m) * dx_m
    ymin = math.floor(float(y.min()) / dx_m) * dx_m
    ymax = math.ceil(float(y.max()) / dx_m) * dx_m
    return GridSpec(
        lat0=lat0,
        lon0=lon0,
        xs_m=np.arange(xmin, xmax + 0.5 * dx_m, dx_m),
        ys_m=np.arange(ymin, ymax + 0.5 * dx_m, dx_m),
        zs_m=np.arange(0.0, zmax_m + 0.5 * dz_m, dz_m),
        dx_m=float(dx_m),
        dz_m=float(dz_m),
    )


def query_grid_neighbours(
    point_xy: np.ndarray, grid: GridSpec, k: int, chunk_size: int = 200_000
) -> tuple[np.ndarray, np.ndarray]:
    """分块计算网格到观测点的近邻，避免一次性产生超大临时数组。"""
    # 先对观测点建立 KDTree，然后一次查询一批网格，节省内存。
    tree = cKDTree(point_xy)
    n_cell = grid.nx * grid.ny
    k_eff = min(k, len(point_xy))
    distances = np.empty((n_cell, k_eff), dtype=np.float32)
    indices = np.empty((n_cell, k_eff), dtype=np.int32)
    for start in range(0, n_cell, chunk_size):
        end = min(n_cell, start + chunk_size)
        flat = np.arange(start, end)
        # 将一维网格序号还原为行列号，再组成 (x, y) 查询点。
        iy, ix = np.divmod(flat, grid.nx)
        query_points = np.column_stack([grid.xs_m[ix], grid.ys_m[iy]])
        distance, index = tree.query(query_points, k=k_eff, workers=-1)
        if k_eff == 1:
            distance, index = distance[:, None], index[:, None]
        distances[start:end] = distance.astype(np.float32)
        indices[start:end] = index.astype(np.int32)
    return distances, indices


def interpolate_from_neighbours(
    values: np.ndarray,
    point_uncertainty: np.ndarray,
    point_confidence: np.ndarray,
    distances: np.ndarray,
    indices: np.ndarray,
    shape: tuple[int, int],
    max_radius_m: float,
    d0_m: float,
    uncertainty_floor: float,
    chunk_size: int = 200_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """带覆盖度和不确定度的 IDW；无支持位置返回 NaN。

    values 是站点指数，point_uncertainty 是站点不确定度，
    point_confidence 是数据完整性/垂直贴近程度。返回值依次为：
    插值场、覆盖度场和不确定度场。
    """
    n_cell = distances.shape[0]
    field = np.full(n_cell, np.nan, dtype=np.float32)
    coverage = np.zeros(n_cell, dtype=np.float32)
    uncertainty = np.full(n_cell, np.nan, dtype=np.float32)
    values = np.asarray(values, dtype=float)
    point_uncertainty = np.asarray(point_uncertainty, dtype=float)
    point_confidence = np.asarray(point_confidence, dtype=float)

    for start in range(0, n_cell, chunk_size):
        end = min(n_cell, start + chunk_size)
        distance = distances[start:end].astype(float)
        index = indices[start:end]
        neighbour_value = values[index]
        neighbour_uncertainty = point_uncertainty[index]
        neighbour_confidence = point_confidence[index]
        # 只使用有效值、有效不确定度、正置信度且半径内的邻点。
        valid = (
            np.isfinite(neighbour_value)
            & np.isfinite(neighbour_uncertainty)
            & (neighbour_confidence > 0)
            & (distance <= max_radius_m)
        )
        # 空间权重 = 高斯距离衰减 × 带平滑项的距离平方反比。
        spatial_weight = np.exp(-0.5 * (distance / max_radius_m) ** 2) / (distance**2 + d0_m**2)
        # 在空间权重上再乘置信度、除以不确定度方差。
        weight = np.where(
            valid,
            spatial_weight
            * neighbour_confidence
            / (neighbour_uncertainty**2 + uncertainty_floor**2),
            0.0,
        )
        weight_sum = weight.sum(axis=1)
        supported = weight_sum > 0
        if not supported.any():
            continue
        mean = np.divide(
            (weight * np.where(valid, neighbour_value, 0.0)).sum(axis=1),
            weight_sum,
            out=np.full(end - start, np.nan),
            where=supported,
        )
        # spread 描述邻站彼此分歧；measurement_variance 描述观测自身误差。
        spread = np.divide(
            (
                weight
                * np.where(valid, (neighbour_value - mean[:, None]) ** 2, 0.0)
            ).sum(axis=1),
            weight_sum,
            out=np.full(end - start, np.nan),
            where=supported,
        )
        measurement_variance = np.divide(
            (
                weight
                * np.where(valid, neighbour_uncertainty**2, 0.0)
            ).sum(axis=1),
            weight_sum,
            out=np.full(end - start, np.nan),
            where=supported,
        )
        valid_count = valid.sum(axis=1)
        nearest = np.min(np.where(valid, distance, np.inf), axis=1)
        local_confidence = np.divide(
            (weight * np.where(valid, neighbour_confidence, 0.0)).sum(axis=1),
            weight_sum,
            out=np.zeros(end - start),
            where=supported,
        )
        # 覆盖度同时受最近站距离、有效邻站数和局地置信度影响。
        cov = (
            np.exp(-0.5 * (nearest / max_radius_m) ** 2)
            * (1.0 - np.exp(-valid_count / 3.0))
            * local_confidence
        )
        # 离最近观测点越远，额外距离惩罚越大。
        distance_penalty = 0.20 * np.minimum(nearest / max_radius_m, 1.0)
        field[start:end][supported] = mean[supported].astype(np.float32)
        coverage[start:end][supported] = np.clip(cov[supported], 0.0, 1.0).astype(np.float32)
        uncertainty[start:end][supported] = np.sqrt(
            measurement_variance[supported] + spread[supported] + distance_penalty[supported] ** 2
        ).astype(np.float32)
    return field.reshape(shape), coverage.reshape(shape), uncertainty.reshape(shape)


def fuse_source_fields(
    fields: list[np.ndarray],
    coverages: list[np.ndarray],
    uncertainties: list[np.ndarray],
    base_weights: list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """将天气雷达、WPR 和地面站三个同高度平面场融合。

    权重核心思想：基础权重越高、覆盖度越大、不确定度越小，该源贡献越大。
    返回：融合指数、总覆盖度、总不确定度、三源贡献率。
    """
    # 第 0 维是数据源，后两维是水平网格。
    stack = np.stack(fields).astype(float)
    coverage_stack = np.stack(coverages).astype(float)
    uncertainty_stack = np.stack(uncertainties).astype(float)
    weights = np.zeros_like(stack)
    for i, base in enumerate(base_weights):
        finite = np.isfinite(stack[i]) & np.isfinite(uncertainty_stack[i]) & (coverage_stack[i] > 0)
        weights[i, finite] = base * coverage_stack[i, finite] / (
            uncertainty_stack[i, finite] ** 2 + 0.10**2
        )
    weight_sum = weights.sum(axis=0)
    fused = np.divide(
        np.nansum(weights * stack, axis=0),
        weight_sum,
        out=np.full(weight_sum.shape, np.nan),
        where=weight_sum > 0,
    )
    # 三源对同一网格的判断越不一致，融合不确定度越高。
    disagreement = np.divide(
        np.nansum(weights * (stack - fused[None, :, :]) ** 2, axis=0),
        weight_sum,
        out=np.full(weight_sum.shape, np.nan),
        where=weight_sum > 0,
    )
    uncertainty = np.sqrt(
        np.divide(1.0, weight_sum, out=np.full_like(weight_sum, np.nan), where=weight_sum > 0)
        + disagreement
    )
    # 并集覆盖度：只要任一数据源覆盖，总覆盖度就会提高。
    total_coverage = 1.0 - np.prod(1.0 - np.clip(coverage_stack, 0.0, 1.0), axis=0)
    contribution = np.divide(
        weights,
        weight_sum[None, :, :],
        out=np.zeros_like(weights),
        where=weight_sum[None, :, :] > 0,
    )
    # 低覆盖网格不强行给值，NaN 明确表示“缺乏观测支撑”。
    unsupported = total_coverage < 0.05
    fused[unsupported] = np.nan
    uncertainty[unsupported] = np.nan
    return (
        fused.astype(np.float32),
        total_coverage.astype(np.float32),
        uncertainty.astype(np.float32),
        contribution.astype(np.float32),
    )


# ========================= 6. 结果可视化 =========================


def save_quicklook(
    path: Path,
    field: np.ndarray,
    coverage: np.ndarray,
    grid: GridSpec,
    z_m: float,
    master: pd.DataFrame,
    wpr_stations: pd.DataFrame,
) -> None:
    """保存某一高度的湍流指数与观测覆盖度对照图。"""
    lon_limits, lat_limits = xy_to_lonlat(
        np.array([grid.xs_m[0], grid.xs_m[-1]]),
        np.array([grid.ys_m[0], grid.ys_m[-1]]),
        grid.lat0,
        grid.lon0,
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150, constrained_layout=True)
    extent = [lon_limits[0], lon_limits[1], lat_limits[0], lat_limits[1]]
    image = axes[0].imshow(field, origin="lower", extent=extent, vmin=0, vmax=1, cmap="turbo")
    axes[0].scatter(
        wpr_stations["longitude_deg"],
        wpr_stations["latitude_deg"],
        marker="^",
        s=28,
        c="black",
        label="WPR",
    )
    axes[0].set(title=f"Fused relative turbulence index — {z_m:.0f} m AGL", xlabel="Longitude", ylabel="Latitude")
    axes[0].legend(loc="upper right")
    fig.colorbar(image, ax=axes[0], label="Relative turbulence index (0–1)")
    image_cov = axes[1].imshow(coverage, origin="lower", extent=extent, vmin=0, vmax=1, cmap="viridis")
    axes[1].set(title="Observation coverage", xlabel="Longitude", ylabel="Latitude")
    fig.colorbar(image_cov, ax=axes[1], label="Coverage (0–1)")
    fig.savefig(path)
    plt.close(fig)


def save_cross_section(path: Path, rows: np.ndarray, grid: GridSpec) -> None:
    """保存研究区中央纬度上的东西向垂直剖面。"""
    longitude, _ = xy_to_lonlat(grid.xs_m, np.full(grid.nx, grid.ys_m[grid.ny // 2]), grid.lat0, grid.lon0)
    fig, ax = plt.subplots(figsize=(11, 5), dpi=160, constrained_layout=True)
    mesh = ax.pcolormesh(longitude, grid.zs_m, rows, shading="auto", vmin=0, vmax=1, cmap="turbo")
    ax.set(title="Central west–east vertical cross-section", xlabel="Longitude", ylabel="Height AGL (m)")
    fig.colorbar(mesh, ax=ax, label="Relative turbulence index (0–1)")
    fig.savefig(path)
    plt.close(fig)


def save_3d_scatter(
    path: Path,
    samples: list[tuple[float, np.ndarray]],
    grid: GridSpec,
    threshold: float = 0.65,
) -> None:
    """对高于阈值的网格降采样，绘制三维高值区散点图。"""
    x_all: list[np.ndarray] = []
    y_all: list[np.ndarray] = []
    z_all: list[np.ndarray] = []
    value_all: list[np.ndarray] = []
    # 正式网格过密，按格点总数自动决定降采样步长，避免图像卡顿。
    stride = max(1, int(math.sqrt((grid.nx * grid.ny) / 4000)))
    x_sample = grid.xs_m[::stride] / 1000.0
    y_sample = grid.ys_m[::stride] / 1000.0
    xx, yy = np.meshgrid(x_sample, y_sample)
    for z_m, array in samples:
        value = array[::stride, ::stride]
        mask = np.isfinite(value) & (value >= threshold)
        if mask.any():
            x_all.append(xx[mask])
            y_all.append(yy[mask])
            z_all.append(np.full(mask.sum(), z_m / 1000.0))
            value_all.append(value[mask])
    fig = plt.figure(figsize=(9, 7), dpi=160, constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    if value_all:
        x = np.concatenate(x_all)
        y = np.concatenate(y_all)
        z = np.concatenate(z_all)
        value = np.concatenate(value_all)
        # 最多绘制 4 万点，只影响图形，不改变完整数值结果。
        if len(value) > 40_000:
            select = np.linspace(0, len(value) - 1, 40_000, dtype=int)
            x, y, z, value = x[select], y[select], z[select], value[select]
        scatter = ax.scatter(x, y, z, c=value, cmap="turbo", vmin=0, vmax=1, s=4, alpha=0.45)
        fig.colorbar(scatter, ax=ax, shrink=0.65, label="Relative turbulence index")
    ax.set(
        title=f"3-D high-turbulence regions (index ≥ {threshold:.2f})",
        xlabel="Local east (km)",
        ylabel="Local north (km)",
        zlabel="Height AGL (km)",
    )
    fig.savefig(path)
    plt.close(fig)


# ========================= 7. 第二问主计算流程 =========================


def run(args: argparse.Namespace) -> Path:
    """执行完整三维融合，并返回结果目录。

    主要步骤：
    1. 读取并质控三类观测；
    2. 构建雷达/WPR 垂直剖面和地面代理指标；
    3. 建立规则三维网格并预计算水平近邻；
    4. 逐高度插值三个单源场并融合；
    5. 保存 NPZ、CSV、JSON 和图片。
    """
    # 输出按“数值高度层”和“图片”分目录保存。
    output_dir = args.output.resolve()
    level_dir = output_dir / "levels"
    figure_dir = output_dir / "figures"
    level_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    # 步骤 1：建立目标高度轴，读取并准备三类数据源。
    zs_m = np.arange(0.0, args.zmax + 0.5 * args.dz, args.dz)
    matches = read_radar_matches(args.matches, args.sheet)
    (
        radar_stations,
        radar_ti,
        radar_unc,
        radar_cov,
        radar_count,
        radar_diagnostics,
        _radars,
        _radar_source_ti,
        _radar_source_unc,
        _radar_source_cov,
    ) = prepare_radar_profiles(matches, zs_m)
    q1_project = args.q1_project.resolve() if args.q1_project and args.q1_project.exists() else None
    wpr_profiles, wpr_calibration, wpr_stations, wpr_ti, wpr_unc, wpr_cov = prepare_wpr_profiles(
        args.data_dir, zs_m, q1_project
    )
    surface = read_surface_stations(args.surface)

    # 步骤 2：合并水平观测点，以全部观测范围建立规则网格。
    master = build_master_points(surface, radar_stations)
    grid = make_grid(master, wpr_stations, args.dx, args.dz, args.zmax)

    # 将站点剖面映射到统一的地面点集合。
    master_radar_ti = np.full((len(master), grid.nz), np.nan, dtype=float)
    master_radar_unc = np.full_like(master_radar_ti, np.nan)
    master_radar_cov = np.zeros_like(master_radar_ti)
    master_radar_count = np.zeros((len(master), grid.nz), dtype=np.int16)
    master_surface_ti = np.full(len(master), np.nan, dtype=float)
    master_surface_unc = np.full(len(master), np.nan, dtype=float)
    master_surface_cov = np.zeros(len(master), dtype=float)
    for i, row in master.iterrows():
        radar_i = int(row["radar_row"])
        if radar_i >= 0:
            master_radar_ti[i] = radar_ti[radar_i]
            master_radar_unc[i] = radar_unc[radar_i]
            master_radar_cov[i] = radar_cov[radar_i]
            master_radar_count[i] = radar_count[radar_i]
        surface_i = int(row["surface_row"])
        if surface_i >= 0:
            master_surface_ti[i] = float(surface.loc[surface_i, "surface_turbulence_index"])
            master_surface_unc[i] = float(surface.loc[surface_i, "surface_uncertainty"])
            master_surface_cov[i] = 1.0

    # 步骤 3：把站点转到局地米制坐标，并预计算每个网格的近邻距离/索引。
    # 这些距离在 41 个高度层共用，只需计算一次。
    master_x, master_y = lonlat_to_xy(
        master["latitude_deg"], master["longitude_deg"], grid.lat0, grid.lon0
    )
    wpr_x, wpr_y = lonlat_to_xy(
        wpr_stations["latitude_deg"], wpr_stations["longitude_deg"], grid.lat0, grid.lon0
    )
    master_distance, master_index = query_grid_neighbours(
        np.column_stack([master_x, master_y]), grid, k=12, chunk_size=args.chunk_size
    )
    wpr_distance, wpr_index = query_grid_neighbours(
        np.column_stack([wpr_x, wpr_y]), grid, k=3, chunk_size=args.chunk_size
    )

    # 地面源的水平场只需计算一次，垂直影响通过覆盖度衰减。
    surface_field, surface_horizontal_cov, surface_field_unc = interpolate_from_neighbours(
        master_surface_ti,
        master_surface_unc,
        master_surface_cov,
        master_distance,
        master_index,
        (grid.ny, grid.nx),
        max_radius_m=args.surface_radius,
        d0_m=5_000.0,
        uncertainty_floor=0.12,
        chunk_size=args.chunk_size,
    )

    # 将雷达站点垂直剖面整理成长表，方便论文统计和二次分析。
    radar_profile_rows: list[dict[str, object]] = []
    for master_i, row in master.iterrows():
        radar_i = int(row["radar_row"])
        if radar_i < 0:
            continue
        for iz, z_m in enumerate(grid.zs_m):
            if np.isfinite(master_radar_ti[master_i, iz]):
                radar_profile_rows.append(
                    {
                        "point_key": row["point_key"],
                        "latitude_deg": row["latitude_deg"],
                        "longitude_deg": row["longitude_deg"],
                        "height_agl_m": z_m,
                        "radar_turbulence_index": master_radar_ti[master_i, iz],
                        "radar_uncertainty": master_radar_unc[master_i, iz],
                        "radar_coverage": master_radar_cov[master_i, iz],
                        "contributing_radars": int(master_radar_count[master_i, iz]),
                    }
                )

    # 只对典型高度生成平面图，避免为 41 层全部绘图浪费时间。
    chosen_quicklooks = (
        set()
        if args.skip_figures
        else {
            float(grid.zs_m[np.argmin(np.abs(grid.zs_m - target))])
            for target in (0, 200, 500, 1000, 1500, 2000)
            if target <= grid.zs_m.max()
        }
    )
    centre_rows = np.full((grid.nz, grid.nx), np.nan, dtype=np.float32)
    samples_3d: list[tuple[float, np.ndarray]] = []
    level_summary: list[dict[str, object]] = []

    # 步骤 4：按高度逐层计算。逐层运行可控制峰值内存。
    for iz, z_m in enumerate(grid.zs_m):
        # 4.1 将该高度的天气雷达站点剖面插值成水平场。
        radar_field, radar_field_cov, radar_field_unc = interpolate_from_neighbours(
            master_radar_ti[:, iz],
            master_radar_unc[:, iz],
            master_radar_cov[:, iz],
            master_distance,
            master_index,
            (grid.ny, grid.nx),
            max_radius_m=args.radar_radius,
            d0_m=5_000.0,
            uncertainty_floor=0.10,
            chunk_size=args.chunk_size,
        )
        # 4.2 将三部 WPR 的该高度指数插值成水平场。
        wpr_field, wpr_field_cov, wpr_field_unc = interpolate_from_neighbours(
            wpr_ti[:, iz],
            wpr_unc[:, iz],
            wpr_cov[:, iz],
            wpr_distance,
            wpr_index,
            (grid.ny, grid.nx),
            max_radius_m=args.wpr_radius,
            d0_m=8_000.0,
            uncertainty_floor=0.15,
            chunk_size=args.chunk_size,
        )
        # 4.3 地面指数的水平分布不变，但覆盖度随高度指数衰减。
        surface_vertical_factor = math.exp(-float(z_m) / args.surface_vertical_scale)
        surface_cov = surface_horizontal_cov * surface_vertical_factor
        surface_unc = surface_field_unc + np.float32(0.15 * (1.0 - surface_vertical_factor))
        # 4.4 按基础权重、覆盖度和不确定度融合三个单源场。
        fused, coverage, uncertainty, contribution = fuse_source_fields(
            [radar_field, wpr_field, surface_field],
            [radar_field_cov, wpr_field_cov, surface_cov],
            [radar_field_unc, wpr_field_unc, surface_unc],
            [args.radar_weight, args.wpr_weight, args.surface_weight],
        )
        # 4.5 每一高度单独保存为标准压缩 NPZ，方便按需读取，无需一次加载 1.4 GB。
        payload: dict[str, np.ndarray] = {
            "turbulence_index": fused,
            "coverage": coverage,
            "uncertainty": uncertainty,
            "source_contribution": contribution,
        }
        # 默认只保存三源贡献率；指定该选项时再附加保存三个完整单源场。
        if args.save_source_fields:
            payload.update(
                {
                    "weather_radar_index": radar_field,
                    "weather_radar_coverage": radar_field_cov,
                    "wpr_index": wpr_field,
                    "wpr_coverage": wpr_field_cov,
                    "surface_index": surface_field,
                    "surface_coverage": surface_cov,
                }
            )
        np.savez_compressed(level_dir / f"z_{int(round(z_m)):04d}m.npz", **payload)
        centre_rows[iz] = fused[grid.ny // 2]
        if not args.skip_figures:
            samples_3d.append((float(z_m), fused.copy()))

        # 对每层计算统计摘要，README 中的逐高度结果来自这张表。
        finite = np.isfinite(fused)
        level_summary.append(
            {
                "height_agl_m": float(z_m),
                "finite_cells": int(finite.sum()),
                "coverage_fraction": float(finite.mean()),
                "mean_observation_coverage": float(np.mean(coverage)),
                "p10_observation_coverage": float(np.quantile(coverage, 0.10)),
                "mean_turbulence_index": float(np.nanmean(fused)) if finite.any() else np.nan,
                "p90_turbulence_index": float(np.nanquantile(fused, 0.90)) if finite.any() else np.nan,
                "mean_uncertainty": float(np.nanmean(uncertainty)) if finite.any() else np.nan,
                "radar_mean_contribution": float(np.nanmean(contribution[0][finite])) if finite.any() else np.nan,
                "wpr_mean_contribution": float(np.nanmean(contribution[1][finite])) if finite.any() else np.nan,
                "surface_mean_contribution": float(np.nanmean(contribution[2][finite])) if finite.any() else np.nan,
            }
        )
        if float(z_m) in chosen_quicklooks:
            save_quicklook(
                figure_dir / f"quicklook_z{int(round(z_m)):04d}m.png",
                fused,
                coverage,
                grid,
                z_m,
                master,
                wpr_stations,
            )
        # 在 Spyder/终端显示当前高度层进度。
        print(f"[{iz + 1:02d}/{grid.nz:02d}] z={z_m:.0f} m")

    # 全部高度结束后才能生成垂直剖面和三维高值图。
    if not args.skip_figures:
        save_cross_section(figure_dir / "cross_section_central_we.png", centre_rows, grid)
        save_3d_scatter(figure_dir / "turbulence_3d_high_regions.png", samples_3d, grid)

    # 步骤 5：科研数据分层保存。原始文件不改，处理结果与诊断分开。
    surface.to_csv(output_dir / "surface_station_proxies.csv", index=False)
    wpr_profiles.to_csv(output_dir / "wpr_profiles.csv", index=False)
    pd.DataFrame(radar_profile_rows).to_csv(output_dir / "radar_station_profiles.csv", index=False)
    pd.DataFrame(level_summary).to_csv(output_dir / "level_summary.csv", index=False)
    # 诊断表记录每部雷达/WPR/地面源的样本数和归一化范围。
    diagnostics = pd.concat(
        [
            radar_diagnostics,
            wpr_calibration.assign(kind="wind_profiler", valid_rows=len(wpr_profiles)),
            pd.DataFrame(
                [
                    {
                        "source": "surface_wind_proxy",
                        "kind": "surface_station",
                        "valid_rows": len(surface),
                        "calibration_low": surface.attrs["calibration_low"],
                        "calibration_high": surface.attrs["calibration_high"],
                        "calibration_unit": "1",
                    }
                ]
            ),
        ],
        ignore_index=True,
        sort=False,
    )
    diagnostics.to_csv(output_dir / "source_diagnostics.csv", index=False)

    lon_axis, _ = xy_to_lonlat(grid.xs_m, np.full(grid.nx, grid.ys_m[0]), grid.lat0, grid.lon0)
    _, lat_axis = xy_to_lonlat(np.full(grid.ny, grid.xs_m[0]), grid.ys_m, grid.lat0, grid.lon0)
    # meta.json 记录输入、网格轴和模型参数，使数值文件可追溯。
    metadata = {
        "title": "2025 D Question 2 multi-source relative turbulence field",
        "target_time": TARGET_TIME.isoformat(),
        "metric": "dimensionless relative turbulence index",
        "metric_range": [0.0, 1.0],
        "warning": "This is a relative index, not calibrated EDR.",
        "grid": {
            "nx": grid.nx,
            "ny": grid.ny,
            "nz": grid.nz,
            "dx_m": grid.dx_m,
            "dy_m": grid.dx_m,
            "dz_m": grid.dz_m,
            "x_min_m": float(grid.xs_m[0]),
            "y_min_m": float(grid.ys_m[0]),
            "z_levels_agl_m": grid.zs_m.tolist(),
            "longitude_axis_deg": lon_axis.tolist(),
            "latitude_axis_deg": lat_axis.tolist(),
            "local_projection_origin": {"latitude_deg": grid.lat0, "longitude_deg": grid.lon0},
        },
        "inputs": {
            "radar_matches": str(args.matches.resolve()),
            "surface_stations": str(args.surface.resolve()),
            "wpr_directory": str(args.data_dir.resolve()),
            "q1_project_for_diagnostic": str(q1_project) if q1_project else None,
        },
        "fusion": {
            "sources": ["weather_radar_SW", "WPR_RAD_SW_plus_Q1_wind_features", "surface_wind_proxy"],
            "base_weights": {
                "weather_radar": args.radar_weight,
                "wind_profiler": args.wpr_weight,
                "surface_station": args.surface_weight,
            },
            "surface_vertical_coverage_scale_m": args.surface_vertical_scale,
            "surface_station_horizontal_radius_m": args.surface_radius,
            "weather_radar_horizontal_radius_m": args.radar_radius,
            "wind_profiler_horizontal_radius_m": args.wpr_radius,
            "minimum_output_coverage": 0.05,
        },
        "data_counts": {
            "valid_radar_match_rows": int(len(matches)),
            "radar_stations": int(len(radar_stations)),
            "valid_surface_stations": int(len(surface)),
            "wind_profiler_stations": int(len(wpr_stations)),
            "master_horizontal_points": int(len(master)),
        },
    }
    (output_dir / "meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output_dir



# ========================= 8. 模型验证 =========================


def regression_metrics(observed: np.ndarray, predicted: np.ndarray, threshold: float = 0.65) -> dict[str, float]:
    """计算连续指数误差、相关性和高值区分类 F1。"""
    # 只比较观测值和预测值同时有效的样本。
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = np.isfinite(observed) & np.isfinite(predicted)
    observed, predicted = observed[valid], predicted[valid]
    if observed.size == 0:
        return {key: np.nan for key in ["n", "mae", "rmse", "bias", "r2", "pearson_r", "spearman_r", "f1_high"]}
    residual = predicted - observed
    # R² 表示对观测方差的解释程度；负值意味着还不如直接预测均值。
    denominator = np.sum((observed - observed.mean()) ** 2)
    r2 = 1.0 - np.sum(residual**2) / denominator if denominator > 0 else np.nan
    pearson = pearsonr(observed, predicted).statistic if np.std(observed) > 0 and np.std(predicted) > 0 else np.nan
    spearman = spearmanr(observed, predicted).statistic if np.std(observed) > 0 and np.std(predicted) > 0 else np.nan
    # 将指数 >= 0.65 视为相对高值，计算高值识别 F1。
    truth = observed >= threshold
    estimate = predicted >= threshold
    true_positive = np.sum(truth & estimate)
    false_positive = np.sum(~truth & estimate)
    false_negative = np.sum(truth & ~estimate)
    denominator_f1 = 2 * true_positive + false_positive + false_negative
    f1 = 2 * true_positive / denominator_f1 if denominator_f1 else np.nan
    return {
        "n": int(observed.size),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "bias": float(np.mean(residual)),
        "r2": float(r2),
        "pearson_r": float(pearson),
        "spearman_r": float(spearman),
        "f1_high": float(f1),
    }


def fuse_radar_sources(
    source_ti: np.ndarray,
    source_unc: np.ndarray,
    source_cov: np.ndarray,
    radars: list[str],
    excluded: int,
) -> tuple[np.ndarray, np.ndarray]:
    """验证专用：排除一部雷达后，用其余雷达重建站点剖面。"""
    weights = np.zeros_like(source_ti)
    for i, radar in enumerate(radars):
        if i == excluded:
            continue
        finite = np.isfinite(source_ti[i]) & np.isfinite(source_unc[i]) & (source_cov[i] > 0)
        weights[i, finite] = (
            RADAR_BASE_RELIABILITY.get(radar, 0.8)
            * source_cov[i, finite]
            / (source_unc[i, finite] ** 2 + 0.08**2)
        )
    weight_sum = weights.sum(axis=0)
    prediction = np.divide(
        np.nansum(weights * source_ti, axis=0),
        weight_sum,
        out=np.full(weight_sum.shape, np.nan),
        where=weight_sum > 0,
    )
    coverage = 1.0 - np.prod(
        1.0 - np.where(np.arange(len(radars))[:, None, None] == excluded, 0.0, source_cov),
        axis=0,
    )
    return prediction, coverage


def radar_leave_one_out(
    stations: pd.DataFrame,
    zs_m: np.ndarray,
    radars: list[str],
    source_ti: np.ndarray,
    source_unc: np.ndarray,
    source_cov: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """逐部留出天气雷达，检验其余雷达能否重建被留出雷达的相对结构。"""
    metrics: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    station_indices, height_indices = np.indices(source_ti.shape[1:])
    for excluded, radar in enumerate(radars):
        # 例如 excluded 对应 S9250 时，预测场中完全不使用 S9250。
        prediction, prediction_coverage = fuse_radar_sources(
            source_ti, source_unc, source_cov, radars, excluded
        )
        observed = source_ti[excluded]
        mask = (
            np.isfinite(observed)
            & np.isfinite(prediction)
            & (source_cov[excluded] >= 0.10)
            & (prediction_coverage >= 0.10)
        )
        values = regression_metrics(observed[mask], prediction[mask])
        metrics.append(
            {
                "validation": "leave_one_weather_radar_out",
                "held_out_source": radar,
                **values,
                "held_out_supported_points": int(np.isfinite(observed).sum()),
                "overlap_fraction": float(mask.sum() / max(np.isfinite(observed).sum(), 1)),
            }
        )
        if mask.any():
            flat_station = station_indices[mask]
            flat_height = height_indices[mask]
            predictions.append(
                pd.DataFrame(
                    {
                        "validation": "leave_one_weather_radar_out",
                        "held_out_source": radar,
                        "station_id": stations.iloc[flat_station]["station_id"].to_numpy(),
                        "height_agl_m": zs_m[flat_height],
                        "observed_index": observed[mask],
                        "predicted_index": prediction[mask],
                        "prediction_coverage": prediction_coverage[mask],
                    }
                )
            )
    return pd.DataFrame(metrics), pd.concat(predictions, ignore_index=True)


def radar_paired_scan_reproducibility(matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """比较每部雷达 02:00 前后两次扫描，检验短时重复性。"""
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    for radar, group in matches.groupby("radar"):
        times = sorted(group["time"].unique())
        if len(times) != 2:
            continue
        low, high = robust_bounds(group["SW_corrected_mps"], 0.10, 0.90)
        calibrated = group.copy()
        calibrated["radar_ti"] = robust_unit(
            calibrated["SW_corrected_mps"].to_numpy(float), low, high
        )
        # 用站点 ID 和仰角配对 02:00 前后两次扫描的同类观测。
        first = calibrated[calibrated["time"] == times[0]][
            ["station_id", "elev_code", "beam_h_agl_station_m", "radar_ti"]
        ]
        second = calibrated[calibrated["time"] == times[1]][
            ["station_id", "elev_code", "beam_h_agl_station_m", "radar_ti"]
        ]
        paired = first.merge(
            second,
            on=["station_id", "elev_code"],
            suffixes=("_first", "_second"),
        )
        metric_rows.append(
            {
                "validation": "paired_scan_reproducibility",
                "held_out_source": str(radar),
                **regression_metrics(paired["radar_ti_first"], paired["radar_ti_second"]),
                "held_out_supported_points": int(len(first)),
                "overlap_fraction": float(len(paired) / max(len(first), 1)),
            }
        )
        prediction_rows.append(
            pd.DataFrame(
                {
                    "validation": "paired_scan_reproducibility",
                    "held_out_source": str(radar),
                    "station_id": paired["station_id"],
                    "height_agl_m": 0.5
                    * (
                        paired["beam_h_agl_station_m_first"]
                        + paired["beam_h_agl_station_m_second"]
                    ),
                    "observed_index": paired["radar_ti_first"],
                    "predicted_index": paired["radar_ti_second"],
                    "prediction_coverage": 1.0,
                }
            )
        )
    return pd.DataFrame(metric_rows), pd.concat(prediction_rows, ignore_index=True)


def wpr_leave_one_out(
    stations: pd.DataFrame,
    zs_m: np.ndarray,
    ti: np.ndarray,
    uncertainty: np.ndarray,
    confidence: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """每次留出一部 WPR，用另两部 WPR 的距离加权剖面做预测。"""
    lat0, lon0 = float(stations["latitude_deg"].mean()), float(stations["longitude_deg"].mean())
    x, y = lonlat_to_xy(stations["latitude_deg"], stations["longitude_deg"], lat0, lon0)
    points = np.column_stack([x, y])
    metrics: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    for held in range(len(stations)):
        # held 是当前完全不参与预测的 WPR 站。
        distance = np.hypot(points[:, 0] - points[held, 0], points[:, 1] - points[held, 1])
        prediction = np.full(len(zs_m), np.nan)
        prediction_coverage = np.zeros(len(zs_m))
        for iz in range(len(zs_m)):
            usable = (
                (np.arange(len(stations)) != held)
                & np.isfinite(ti[:, iz])
                & np.isfinite(uncertainty[:, iz])
                & (confidence[:, iz] > 0)
                & (distance <= 70_000.0)
            )
            if not usable.any():
                continue
            weight = (
                np.exp(-0.5 * (distance[usable] / 70_000.0) ** 2)
                / (distance[usable] ** 2 + 8_000.0**2)
                * confidence[usable, iz]
                / (uncertainty[usable, iz] ** 2 + 0.15**2)
            )
            prediction[iz] = np.sum(weight * ti[usable, iz]) / np.sum(weight)
            prediction_coverage[iz] = (1.0 - np.exp(-usable.sum() / 2.0)) * np.exp(
                -0.5 * (distance[usable].min() / 70_000.0) ** 2
            )
        observed = ti[held]
        mask = np.isfinite(observed) & np.isfinite(prediction)
        station_name = str(stations.loc[held, "station_name"])
        metrics.append(
            {
                "validation": "leave_one_wpr_out",
                "held_out_source": f"WPR_{station_name.upper()}",
                **regression_metrics(observed[mask], prediction[mask]),
                "held_out_supported_points": int(np.isfinite(observed).sum()),
                "overlap_fraction": float(mask.sum() / max(np.isfinite(observed).sum(), 1)),
            }
        )
        for iz in np.flatnonzero(mask):
            rows.append(
                {
                    "validation": "leave_one_wpr_out",
                    "held_out_source": f"WPR_{station_name.upper()}",
                    "station_id": station_name,
                    "height_agl_m": zs_m[iz],
                    "observed_index": observed[iz],
                    "predicted_index": prediction[iz],
                    "prediction_coverage": prediction_coverage[iz],
                }
            )
    return pd.DataFrame(metrics), pd.DataFrame(rows)


def surface_leave_one_out(surface: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """地面站逐站留一：用周围站预测被留出站的代理指数。"""
    lat0, lon0 = float(surface["Lat"].mean()), float(surface["Lon"].mean())
    x, y = lonlat_to_xy(surface["Lat"], surface["Lon"], lat0, lon0)
    points = np.column_stack([x, y])
    tree = cKDTree(points)
    k_eff = min(9, len(points))
    distances, indices = tree.query(points, k=k_eff)
    values = surface["surface_turbulence_index"].to_numpy(float)
    uncertainty = surface["surface_uncertainty"].to_numpy(float)
    prediction = np.full(len(surface), np.nan)
    coverage = np.zeros(len(surface))
    for i in range(len(surface)):
        # KDTree 结果第 0 个邻点是站点自身，留一时必须从第 1 个邻点开始。
        neighbour = indices[i, 1:]
        distance = distances[i, 1:]
        usable = np.isfinite(values[neighbour]) & (distance <= 30_000.0)
        if not usable.any():
            continue
        weight = (
            np.exp(-0.5 * (distance[usable] / 30_000.0) ** 2)
            / (distance[usable] ** 2 + 5_000.0**2)
            / (uncertainty[neighbour[usable]] ** 2 + 0.12**2)
        )
        prediction[i] = np.sum(weight * values[neighbour[usable]]) / np.sum(weight)
        coverage[i] = (1.0 - np.exp(-usable.sum() / 3.0)) * np.exp(
            -0.5 * (distance[usable].min() / 30_000.0) ** 2
        )
    mask = np.isfinite(prediction)
    metrics = pd.DataFrame(
        [
            {
                "validation": "leave_one_surface_station_out",
                "held_out_source": "all_surface_stations",
                **regression_metrics(values[mask], prediction[mask]),
                "held_out_supported_points": int(len(surface)),
                "overlap_fraction": float(mask.mean()),
            }
        ]
    )
    rows = pd.DataFrame(
        {
            "validation": "leave_one_surface_station_out",
            "held_out_source": surface["surface_id"],
            "station_id": surface["surface_id"],
            "height_agl_m": 0.0,
            "observed_index": values,
            "predicted_index": prediction,
            "prediction_coverage": coverage,
        }
    )
    return metrics, rows


def radar_wpr_agreement(
    radar_stations: pd.DataFrame,
    radar_ti: np.ndarray,
    wpr_stations: pd.DataFrame,
    wpr_ti: np.ndarray,
    zs_m: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """比较每部 WPR 与最近天气雷达站点剖面，检查跨传感器一致性。"""
    lat0 = float(pd.concat([radar_stations["station_lat"], wpr_stations["latitude_deg"]]).mean())
    lon0 = float(pd.concat([radar_stations["station_lon"], wpr_stations["longitude_deg"]]).mean())
    rx, ry = lonlat_to_xy(radar_stations["station_lat"], radar_stations["station_lon"], lat0, lon0)
    wx, wy = lonlat_to_xy(wpr_stations["latitude_deg"], wpr_stations["longitude_deg"], lat0, lon0)
    # 对每个 WPR 位置找最近的天气雷达站点剖面。
    tree = cKDTree(np.column_stack([rx, ry]))
    distance, index = tree.query(np.column_stack([wx, wy]), k=1)
    rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    for i, station_name in enumerate(wpr_stations["station_name"]):
        observed = wpr_ti[i]
        predicted = radar_ti[index[i]]
        mask = np.isfinite(observed) & np.isfinite(predicted)
        metric_rows.append(
            {
                "validation": "radar_wpr_cross_sensor_agreement",
                "held_out_source": f"WPR_{str(station_name).upper()}",
                **regression_metrics(observed[mask], predicted[mask]),
                "held_out_supported_points": int(np.isfinite(observed).sum()),
                "overlap_fraction": float(mask.sum() / max(np.isfinite(observed).sum(), 1)),
                "nearest_radar_profile_distance_km": float(distance[i] / 1000.0),
            }
        )
        for iz in np.flatnonzero(mask):
            rows.append(
                {
                    "validation": "radar_wpr_cross_sensor_agreement",
                    "held_out_source": f"WPR_{str(station_name).upper()}",
                    "station_id": radar_stations.iloc[index[i]]["station_id"],
                    "height_agl_m": zs_m[iz],
                    "observed_index": observed[iz],
                    "predicted_index": predicted[iz],
                    "prediction_coverage": np.nan,
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(rows)


def save_validation_figure(path: Path, predictions: pd.DataFrame) -> None:
    """绘制四类验证的观测—预测散点图。"""
    groups = [
        ("paired_scan_reproducibility", "Weather-radar paired scans"),
        ("leave_one_wpr_out", "WPR leave-one-out"),
        ("leave_one_surface_station_out", "Surface-station leave-one-out"),
        ("radar_wpr_cross_sensor_agreement", "Radar–WPR agreement"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), dpi=160, constrained_layout=True)
    for ax, (name, title) in zip(axes.flat, groups):
        selected = predictions[predictions["validation"] == name]
        # 散点过多时随机抽样，只影响绘图速度，指标仍使用全部数据。
        if len(selected) > 20_000:
            selected = selected.sample(20_000, random_state=2025)
        for source, group in selected.groupby("held_out_source"):
            ax.scatter(
                group["observed_index"],
                group["predicted_index"],
                s=8,
                alpha=0.28,
                label=str(source),
            )
        ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
        ax.set(xlim=(0, 1), ylim=(0, 1), title=title, xlabel="Observed index", ylabel="Predicted index")
        if selected["held_out_source"].nunique() <= 5:
            ax.legend(fontsize=7)
    fig.savefig(path)
    plt.close(fig)



# ========================= 9. 敏感性分析 =========================

# 除 baseline 外，每个场景只改动一类参数，便于判断单因素影响。
SCENARIOS = {
    "baseline": {},
    "radar_radius_30km": {"radar_radius": 30_000.0},
    "radar_radius_50km": {"radar_radius": 50_000.0},
    "wpr_weight_0.60": {"wpr_weight": 0.60},
    "wpr_weight_1.10": {"wpr_weight": 1.10},
    "surface_scale_200m": {"surface_vertical_scale": 200.0},
    "surface_scale_500m": {"surface_vertical_scale": 500.0},
}


def load_level(path: Path, height_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读取某场景、某高度的湍流指数、覆盖度和不确定度。"""
    with np.load(path / "levels" / f"z_{int(round(height_m)):04d}m.npz") as archive:
        return (
            archive["turbulence_index"].astype(float),
            archive["coverage"].astype(float),
            archive["uncertainty"].astype(float),
        )



# ========================= 10. GIS 导出 =========================

# GeoTIFF 标签中的坐标系定义：地理坐标、像元表示面积、EPSG:4326、角度单位为度。
GEOKEY_DIRECTORY = (
    1,
    1,
    0,
    4,
    1024,
    0,
    1,
    2,  # GTModelTypeGeoKey = Geographic
    1025,
    0,
    1,
    1,  # GTRasterTypeGeoKey = PixelIsArea
    2048,
    0,
    1,
    4326,  # GeographicTypeGeoKey = EPSG:4326
    2054,
    0,
    1,
    9102,  # GeogAngularUnitsGeoKey = degree
)
EPSG4326_WKT = (
    'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],'
    'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433],AUTHORITY["EPSG","4326"]]'
)


def write_geotiff(
    path: Path,
    array: np.ndarray,
    longitude: np.ndarray,
    latitude: np.ndarray,
    variable: str,
    height_m: float,
    nodata: float = -9999.0,
) -> None:
    """将二维网格写成带 EPSG:4326 地理标签的单波段 GeoTIFF。"""
    # 经纬度轴等间距，中位差分可避免浮点微小误差。
    dlon = float(np.median(np.diff(longitude)))
    dlat = float(np.median(np.diff(latitude)))
    west = float(longitude[0] - 0.5 * dlon)
    north = float(latitude[-1] + 0.5 * dlat)
    # NumPy 数组原点在南西，GeoTIFF 通常从北西开始逐行存储，因此需要上下翻转。
    north_up = np.flipud(np.asarray(array, dtype=np.float32))
    # GIS 栅格不能直接保留 NaN 语义时，使用 -9999 作为 NoData。
    north_up = np.where(np.isfinite(north_up), north_up, nodata).astype(np.float32)
    # 33550=像元尺度，33922=左上角坐标，34735=坐标系，42113=NoData。
    extratags = [
        (33550, "d", 3, (abs(dlon), abs(dlat), 0.0), False),
        (33922, "d", 6, (0.0, 0.0, 0.0, west, north, 0.0), False),
        (34735, "H", len(GEOKEY_DIRECTORY), GEOKEY_DIRECTORY, False),
        (42113, "s", 0, str(nodata), False),
    ]
    description = json.dumps(
        {
            "variable": variable,
            "height_agl_m": float(height_m),
            "crs": "EPSG:4326",
            "nodata": nodata,
        },
        ensure_ascii=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        path,
        north_up,
        dtype=np.float32,
        photometric="minisblack",
        compression="deflate",
        metadata=None,
        description=description,
        extratags=extratags,
    )


def sensor_geojson(input_dir: Path, legacy_dir: Path) -> dict[str, object]:
    """将地面站、WPR 和天气雷达位置组装成 GeoJSON 点要素。"""
    features: list[dict[str, object]] = []
    surface_path = input_dir / "surface_station_proxies.csv"
    if surface_path.exists():
        surface = pd.read_csv(surface_path)
        for _, row in surface.iterrows():
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [float(row["Lon"]), float(row["Lat"])]},
                    "properties": {
                        "sensor_type": "surface_station",
                        "sensor_id": str(row["surface_id"]),
                        "altitude_m": float(row["Alti"]),
                        "turbulence_index": float(row["surface_turbulence_index"]),
                    },
                }
            )
    wpr_path = input_dir / "wpr_profiles.csv"
    if wpr_path.exists():
        wpr = pd.read_csv(wpr_path).drop_duplicates("station_name")
        for _, row in wpr.iterrows():
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [float(row["longitude_deg"]), float(row["latitude_deg"])],
                    },
                    "properties": {
                        "sensor_type": "wind_profiler",
                        "sensor_id": f"WPR_{str(row['station_name']).upper()}",
                        "altitude_m": float(row["site_altitude_m"]),
                    },
                }
            )
    radar_path = legacy_dir / "radar_sites_local_coords.csv"
    if radar_path.exists():
        radar = pd.read_csv(radar_path, encoding="utf-8-sig")
        for _, row in radar.iterrows():
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [float(row["lon"]), float(row["lat"])]},
                    "properties": {
                        "sensor_type": "weather_radar",
                        "sensor_id": str(row["site_id"]),
                        "description": str(row["desc"]),
                        "altitude_m": float(row["alt_m"]),
                    },
                }
            )
    return {"type": "FeatureCollection", "name": "question2_sensors", "features": features}


def high_turbulence_geojson(
    input_dir: Path,
    longitude: np.ndarray,
    latitude: np.ndarray,
    heights: list[float],
    threshold: float,
    stride: int,
    maximum_points: int,
) -> dict[str, object]:
    """将高于指定阈值且覆盖度足够的三维格点导出为 GeoJSON。"""
    features: list[dict[str, object]] = []
    # 完整 6438 万网格不适合全部写入 JSON，先按 stride 水平降采样。
    lon_sample = longitude[::stride]
    lat_sample = latitude[::stride]
    lon_grid, lat_grid = np.meshgrid(lon_sample, lat_sample)
    candidates: list[tuple[float, float, float, float, float, float]] = []
    for z_m in heights:
        with np.load(input_dir / "levels" / f"z_{int(round(z_m)):04d}m.npz") as archive:
            ti = archive["turbulence_index"][::stride, ::stride]
            coverage = archive["coverage"][::stride, ::stride]
            uncertainty = archive["uncertainty"][::stride, ::stride]
        # 除湍流高值外还要求 coverage>=0.20，排除支撑过弱的“伪高值”。
        mask = np.isfinite(ti) & (ti >= threshold) & (coverage >= 0.20)
        for lon, lat, value, cov, unc in zip(
            lon_grid[mask], lat_grid[mask], ti[mask], coverage[mask], uncertainty[mask]
        ):
            candidates.append((float(value), float(lon), float(lat), float(z_m), float(cov), float(unc)))
    # 先按指数从高到低排序；超过上限时均匀抽样，避免 GeoJSON 过大。
    candidates.sort(reverse=True)
    if len(candidates) > maximum_points:
        selected = np.linspace(0, len(candidates) - 1, maximum_points, dtype=int)
        candidates = [candidates[i] for i in selected]
    for value, lon, lat, z_m, coverage, uncertainty in candidates:
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat, z_m]},
                "properties": {
                    "height_agl_m": z_m,
                    "turbulence_index": value,
                    "coverage": coverage,
                    "uncertainty": uncertainty,
                },
            }
        )
    return {"type": "FeatureCollection", "name": "high_turbulence_voxels", "features": features}



# ========================= 11. 命令行参数与各子流程 =========================


def build_fusion_parser(quick: bool = False) -> argparse.ArgumentParser:
    """构建主融合命令参数。"""
    parser = argparse.ArgumentParser(
        description="生成 2025 D 题第二问 02:00 多源三维相对湍流场"
    )
    # 输入路径：默认全部指向仓库 data/ 目录。
    parser.add_argument("--matches", type=Path, default=DATA_DIR / "radar_matches.xlsx")
    parser.add_argument("--sheet", default="merged")
    parser.add_argument("--surface", type=Path, default=DATA_DIR / "surface_stations.xlsx")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--q1-project", type=Path, default=DEFAULT_Q1_PROJECT)
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "outputs" / ("quick_test" if quick else "q2_0200_fusion"),
    )
    # quick 使用粗网格只做环境检查；run 使用题目要求的正式分辨率。
    parser.add_argument("--dx", type=float, default=5000.0 if quick else 100.0)
    parser.add_argument("--dz", type=float, default=250.0 if quick else 50.0)
    parser.add_argument("--zmax", type=float, default=500.0 if quick else 2000.0)
    parser.add_argument("--chunk-size", type=int, default=200_000)
    # 可选开关：单源场会大幅增加磁盘占用，因此默认不保存。
    parser.add_argument("--save-source-fields", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    # 三类数据源的基础融合权重。
    parser.add_argument("--radar-weight", type=float, default=1.0)
    parser.add_argument("--wpr-weight", type=float, default=0.85)
    parser.add_argument("--surface-weight", type=float, default=0.65)
    # 三类数据源的最大水平影响半径，单位为 m。
    parser.add_argument("--radar-radius", type=float, default=40_000.0)
    parser.add_argument("--wpr-radius", type=float, default=70_000.0)
    parser.add_argument("--surface-radius", type=float, default=30_000.0)
    parser.add_argument("--surface-vertical-scale", type=float, default=300.0)
    return parser


def run_validation(output: Path) -> Path:
    """运行留一交叉验证、雷达重复扫描及跨传感器一致性检查。"""
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # 验证使用与正式运行相同的 0--2000 m、50 m 高度轴。
    zs_m = np.arange(0.0, 2000.0 + 25.0, 50.0)
    matches = read_radar_matches(DATA_DIR / "radar_matches.xlsx")
    (
        radar_stations,
        radar_ti,
        _radar_unc,
        _radar_cov,
        _radar_count,
        _radar_diagnostics,
        radars,
        source_ti,
        source_unc,
        source_cov,
    ) = prepare_radar_profiles(matches, zs_m)
    q1_project = DEFAULT_Q1_PROJECT if DEFAULT_Q1_PROJECT.exists() else None
    (
        _wpr_profiles,
        _wpr_calibration,
        wpr_stations,
        wpr_ti,
        wpr_unc,
        wpr_cov,
    ) = prepare_wpr_profiles(DATA_DIR, zs_m, q1_project)
    surface = read_surface_stations(DATA_DIR / "surface_stations.xlsx")

    # 五类检验相互补充：留一设备、短时重复性和跨传感器一致性。
    radar_metrics, radar_predictions = radar_leave_one_out(
        radar_stations, zs_m, radars, source_ti, source_unc, source_cov
    )
    scan_metrics, scan_predictions = radar_paired_scan_reproducibility(matches)
    wpr_metrics, wpr_predictions = wpr_leave_one_out(
        wpr_stations, zs_m, wpr_ti, wpr_unc, wpr_cov
    )
    surface_metrics, surface_predictions = surface_leave_one_out(surface)
    agreement_metrics, agreement_predictions = radar_wpr_agreement(
        radar_stations, radar_ti, wpr_stations, wpr_ti, zs_m
    )
    metrics = pd.concat(
        [radar_metrics, scan_metrics, wpr_metrics, surface_metrics, agreement_metrics],
        ignore_index=True,
        sort=False,
    )
    predictions = pd.concat(
        [radar_predictions, scan_predictions, wpr_predictions, surface_predictions, agreement_predictions],
        ignore_index=True,
        sort=False,
    )
    # metrics 是按验证对象汇总的指标；predictions 保留每一个观测—预测对。
    metrics.to_csv(output / "validation_metrics.csv", index=False)
    predictions.to_csv(output / "validation_predictions.csv", index=False)
    save_validation_figure(output / "validation_scatter.png", predictions)
    summary = {
        "target_time": TARGET_TIME.isoformat(),
        "high_turbulence_threshold": 0.65,
        "validation_methods": metrics["validation"].drop_duplicates().tolist(),
        "records": int(len(predictions)),
    }
    (output / "validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Validation saved to {output}")
    return output


def run_sensitivity_analysis(output: Path, dx: float = 2000.0, dz: float = 100.0) -> Path:
    """对雷达半径、WPR 权重和地面站垂直尺度做单因素敏感性分析。"""
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    scenario_root = output / "scenarios"
    scenario_root.mkdir(parents=True, exist_ok=True)
    # 基于正式默认参数创建每个场景，但用粗网格降低 7 次重复计算成本。
    base_namespace = build_fusion_parser().parse_args([])
    scenario_paths: dict[str, Path] = {}
    for name, overrides in SCENARIOS.items():
        # 复制参数命名空间，确保上一场景的修改不泄漏到下一场景。
        namespace = argparse.Namespace(**vars(base_namespace))
        namespace.dx = dx
        namespace.dz = dz
        namespace.zmax = 2000.0
        namespace.skip_figures = True
        namespace.save_source_fields = False
        namespace.output = scenario_root / name
        for key, value in overrides.items():
            setattr(namespace, key, value)
        print(f"Running sensitivity scenario: {name}")
        scenario_paths[name] = run(namespace)

    # baseline 作为参考场，其他场景均与它做逐网格差分。
    baseline_summary = pd.read_csv(scenario_paths["baseline"] / "level_summary.csv")
    heights = baseline_summary["height_agl_m"].to_numpy(float)
    by_height: list[dict[str, object]] = []
    metrics: list[dict[str, object]] = []
    baseline_levels = {z: load_level(scenario_paths["baseline"], z) for z in heights}
    for name, path in scenario_paths.items():
        absolute_differences: list[np.ndarray] = []
        signed_differences: list[np.ndarray] = []
        scenario_values: list[np.ndarray] = []
        baseline_values: list[np.ndarray] = []
        high_count = 0
        finite_count = 0
        coverage_values: list[float] = []
        uncertainty_values: list[float] = []
        for z in heights:
            baseline_ti, _, _ = baseline_levels[z]
            ti, coverage, uncertainty = load_level(path, z)
            valid = np.isfinite(baseline_ti) & np.isfinite(ti)
            # 只在两个场景都有效的公共网格上计算差异。
            difference = ti[valid] - baseline_ti[valid]
            absolute_differences.append(np.abs(difference))
            signed_differences.append(difference)
            scenario_values.append(ti[valid])
            baseline_values.append(baseline_ti[valid])
            high_count += int(np.sum(ti[valid] >= 0.65))
            finite_count += int(valid.sum())
            coverage_values.append(float(np.mean(coverage)))
            uncertainty_values.append(float(np.nanmean(uncertainty)))
            by_height.append(
                {
                    "scenario": name,
                    "height_agl_m": z,
                    "mean_turbulence_index": float(np.nanmean(ti)),
                    "p90_turbulence_index": float(np.nanquantile(ti, 0.90)),
                    "mae_from_baseline": float(np.mean(np.abs(difference))) if valid.any() else np.nan,
                    "mean_delta_from_baseline": float(np.mean(difference)) if valid.any() else np.nan,
                    "mean_coverage": float(np.mean(coverage)),
                    "mean_uncertainty": float(np.nanmean(uncertainty)),
                }
            )
        absolute = np.concatenate(absolute_differences)
        signed = np.concatenate(signed_differences)
        scenario_all = np.concatenate(scenario_values)
        baseline_all = np.concatenate(baseline_values)
        correlation = np.corrcoef(scenario_all, baseline_all)[0, 1]
        metrics.append(
            {
                "scenario": name,
                "changed_parameters": json.dumps(SCENARIOS[name], ensure_ascii=False, sort_keys=True),
                "grid_dx_m": dx,
                "grid_dz_m": dz,
                "n_common_cells": int(len(absolute)),
                "mae_from_baseline": float(np.mean(absolute)),
                "rmse_from_baseline": float(np.sqrt(np.mean(signed**2))),
                "max_abs_difference": float(np.max(absolute)),
                "mean_delta_from_baseline": float(np.mean(signed)),
                "correlation_with_baseline": float(correlation),
                "high_risk_fraction": float(high_count / max(finite_count, 1)),
                "mean_coverage": float(np.mean(coverage_values)),
                "mean_uncertainty": float(np.mean(uncertainty_values)),
            }
        )

    metrics_table = pd.DataFrame(metrics)
    height_table = pd.DataFrame(by_height)
    metrics_table.to_csv(output / "sensitivity_metrics.csv", index=False)
    height_table.to_csv(output / "sensitivity_by_height.csv", index=False)
    # 左图比较平均垂直剖面，右图显示每层相对基准场的 MAE。
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=160, constrained_layout=True)
    for name, group in height_table.groupby("scenario"):
        axes[0].plot(group["mean_turbulence_index"], group["height_agl_m"], label=name)
        if name != "baseline":
            axes[1].plot(group["mae_from_baseline"], group["height_agl_m"], label=name)
    axes[0].set(
        xlabel="Domain-mean turbulence index",
        ylabel="Height AGL (m)",
        title="Sensitivity of mean vertical profile",
    )
    axes[1].set(
        xlabel="MAE from baseline",
        ylabel="Height AGL (m)",
        title="One-at-a-time parameter sensitivity",
    )
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    fig.savefig(output / "sensitivity_profiles.png")
    plt.close(fig)
    print(metrics_table.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print(f"Sensitivity results saved to {output}")
    return output


def export_gis_outputs(
    input_dir: Path,
    output: Path | None = None,
    all_auxiliary: bool = False,
    threshold: float = 0.65,
    voxel_stride: int = 20,
    maximum_voxel_points: int = 50_000,
) -> Path:
    """将融合结果导出为 EPSG:4326 GeoTIFF 和 GeoJSON。"""
    input_dir = input_dir.resolve()
    output = output.resolve() if output else input_dir / "gis"
    output.mkdir(parents=True, exist_ok=True)
    # meta.json 中保存了经度轴、纬度轴和全部高度层。
    meta = json.loads((input_dir / "meta.json").read_text(encoding="utf-8"))
    longitude = np.asarray(meta["grid"]["longitude_axis_deg"], dtype=float)
    latitude = np.asarray(meta["grid"]["latitude_axis_deg"], dtype=float)
    heights = [float(value) for value in meta["grid"]["z_levels_agl_m"]]
    # 湍流指数导出全部高度；覆盖度/不确定度默认只导出代表高度。
    selected_auxiliary = [0.0, 200.0, 500.0, 1000.0, 1500.0, 2000.0]
    auxiliary_heights = set(heights if all_auxiliary else selected_auxiliary)
    manifest: list[dict[str, object]] = []
    for z_m in heights:
        level_path = input_dir / "levels" / f"z_{int(round(z_m)):04d}m.npz"
        with np.load(level_path) as archive:
            variables = ["turbulence_index"]
            if z_m in auxiliary_heights:
                variables.extend(["coverage", "uncertainty"])
            for variable in variables:
                array = archive[variable]
                path = output / variable / f"{variable}_z{int(round(z_m)):04d}m.tif"
                write_geotiff(path, array, longitude, latitude, variable, z_m)
                finite = np.isfinite(array)
                manifest.append(
                    {
                        "file": str(path.relative_to(output)),
                        "variable": variable,
                        "height_agl_m": z_m,
                        "crs": "EPSG:4326",
                        "rows": int(array.shape[0]),
                        "columns": int(array.shape[1]),
                        "minimum": float(np.nanmin(array)) if finite.any() else np.nan,
                        "maximum": float(np.nanmax(array)) if finite.any() else np.nan,
                        "nodata": -9999.0,
                    }
                )
        print(f"GeoTIFF z={z_m:.0f} m")
    # 除栅格外，同时导出传感器位置和降采样的三维高值点。
    (output / "sensor_points.geojson").write_text(
        json.dumps(sensor_geojson(input_dir, DATA_DIR), ensure_ascii=False), encoding="utf-8"
    )
    high_risk = high_turbulence_geojson(
        input_dir,
        longitude,
        latitude,
        heights,
        threshold,
        voxel_stride,
        maximum_voxel_points,
    )
    (output / "high_turbulence_voxels.geojson").write_text(
        json.dumps(high_risk, ensure_ascii=False), encoding="utf-8"
    )
    pd.DataFrame(manifest).to_csv(output / "gis_manifest.csv", index=False)
    (output / "EPSG4326.prj").write_text(EPSG4326_WKT, encoding="ascii")
    print(f"GIS outputs saved to {output}")
    return output


def print_main_help() -> None:
    """打印面向用户的简要命令说明。"""
    print(
        """2025 D 题第二问——多源三维湍流融合

用法：
  python main.py              直接执行正式网格计算
  python main.py quick        快速检查环境与数据
  python main.py run          正式网格计算（100 m × 100 m × 50 m）
  python main.py validate     交叉验证与跨传感器一致性检查
  python main.py sensitivity  关键参数敏感性分析
  python main.py gis          导出 GeoTIFF/GeoJSON（需先 run）
  python main.py all          按顺序执行上述正式流程

在命令后加 --help 查看可选参数。
"""
    )


def main() -> None:
    """程序统一入口：识别命令并调用对应的主流程。"""
    argv = sys.argv[1:]
    # 便于在 Spyder/PyCharm 中直接点击“运行”：无命令行参数时
    # 默认生成题目要求的 100 m × 100 m × 50 m 正式结果。
    if not argv:
        argv = ["run"]
    if argv[0] in {"-h", "--help", "help"}:
        print_main_help()
        return
    # 第一个参数是子命令，后续参数交给该子命令自己解析。
    command, command_args = argv[0].lower(), argv[1:]
    if command in {"run", "quick"}:
        args = build_fusion_parser(quick=command == "quick").parse_args(command_args)
        output = run(args)
        print(f"Results saved to {output}")
        return
    if command == "validate":
        parser = argparse.ArgumentParser(description="第二问交叉验证")
        parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "outputs" / "validation")
        args = parser.parse_args(command_args)
        run_validation(args.output)
        return
    if command == "sensitivity":
        parser = argparse.ArgumentParser(description="第二问参数敏感性分析")
        parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "outputs" / "sensitivity")
        parser.add_argument("--dx", type=float, default=2000.0)
        parser.add_argument("--dz", type=float, default=100.0)
        args = parser.parse_args(command_args)
        run_sensitivity_analysis(args.output, args.dx, args.dz)
        return
    if command == "gis":
        parser = argparse.ArgumentParser(description="导出第二问 GIS 数据")
        parser.add_argument("--input", type=Path, default=SCRIPT_DIR / "outputs" / "q2_0200_fusion")
        parser.add_argument("--output", type=Path)
        parser.add_argument("--all-auxiliary", action="store_true")
        parser.add_argument("--threshold", type=float, default=0.65)
        parser.add_argument("--voxel-stride", type=int, default=20)
        parser.add_argument("--maximum-voxel-points", type=int, default=50_000)
        args = parser.parse_args(command_args)
        export_gis_outputs(
            args.input,
            args.output,
            args.all_auxiliary,
            args.threshold,
            args.voxel_stride,
            args.maximum_voxel_points,
        )
        return
    if command == "all":
        # all 顺序执行正式融合、验证、敏感性和 GIS 导出，耗时最长。
        args = build_fusion_parser().parse_args(command_args)
        fusion_output = run(args)
        run_validation(SCRIPT_DIR / "outputs" / "validation")
        run_sensitivity_analysis(SCRIPT_DIR / "outputs" / "sensitivity")
        export_gis_outputs(fusion_output)
        return
    raise SystemExit(f"未知命令：{command}\n请运行 python main.py --help")


if __name__ == "__main__":
    main()
