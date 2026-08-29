#!/usr/bin/env python3
"""2025 D 题第三问：低空湍流预警、非线性外推与三维航路规划。

本程序在前两问口径上完成三个任务：

1. 将 02:00--05:00 的天气雷达、风廓线雷达和地面站融合结果采样到
   数值预报原生 1 km 网格，作为模型 c 验证场；
2. 从 WRF 位温和三维风场构造 Ri、风切变、Ellrod、局地脉动和地形强迫等
   物理特征，建立模型 d，并用模型 c 做时序阻塞验证与非线性订正；
3. 仅利用模型 c 的历史场训练非线性时空外推模型 e，递归外推到 06:00，
   对模型 d/e 的 06:00 场分别做带湍流代价的三维 A* 航路规划。

所有场均为 0--1 相对湍流指数，不解释为 EDR 绝对真值。
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import re
import struct
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter
from scipy.spatial import cKDTree
from scipy.stats import pearsonr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


# ============================= 0. 全局配置 =============================

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_OUTPUT = SCRIPT_DIR / "results"

VALIDATION_TIMES = pd.date_range("2025-07-31 02:00", "2025-07-31 05:00", freq="30min")
TARGET_HEIGHTS_M = np.arange(0.0, 2000.0 + 25.0, 50.0, dtype=np.float32)
SURFACE_BOUNDS = (118.2994, 119.3002, 31.2028, 32.6931)  # lon_min, lon_max, lat_min, lat_max
HIGH_RISK_THRESHOLD = 0.65
HIGH_RISK_CLASS_WEIGHT = 3.0

RADAR_CONFIG = {
    "S9250": {
        "directory": "S波段多普勒天气雷达9250站",
        "lat": 32.191,
        "lon": 118.698,
        "alt_m": 138.2,
        "reliability": 1.00,
        "range_scale_km": 90.0,
    },
    "X205": {
        "directory": "X波段多普勒天气雷达205站",
        "lat": 32.514,
        "lon": 118.767,
        "alt_m": 115.0,
        "reliability": 0.85,
        "range_scale_km": 55.0,
    },
    "X206": {
        "directory": "X波段多普勒天气雷达206站",
        "lat": 32.061,
        "lon": 118.540,
        "alt_m": 425.0,
        "reliability": 0.85,
        "range_scale_km": 55.0,
    },
    "X207": {
        "directory": "X波段多普勒天气雷达207站",
        "lat": 32.006,
        "lon": 118.958,
        "alt_m": 295.0,
        "reliability": 0.85,
        "range_scale_km": 55.0,
    },
}

FEATURE_NAMES = [
    "vertical_shear",
    "ri_risk",
    "ellrod",
    "subgrid_tke",
    "vertical_motion",
    "terrain_forcing",
]
MODEL_D_FEATURE_NAMES = FEATURE_NAMES + ["physical_proxy", "proxy_tendency", "height_fraction"]


@dataclass(frozen=True)
class WrfGrid:
    lat: np.ndarray
    lon: np.ndarray
    terrain_m: np.ndarray
    mask: np.ndarray
    y_slice: slice
    x_slice: slice
    dx_m: float
    dy_m: float

    @property
    def shape(self) -> tuple[int, int]:
        return self.lat.shape


@dataclass
class RouteResult:
    nodes: list[tuple[int, int, int]]
    total_cost: float
    reached: bool


# ============================= 1. 通用工具 =============================


def robust_bounds(values: Iterable[float], low_q: float = 0.10, high_q: float = 0.90) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        raise ValueError("Cannot calibrate an empty variable")
    low, high = np.quantile(array, [low_q, high_q])
    if high - low < 1e-8:
        pad = max(abs(float(low)) * 0.1, 1e-3)
        low, high = float(low) - pad, float(high) + pad
    return float(low), float(high)


def robust_unit(values: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip((np.asarray(values, dtype=float) - low) / (high - low), 0.0, 1.0)


def wind_components(direction_deg: np.ndarray, speed_mps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    angle = np.deg2rad(np.asarray(direction_deg, dtype=float))
    speed = np.asarray(speed_mps, dtype=float)
    return -speed * np.sin(angle), -speed * np.cos(angle)


def lonlat_to_xy(
    latitude: np.ndarray, longitude: np.ndarray, lat0: float, lon0: float
) -> tuple[np.ndarray, np.ndarray]:
    metres_per_degree_lat = 111_320.0
    metres_per_degree_lon = metres_per_degree_lat * math.cos(math.radians(lat0))
    x = (np.asarray(longitude, dtype=float) - lon0) * metres_per_degree_lon
    y = (np.asarray(latitude, dtype=float) - lat0) * metres_per_degree_lat
    return x, y


def timestamp_from_name(path: Path) -> pd.Timestamp:
    second_match = re.search(r"(20\d{6})[_.]?(\d{6})", path.name)
    if second_match:
        return pd.to_datetime("".join(second_match.groups()), format="%Y%m%d%H%M%S")
    minute_match = re.search(r"(20\d{6})[_.]?(\d{4})", path.name)
    if minute_match:
        return pd.to_datetime("".join(minute_match.groups()), format="%Y%m%d%H%M")
    raise ValueError(f"Timestamp not found in {path.name}")


def nearest_time_index(times: Sequence[pd.Timestamp], target: pd.Timestamp) -> int:
    delta = np.array([abs((pd.Timestamp(t) - target).total_seconds()) for t in times])
    return int(np.argmin(delta))


def safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 3 or np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return float("nan")
    return float(pearsonr(y_true, y_pred).statistic)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = np.asarray(y_true)[valid]
    y_pred = np.asarray(y_pred)[valid]
    if len(y_true) == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "pearson_r": np.nan, "high_f1": np.nan}
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    truth = y_true >= HIGH_RISK_THRESHOLD
    pred = y_pred >= HIGH_RISK_THRESHOLD
    tp = int(np.sum(truth & pred))
    fp = int(np.sum(~truth & pred))
    fn = int(np.sum(truth & ~pred))
    f1 = float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else np.nan
    return {"n": int(len(y_true)), "mae": mae, "rmse": rmse, "pearson_r": safe_pearson(y_true, y_pred), "high_f1": f1}


def warning_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(y_true) & np.isfinite(probability)
    truth = np.asarray(y_true)[valid] >= HIGH_RISK_THRESHOLD
    probability = np.asarray(probability)[valid]
    predicted = probability >= 0.5
    tp = int(np.sum(truth & predicted))
    fp = int(np.sum(~truth & predicted))
    fn = int(np.sum(truth & ~predicted))
    precision = float(tp / (tp + fp)) if tp + fp else np.nan
    recall = float(tp / (tp + fn)) if tp + fn else np.nan
    f1 = float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else np.nan
    return {
        "n": int(len(truth)),
        "brier": float(np.mean((truth.astype(float) - probability) ** 2)),
        "precision": precision,
        "recall": recall,
        "high_f1": f1,
    }


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": ["Arial Unicode MS", "PingFang SC", "Heiti SC", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 140,
            "savefig.dpi": 180,
        }
    )


# ============================= 2. WRF 与模型 d =============================


def discover_nwp_files(data_root: Path) -> list[tuple[pd.Timestamp, Path]]:
    paths = sorted((data_root / "数值天气预报数据").glob("*.nc"))
    result = [(timestamp_from_name(path), path) for path in paths]
    if not result:
        raise FileNotFoundError(f"No NWP NetCDF files under {data_root / '数值天气预报数据'}")
    return sorted(result)


def load_compact_grid(path: Path) -> WrfGrid:
    with np.load(path) as saved:
        required = {"latitude", "longitude", "terrain_m", "mask", "dx_m", "dy_m", "heights_m"}
        if missing := required - set(saved.files):
            raise ValueError(f"Compact grid is missing variables: {sorted(missing)}")
        heights = np.asarray(saved["heights_m"], dtype=np.float32)
        if heights.shape != TARGET_HEIGHTS_M.shape or not np.allclose(heights, TARGET_HEIGHTS_M):
            raise ValueError("Compact NWP heights do not match the required 0--2000 m / 50 m grid")
        latitude = np.asarray(saved["latitude"], dtype=np.float32)
        longitude = np.asarray(saved["longitude"], dtype=np.float32)
        terrain = np.asarray(saved["terrain_m"], dtype=np.float32)
        mask = np.asarray(saved["mask"], dtype=bool)
        dx_m = float(saved["dx_m"])
        dy_m = float(saved["dy_m"])
    if latitude.shape != longitude.shape or latitude.shape != terrain.shape or latitude.shape != mask.shape:
        raise ValueError("Compact grid latitude, longitude, terrain and mask shapes differ")
    return WrfGrid(
        latitude,
        longitude,
        terrain,
        mask,
        slice(0, latitude.shape[0]),
        slice(0, latitude.shape[1]),
        dx_m,
        dy_m,
    )


def discover_nwp_inputs(data_root: Path) -> tuple[list[tuple[pd.Timestamp, Path]], WrfGrid, str]:
    """Prefer the GitHub-sized compact inputs, with raw files as a local-data fallback."""
    compact_dir = data_root / "nwp"
    compact_files = sorted(compact_dir.glob("nwp_regular_*.npz"))
    compact_grid = compact_dir / "grid.npz"
    if compact_files and compact_grid.exists():
        return (
            sorted((timestamp_from_name(path), path) for path in compact_files),
            load_compact_grid(compact_grid),
            "compact",
        )
    raw_files = discover_nwp_files(data_root)
    return raw_files, build_wrf_grid(raw_files[0][1]), "raw"


def build_wrf_grid(first_nwp: Path, bounds: tuple[float, float, float, float] = SURFACE_BOUNDS) -> WrfGrid:
    lon_min, lon_max, lat_min, lat_max = bounds
    with h5py.File(first_nwp, "r") as handle:
        lat_full = np.asarray(handle["XLAT"][0], dtype=np.float32)
        lon_full = np.asarray(handle["XLONG"][0], dtype=np.float32)
        terrain_full = np.asarray(handle["ter"][0], dtype=np.float32)
        dx_m = float(np.ravel(handle.attrs["DX"])[0])
        dy_m = float(np.ravel(handle.attrs["DY"])[0])
    inside = (lon_full >= lon_min) & (lon_full <= lon_max) & (lat_full >= lat_min) & (lat_full <= lat_max)
    rows, cols = np.where(inside)
    if not len(rows):
        raise ValueError("Requested observation rectangle does not intersect the WRF grid")
    y_slice = slice(max(int(rows.min()) - 1, 0), min(int(rows.max()) + 2, lat_full.shape[0]))
    x_slice = slice(max(int(cols.min()) - 1, 0), min(int(cols.max()) + 2, lat_full.shape[1]))
    lat = lat_full[y_slice, x_slice]
    lon = lon_full[y_slice, x_slice]
    terrain = terrain_full[y_slice, x_slice]
    mask = (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)
    return WrfGrid(lat, lon, terrain, mask, y_slice, x_slice, dx_m, dy_m)


def interpolate_columns(z_agl: np.ndarray, values: np.ndarray, targets_m: np.ndarray) -> np.ndarray:
    """Interpolate a (level, y, x) field to common AGL targets."""
    z = np.asarray(z_agl, dtype=np.float32)
    v = np.asarray(values, dtype=np.float32)
    if z.shape != v.shape or z.ndim != 3:
        raise ValueError("z_agl and values must be same-shaped 3-D arrays")
    result = np.empty((len(targets_m),) + z.shape[1:], dtype=np.float32)
    for out_i, target in enumerate(np.asarray(targets_m, dtype=float)):
        lower = np.sum(z <= target, axis=0) - 1
        lower = np.clip(lower, 0, z.shape[0] - 2)
        upper = lower + 1
        z0 = np.take_along_axis(z, lower[None, :, :], axis=0)[0]
        z1 = np.take_along_axis(z, upper[None, :, :], axis=0)[0]
        v0 = np.take_along_axis(v, lower[None, :, :], axis=0)[0]
        v1 = np.take_along_axis(v, upper[None, :, :], axis=0)[0]
        fraction = np.divide(target - z0, z1 - z0, out=np.zeros_like(z0), where=np.abs(z1 - z0) > 1e-6)
        fraction = np.clip(fraction, 0.0, 1.0)
        result[out_i] = v0 + fraction * (v1 - v0)
    return result


def load_nwp_regular(path: Path, grid: WrfGrid, targets_m: np.ndarray = TARGET_HEIGHTS_M) -> dict[str, np.ndarray]:
    ys, xs = grid.y_slice, grid.x_slice
    with h5py.File(path, "r") as handle:
        z_msl = np.asarray(handle["Z"][0, :, ys, xs], dtype=np.float32)
        theta = np.asarray(handle["TT"][0, :, ys, xs], dtype=np.float32)
        u = np.asarray(handle["Ua"][0, :, ys, xs], dtype=np.float32)
        v = np.asarray(handle["Va"][0, :, ys, xs], dtype=np.float32)
        w = np.asarray(handle["Wa"][0, :, ys, xs], dtype=np.float32)
    z_agl = z_msl - grid.terrain_m[None, :, :]
    return {
        "theta": interpolate_columns(z_agl, theta, targets_m),
        "u": interpolate_columns(z_agl, u, targets_m),
        "v": interpolate_columns(z_agl, v, targets_m),
        "w": interpolate_columns(z_agl, w, targets_m),
    }


def load_compact_nwp_regular(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as saved:
        required = {"heights_m", "theta", "u", "v", "w"}
        if missing := required - set(saved.files):
            raise ValueError(f"Compact NWP file {path.name} is missing variables: {sorted(missing)}")
        heights = np.asarray(saved["heights_m"], dtype=np.float32)
        if heights.shape != TARGET_HEIGHTS_M.shape or not np.allclose(heights, TARGET_HEIGHTS_M):
            raise ValueError(f"Compact NWP file {path.name} has incompatible height levels")
        return {name: np.asarray(saved[name], dtype=np.float32) for name in ("theta", "u", "v", "w")}


def compute_nwp_raw_features(regular: dict[str, np.ndarray], grid: WrfGrid) -> dict[str, np.ndarray]:
    theta, u, v, w = (regular[name].astype(float) for name in ("theta", "u", "v", "w"))
    dz = float(TARGET_HEIGHTS_M[1] - TARGET_HEIGHTS_M[0])
    dtheta_dz = np.gradient(theta, dz, axis=0, edge_order=2)
    du_dz = np.gradient(u, dz, axis=0, edge_order=2)
    dv_dz = np.gradient(v, dz, axis=0, edge_order=2)
    shear = np.hypot(du_dz, dv_dz)
    n2 = 9.80665 / np.maximum(theta, 180.0) * dtheta_dz
    ri = n2 / (shear**2 + 1e-6)
    # Ri < 0.25 的风切变不稳定层风险上升；用 logistic 避免无界尖峰。
    ri_risk = 1.0 / (1.0 + np.exp(np.clip((ri - 0.25) / 0.25, -30.0, 30.0)))

    dudx = np.gradient(u, grid.dx_m, axis=2, edge_order=2)
    dudy = np.gradient(u, grid.dy_m, axis=1, edge_order=2)
    dvdx = np.gradient(v, grid.dx_m, axis=2, edge_order=2)
    dvdy = np.gradient(v, grid.dy_m, axis=1, edge_order=2)
    deformation = np.hypot(dudx - dvdy, dvdx + dudy)
    divergence = np.abs(dudx + dvdy)
    ellrod = shear * (deformation + divergence)

    # 3x3x3 局地方差对应数值模式可解尺度的风场脉动代理量。
    local_energy = np.zeros_like(u)
    for component in (u, v, w):
        mean = uniform_filter(component, size=(3, 3, 3), mode="nearest")
        mean_square = uniform_filter(component**2, size=(3, 3, 3), mode="nearest")
        local_energy += np.maximum(mean_square - mean**2, 0.0)
    subgrid_tke = 0.5 * local_energy

    terrain_slope = np.hypot(
        np.gradient(grid.terrain_m.astype(float), grid.dx_m, axis=1, edge_order=2),
        np.gradient(grid.terrain_m.astype(float), grid.dy_m, axis=0, edge_order=2),
    )
    terrain_forcing = terrain_slope[None, :, :] * np.exp(-TARGET_HEIGHTS_M[:, None, None] / 400.0)
    features = {
        "vertical_shear": shear,
        "ri_risk": ri_risk,
        "ellrod": ellrod,
        "subgrid_tke": subgrid_tke,
        "vertical_motion": np.abs(w),
        "terrain_forcing": terrain_forcing,
    }
    for name in features:
        array = np.asarray(features[name], dtype=np.float32)
        array[:, ~grid.mask] = np.nan
        features[name] = array
    return features


def compute_feature_scaler(
    raw_features: list[dict[str, np.ndarray]], validation_indices: Sequence[int], grid: WrfGrid
) -> dict[str, tuple[float, float]]:
    scaler: dict[str, tuple[float, float]] = {}
    rng = np.random.default_rng(2025)
    for name in FEATURE_NAMES:
        if name == "ri_risk":
            scaler[name] = (0.0, 1.0)
            continue
        samples: list[np.ndarray] = []
        for time_i in validation_indices:
            values = raw_features[time_i][name][:, grid.mask]
            values = values[np.isfinite(values)]
            if values.size > 80_000:
                values = rng.choice(values, size=80_000, replace=False)
            samples.append(values)
        scaler[name] = robust_bounds(np.concatenate(samples), 0.10, 0.90)
    return scaler


def normalize_nwp_features(
    raw: dict[str, np.ndarray], scaler: dict[str, tuple[float, float]], grid: WrfGrid
) -> tuple[np.ndarray, np.ndarray]:
    normalized = []
    for name in FEATURE_NAMES:
        low, high = scaler[name]
        unit = robust_unit(raw[name], low, high).astype(np.float32)
        unit[:, ~grid.mask] = np.nan
        normalized.append(unit)
    cube = np.stack(normalized, axis=0)
    # 权重与前两问一致：风切变/稳定度和局地速度差是主体，垂直运动和地形为辅。
    weights = np.array([0.25, 0.20, 0.20, 0.15, 0.10, 0.10], dtype=np.float32)
    proxy = np.nansum(cube * weights[:, None, None, None], axis=0)
    proxy[:, ~grid.mask] = np.nan
    return cube, proxy.astype(np.float32)


def temporal_derivative(fields: np.ndarray, times: Sequence[pd.Timestamp]) -> np.ndarray:
    seconds = np.array([(pd.Timestamp(t) - pd.Timestamp(times[0])).total_seconds() for t in times], dtype=float)
    result = np.empty_like(fields, dtype=np.float32)
    for i in range(len(times)):
        if i == 0:
            result[i] = (fields[1] - fields[0]) / max(seconds[1] - seconds[0], 1.0)
        elif i == len(times) - 1:
            result[i] = (fields[-1] - fields[-2]) / max(seconds[-1] - seconds[-2], 1.0)
        else:
            result[i] = (fields[i + 1] - fields[i - 1]) / max(seconds[i + 1] - seconds[i - 1], 1.0)
    # 换成“每 30 min 指数变化”，便于模型数值尺度稳定。
    return result * 1800.0


# ============================= 3. 模型 c 观测验证场 =============================


def read_surface_file(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, sep=r"\s+")
    required = {"Lat", "Lon", "Alti", "WIN_D_Avg_2mi", "WIN_S_Avg_2mi"}
    if missing := required - set(table.columns):
        raise ValueError(f"Surface file missing columns {sorted(missing)}: {path}")
    for name in required:
        table[name] = pd.to_numeric(table[name], errors="coerce")
    table["station_key"] = table["Lat"].round(4).astype(str) + "_" + table["Lon"].round(4).astype(str)
    valid = (
        table["Lat"].between(20, 45)
        & table["Lon"].between(100, 130)
        & table["WIN_D_Avg_2mi"].between(0, 360)
        & table["WIN_S_Avg_2mi"].between(0, 60)
    )
    table["surface_valid"] = valid
    u, v = wind_components(table["WIN_D_Avg_2mi"], table["WIN_S_Avg_2mi"])
    table["u_ms"], table["v_ms"] = u, v
    return table


def add_surface_proxy(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    valid = result["surface_valid"].to_numpy(bool)
    result["surface_proxy_raw"] = np.nan
    result["surface_uncertainty"] = np.nan
    if np.sum(valid) < 3:
        return result
    lat0, lon0 = float(result.loc[valid, "Lat"].mean()), float(result.loc[valid, "Lon"].mean())
    x, y = lonlat_to_xy(result.loc[valid, "Lat"], result.loc[valid, "Lon"], lat0, lon0)
    points = np.column_stack([x, y])
    tree = cKDTree(points)
    distances, indices = tree.query(points, k=min(7, len(points)))
    if distances.ndim == 1:
        distances, indices = distances[:, None], indices[:, None]
    distances, indices = distances[:, 1:], indices[:, 1:]
    u = result.loc[valid, "u_ms"].to_numpy(float)
    v = result.loc[valid, "v_ms"].to_numpy(float)
    weights = 1.0 / (distances**2 + 5000.0**2)
    weights[distances > 30_000.0] = 0.0
    vector_difference2 = (u[indices] - u[:, None]) ** 2 + (v[indices] - v[:, None]) ** 2
    weight_sum = weights.sum(axis=1)
    variability = np.sqrt(
        np.divide((weights * vector_difference2).sum(axis=1), weight_sum, out=np.zeros_like(weight_sum), where=weight_sum > 0)
    )
    raw = np.hypot(result.loc[valid, "WIN_S_Avg_2mi"].to_numpy(float) / 5.0, variability / 3.0)
    result.loc[valid, "surface_proxy_raw"] = raw
    neighbour_count = np.sum(weights > 0, axis=1)
    result.loc[valid, "surface_uncertainty"] = np.clip(0.12 + 0.20 / np.sqrt(np.maximum(neighbour_count, 1)), 0.12, 0.35)
    return result


def discover_surface_tables(data_root: Path) -> dict[pd.Timestamp, pd.DataFrame]:
    tables: dict[pd.Timestamp, pd.DataFrame] = {}
    for path in sorted((data_root / "地面自动气象站数据").glob("*.txt")):
        time = timestamp_from_name(path)
        tables[time] = add_surface_proxy(read_surface_file(path))
    missing = [time for time in VALIDATION_TIMES if time not in tables]
    if missing:
        raise FileNotFoundError(f"Missing surface observations at {missing}")
    return tables


def collapse_surface_locations(table: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate coordinates while retaining conservative uncertainty."""
    return (
        table.groupby("station_key", as_index=False)
        .agg(
            Lat=("Lat", "mean"),
            Lon=("Lon", "mean"),
            Alti=("Alti", "mean"),
            surface_proxy_raw=("surface_proxy_raw", "mean"),
            surface_uncertainty=("surface_uncertainty", "max"),
            surface_valid=("surface_valid", "max"),
        )
        .sort_values("station_key")
        .reset_index(drop=True)
    )


def parse_robs_file(path: Path, station_name: str) -> pd.DataFrame:
    lines = path.read_text(encoding="ascii", errors="replace").splitlines()
    if len(lines) < 4 or not lines[0].startswith("WNDROBS"):
        raise ValueError(f"Unsupported ROBS file: {path}")
    header = lines[1].split()
    station_id, longitude, latitude, site_altitude, timezone, timestamp = header[:6]
    start = lines.index("ROBS") + 1
    rows = []
    for line in lines[start:]:
        if line.strip() == "NNNN":
            break
        fields = line.split()
        if len(fields) != 7 or any("/" in value for value in fields):
            continue
        try:
            height, direction, speed, vertical_speed = map(float, fields[:4])
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
            }
        )
    profile = pd.DataFrame(rows).sort_values("height_m").reset_index(drop=True)
    if profile.empty:
        raise ValueError(f"No valid ROBS rows in {path}")
    u, v = wind_components(profile["wind_dir_deg"], profile["wind_speed_mps"])
    heights = profile["height_m"].to_numpy(float)
    u_smooth = pd.Series(u).rolling(5, center=True, min_periods=1).mean().to_numpy(float)
    v_smooth = pd.Series(v).rolling(5, center=True, min_periods=1).mean().to_numpy(float)
    profile["model_b_shear_per_s"] = np.hypot(np.gradient(u_smooth, heights), np.gradient(v_smooth, heights))
    return profile


def parse_wpr_rad_low_mode(path: Path, station_name: str) -> pd.DataFrame:
    lines = path.read_text(encoding="ascii", errors="replace").splitlines()
    header = lines[1].split()
    station_id, longitude, latitude, site_altitude, timezone = header[:5]
    rows, block_count, beam = [], 0, ""
    for line in lines[2:]:
        stripped = line.strip()
        if stripped.startswith("RAD "):
            block_count += 1
            if block_count > 5:
                break
            beam = stripped.split(maxsplit=1)[1]
            continue
        fields = stripped.split()
        if not beam or len(fields) != 4 or any("/" in value for value in fields):
            continue
        try:
            height, radial_velocity, snr, spectrum_width = map(float, fields)
        except ValueError:
            continue
        if 0 <= height <= 4000 and abs(radial_velocity) < 100 and 0 <= snr < 100 and 0 <= spectrum_width < 20:
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
                    "spectrum_width_mps": spectrum_width,
                }
            )
    raw = pd.DataFrame(rows)
    if raw.empty:
        raise ValueError(f"No valid low-mode RAD rows in {path}")

    def mad(series: pd.Series) -> float:
        values = series.to_numpy(float)
        return float(np.median(np.abs(values - np.median(values))))

    return (
        raw.groupby(
            ["station_name", "station_id", "longitude_deg", "latitude_deg", "site_altitude_m", "timezone", "height_m"],
            as_index=False,
        )
        .agg(
            wpr_spectrum_width_mps=("spectrum_width_mps", "median"),
            wpr_spectrum_width_mad=("spectrum_width_mps", mad),
            valid_beams=("beam", "nunique"),
        )
        .sort_values("height_m")
        .reset_index(drop=True)
    )


def discover_wpr_profiles(data_root: Path) -> dict[pd.Timestamp, list[pd.DataFrame]]:
    profiles: dict[pd.Timestamp, list[pd.DataFrame]] = {time: [] for time in VALIDATION_TIMES}
    for station in ("a", "b", "c"):
        directory = data_root / f"风廓线雷达{station}站点"
        robs_by_time = {timestamp_from_name(path): path for path in directory.glob("*_ROBS.TXT")}
        rad_by_time = {timestamp_from_name(path): path for path in directory.glob("*_RAD.TXT")}
        for time in VALIDATION_TIMES:
            if time not in robs_by_time or time not in rad_by_time:
                raise FileNotFoundError(f"Missing WPR {station} at {time}")
            robs = parse_robs_file(robs_by_time[time], station)
            rad = parse_wpr_rad_low_mode(rad_by_time[time], station)
            merged = pd.merge(
                robs,
                rad[["height_m", "wpr_spectrum_width_mps", "wpr_spectrum_width_mad", "valid_beams"]],
                on="height_m",
                how="inner",
            )
            merged["dynamic_proxy_raw"] = np.hypot(
                merged["model_b_shear_per_s"] / 0.01,
                np.abs(merged["vertical_speed_mps"]) / 3.0,
            )
            profiles[time].append(merged)
    return profiles


def prepare_wpr_cubes(
    profiles: dict[pd.Timestamp, list[pd.DataFrame]], targets_m: np.ndarray = TARGET_HEIGHTS_M
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, dict[str, tuple[float, float]]]:
    all_profiles = pd.concat([profile for rows in profiles.values() for profile in rows], ignore_index=True)
    low_height = all_profiles[all_profiles["height_m"].between(0, float(targets_m.max()))]
    sw_bounds = robust_bounds(low_height["wpr_spectrum_width_mps"], 0.10, 0.90)
    dyn_bounds = robust_bounds(low_height["dynamic_proxy_raw"], 0.10, 0.90)
    times = list(VALIDATION_TIMES)
    first_profiles = profiles[times[0]]
    stations = pd.DataFrame(
        [
            {
                "station_name": p["station_name"].iloc[0],
                "latitude_deg": p["latitude_deg"].iloc[0],
                "longitude_deg": p["longitude_deg"].iloc[0],
                "site_altitude_m": p["site_altitude_m"].iloc[0],
            }
            for p in first_profiles
        ]
    ).sort_values("station_name").reset_index(drop=True)
    ti = np.full((len(times), len(stations), len(targets_m)), np.nan, dtype=np.float32)
    unc = np.full_like(ti, np.nan)
    cov = np.zeros_like(ti)
    for time_i, time in enumerate(times):
        by_station = {p["station_name"].iloc[0]: p for p in profiles[time]}
        for station_i, station_name in enumerate(stations["station_name"]):
            profile = by_station[station_name].sort_values("height_m").copy()
            spectrum = robust_unit(profile["wpr_spectrum_width_mps"], *sw_bounds)
            dynamic = robust_unit(profile["dynamic_proxy_raw"], *dyn_bounds)
            value = 0.75 * spectrum + 0.25 * dynamic
            scale = max(sw_bounds[1] - sw_bounds[0], 0.5)
            uncertainty = np.clip(
                0.10
                + 0.30 * profile["wpr_spectrum_width_mad"].to_numpy(float) / scale
                + 0.15 * (5 - profile["valid_beams"].to_numpy(float)) / 5,
                0.08,
                0.70,
            )
            heights = profile["height_m"].to_numpy(float)
            inside = (targets_m >= heights.min()) & (targets_m <= heights.max())
            ti[time_i, station_i, inside] = np.interp(targets_m[inside], heights, value)
            unc[time_i, station_i, inside] = np.interp(targets_m[inside], heights, uncertainty)
            gap = np.min(np.abs(targets_m[inside, None] - heights[None, :]), axis=1)
            cov[time_i, station_i, inside] = np.exp(-0.5 * (gap / 90.0) ** 2)
    return ti, unc, cov, stations, {"spectrum_width": sw_bounds, "dynamic_proxy": dyn_bounds}


def group_radar_scans(directory: Path) -> dict[pd.Timestamp, list[Path]]:
    groups: dict[pd.Timestamp, list[Path]] = {}
    for path in directory.glob("*-SW.csv"):
        groups.setdefault(timestamp_from_name(path), []).append(path)
    return {time: sorted(paths) for time, paths in groups.items()}


def sample_polar_csv(path: Path, target_azimuth_deg: np.ndarray, target_range_km: np.ndarray) -> np.ndarray:
    """Read only columns needed for station sampling from a polar radar CSV."""
    with path.open("r", encoding="ascii", errors="replace") as handle:
        header = handle.readline().strip()
    fields = [field.strip() for field in header.split(",")]
    ranges = np.array([float(value) for value in fields[1:] if value], dtype=float)
    range_index = np.searchsorted(ranges, target_range_km)
    range_index = np.clip(range_index, 0, len(ranges) - 1)
    previous = np.maximum(range_index - 1, 0)
    choose_previous = np.abs(ranges[previous] - target_range_km) < np.abs(ranges[range_index] - target_range_km)
    range_index = np.where(choose_previous, previous, range_index)
    positions = sorted({0, *[int(i) + 1 for i in range_index]})
    frame = pd.read_csv(path, usecols=positions, dtype=np.float32)
    data = frame.to_numpy(dtype=np.float32)
    loaded_column = {position: i for i, position in enumerate(positions)}
    azimuth = data[:, loaded_column[0]].astype(float)
    circular = np.abs(((azimuth[:, None] - target_azimuth_deg[None, :] + 180.0) % 360.0) - 180.0)
    row_index = np.argmin(circular, axis=0)
    column_index = np.array([loaded_column[int(i) + 1] for i in range_index])
    values = data[row_index, column_index].astype(float)
    values[(values < 0.0) | (values >= 20.0) | (circular[row_index, np.arange(len(values))] > 2.0)] = np.nan
    values[target_range_km > ranges.max() + 1e-6] = np.nan
    return values.astype(np.float32)


def radar_geometry(stations: pd.DataFrame, config: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    lat0 = float(config["lat"])
    x, y = lonlat_to_xy(stations["Lat"], stations["Lon"], lat0, float(config["lon"]))
    ranges_m = np.hypot(x, y)
    azimuth = (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0
    return azimuth, ranges_m


def local_velocity_gradient(values: np.ndarray, neighbour_dist: np.ndarray, neighbour_idx: np.ndarray) -> np.ndarray:
    neighbour = values[neighbour_idx]
    valid = np.isfinite(neighbour) & np.isfinite(values[:, None]) & (neighbour_dist > 0) & (neighbour_dist <= 30_000)
    gradient2 = np.divide(
        (neighbour - values[:, None]) ** 2,
        neighbour_dist**2 + 1000.0**2,
        out=np.zeros_like(neighbour_dist),
        where=valid,
    )
    count = valid.sum(axis=1)
    return np.sqrt(np.divide((gradient2 * valid).sum(axis=1), count, out=np.zeros(len(values)), where=count > 0))


def elevation_for_file(path: Path, angles_path: Path) -> float:
    match = re.search(r"-el(\d+)-SW\.csv$", path.name)
    if not match:
        raise ValueError(f"Elevation code missing in {path.name}")
    code = int(match.group(1))
    angles = np.loadtxt(angles_path, dtype=float)
    if code >= len(angles):
        raise IndexError(f"Elevation index {code} exceeds {angles_path}")
    return float(angles[code])


def prepare_radar_samples(
    data_root: Path,
    stations: pd.DataFrame,
    cache_path: Path | None = None,
) -> tuple[list[dict[str, object]], dict[str, tuple[float, float]]]:
    """Sample all 02--05 radar scans at surface station positions.

    Returned records hold corrected spectrum width on each radar elevation.  The
    compact cache avoids rereading about one gigabyte of text on repeat runs.
    """
    if cache_path and cache_path.exists():
        payload = np.load(cache_path, allow_pickle=True)
        records = list(payload["records"])
        bounds = dict(payload["bounds"].item())
        return records, bounds

    lat0, lon0 = float(stations["Lat"].mean()), float(stations["Lon"].mean())
    sx, sy = lonlat_to_xy(stations["Lat"], stations["Lon"], lat0, lon0)
    tree = cKDTree(np.column_stack([sx, sy]))
    neighbour_dist, neighbour_idx = tree.query(np.column_stack([sx, sy]), k=min(7, len(stations)))
    neighbour_dist, neighbour_idx = neighbour_dist[:, 1:], neighbour_idx[:, 1:]
    records: list[dict[str, object]] = []
    calibration: dict[str, list[np.ndarray]] = {name: [] for name in RADAR_CONFIG}
    for radar, config in RADAR_CONFIG.items():
        directory = data_root / str(config["directory"])
        scan_groups = group_radar_scans(directory)
        scan_times = sorted(scan_groups)
        azimuth, ranges_m = radar_geometry(stations, config)
        for target in VALIDATION_TIMES:
            nearest = scan_times[nearest_time_index(scan_times, target)]
            if abs((nearest - target).total_seconds()) > 7 * 60:
                continue
            sw_files = scan_groups[nearest]
            angle_candidates = sorted(directory.glob(f"*{nearest.strftime('%Y%m%d%H%M%S')}*elev_angles.txt"))
            if not angle_candidates:
                continue
            angles_path = angle_candidates[0]
            for sw_path in sw_files:
                vel_path = Path(str(sw_path).replace("-SW.csv", "-VEL.csv"))
                if not vel_path.exists():
                    continue
                sw = sample_polar_csv(sw_path, azimuth, ranges_m / 1000.0).astype(float)
                vel = sample_polar_csv(vel_path, azimuth, ranges_m / 1000.0).astype(float)
                gradient = local_velocity_gradient(vel, neighbour_dist, neighbour_idx)
                beam_diameter = ranges_m * math.radians(1.0)
                shear_broadening = beam_diameter * gradient / math.sqrt(12.0)
                positive = sw[np.isfinite(sw) & (sw > 0)]
                noise = float(np.clip(0.25 * np.median(positive), 0.10, 0.50)) if positive.size else 0.20
                corrected = np.sqrt(np.maximum(sw**2 - shear_broadening**2 - noise**2, 0.0))
                elevation = elevation_for_file(sw_path, angles_path)
                effective_earth = 4.0 / 3.0 * 6_371_000.0
                beam_msl = (
                    float(config["alt_m"])
                    + ranges_m * math.sin(math.radians(elevation))
                    + ranges_m**2 / (2.0 * effective_earth)
                )
                beam_agl = beam_msl - stations["Alti"].to_numpy(float)
                record = {
                    "target_time": target,
                    "scan_time": nearest,
                    "radar": radar,
                    "elevation_deg": elevation,
                    "range_m": ranges_m.astype(np.float32),
                    "beam_agl_m": beam_agl.astype(np.float32),
                    "corrected_sw_mps": corrected.astype(np.float32),
                    "noise_floor_mps": noise,
                }
                records.append(record)
                calibration[radar].append(corrected[np.isfinite(corrected)])
    bounds = {
        radar: robust_bounds(np.concatenate(values), 0.10, 0.90)
        for radar, values in calibration.items()
        if values and sum(len(value) for value in values) > 0
    }
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, records=np.array(records, dtype=object), bounds=np.array(bounds, dtype=object))
    return records, bounds


def radar_station_profiles(
    records: list[dict[str, object]],
    bounds: dict[str, tuple[float, float]],
    n_station: int,
    targets_m: np.ndarray = TARGET_HEIGHTS_M,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = list(VALIDATION_TIMES)
    ti = np.full((len(times), n_station, len(targets_m)), np.nan, dtype=np.float32)
    unc = np.full_like(ti, np.nan)
    cov = np.zeros_like(ti)
    for time_i, target in enumerate(times):
        selected = [record for record in records if pd.Timestamp(record["target_time"]) == target]
        if not selected:
            continue
        source_values, source_weights = [], []
        for record in selected:
            radar = str(record["radar"])
            if radar not in bounds:
                continue
            value = robust_unit(np.asarray(record["corrected_sw_mps"], dtype=float), *bounds[radar])
            beam = np.asarray(record["beam_agl_m"], dtype=float)
            ranges = np.asarray(record["range_m"], dtype=float)
            dz = targets_m[None, :] - beam[:, None]
            vertical = np.exp(-0.5 * (dz / 250.0) ** 2)
            vertical[np.abs(dz) > 600.0] = 0.0
            config = RADAR_CONFIG[radar]
            range_weight = np.exp(-0.5 * (ranges / (1000.0 * float(config["range_scale_km"]))) ** 2)
            time_weight = math.exp(
                -0.5 * ((pd.Timestamp(record["scan_time"]) - target).total_seconds() / 360.0) ** 2
            )
            weight = vertical * range_weight[:, None] * time_weight * float(config["reliability"])
            weight[~np.isfinite(value), :] = 0.0
            source_values.append(np.broadcast_to(value[:, None], weight.shape))
            source_weights.append(weight)
        if not source_values:
            continue
        values = np.stack(source_values)
        weights = np.stack(source_weights)
        weight_sum = weights.sum(axis=0)
        mean = np.divide(
            np.nansum(weights * values, axis=0),
            weight_sum,
            out=np.full(weight_sum.shape, np.nan),
            where=weight_sum > 0,
        )
        variance = np.divide(
            np.nansum(weights * (values - mean[None, :, :]) ** 2, axis=0),
            weight_sum,
            out=np.full(weight_sum.shape, np.nan),
            where=weight_sum > 0,
        )
        ti[time_i] = mean.astype(np.float32)
        unc[time_i] = np.sqrt(variance + 0.15**2).astype(np.float32)
        cov[time_i] = (1.0 - np.exp(-weight_sum / 2.0)).astype(np.float32)
    return ti, unc, cov


def query_neighbours(
    point_lat: np.ndarray,
    point_lon: np.ndarray,
    grid: WrfGrid,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    lat0, lon0 = float(grid.lat.mean()), float(grid.lon.mean())
    px, py = lonlat_to_xy(point_lat, point_lon, lat0, lon0)
    gx, gy = lonlat_to_xy(grid.lat.ravel(), grid.lon.ravel(), lat0, lon0)
    tree = cKDTree(np.column_stack([px, py]))
    distance, index = tree.query(np.column_stack([gx, gy]), k=min(k, len(px)))
    if distance.ndim == 1:
        distance, index = distance[:, None], index[:, None]
    return distance.astype(np.float32), index.astype(np.int32)


def idw_grid(
    values: np.ndarray,
    point_uncertainty: np.ndarray,
    point_confidence: np.ndarray,
    distances: np.ndarray,
    indices: np.ndarray,
    grid: WrfGrid,
    radius_m: float,
    d0_m: float = 3000.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    neighbour_value = np.asarray(values, dtype=float)[indices]
    neighbour_unc = np.asarray(point_uncertainty, dtype=float)[indices]
    neighbour_conf = np.asarray(point_confidence, dtype=float)[indices]
    valid = (
        np.isfinite(neighbour_value)
        & np.isfinite(neighbour_unc)
        & (neighbour_conf > 0)
        & (distances <= radius_m)
    )
    spatial = np.exp(-0.5 * (distances / radius_m) ** 2) / (distances**2 + d0_m**2)
    weights = np.where(valid, spatial * neighbour_conf / (neighbour_unc**2 + 0.10**2), 0.0)
    weight_sum = weights.sum(axis=1)
    mean = np.divide(
        (weights * np.where(valid, neighbour_value, 0.0)).sum(axis=1),
        weight_sum,
        out=np.full(len(weight_sum), np.nan),
        where=weight_sum > 0,
    )
    spread = np.divide(
        (weights * np.where(valid, (neighbour_value - mean[:, None]) ** 2, 0.0)).sum(axis=1),
        weight_sum,
        out=np.full(len(weight_sum), np.nan),
        where=weight_sum > 0,
    )
    measurement = np.divide(
        (weights * np.where(valid, neighbour_unc**2, 0.0)).sum(axis=1),
        weight_sum,
        out=np.full(len(weight_sum), np.nan),
        where=weight_sum > 0,
    )
    count = valid.sum(axis=1)
    nearest = np.min(np.where(valid, distances, np.inf), axis=1)
    local_conf = np.divide(
        (weights * np.where(valid, neighbour_conf, 0.0)).sum(axis=1),
        weight_sum,
        out=np.zeros(len(weight_sum)),
        where=weight_sum > 0,
    )
    coverage = np.exp(-0.5 * (nearest / radius_m) ** 2) * (1.0 - np.exp(-count / 3.0)) * local_conf
    uncertainty = np.sqrt(measurement + spread + (0.20 * np.minimum(nearest / radius_m, 1.0)) ** 2)
    shape = grid.shape
    field = mean.reshape(shape).astype(np.float32)
    coverage = np.clip(coverage.reshape(shape), 0, 1).astype(np.float32)
    uncertainty = uncertainty.reshape(shape).astype(np.float32)
    field[~grid.mask] = np.nan
    coverage[~grid.mask] = 0.0
    uncertainty[~grid.mask] = np.nan
    return field, coverage, uncertainty


def fuse_fields(
    fields: Sequence[np.ndarray],
    coverages: Sequence[np.ndarray],
    uncertainties: Sequence[np.ndarray],
    base_weights: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.stack(fields).astype(float)
    cov = np.stack(coverages).astype(float)
    unc = np.stack(uncertainties).astype(float)
    weights = np.zeros_like(values)
    for source_i, base in enumerate(base_weights):
        finite = np.isfinite(values[source_i]) & np.isfinite(unc[source_i]) & (cov[source_i] > 0)
        weights[source_i, finite] = base * cov[source_i, finite] / (unc[source_i, finite] ** 2 + 0.10**2)
    weight_sum = weights.sum(axis=0)
    fused = np.divide(
        np.nansum(weights * values, axis=0),
        weight_sum,
        out=np.full(weight_sum.shape, np.nan),
        where=weight_sum > 0,
    )
    disagreement = np.divide(
        np.nansum(weights * (values - fused[None, ...]) ** 2, axis=0),
        weight_sum,
        out=np.full(weight_sum.shape, np.nan),
        where=weight_sum > 0,
    )
    total_cov = 1.0 - np.prod(1.0 - np.clip(cov, 0, 1), axis=0)
    total_unc = np.sqrt(
        np.divide(1.0, weight_sum, out=np.full_like(weight_sum, np.nan), where=weight_sum > 0)
        + disagreement
    )
    unsupported = total_cov < 0.05
    fused[unsupported] = np.nan
    total_unc[unsupported] = np.nan
    return fused.astype(np.float32), total_cov.astype(np.float32), total_unc.astype(np.float32)


def build_observed_c(
    data_root: Path,
    grid: WrfGrid,
    output_dir: Path,
    include_radar: bool = True,
    use_cache: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    cache = output_dir / "cache" / ("observed_c_with_radar.npz" if include_radar else "observed_c_no_radar.npz")
    if use_cache and cache.exists():
        saved = np.load(cache, allow_pickle=True)
        return saved["field"], saved["coverage"], saved["uncertainty"], dict(saved["metadata"].item())

    surface_tables = discover_surface_tables(data_root)
    all_surface_raw = np.concatenate(
        [table["surface_proxy_raw"].dropna().to_numpy(float) for time, table in surface_tables.items() if time in VALIDATION_TIMES]
    )
    surface_bounds = robust_bounds(all_surface_raw, 0.10, 0.90)
    master = collapse_surface_locations(surface_tables[VALIDATION_TIMES[0]])
    surface_distance, surface_index = query_neighbours(master["Lat"], master["Lon"], grid, k=8)

    wpr_profiles = discover_wpr_profiles(data_root)
    wpr_ti, wpr_unc, wpr_cov, wpr_stations, wpr_bounds = prepare_wpr_cubes(wpr_profiles)
    wpr_distance, wpr_index = query_neighbours(
        wpr_stations["latitude_deg"], wpr_stations["longitude_deg"], grid, k=3
    )

    radar_ti = np.full((len(VALIDATION_TIMES), len(master), len(TARGET_HEIGHTS_M)), np.nan, dtype=np.float32)
    radar_unc = np.full_like(radar_ti, np.nan)
    radar_cov = np.zeros_like(radar_ti)
    radar_bounds: dict[str, tuple[float, float]] = {}
    if include_radar:
        radar_cache = output_dir / "cache" / "radar_station_samples.npz" if use_cache else None
        records, radar_bounds = prepare_radar_samples(data_root, master, radar_cache)
        radar_ti, radar_unc, radar_cov = radar_station_profiles(records, radar_bounds, len(master))

    field = np.full((len(VALIDATION_TIMES), len(TARGET_HEIGHTS_M)) + grid.shape, np.nan, dtype=np.float32)
    coverage = np.zeros_like(field)
    uncertainty = np.full_like(field, np.nan)
    for time_i, time in enumerate(VALIDATION_TIMES):
        surface = (
            collapse_surface_locations(surface_tables[time])
            .set_index("station_key")
            .reindex(master["station_key"])
            .reset_index()
        )
        surface_value = robust_unit(surface["surface_proxy_raw"].to_numpy(float), *surface_bounds)
        surface_conf = surface["surface_valid"].fillna(False).to_numpy(float)
        surface_unc_value = surface["surface_uncertainty"].to_numpy(float)
        surface_plane, surface_cov0, surface_unc_plane = idw_grid(
            surface_value,
            surface_unc_value,
            surface_conf,
            surface_distance,
            surface_index,
            grid,
            radius_m=30_000.0,
        )
        for z_i, height in enumerate(TARGET_HEIGHTS_M):
            decay = float(math.exp(-height / 300.0))
            source_fields, source_covs, source_uncs, source_weights = [], [], [], []
            if include_radar:
                radar_plane, radar_cov_plane, radar_unc_plane = idw_grid(
                    radar_ti[time_i, :, z_i],
                    radar_unc[time_i, :, z_i],
                    radar_cov[time_i, :, z_i],
                    surface_distance,
                    surface_index,
                    grid,
                    radius_m=40_000.0,
                )
                source_fields.append(radar_plane)
                source_covs.append(radar_cov_plane)
                source_uncs.append(radar_unc_plane)
                source_weights.append(1.00)
            wpr_plane, wpr_cov_plane, wpr_unc_plane = idw_grid(
                wpr_ti[time_i, :, z_i],
                wpr_unc[time_i, :, z_i],
                wpr_cov[time_i, :, z_i],
                wpr_distance,
                wpr_index,
                grid,
                radius_m=70_000.0,
                d0_m=5000.0,
            )
            source_fields.extend([wpr_plane, surface_plane])
            source_covs.extend([wpr_cov_plane, surface_cov0 * decay])
            source_uncs.extend([wpr_unc_plane, surface_unc_plane + 0.15 * (1.0 - decay)])
            source_weights.extend([0.85, 0.65])
            field[time_i, z_i], coverage[time_i, z_i], uncertainty[time_i, z_i] = fuse_fields(
                source_fields, source_covs, source_uncs, source_weights
            )
    metadata = {
        "times": [time.isoformat() for time in VALIDATION_TIMES],
        "surface_calibration": surface_bounds,
        "wpr_calibration": wpr_bounds,
        "radar_calibration": radar_bounds,
        "include_radar": include_radar,
        "description": "Model-c field sampled on the native WRF horizontal grid",
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, field=field, coverage=coverage, uncertainty=uncertainty, metadata=np.array(metadata, dtype=object))
    return field, coverage, uncertainty, metadata


def load_compact_observed_c(
    data_root: Path, grid: WrfGrid
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]] | None:
    """Load the portable model-c input bundled in data/, if present."""
    path = data_root / "model_c_validation_1km_50m.npz"
    if not path.exists():
        return None
    with np.load(path) as saved:
        required = {
            "times",
            "heights_m",
            "latitude",
            "longitude",
            "mask",
            "turbulence_index",
            "coverage",
            "uncertainty",
        }
        if missing := required - set(saved.files):
            raise ValueError(f"Compact model-c input is missing variables: {sorted(missing)}")
        times = [pd.Timestamp(str(value)) for value in saved["times"]]
        heights = np.asarray(saved["heights_m"], dtype=np.float32)
        latitude = np.asarray(saved["latitude"], dtype=np.float32)
        longitude = np.asarray(saved["longitude"], dtype=np.float32)
        mask = np.asarray(saved["mask"], dtype=bool)
        field = np.asarray(saved["turbulence_index"], dtype=np.float32)
        coverage = np.asarray(saved["coverage"], dtype=np.float32)
        uncertainty = np.asarray(saved["uncertainty"], dtype=np.float32)
    if times != list(VALIDATION_TIMES):
        raise ValueError("Compact model-c times do not match the 02:00--05:00 validation window")
    if heights.shape != TARGET_HEIGHTS_M.shape or not np.allclose(heights, TARGET_HEIGHTS_M):
        raise ValueError("Compact model-c heights do not match the common vertical grid")
    expected_shape = (len(VALIDATION_TIMES), len(TARGET_HEIGHTS_M)) + grid.shape
    if field.shape != expected_shape or coverage.shape != expected_shape or uncertainty.shape != expected_shape:
        raise ValueError("Compact model-c field shape does not match the compact NWP grid")
    if not np.allclose(latitude, grid.lat) or not np.allclose(longitude, grid.lon) or not np.array_equal(mask, grid.mask):
        raise ValueError("Compact model-c horizontal grid differs from the compact NWP grid")
    metadata = {
        "times": [time.isoformat() for time in VALIDATION_TIMES],
        "include_radar": True,
        "input": "data/model_c_validation_1km_50m.npz",
        "description": "Portable model-c validation field on the native WRF horizontal grid",
    }
    return field, coverage, uncertainty, metadata


# ============================= 4. d/e 校准、验证和预报 =============================


def stack_model_d_features(
    normalized_cube: np.ndarray,
    proxy: np.ndarray,
    tendency: np.ndarray,
) -> np.ndarray:
    height_fraction = np.broadcast_to(
        (TARGET_HEIGHTS_M / TARGET_HEIGHTS_M.max())[:, None, None], proxy.shape
    ).astype(np.float32)
    return np.concatenate(
        [normalized_cube, proxy[None, ...], tendency[None, ...], height_fraction[None, ...]],
        axis=0,
    ).astype(np.float32)


def sample_training_rows(
    features: np.ndarray,
    target: np.ndarray,
    coverage: np.ndarray | None,
    max_rows: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.moveaxis(features, 0, -1).reshape(-1, features.shape[0])
    y = target.reshape(-1)
    valid = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    if coverage is not None:
        valid &= coverage.reshape(-1) >= 0.15
    indices = np.flatnonzero(valid)
    rng = np.random.default_rng(seed)
    if len(indices) > max_rows:
        indices = rng.choice(indices, size=max_rows, replace=False)
    return x[indices], y[indices]


def fit_model_d(
    feature_cubes: list[np.ndarray],
    proxy_fields: np.ndarray,
    observed_c: np.ndarray,
    observed_coverage: np.ndarray,
) -> tuple[
    HistGradientBoostingRegressor,
    HistGradientBoostingClassifier,
    pd.DataFrame,
    dict[str, float],
    np.ndarray,
    np.ndarray,
]:
    """Fit an intensity regressor and a cost-sensitive high-risk warning head."""
    x_by_time, y_by_time = [], []
    for i in range(len(VALIDATION_TIMES)):
        x, y = sample_training_rows(feature_cubes[i], observed_c[i], observed_coverage[i], 30_000, 2025 + i)
        x_by_time.append(x)
        y_by_time.append(y)
    predictions, warning_probabilities, truths, raw_predictions = [], [], [], []
    fold_rows = []
    # 至少三个时次后开始滚动预报，防止将待验证时次泄漏进训练集。
    for test_i in range(3, len(VALIDATION_TIMES)):
        x_train = np.concatenate(x_by_time[:test_i])
        y_train = np.concatenate(y_by_time[:test_i])
        model = HistGradientBoostingRegressor(
            max_iter=120,
            learning_rate=0.06,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=1.0,
            random_state=2025,
        )
        model.fit(x_train, y_train)
        warning_model = HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.06,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=1.0,
            random_state=2025,
        )
        warning_weight = np.where(y_train >= HIGH_RISK_THRESHOLD, HIGH_RISK_CLASS_WEIGHT, 1.0)
        warning_model.fit(x_train, y_train >= HIGH_RISK_THRESHOLD, sample_weight=warning_weight)
        pred = np.clip(model.predict(x_by_time[test_i]), 0.0, 1.0)
        warning_probability = warning_model.predict_proba(x_by_time[test_i])[:, 1]
        raw = x_by_time[test_i][:, FEATURE_NAMES.__len__()]
        metric = regression_metrics(y_by_time[test_i], pred)
        warning_metric = warning_metrics(y_by_time[test_i], warning_probability)
        raw_metric = regression_metrics(y_by_time[test_i], raw)
        fold_rows.append(
            {
                "test_time": VALIDATION_TIMES[test_i],
                **{f"calibrated_{key}": value for key, value in metric.items()},
                **{f"warning_{key}": value for key, value in warning_metric.items()},
                **{f"physical_{key}": value for key, value in raw_metric.items()},
            }
        )
        predictions.append(pred)
        warning_probabilities.append(warning_probability)
        truths.append(y_by_time[test_i])
        raw_predictions.append(raw)
    pooled_pred = np.concatenate(predictions)
    pooled_warning = np.concatenate(warning_probabilities)
    pooled_truth = np.concatenate(truths)
    pooled_raw = np.concatenate(raw_predictions)
    summary = {
        **{f"calibrated_{key}": value for key, value in regression_metrics(pooled_truth, pooled_pred).items()},
        **{f"warning_{key}": value for key, value in warning_metrics(pooled_truth, pooled_warning).items()},
        **{f"physical_{key}": value for key, value in regression_metrics(pooled_truth, pooled_raw).items()},
    }
    final = HistGradientBoostingRegressor(
        max_iter=160,
        learning_rate=0.055,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=2025,
    )
    final_x = np.concatenate(x_by_time)
    final_y = np.concatenate(y_by_time)
    final.fit(final_x, final_y)
    final_warning = HistGradientBoostingClassifier(
        max_iter=160,
        learning_rate=0.055,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=2025,
    )
    final_warning_weight = np.where(final_y >= HIGH_RISK_THRESHOLD, HIGH_RISK_CLASS_WEIGHT, 1.0)
    final_warning.fit(final_x, final_y >= HIGH_RISK_THRESHOLD, sample_weight=final_warning_weight)
    return final, final_warning, pd.DataFrame(fold_rows), summary, pooled_truth, pooled_pred


def predict_field(model: HistGradientBoostingRegressor, features: np.ndarray, chunk: int = 250_000) -> np.ndarray:
    x = np.moveaxis(features, 0, -1).reshape(-1, features.shape[0])
    valid = np.all(np.isfinite(x), axis=1)
    prediction = np.full(len(x), np.nan, dtype=np.float32)
    selected = np.flatnonzero(valid)
    for start in range(0, len(selected), chunk):
        indices = selected[start : start + chunk]
        prediction[indices] = np.clip(model.predict(x[indices]), 0.0, 1.0).astype(np.float32)
    return prediction.reshape(features.shape[1:])


def predict_warning_field(
    model: HistGradientBoostingClassifier, features: np.ndarray, chunk: int = 250_000
) -> np.ndarray:
    x = np.moveaxis(features, 0, -1).reshape(-1, features.shape[0])
    valid = np.all(np.isfinite(x), axis=1)
    probability = np.full(len(x), np.nan, dtype=np.float32)
    selected = np.flatnonzero(valid)
    for start in range(0, len(selected), chunk):
        indices = selected[start : start + chunk]
        probability[indices] = model.predict_proba(x[indices])[:, 1].astype(np.float32)
    return probability.reshape(features.shape[1:])


def local_mean_and_gradient(field: np.ndarray, dx_m: float, dy_m: float) -> tuple[np.ndarray, np.ndarray]:
    filled = np.nan_to_num(field, nan=float(np.nanmedian(field)))
    local_mean = uniform_filter(filled, size=(1, 3, 3), mode="nearest")
    gx = np.gradient(filled, dx_m, axis=2, edge_order=2)
    gy = np.gradient(filled, dy_m, axis=1, edge_order=2)
    return local_mean.astype(np.float32), np.hypot(gx, gy).astype(np.float32)


def model_e_features(current: np.ndarray, lag1: np.ndarray, lag2: np.ndarray, grid: WrfGrid) -> np.ndarray:
    delta1 = current - lag1
    delta2 = lag1 - lag2
    acceleration = delta1 - delta2
    local_mean, spatial_gradient = local_mean_and_gradient(current, grid.dx_m, grid.dy_m)
    # 水平梯度乘 1 km 使其成为无量纲的单网格变化量。
    spatial_gradient *= 1000.0
    height_fraction = np.broadcast_to(
        (TARGET_HEIGHTS_M / TARGET_HEIGHTS_M.max())[:, None, None], current.shape
    ).astype(np.float32)
    return np.stack(
        [current, lag1, lag2, delta1, delta2, acceleration, local_mean, spatial_gradient, height_fraction]
    ).astype(np.float32)


def fit_and_forecast_model_e(
    observed_c: np.ndarray,
    observed_coverage: np.ndarray,
    grid: WrfGrid,
) -> tuple[np.ndarray, np.ndarray, HistGradientBoostingRegressor, dict[str, float], np.ndarray, np.ndarray]:
    transition_x, transition_y = [], []
    # i 时次用 i,i-1,i-2 预测 i+1。最后一个可验证转移留作严格测试。
    for i in range(2, len(observed_c) - 1):
        features = model_e_features(observed_c[i], observed_c[i - 1], observed_c[i - 2], grid)
        x, y = sample_training_rows(features, observed_c[i + 1], observed_coverage[i + 1], 35_000, 3050 + i)
        transition_x.append(x)
        transition_y.append(y)
    if len(transition_x) < 2:
        raise ValueError("At least two trainable model-e transitions are required")
    validation_model = HistGradientBoostingRegressor(
        max_iter=120,
        learning_rate=0.06,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=2026,
    )
    validation_model.fit(np.concatenate(transition_x[:-1]), np.concatenate(transition_y[:-1]))
    validation_prediction = np.clip(validation_model.predict(transition_x[-1]), 0, 1)
    validation_truth = transition_y[-1]
    persistence = transition_x[-1][:, 0]
    metrics = {
        **{f"model_e_{key}": value for key, value in regression_metrics(validation_truth, validation_prediction).items()},
        **{f"persistence_{key}": value for key, value in regression_metrics(validation_truth, persistence).items()},
    }

    final = HistGradientBoostingRegressor(
        max_iter=140,
        learning_rate=0.055,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=2026,
    )
    final.fit(np.concatenate(transition_x), np.concatenate(transition_y))
    features_0530 = model_e_features(observed_c[-1], observed_c[-2], observed_c[-3], grid)
    forecast_0530 = predict_field(final, features_0530)
    features_0600 = model_e_features(forecast_0530, observed_c[-1], observed_c[-2], grid)
    forecast_0600 = predict_field(final, features_0600)
    rmse = float(metrics["model_e_rmse"])
    uncertainty = np.stack(
        [
            np.clip(rmse + 0.20 * np.abs(forecast_0530 - observed_c[-1]), 0.08, 0.80),
            np.clip(rmse * math.sqrt(2.0) + 0.20 * np.abs(forecast_0600 - forecast_0530), 0.10, 0.90),
        ]
    ).astype(np.float32)
    return forecast_0530, forecast_0600, final, metrics, validation_truth, validation_prediction


# ============================= 5. 三维 A* 航路 =============================


def nearest_grid_cell(grid: WrfGrid, lon: float, lat: float) -> tuple[int, int]:
    distance2 = (grid.lon - lon) ** 2 + (grid.lat - lat) ** 2
    distance2 = np.where(grid.mask, distance2, np.inf)
    y, x = np.unravel_index(np.argmin(distance2), distance2.shape)
    return int(y), int(x)


def reconstruct_route(parent: dict[int, int], current: int, shape: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    flat_nodes = [current]
    while current in parent:
        current = parent[current]
        flat_nodes.append(current)
    flat_nodes.reverse()
    return [tuple(map(int, np.unravel_index(index, shape))) for index in flat_nodes]


def astar_route(
    turbulence: np.ndarray,
    horizontal_mask: np.ndarray,
    start: tuple[int, int, int],
    goal: tuple[int, int, int],
    dx_m: float,
    dy_m: float,
    dz_m: float,
    risk_weight: float = 8.0,
) -> RouteResult:
    """3-D A* with 8 horizontal directions and at most one vertical level per step."""
    shape = turbulence.shape
    start_flat = int(np.ravel_multi_index(start, shape))
    goal_flat = int(np.ravel_multi_index(goal, shape))
    g_score = np.full(np.prod(shape), np.inf, dtype=np.float64)
    g_score[start_flat] = 0.0
    parent: dict[int, int] = {}

    def heuristic(node: tuple[int, int, int]) -> float:
        z, y, x = node
        gz, gy, gx = goal
        return math.sqrt(((x - gx) * dx_m) ** 2 + ((y - gy) * dy_m) ** 2 + ((z - gz) * dz_m) ** 2)

    queue: list[tuple[float, int]] = [(heuristic(start), start_flat)]
    closed = np.zeros(np.prod(shape), dtype=bool)
    horizontal_moves = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]
    while queue:
        _, current_flat = heapq.heappop(queue)
        if closed[current_flat]:
            continue
        if current_flat == goal_flat:
            return RouteResult(reconstruct_route(parent, current_flat, shape), float(g_score[current_flat]), True)
        closed[current_flat] = True
        z, y, x = map(int, np.unravel_index(current_flat, shape))
        current_risk = float(turbulence[z, y, x])
        for dy, dx in horizontal_moves:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < shape[1] and 0 <= nx < shape[2]) or not horizontal_mask[ny, nx]:
                continue
            for dz in (-1, 0, 1):
                nz = z + dz
                if not (0 <= nz < shape[0]):
                    continue
                next_risk = float(turbulence[nz, ny, nx])
                if not np.isfinite(next_risk):
                    continue
                distance = math.sqrt((dx * dx_m) ** 2 + (dy * dy_m) ** 2 + (dz * dz_m) ** 2)
                mean_risk = 0.5 * (current_risk + next_risk)
                edge_cost = distance * (1.0 + risk_weight * mean_risk**3)
                neighbour_flat = int(np.ravel_multi_index((nz, ny, nx), shape))
                tentative = g_score[current_flat] + edge_cost
                if tentative < g_score[neighbour_flat]:
                    g_score[neighbour_flat] = tentative
                    parent[neighbour_flat] = current_flat
                    heapq.heappush(queue, (tentative + heuristic((nz, ny, nx)), neighbour_flat))
    return RouteResult([], float("inf"), False)


def straight_route_nodes(start: tuple[int, int, int], goal: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    count = max(abs(goal[i] - start[i]) for i in range(3)) + 1
    coordinates = [np.rint(np.linspace(start[i], goal[i], count)).astype(int) for i in range(3)]
    result = []
    for node in zip(*coordinates):
        node_tuple = tuple(int(value) for value in node)
        if not result or result[-1] != node_tuple:
            result.append(node_tuple)
    return result


def route_metrics(
    nodes: list[tuple[int, int, int]], turbulence: np.ndarray, dx_m: float, dy_m: float, dz_m: float
) -> dict[str, float]:
    if not nodes:
        return {"node_count": 0, "length_km": np.nan, "mean_turbulence": np.nan, "p90_turbulence": np.nan, "max_turbulence": np.nan, "high_risk_fraction": np.nan, "integrated_risk_km": np.nan}
    values = np.array([turbulence[node] for node in nodes], dtype=float)
    distances = []
    for a, b in zip(nodes[:-1], nodes[1:]):
        distances.append(math.sqrt(((b[2] - a[2]) * dx_m) ** 2 + ((b[1] - a[1]) * dy_m) ** 2 + ((b[0] - a[0]) * dz_m) ** 2))
    distances = np.asarray(distances, dtype=float)
    segment_risk = 0.5 * (values[:-1] + values[1:]) if len(values) > 1 else np.array([])
    return {
        "node_count": int(len(nodes)),
        "length_km": float(distances.sum() / 1000.0),
        "mean_turbulence": float(np.nanmean(values)),
        "p90_turbulence": float(np.nanquantile(values, 0.90)),
        "max_turbulence": float(np.nanmax(values)),
        "high_risk_fraction": float(np.nanmean(values >= HIGH_RISK_THRESHOLD)),
        "integrated_risk_km": float(np.nansum(distances * segment_risk) / 1000.0),
    }


def route_to_table(
    nodes: list[tuple[int, int, int]], turbulence: np.ndarray, grid: WrfGrid, time: pd.Timestamp, model: str
) -> pd.DataFrame:
    rows = []
    for sequence, (z_i, y_i, x_i) in enumerate(nodes):
        rows.append(
            {
                "sequence": sequence,
                "model": model,
                "valid_time": time,
                "longitude_deg": float(grid.lon[y_i, x_i]),
                "latitude_deg": float(grid.lat[y_i, x_i]),
                "altitude_agl_m": float(TARGET_HEIGHTS_M[z_i]),
                "turbulence_index": float(turbulence[z_i, y_i, x_i]),
                "grid_z": z_i,
                "grid_y": y_i,
                "grid_x": x_i,
            }
        )
    return pd.DataFrame(rows)


def write_route_geojson(tables: Sequence[pd.DataFrame], path: Path) -> None:
    features = []
    for table in tables:
        if table.empty:
            continue
        properties = {"model": str(table["model"].iloc[0]), "valid_time": str(table["valid_time"].iloc[0])}
        coordinates = table[["longitude_deg", "latitude_deg", "altitude_agl_m"]].to_numpy(float).tolist()
        features.append({"type": "Feature", "properties": properties, "geometry": {"type": "LineString", "coordinates": coordinates}})
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False, indent=2), encoding="utf-8")


# ============================= 6. 图表与输出 =============================


def read_polygon_shapefile(path: Path) -> list[np.ndarray]:
    """Minimal Polygon/PolygonZ SHP reader used only for a map outline."""
    shapes: list[np.ndarray] = []
    if not path.exists():
        return shapes
    with path.open("rb") as handle:
        handle.seek(100)
        while True:
            record_header = handle.read(8)
            if len(record_header) < 8:
                break
            _, content_words = struct.unpack(">2i", record_header)
            content = handle.read(content_words * 2)
            if len(content) < 44:
                continue
            shape_type = struct.unpack("<i", content[:4])[0]
            if shape_type not in (5, 15, 25):
                continue
            num_parts, num_points = struct.unpack("<2i", content[36:44])
            parts_start = 44
            points_start = parts_start + 4 * num_parts
            parts = list(struct.unpack(f"<{num_parts}i", content[parts_start:points_start])) + [num_points]
            points = np.frombuffer(content[points_start : points_start + 16 * num_points], dtype="<f8").reshape(-1, 2).copy()
            for a, b in zip(parts[:-1], parts[1:]):
                shapes.append(points[a:b])
    return shapes


def find_boundary_shapefile(geo_root: Path) -> Path | None:
    candidates = list(geo_root.glob("**/县.shp")) + list(geo_root.glob("**/市.shp"))
    return candidates[0] if candidates else None


def plot_boundaries(ax: plt.Axes, shapes: Sequence[np.ndarray]) -> None:
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    for points in shapes:
        if points.size:
            intersects = (
                (points[:, 0] >= min(xlim))
                & (points[:, 0] <= max(xlim))
                & (points[:, 1] >= min(ylim))
                & (points[:, 1] <= max(ylim))
            )
            if np.any(intersects):
                ax.plot(points[:, 0], points[:, 1], color="0.25", linewidth=0.45, alpha=0.55, zorder=3)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def plot_validation(
    truth_d: np.ndarray,
    pred_d: np.ndarray,
    truth_e: np.ndarray,
    pred_e: np.ndarray,
    output: Path,
) -> None:
    configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    for ax, truth, pred, title in [
        (axes[0], truth_d, pred_d, "模型 d：时序阻塞验证"),
        (axes[1], truth_e, pred_e, "模型 e：05:00 后验外推"),
    ]:
        if len(truth) > 20_000:
            rng = np.random.default_rng(2025)
            idx = rng.choice(len(truth), 20_000, replace=False)
            truth, pred = truth[idx], pred[idx]
        ax.hexbin(truth, pred, gridsize=55, mincnt=1, cmap="viridis")
        ax.plot([0, 1], [0, 1], "r--", linewidth=1.2)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="模型 c 相对指数", ylabel="预测指数", title=title)
        ax.grid(alpha=0.2)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_forecast_summary(
    times: Sequence[pd.Timestamp],
    model_d: np.ndarray,
    e_times: Sequence[pd.Timestamp],
    model_e: np.ndarray,
    grid: WrfGrid,
    output: Path,
) -> None:
    configure_matplotlib()
    d_mean = np.array([[np.nanmean(field[z][grid.mask]) for z in range(field.shape[0])] for field in model_d])
    e_mean = np.array([[np.nanmean(field[z][grid.mask]) for z in range(field.shape[0])] for field in model_e])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True, sharey=True)
    im0 = axes[0].pcolormesh(pd.to_datetime(times), TARGET_HEIGHTS_M, d_mean.T, shading="auto", vmin=0, vmax=1, cmap="turbo")
    axes[0].set(title="模型 d 区域平均湍流预报", ylabel="离地高度 (m)")
    axes[0].set_xticks(pd.to_datetime(times)[::2])
    axes[0].tick_params(axis="x", rotation=35)
    im1 = axes[1].pcolormesh(pd.to_datetime(e_times), TARGET_HEIGHTS_M, e_mean.T, shading="auto", vmin=0, vmax=1, cmap="turbo")
    axes[1].set(title="模型 e 无数值预报外推")
    axes[1].set_xticks(pd.to_datetime(e_times))
    axes[1].tick_params(axis="x", rotation=35)
    fig.colorbar(im1, ax=axes, label="相对湍流指数", shrink=0.9)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_routes(
    fields: Sequence[np.ndarray],
    route_tables: Sequence[pd.DataFrame],
    straight_tables: Sequence[pd.DataFrame],
    grid: WrfGrid,
    boundary_shapes: Sequence[np.ndarray],
    output: Path,
) -> None:
    configure_matplotlib()
    z_i = int(np.argmin(np.abs(TARGET_HEIGHTS_M - 300.0)))
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True, sharex=True, sharey=True)
    titles = ["模型 d：06:00 最优航路", "模型 e：06:00 最优航路"]
    for ax, field, route, straight, title in zip(axes, fields, route_tables, straight_tables, titles):
        image = ax.pcolormesh(grid.lon, grid.lat, field[z_i], shading="auto", cmap="turbo", vmin=0, vmax=1)
        plot_boundaries(ax, boundary_shapes)
        ax.plot(straight["longitude_deg"], straight["latitude_deg"], "w--", linewidth=1.5, label="直线基准")
        ax.plot(route["longitude_deg"], route["latitude_deg"], color="#00ffbb", linewidth=2.3, label="3D A* 航路")
        ax.scatter(route["longitude_deg"].iloc[[0, -1]], route["latitude_deg"].iloc[[0, -1]], c=["white", "black"], s=45, edgecolor="0.2", zorder=5)
        ax.set(title=f"{title}\n底图为 300 m AGL", xlabel="经度", ylabel="纬度")
        ax.legend(loc="best")
    fig.colorbar(image, ax=axes, label="相对湍流指数", shrink=0.82)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_route_profiles(route_tables: Sequence[pd.DataFrame], grid: WrfGrid, output: Path) -> None:
    configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True, sharey=True)
    for ax, table, title in zip(axes, route_tables, ("模型 d", "模型 e")):
        dy = np.diff(table["grid_y"].to_numpy(float)) * grid.dy_m
        dx = np.diff(table["grid_x"].to_numpy(float)) * grid.dx_m
        dz = np.diff(table["altitude_agl_m"].to_numpy(float))
        distance = np.r_[0.0, np.cumsum(np.sqrt(dx**2 + dy**2 + dz**2))] / 1000.0
        risk = table["turbulence_index"].to_numpy(float)
        altitude = table["altitude_agl_m"].to_numpy(float)
        risk_axis = ax.twinx()
        ax.plot(distance, altitude, color="#1769aa", linewidth=2.0, label="飞行高度")
        risk_axis.plot(distance, risk, color="#ef6c00", linewidth=1.6, label="湍流指数")
        risk_axis.axhline(HIGH_RISK_THRESHOLD, color="#c62828", linestyle="--", linewidth=1.0, label="高风险阈值")
        ax.set(xlabel="累计航程 (km)", ylabel="离地高度 (m)", title=f"{title} 06:00 三维航路剖面")
        risk_axis.set(ylabel="相对湍流指数", ylim=(0, 1))
        ax.grid(alpha=0.2)
        lines = ax.get_lines() + risk_axis.get_lines()
        ax.legend(lines, [line.get_label() for line in lines], loc="best", fontsize=8)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def level_summary(times: Sequence[pd.Timestamp], fields: np.ndarray, uncertainties: np.ndarray | None, model: str, grid: WrfGrid) -> pd.DataFrame:
    rows = []
    for time_i, time in enumerate(times):
        for z_i, height in enumerate(TARGET_HEIGHTS_M):
            values = fields[time_i, z_i][grid.mask]
            values = values[np.isfinite(values)]
            if not len(values):
                continue
            row = {
                "model": model,
                "valid_time": time,
                "height_agl_m": float(height),
                "mean_turbulence": float(np.mean(values)),
                "p90_turbulence": float(np.quantile(values, 0.90)),
                "high_risk_fraction": float(np.mean(values >= HIGH_RISK_THRESHOLD)),
            }
            if uncertainties is not None:
                u = uncertainties[time_i, z_i][grid.mask]
                row["mean_uncertainty"] = float(np.nanmean(u))
            rows.append(row)
    return pd.DataFrame(rows)


# ============================= 7. 主流程 =============================


def run_pipeline(args: argparse.Namespace) -> dict[str, object]:
    data_root = DATA_DIR
    output_dir = DEFAULT_OUTPUT
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    if not data_root.exists():
        raise FileNotFoundError(f"Question-3 data root not found: {data_root}")

    nwp_files, grid, nwp_input_mode = discover_nwp_inputs(data_root)
    nwp_times = [time for time, _ in nwp_files]
    raw_features = []
    for time, path in nwp_files:
        print(f"[NWP] {time:%H:%M}  {path.name}", flush=True)
        regular = load_compact_nwp_regular(path) if nwp_input_mode == "compact" else load_nwp_regular(path, grid)
        raw_features.append(compute_nwp_raw_features(regular, grid))
    validation_indices = [nearest_time_index(nwp_times, time) for time in VALIDATION_TIMES]
    for target, index in zip(VALIDATION_TIMES, validation_indices):
        if nwp_times[index] != target:
            raise ValueError(f"No exact NWP validation field at {target}; nearest is {nwp_times[index]}")
    scaler = compute_feature_scaler(raw_features, validation_indices, grid)
    normalized_cubes, proxy_fields = [], []
    for raw in raw_features:
        cube, proxy = normalize_nwp_features(raw, scaler, grid)
        normalized_cubes.append(cube)
        proxy_fields.append(proxy)
    proxy_stack = np.stack(proxy_fields)
    tendency_stack = temporal_derivative(proxy_stack, nwp_times)
    model_d_features = [
        stack_model_d_features(cube, proxy_stack[i], tendency_stack[i]) for i, cube in enumerate(normalized_cubes)
    ]

    print("[OBS] building model-c validation fields", flush=True)
    compact_observed = load_compact_observed_c(data_root, grid)
    if compact_observed is not None:
        observed_c, observed_cov, observed_unc, c_meta = compact_observed
    else:
        observed_c, observed_cov, observed_unc, c_meta = build_observed_c(
            data_root,
            grid,
            output_dir,
            include_radar=True,
            use_cache=True,
        )
    validation_features = [model_d_features[index] for index in validation_indices]
    validation_proxy = proxy_stack[validation_indices]
    print("[MODEL D] blocked temporal validation and calibration", flush=True)
    model_d, model_d_warning, fold_metrics, d_metrics, truth_d, pred_d = fit_model_d(
        validation_features, validation_proxy, observed_c, observed_cov
    )
    joblib.dump(model_d, output_dir / "model_d_calibrator.joblib")
    joblib.dump(model_d_warning, output_dir / "model_d_warning_classifier.joblib")
    fold_metrics.to_csv(output_dir / "model_d_validation_by_time.csv", index=False)
    model_d_fields = np.stack([predict_field(model_d, features) for features in model_d_features])
    model_d_warning_fields = np.stack(
        [predict_warning_field(model_d_warning, features) for features in model_d_features]
    )
    d_rmse = float(d_metrics["calibrated_rmse"])
    forecast_lead = np.array([max((time - pd.Timestamp("2025-07-31 05:00")).total_seconds() / 3600.0, 0.0) for time in nwp_times])
    model_d_unc = np.stack(
        [
            np.clip(
                d_rmse + 0.15 * np.abs(model_d_fields[i] - proxy_stack[i]) + 0.025 * forecast_lead[i],
                0.08,
                0.90,
            )
            for i in range(len(nwp_times))
        ]
    ).astype(np.float32)

    # 官方目录缺 06:00；在 05:30 与 06:30 之间插入明确标记的诊断场。
    target_0600 = pd.Timestamp("2025-07-31 06:00")
    if target_0600 not in nwp_times:
        left = nwp_times.index(pd.Timestamp("2025-07-31 05:30"))
        right = nwp_times.index(pd.Timestamp("2025-07-31 06:30"))
        insert_at = right
        nwp_times.insert(insert_at, target_0600)
        model_d_fields = np.insert(model_d_fields, insert_at, 0.5 * (model_d_fields[left] + model_d_fields[right]), axis=0)
        model_d_unc = np.insert(
            model_d_unc,
            insert_at,
            np.clip(0.5 * (model_d_unc[left] + model_d_unc[right]) + 0.05, 0, 1),
            axis=0,
        )
        model_d_warning_fields = np.insert(
            model_d_warning_fields,
            insert_at,
            0.5 * (model_d_warning_fields[left] + model_d_warning_fields[right]),
            axis=0,
        )
        nwp_0600_interpolated = True
    else:
        nwp_0600_interpolated = False

    print("[MODEL E] nonlinear observation-only extrapolation", flush=True)
    e0530, e0600, model_e, e_metrics, truth_e, pred_e = fit_and_forecast_model_e(observed_c, observed_cov, grid)
    joblib.dump(model_e, output_dir / "model_e_extrapolator.joblib")
    e_fields = np.stack([e0530, e0600])
    e_times = [pd.Timestamp("2025-07-31 05:30"), target_0600]
    e_rmse = float(e_metrics["model_e_rmse"])
    e_unc = np.stack(
        [
            np.clip(e_rmse + 0.20 * np.abs(e0530 - observed_c[-1]), 0.08, 0.80),
            np.clip(e_rmse * math.sqrt(2.0) + 0.20 * np.abs(e0600 - e0530), 0.10, 0.90),
        ]
    ).astype(np.float32)

    # 路径规划默认只允许 100--1000 m AGL；起终点可通过 CLI 覆盖。
    start_lon, start_lat, start_alt = args.start
    end_lon, end_lat, end_alt = args.end
    min_z = int(np.searchsorted(TARGET_HEIGHTS_M, args.min_altitude, side="left"))
    max_z = int(np.searchsorted(TARGET_HEIGHTS_M, args.max_altitude, side="right") - 1)
    start_y, start_x = nearest_grid_cell(grid, start_lon, start_lat)
    end_y, end_x = nearest_grid_cell(grid, end_lon, end_lat)
    start_z_global = int(np.argmin(np.abs(TARGET_HEIGHTS_M - start_alt)))
    end_z_global = int(np.argmin(np.abs(TARGET_HEIGHTS_M - end_alt)))
    start_z_global = int(np.clip(start_z_global, min_z, max_z))
    end_z_global = int(np.clip(end_z_global, min_z, max_z))
    d0600 = model_d_fields[nwp_times.index(target_0600)]
    route_tables, straight_tables, route_metric_rows = [], [], []
    for model_name, field in (("model_d", d0600), ("model_e", e0600)):
        allowed = field[min_z : max_z + 1]
        start = (start_z_global - min_z, start_y, start_x)
        goal = (end_z_global - min_z, end_y, end_x)
        result = astar_route(
            allowed,
            grid.mask,
            start,
            goal,
            grid.dx_m,
            grid.dy_m,
            float(TARGET_HEIGHTS_M[1] - TARGET_HEIGHTS_M[0]),
            risk_weight=args.risk_weight,
        )
        if not result.reached:
            raise RuntimeError(f"A* could not find a route for {model_name}")
        global_nodes = [(z + min_z, y, x) for z, y, x in result.nodes]
        straight_nodes = straight_route_nodes((start_z_global, start_y, start_x), (end_z_global, end_y, end_x))
        route_table = route_to_table(global_nodes, field, grid, target_0600, model_name)
        straight_table = route_to_table(straight_nodes, field, grid, target_0600, model_name + "_straight")
        route_table.to_csv(output_dir / f"optimal_route_{model_name}_0600.csv", index=False)
        route_tables.append(route_table)
        straight_tables.append(straight_table)
        route_metric_rows.append({"model": model_name, "route": "optimal", **route_metrics(global_nodes, field, grid.dx_m, grid.dy_m, 50.0)})
        route_metric_rows.append({"model": model_name, "route": "straight", **route_metrics(straight_nodes, field, grid.dx_m, grid.dy_m, 50.0)})
    route_metrics_table = pd.DataFrame(route_metric_rows)
    route_metrics_table.to_csv(output_dir / "route_metrics.csv", index=False)
    write_route_geojson(route_tables, output_dir / "optimal_routes_0600.geojson")

    level_d = level_summary(nwp_times, model_d_fields, model_d_unc, "model_d", grid)
    level_e = level_summary(e_times, e_fields, e_unc, "model_e", grid)
    pd.concat([level_d, level_e], ignore_index=True).to_csv(output_dir / "forecast_level_summary.csv", index=False)
    np.savez_compressed(
        output_dir / "model_c_validation_1km_50m.npz",
        times=np.array([time.isoformat() for time in VALIDATION_TIMES]),
        heights_m=TARGET_HEIGHTS_M,
        latitude=grid.lat,
        longitude=grid.lon,
        mask=grid.mask,
        turbulence_index=observed_c,
        coverage=observed_cov,
        uncertainty=observed_unc,
    )
    np.savez_compressed(
        output_dir / "model_d_forecast_1km_50m.npz",
        times=np.array([time.isoformat() for time in nwp_times]),
        heights_m=TARGET_HEIGHTS_M,
        latitude=grid.lat,
        longitude=grid.lon,
        mask=grid.mask,
        turbulence_index=model_d_fields,
        high_risk_warning_score=model_d_warning_fields,
        uncertainty=model_d_unc,
    )
    np.savez_compressed(
        output_dir / "model_e_forecast_1km_50m.npz",
        times=np.array([time.isoformat() for time in e_times]),
        heights_m=TARGET_HEIGHTS_M,
        latitude=grid.lat,
        longitude=grid.lon,
        mask=grid.mask,
        turbulence_index=e_fields,
        uncertainty=e_unc,
    )

    boundary_path = find_boundary_shapefile(data_root / "地理信息数据")
    boundary_shapes = read_polygon_shapefile(boundary_path) if boundary_path else []
    plot_validation(truth_d, pred_d, truth_e, pred_e, output_dir / "figures" / "validation_scatter.png")
    plot_forecast_summary(nwp_times, model_d_fields, e_times, e_fields, grid, output_dir / "figures" / "forecast_time_height.png")
    plot_routes(
        [d0600, e0600],
        route_tables,
        straight_tables,
        grid,
        boundary_shapes,
        output_dir / "figures" / "optimal_routes_0600.png",
    )
    plot_route_profiles(route_tables, grid, output_dir / "figures" / "route_vertical_profiles_0600.png")

    metrics_table = pd.DataFrame(
        [
            {"model": "d_calibrated", **{k.replace("calibrated_", ""): v for k, v in d_metrics.items() if k.startswith("calibrated_")}},
            {"model": "d_warning", **{k.replace("warning_", ""): v for k, v in d_metrics.items() if k.startswith("warning_")}},
            {"model": "d_physical", **{k.replace("physical_", ""): v for k, v in d_metrics.items() if k.startswith("physical_")}},
            {"model": "e_nonlinear", **{k.replace("model_e_", ""): v for k, v in e_metrics.items() if k.startswith("model_e_")}},
            {"model": "e_persistence", **{k.replace("persistence_", ""): v for k, v in e_metrics.items() if k.startswith("persistence_")}},
        ]
    )
    metrics_table.to_csv(output_dir / "validation_metrics.csv", index=False)
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_root": "data",
        "nwp_input": f"data/{'nwp' if nwp_input_mode == 'compact' else '数值天气预报数据'}",
        "grid": {
            "shape_yx": list(grid.shape),
            "horizontal_resolution_m": grid.dx_m,
            "vertical_resolution_m": 50.0,
            "height_range_m": [0.0, 2000.0],
            "bounds": SURFACE_BOUNDS,
        },
        "model_c": c_meta,
        "model_d_feature_scaler": scaler,
        "model_d_high_risk_class_weight": HIGH_RISK_CLASS_WEIGHT,
        "model_d_validation": d_metrics,
        "model_e_validation": e_metrics,
        "nwp_0600_interpolated_from_0530_0630": nwp_0600_interpolated,
        "route": {
            "start_requested": args.start,
            "end_requested": args.end,
            "start_snapped": [float(grid.lon[start_y, start_x]), float(grid.lat[start_y, start_x]), float(TARGET_HEIGHTS_M[start_z_global])],
            "end_snapped": [float(grid.lon[end_y, end_x]), float(grid.lat[end_y, end_x]), float(TARGET_HEIGHTS_M[end_z_global])],
            "allowed_altitude_m": [float(args.min_altitude), float(args.max_altitude)],
            "risk_weight": float(args.risk_weight),
            "official_route_coordinates_present": False,
            "assumption": "Official files contain no machine-readable start/end coordinates; configurable defaults were used.",
        },
    }
    (output_dir / "run_summary.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Completed. Results: {output_dir}", flush=True)
    return metadata


def quick_check(args: argparse.Namespace) -> None:
    files, grid, input_mode = discover_nwp_inputs(DATA_DIR)
    regular = load_compact_nwp_regular(files[0][1]) if input_mode == "compact" else load_nwp_regular(files[0][1], grid)
    features = compute_nwp_raw_features(regular, grid)
    print(
        json.dumps(
            {
                "nwp_files": len(files),
                "nwp_input_mode": input_mode,
                "grid_shape": grid.shape,
                "valid_horizontal_cells": int(grid.mask.sum()),
                "height_levels": len(TARGET_HEIGHTS_M),
                "feature_finite_fraction": {
                    name: float(np.mean(np.isfinite(value[:, grid.mask]))) for name, value in features.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest="command")
    for command in ("all", "quick"):
        sub = subparsers.add_parser(command)
        if command == "all":
            sub.add_argument("--start", type=float, nargs=3, metavar=("LON", "LAT", "AGL_M"), default=(118.36, 31.28, 300.0))
            sub.add_argument("--end", type=float, nargs=3, metavar=("LON", "LAT", "AGL_M"), default=(119.24, 32.60, 300.0))
            sub.add_argument("--min-altitude", type=float, default=100.0)
            sub.add_argument("--max-altitude", type=float, default=1000.0)
            sub.add_argument("--risk-weight", type=float, default=8.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        args = parser.parse_args(["all"])
    try:
        if args.command == "quick":
            quick_check(args)
        else:
            run_pipeline(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
