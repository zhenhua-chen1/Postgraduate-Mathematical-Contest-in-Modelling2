#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""车辆采样数据的可测试预处理函数。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PreprocessConfig:
    """全部阈值都来自题面或旧程序，并明确单位。"""

    interpolate_gap_seconds: int = 2
    max_acceleration_mps2: float = 3.96
    max_deceleration_mps2: float = 8.0
    low_speed_kmh: float = 10.0
    max_low_speed_records: int = 180


REQUIRED_COLUMNS = ["时间", "GPS车速"]


def _parse_and_validate(raw: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError(f"缺少必需字段：{missing}")
    if raw.empty:
        raise ValueError("输入数据为空。")

    data = raw.copy()
    data["时间"] = pd.to_datetime(data["时间"], format="%Y/%m/%d %H:%M:%S.000.", errors="coerce")
    if data["时间"].isna().any():
        bad = (data.index[data["时间"].isna()] + 2).tolist()[:5]
        raise ValueError(f"时间格式无效，Excel 行示例：{bad}")
    if data["时间"].duplicated().any():
        raise ValueError("时间存在重复，不能确定记录顺序。")
    if not data["时间"].is_monotonic_increasing:
        raise ValueError("时间不是严格递增，不能静默排序。")
    data["GPS车速"] = pd.to_numeric(data["GPS车速"], errors="coerce")
    if not np.isfinite(data["GPS车速"]).all() or (data["GPS车速"] < 0).any():
        raise ValueError("GPS车速存在缺失、非数值或负值。")
    data["原GPS车速"] = data["GPS车速"].astype(float)
    data["原始行号"] = np.arange(2, len(data) + 2, dtype=float)
    data["处理标记"] = "原始记录"
    return data


def _interpolate_two_second_gaps(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """只补恰好缺少一个采样点的间隔，并对全部数值传感器线性插值。"""
    delta = data["时间"].diff().dt.total_seconds()
    indices = np.flatnonzero(delta.to_numpy() == 2)
    if not len(indices):
        return data, pd.DataFrame(columns=["时间", "处理类型", "原GPS车速", "处理后GPS车速", "说明"])

    inserts = []
    details = []
    numeric = [
        column for column in data.select_dtypes(include=[np.number]).columns
        if column not in ["原始行号"]
    ]
    for index in indices:
        previous = data.iloc[index - 1]
        current = data.iloc[index]
        row = previous.copy()
        row["时间"] = previous["时间"] + pd.Timedelta(seconds=1)
        for column in numeric:
            left, right = previous[column], current[column]
            row[column] = (left + right) / 2 if pd.notna(left) and pd.notna(right) else np.nan
        row["原GPS车速"] = np.nan
        row["原始行号"] = np.nan
        row["处理标记"] = "两秒缺口线性插值"
        inserts.append(row)
        details.append(
            {
                "时间": row["时间"],
                "处理类型": "新增插值记录",
                "原GPS车速": np.nan,
                "处理后GPS车速": row["GPS车速"],
                "说明": f"相邻速度 {previous['GPS车速']:g} 与 {current['GPS车速']:g} km/h 的线性中点",
            }
        )
    combined = pd.concat([data, pd.DataFrame(inserts)], ignore_index=True).sort_values("时间", kind="stable")
    return combined.reset_index(drop=True), pd.DataFrame(details)


def _limit_acceleration(data: pd.DataFrame, config: PreprocessConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按连续采样的真实单位限制速度变化，避免 m/s 与 km/h 混写。"""
    result = data.copy()
    speed = result["GPS车速"].to_numpy(dtype=float, copy=True)
    times = result["时间"].to_numpy(dtype="datetime64[s]")
    details = []
    max_up = config.max_acceleration_mps2 * 3.6
    max_down = config.max_deceleration_mps2 * 3.6

    for index in range(1, len(speed)):
        seconds = float((times[index] - times[index - 1]) / np.timedelta64(1, "s"))
        if seconds != 1:
            continue
        original = speed[index]
        lower = max(0.0, speed[index - 1] - max_down)
        upper = speed[index - 1] + max_up
        speed[index] = np.clip(speed[index], lower, upper)
        if not np.isclose(original, speed[index], rtol=0, atol=1e-12):
            old_tag = result.at[index, "处理标记"]
            result.at[index, "处理标记"] = (
                "异常加减速度限幅" if old_tag == "原始记录" else f"{old_tag}；异常加减速度限幅"
            )
            details.append(
                {
                    "时间": result.at[index, "时间"],
                    "处理类型": "车速限幅",
                    "原GPS车速": original,
                    "处理后GPS车速": speed[index],
                    "说明": f"连续1秒允许范围 [{lower:.6f}, {upper:.6f}] km/h",
                }
            )
    result["GPS车速"] = speed
    return result, pd.DataFrame(details)


def _trim_long_low_speed(data: pd.DataFrame, config: PreprocessConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """每个连续低速段最多保留题目允许的180条1Hz记录。"""
    low = data["GPS车速"].lt(config.low_speed_kmh)
    continuous = data["时间"].diff().dt.total_seconds().eq(1)
    group_start = low.ne(low.shift(fill_value=False)) | ~continuous
    groups = group_start.cumsum()
    rank = data.groupby(groups).cumcount() + 1
    remove = low & rank.gt(config.max_low_speed_records)
    removed = data.loc[remove]
    detail = pd.DataFrame(
        {
            "时间": removed["时间"],
            "处理类型": "删除长低速记录",
            "原GPS车速": removed["原GPS车速"],
            "处理后GPS车速": removed["GPS车速"],
            "说明": f"连续低于{config.low_speed_kmh:g} km/h的记录超过{config.max_low_speed_records}条",
        }
    )
    return data.loc[~remove].reset_index(drop=True), detail.reset_index(drop=True)


def _metrics(raw_rows: int, data: pd.DataFrame, detail: pd.DataFrame, config: PreprocessConfig) -> dict:
    delta = data["时间"].diff().dt.total_seconds()
    acceleration = data["GPS车速"].diff().div(3.6).where(delta.eq(1))
    low = data["GPS车速"].lt(config.low_speed_kmh)
    groups = (low.ne(low.shift(fill_value=False)) | ~delta.eq(1)).cumsum()
    max_low = int(data.loc[low].groupby(groups[low]).size().max()) if low.any() else 0
    counts = detail["处理类型"].value_counts()
    return {
        "原始记录数": int(raw_rows),
        "插值新增数": int(counts.get("新增插值记录", 0)),
        "车速修正数": int(counts.get("车速限幅", 0)),
        "长低速删除数": int(counts.get("删除长低速记录", 0)),
        "处理后记录数": int(len(data)),
        "处理后连续段数": int((~delta.eq(1)).sum()),
        "保留的长间隔数": int(delta.gt(config.interpolate_gap_seconds).sum()),
        "最长连续低速记录数": max_low,
        "最大连续加速度_mps2": round(float(acceleration.max()), 6),
        "最大连续减速度_mps2": round(float(acceleration.min()), 6),
    }


def preprocess_vehicle_data(
    raw: pd.DataFrame, config: PreprocessConfig | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    config = config or PreprocessConfig()
    data = _parse_and_validate(raw)
    raw_rows = len(data)
    data, inserted = _interpolate_two_second_gaps(data)
    data, limited = _limit_acceleration(data, config)
    data, removed = _trim_long_low_speed(data, config)
    detail_frames = [frame for frame in [inserted, limited, removed] if not frame.empty]
    if detail_frames:
        detail = pd.concat(detail_frames, ignore_index=True).sort_values(
            ["时间", "处理类型"], kind="stable"
        ).reset_index(drop=True)
    else:
        detail = pd.DataFrame(
            columns=["时间", "处理类型", "原GPS车速", "处理后GPS车速", "说明"]
        )
    metrics = _metrics(raw_rows, data, detail, config)
    return data, detail, metrics


def validate_processed_data(data: pd.DataFrame, config: PreprocessConfig | None = None) -> None:
    config = config or PreprocessConfig()
    required = REQUIRED_COLUMNS + ["原GPS车速", "原始行号", "处理标记"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise AssertionError(f"结果缺少字段：{missing}")
    frame = data.copy()
    frame["时间"] = pd.to_datetime(frame["时间"], errors="coerce")
    if frame["时间"].isna().any() or frame["时间"].duplicated().any() or not frame["时间"].is_monotonic_increasing:
        raise AssertionError("结果时间必须有效、唯一且严格递增。")
    speed = pd.to_numeric(frame["GPS车速"], errors="coerce")
    if not np.isfinite(speed).all() or (speed < 0).any():
        raise AssertionError("结果车速必须为非负有限数。")
    delta = frame["时间"].diff().dt.total_seconds()
    acceleration = speed.diff().div(3.6).where(delta.eq(1))
    if (acceleration > config.max_acceleration_mps2 + 1e-9).any():
        raise AssertionError("结果仍存在超限加速度。")
    if (acceleration < -config.max_deceleration_mps2 - 1e-9).any():
        raise AssertionError("结果仍存在超限减速度。")
    low = speed.lt(config.low_speed_kmh)
    groups = (low.ne(low.shift(fill_value=False)) | ~delta.eq(1)).cumsum()
    if low.any() and frame.loc[low].groupby(groups[low]).size().max() > config.max_low_speed_records:
        raise AssertionError("结果仍存在超过180条的连续低速记录。")
