#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract compact, reproducible features for Question 4.

The official workbooks contain 1024 flux-density samples per observation.  This
script converts each waveform into physical, statistical and spectral features
while preserving the operating conditions and the training target.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EPS = 1e-12


def waveform_features(flux: np.ndarray) -> pd.DataFrame:
    """Return vectorised features for rows of 1024-point flux waveforms."""
    flux = np.asarray(flux, dtype=np.float64)
    if flux.ndim != 2 or flux.shape[1] < 16:
        raise ValueError("磁通密度数据必须是每行一个波形的二维数组。")
    if not np.isfinite(flux).all():
        raise ValueError("磁通密度数据中存在缺失或无穷大。")

    b_max = flux.max(axis=1)
    b_min = flux.min(axis=1)
    b_pp = b_max - b_min
    b_m = b_pp / 2.0
    b_mean = flux.mean(axis=1)
    centered = flux - b_mean[:, None]
    b_std = centered.std(axis=1)
    b_rms = np.sqrt(np.mean(flux**2, axis=1))
    b_abs_mean = np.mean(np.abs(flux), axis=1)
    b_max_abs = np.max(np.abs(flux), axis=1)

    z = centered / np.maximum(b_std[:, None], EPS)
    skewness = np.mean(z**3, axis=1)
    excess_kurtosis = np.mean(z**4, axis=1) - 3.0

    cyclic_diff = np.diff(np.concatenate([flux, flux[:, :1]], axis=1), axis=1)
    abs_diff = np.abs(cyclic_diff)
    slope_rms_norm = np.sqrt(np.mean(cyclic_diff**2, axis=1)) / np.maximum(b_pp, EPS)
    slope_max_norm = abs_diff.max(axis=1) / np.maximum(b_pp, EPS)
    variation_norm = abs_diff.sum(axis=1) / np.maximum(b_pp, EPS)

    spectrum = np.abs(np.fft.rfft(centered, axis=1))
    fundamental = np.maximum(spectrum[:, 1], EPS)
    harmonic_ratios = {
        f"H{order}_ratio": spectrum[:, order] / fundamental
        for order in range(2, 11)
    }
    thd_2_10 = np.sqrt(
        np.sum((spectrum[:, 2:11] / fundamental[:, None]) ** 2, axis=1)
    )
    spectral_mass = np.maximum(spectrum[:, 1:].sum(axis=1), EPS)
    harmonic_index = np.arange(1, spectrum.shape[1], dtype=float)
    spectral_centroid = (
        spectrum[:, 1:] @ harmonic_index / spectral_mass / harmonic_index[-1]
    )

    signs = np.signbit(centered)
    zero_crossings = np.sum(signs != np.roll(signs, 1, axis=1), axis=1)
    near_peak_fraction = np.mean(
        np.abs(centered) >= 0.9 * np.maximum(b_max_abs, EPS)[:, None], axis=1
    )

    features = {
        "磁通密度峰值B_m_T": b_m,
        "对数磁通密度峰值": np.log(np.maximum(b_m, EPS)),
        "磁通密度_RMS_T": b_rms,
        "磁通密度_绝对均值_T": b_abs_mean,
        "磁通密度_标准差_T": b_std,
        "磁通密度_均值_T": b_mean,
        "磁通密度_最大绝对值_T": b_max_abs,
        "磁通密度_峰峰值_T": b_pp,
        "波峰因子": b_max_abs / np.maximum(b_rms, EPS),
        "波形因子": b_rms / np.maximum(b_abs_mean, EPS),
        "偏度": skewness,
        "超额峰度": excess_kurtosis,
        "斜率RMS归一化": slope_rms_norm,
        "最大斜率归一化": slope_max_norm,
        "总变差归一化": variation_norm,
        "THD_2_10": thd_2_10,
        "频谱质心归一化": spectral_centroid,
        "近峰值点比例": near_peak_fraction,
        "过零数": zero_crossings.astype(float),
        **harmonic_ratios,
    }
    return pd.DataFrame(features)


def add_operating_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["对数频率"] = np.log(data["频率_Hz"].astype(float))
    data["对数频率×对数B_m"] = (
        data["对数频率"] * data["对数磁通密度峰值"]
    )
    data["频率×B_m"] = data["频率_Hz"] * data["磁通密度峰值B_m_T"]
    return data


def extract_training(path: Path) -> pd.DataFrame:
    sheets = pd.read_excel(path, sheet_name=None)
    blocks: list[pd.DataFrame] = []
    sample_start = 1
    for sheet_name, raw in sheets.items():
        if raw.shape[1] < 1028:
            raise ValueError(f"{sheet_name} 列数异常：{raw.shape[1]}")
        flux = raw.iloc[:, 4:].to_numpy(dtype=float)
        block = pd.DataFrame(
            {
                "样本编号": np.arange(sample_start, sample_start + len(raw)),
                "温度_oC": raw.iloc[:, 0].astype(int).to_numpy(),
                "频率_Hz": raw.iloc[:, 1].astype(float).to_numpy(),
                "磁芯损耗_w每m3": raw.iloc[:, 2].astype(float).to_numpy(),
                "磁芯材料": str(sheet_name).strip(),
                "励磁波形": raw.iloc[:, 3].astype(str).str.strip().to_numpy(),
            }
        )
        block = pd.concat([block.reset_index(drop=True), waveform_features(flux)], axis=1)
        blocks.append(add_operating_features(block))
        sample_start += len(raw)
    result = pd.concat(blocks, ignore_index=True)
    if (result["磁芯损耗_w每m3"] <= 0).any():
        raise ValueError("训练集中的磁芯损耗必须为正数。")
    return result


def extract_test(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=0)
    if raw.shape[1] < 1029:
        raise ValueError(f"附件三列数异常：{raw.shape[1]}")
    flux = raw.iloc[:, 5:].to_numpy(dtype=float)
    result = pd.DataFrame(
        {
            "样本编号": raw.iloc[:, 0].astype(int).to_numpy(),
            "温度_oC": raw.iloc[:, 1].astype(int).to_numpy(),
            "频率_Hz": raw.iloc[:, 2].astype(float).to_numpy(),
            "磁芯材料": raw.iloc[:, 3].astype(str).str.strip().to_numpy(),
            "励磁波形": raw.iloc[:, 4].astype(str).str.strip().to_numpy(),
        }
    )
    result = pd.concat([result.reset_index(drop=True), waveform_features(flux)], axis=1)
    return add_operating_features(result)


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="提取第四问训练集和测试集特征")
    parser.add_argument("--train-input", type=Path, required=True, help="官方附件一")
    parser.add_argument(
        "--test-input", type=Path, default=base / "附件三（测试集）.xlsx", help="官方附件三"
    )
    parser.add_argument(
        "--train-output", type=Path, default=base / "附件一_第四问特征数据.csv"
    )
    parser.add_argument(
        "--test-output", type=Path, default=base / "附件三_第四问特征数据.csv"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    training = extract_training(args.train_input)
    testing = extract_test(args.test_input)
    args.train_output.parent.mkdir(parents=True, exist_ok=True)
    training.to_csv(args.train_output, index=False, encoding="utf-8-sig")
    testing.to_csv(args.test_output, index=False, encoding="utf-8-sig")
    print(f"训练特征：{training.shape} -> {args.train_output}")
    print(f"测试特征：{testing.shape} -> {args.test_output}")


if __name__ == "__main__":
    main()
