#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare the compact optimization input for Question 5."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    q4 = base.parent / "2024-C-Question4-Core-Loss-Prediction"
    parser = argparse.ArgumentParser(description="生成第五问优化输入数据")
    parser.add_argument(
        "--features",
        type=Path,
        default=q4 / "附件一_第四问特征数据.csv",
        help="第四问附件一特征表",
    )
    parser.add_argument(
        "--oof-predictions",
        type=Path,
        default=q4 / "模型五折预测.csv",
        help="第四问五折折外预测",
    )
    parser.add_argument(
        "--output", type=Path, default=base / "附件一_第五问优化数据.csv"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = pd.read_csv(args.features)
    predictions = pd.read_csv(args.oof_predictions)
    required_features = [
        "样本编号",
        "温度_oC",
        "频率_Hz",
        "磁芯材料",
        "励磁波形",
        "磁通密度峰值B_m_T",
        "磁芯损耗_w每m3",
    ]
    required_predictions = ["样本编号", "五折预测磁芯损耗_w每m3"]
    missing_features = [c for c in required_features if c not in features.columns]
    missing_predictions = [c for c in required_predictions if c not in predictions.columns]
    if missing_features or missing_predictions:
        raise ValueError(
            f"缺少必要列：特征表{missing_features}，预测表{missing_predictions}"
        )
    if features["样本编号"].duplicated().any() or predictions["样本编号"].duplicated().any():
        raise ValueError("样本编号不得重复。")

    result = features[required_features].merge(
        predictions[required_predictions], on="样本编号", validate="one_to_one"
    )
    result = result.rename(
        columns={
            "磁芯损耗_w每m3": "实际磁芯损耗_w每m3",
            "五折预测磁芯损耗_w每m3": "第四问OOF预测损耗_w每m3",
        }
    )
    if len(result) != len(features) or len(result) != len(predictions):
        raise ValueError("特征表和预测表的样本编号不完全一致。")
    numeric = [
        "温度_oC",
        "频率_Hz",
        "磁通密度峰值B_m_T",
        "实际磁芯损耗_w每m3",
        "第四问OOF预测损耗_w每m3",
    ]
    if result[numeric].isna().any().any() or not np.isfinite(result[numeric]).all().all():
        raise ValueError("优化输入中存在缺失值或非有限数。")
    if (result[["频率_Hz", "磁通密度峰值B_m_T", "实际磁芯损耗_w每m3", "第四问OOF预测损耗_w每m3"]] <= 0).any().any():
        raise ValueError("频率、B_m和损耗必须为正数。")

    result["第四问预测相对误差_percent"] = (
        (result["第四问OOF预测损耗_w每m3"] - result["实际磁芯损耗_w每m3"])
        / result["实际磁芯损耗_w每m3"]
        * 100
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.sort_values("样本编号").to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"已生成 {len(result)} 条第五问优化输入：{args.output}")


if __name__ == "__main__":
    main()
