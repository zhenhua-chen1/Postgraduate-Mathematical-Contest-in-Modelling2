#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从官方附件一提取第三问所需的紧凑分析数据。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="官方附件一 Excel 文件")
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "附件一_第三问分析数据.csv",
        help="紧凑数据输出路径",
    )
    args = parser.parse_args()

    workbook = pd.ExcelFile(args.input)
    frames: list[pd.DataFrame] = []
    sample_start = 1
    for sheet_name in workbook.sheet_names:
        raw = pd.read_excel(args.input, sheet_name=sheet_name)
        if raw.shape[1] < 5:
            raise ValueError(f"{sheet_name} 的列数不足，无法读取磁通密度序列。")
        flux = raw.iloc[:, 4:].to_numpy(dtype=float)
        flux_peak = (flux.max(axis=1) - flux.min(axis=1)) / 2.0
        frame = pd.DataFrame(
            {
                "样本编号": np.arange(sample_start, sample_start + len(raw)),
                "温度_oC": raw.iloc[:, 0].astype(int),
                "频率_Hz": raw.iloc[:, 1].astype(float),
                "磁芯损耗_w每m3": raw.iloc[:, 2].astype(float),
                "励磁波形": raw.iloc[:, 3].astype(str),
                "磁芯材料": str(sheet_name),
                "磁通密度峰值B_m_T": flux_peak,
            }
        )
        frames.append(frame)
        sample_start += len(raw)

    result = pd.concat(frames, ignore_index=True)
    if result.isna().any().any():
        raise ValueError("提取结果存在缺失值。")
    if (
        (result["频率_Hz"] <= 0).any()
        or (result["磁芯损耗_w每m3"] <= 0).any()
        or (result["磁通密度峰值B_m_T"] <= 0).any()
    ):
        raise ValueError("频率、磁芯损耗和磁通密度峰值必须为正数。")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"已提取 {len(result)} 条样本：{args.output.resolve()}")


if __name__ == "__main__":
    main()
