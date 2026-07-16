#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


HERE = Path(__file__).resolve().parent
DEFAULT_METRICS = HERE.parent / "2023年C题第3问" / "两个数据集专家专业性指标和作品创新型指标I.xlsx"


@dataclass(frozen=True)
class Config:
    advance_rate: float = 0.30
    expert_count: int = 30
    reviewers_per_work: int = 3
    generations: int = 80
    attempts_per_generation: int = 3000
    seed: int = 2023
    professionalism_weight: float = 0.15


def minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    span = np.nanmax(values) - np.nanmin(values)
    if not np.isfinite(span) or span == 0:
        return np.zeros_like(values)
    return (values - np.nanmin(values)) / span


def load_inputs(path: Path, dataset: int, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取问题三结果，并按论文实验设计筛选入围作品和候选专家。"""
    work_sheet = f"数据集{dataset}创新性指标I"
    expert_sheet = f"数据集{dataset}专家专业指标"
    works = pd.read_excel(path, sheet_name=work_sheet)
    experts = pd.read_excel(path, sheet_name=expert_sheet)

    required_work = {"名次", "最终成绩", "创新性指标I"}
    required_expert = {"专家编码", "专业性得分", "极评指标"}
    if missing := required_work.difference(works.columns):
        raise KeyError(f"{work_sheet} 缺少列：{sorted(missing)}")
    if missing := required_expert.difference(experts.columns):
        raise KeyError(f"{expert_sheet} 缺少列：{sorted(missing)}")

    works = works.sort_values("名次").head(max(1, int(round(len(works) * cfg.advance_rate)))).copy()
    experts = experts.dropna(subset=list(required_expert)).copy()
    if len(experts) < cfg.expert_count:
        raise ValueError(f"有效专家只有 {len(experts)} 人，少于要求的 {cfg.expert_count} 人")
    experts = experts.nlargest(cfg.expert_count, "专业性得分").reset_index(drop=True)
    return works.reset_index(drop=True), experts


def initialize_balanced(n_works: int, n_experts: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """构造满足每件作品恰好 k 人且专家负载之差不超过 1 的初始解。"""
    if k > n_experts:
        raise ValueError("每件作品所需专家数不能超过候选专家总数")
    total = n_works * k
    # 将一个随机专家排列循环铺开。任意连续 k 个位置均不重复，且每位
    # 专家在总槽位中的出现次数仅可能相差 1。
    order = rng.permutation(n_experts)
    slots = np.resize(order, total)
    return slots.reshape(n_works, k).copy()


def score_assignment(
    assignment: np.ndarray,
    innovation: np.ndarray,
    extremity: np.ndarray,
    professionalism: np.ndarray,
    professionalism_weight: float,
) -> float:
    """论文目标（创新性×极评指标）减去专家组专业性不均衡惩罚。"""
    match = (innovation[:, None] * extremity[assignment]).sum()
    spread = np.ptp(professionalism[assignment], axis=1).sum()
    return float(match - professionalism_weight * spread)


def optimize_by_feasible_swaps(
    assignment: np.ndarray,
    innovation: np.ndarray,
    extremity: np.ndarray,
    professionalism: np.ndarray,
    cfg: Config,
) -> tuple[np.ndarray, list[float]]:
    """以交换为变异算子进行遗传式局部进化；交换始终保持所有硬约束。"""
    rng = np.random.default_rng(cfg.seed + 1)
    current = assignment.copy()
    current_score = score_assignment(
        current, innovation, extremity, professionalism, cfg.professionalism_weight
    )
    history = [current_score]

    for generation in range(cfg.generations):
        temperature = max(0.001, 0.03 * (1 - generation / max(cfg.generations, 1)))
        for _ in range(cfg.attempts_per_generation):
            a, b = rng.integers(0, len(current), size=2)
            if a == b:
                continue
            pa, pb = rng.integers(0, current.shape[1], size=2)
            ea, eb = current[a, pa], current[b, pb]
            if ea == eb or eb in current[a] or ea in current[b]:
                continue

            old_rows = current[[a, b]].copy()
            old_value = score_assignment(
                old_rows, innovation[[a, b]], extremity, professionalism, cfg.professionalism_weight
            )
            current[a, pa], current[b, pb] = eb, ea
            new_value = score_assignment(
                current[[a, b]], innovation[[a, b]], extremity, professionalism, cfg.professionalism_weight
            )
            delta = new_value - old_value
            if delta >= 0 or rng.random() < np.exp(delta / temperature):
                current_score += delta
            else:
                current[a, pa], current[b, pb] = ea, eb
        history.append(current_score)
    return current, history


def validate(assignment: np.ndarray, expert_count: int, k: int) -> dict[str, float]:
    loads = np.bincount(assignment.ravel(), minlength=expert_count)
    if assignment.shape[1] != k or any(len(set(row)) != k for row in assignment):
        raise AssertionError("每件作品必须由互不重复的 3 位专家评审")
    if loads.max() - loads.min() > 1:
        raise AssertionError("专家工作量不均衡")
    return {"最小负载": int(loads.min()), "最大负载": int(loads.max()), "负载标准差": float(loads.std())}


def save_results(
    works: pd.DataFrame,
    experts: pd.DataFrame,
    assignment: np.ndarray,
    history: list[float],
    output: Path,
    cfg: Config,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    expert_codes = experts["专家编码"].astype(str).to_numpy()
    result = works[[c for c in ["名次", "奖项", "最终成绩", "创新性指标I"] if c in works]].copy()
    for position in range(cfg.reviewers_per_work):
        result[f"专家{position + 1}"] = expert_codes[assignment[:, position]]
    result["专家组专业性均值"] = experts["专业性得分"].to_numpy()[assignment].mean(axis=1)
    result["专家组专业性极差"] = np.ptp(experts["专业性得分"].to_numpy()[assignment], axis=1)
    result["创新匹配得分"] = (
        minmax(works["创新性指标I"].to_numpy())[:, None]
        * minmax(experts["极评指标"].to_numpy())[assignment]
    ).sum(axis=1)

    loads = np.bincount(assignment.ravel(), minlength=len(experts))
    expert_result = experts.copy()
    expert_result["第二阶段分配作品数"] = loads
    summary = pd.DataFrame([
        {"指标": "入围作品数", "值": len(works)},
        {"指标": "候选专家数", "值": len(experts)},
        {"指标": "每件作品专家数", "值": cfg.reviewers_per_work},
        {"指标": "专家最小负载", "值": loads.min()},
        {"指标": "专家最大负载", "值": loads.max()},
        {"指标": "初始适应度", "值": history[0]},
        {"指标": "最终适应度", "值": history[-1]},
        {"指标": "适应度提升", "值": history[-1] - history[0]},
    ])
    convergence = pd.DataFrame({"迭代次数": np.arange(len(history)), "适应度": history})

    with pd.ExcelWriter(output / "第二阶段评审分配方案.xlsx", engine="openpyxl") as writer:
        result.to_excel(writer, sheet_name="分配方案", index=False)
        expert_result.to_excel(writer, sheet_name="专家负载", index=False)
        summary.to_excel(writer, sheet_name="模型汇总", index=False)
        convergence.to_excel(writer, sheet_name="收敛过程", index=False)

    # 仅用 Pillow 绘制，避免程序依赖特定图形后端。
    width, height, margin = 1200, 720, 90
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.line((margin, height - margin, width - margin, height - margin), fill="#333333", width=3)
    draw.line((margin, margin, margin, height - margin), fill="#333333", width=3)
    y = np.asarray(history, dtype=float)
    y_span = max(float(y.max() - y.min()), 1e-12)
    xs = np.linspace(margin, width - margin, len(y))
    ys = height - margin - (y - y.min()) / y_span * (height - 2 * margin)
    points = [(int(x), int(v)) for x, v in zip(xs, ys)]
    if len(points) > 1:
        draw.line(points, fill="#1769aa", width=5)
    draw.text((width // 2 - 150, 25), "Stage-2 allocation convergence", fill="#111111")
    draw.text((width // 2 - 35, height - 50), "Generation", fill="#111111")
    draw.text((12, 28), f"max={y.max():.3f}", fill="#111111")
    draw.text((12, height - 80), f"min={y.min():.3f}", fill="#111111")
    image.save(output / "遗传算法收敛曲线.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="2023 年 C 题第 4 问：两阶段创新类竞赛评审模型")
    parser.add_argument("--input", type=Path, default=DEFAULT_METRICS, help="问题三输出的指标工作簿")
    parser.add_argument("--dataset", type=int, choices=(1, 2), default=1)
    parser.add_argument("--output", type=Path, default=HERE, help="输出目录，默认直接写入第4问目录")
    parser.add_argument("--generations", type=int, default=80)
    parser.add_argument("--attempts", type=int, default=3000, help="每代可行交换尝试次数")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(generations=args.generations, attempts_per_generation=args.attempts)
    works, experts = load_inputs(args.input, args.dataset, cfg)
    innovation = minmax(works["创新性指标I"].to_numpy())
    extremity = minmax(experts["极评指标"].to_numpy())
    professionalism = minmax(experts["专业性得分"].to_numpy())
    initial = initialize_balanced(
        len(works), len(experts), cfg.reviewers_per_work, np.random.default_rng(cfg.seed)
    )
    best, history = optimize_by_feasible_swaps(
        initial, innovation, extremity, professionalism, cfg
    )
    checks = validate(best, len(experts), cfg.reviewers_per_work)
    save_results(works, experts, best, history, args.output, cfg)
    print(f"入围作品：{len(works)}，候选专家：{len(experts)}")
    print(f"约束检验：{checks}")
    print(f"适应度：{history[0]:.6f} -> {history[-1]:.6f}")
    print(f"结果已保存至：{args.output.resolve()}")


if __name__ == "__main__":
    main()
