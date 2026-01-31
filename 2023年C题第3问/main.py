#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 28 15:39:39 2025

@author: chenzhenhua
"""

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


data1 = pd.read_excel('数据2.1.xlsx')
data2 = pd.read_excel('数据2.2.xlsx')

def explain_innovation_linear(best_model, feature_cols, innov_col="创新性指标I"):
    # 只有当 best_model 是 LinearRegression pipeline 才适用
    if "model" not in best_model.named_steps:
        return None

    mdl = best_model.named_steps["model"]
    if not hasattr(mdl, "coef_"):
        return None

    coefs = pd.Series(mdl.coef_, index=feature_cols).sort_values(key=np.abs, ascending=False)
    return coefs.loc[[innov_col]] if innov_col in coefs.index else coefs

def fit_and_compare_models(data1,
                           target_col="最终成绩",
                           std_cols=None,
                           innov_col="创新性指标I",
                           prof_cols=None,
                           test_size=0.2,
                           random_state=42):
    """
    训练/测试划分 + 拟合：线性回归、决策树、随机森林、XGBoost（可选）
    输出：评估表、最佳模型、X_train/X_test/y_train/y_test、特征列名
    """
    if std_cols is None:
        std_cols = [f"标准分{k}" for k in range(1, 6)]
    if prof_cols is None:
        prof_cols = [f"专家专业性得分{k}" for k in range(1, 6)]

    feature_cols = std_cols + [innov_col] + prof_cols

    # 1) 检查列是否存在
    missing = [c for c in [target_col] + feature_cols if c not in data1.columns]
    if missing:
        raise KeyError(f"data1 缺少列：{missing}")

    # 2) 取 X,y 并转数值
    df = data1.copy()
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    X = df[feature_cols]
    y = df[target_col]

    # 3) 划分训练/测试
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 4) 定义模型（统一做缺失值填补；线性回归再做标准化）
    models = {}

    models["LinearRegression"] = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])

    models["DecisionTree"] = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("model", DecisionTreeRegressor(
            random_state=random_state,
            max_depth=None
        ))
    ])

    models["RandomForest"] = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            random_state=random_state,
            n_estimators=500,
            max_depth=None,
            n_jobs=-1
        ))
    ])

    # 5) 训练并评估
    rows = []
    fitted = {}

    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        fitted[name] = pipe

        pred = pipe.predict(X_test)

        r2 = r2_score(y_test, pred)
        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))

        rows.append({
            "Model": name,
            "R2": r2,
            "MAE": mae,
            "RMSE": rmse
        })

    results = pd.DataFrame(rows).sort_values(by="R2", ascending=False).reset_index(drop=True)

    best_name = results.loc[0, "Model"]
    best_model = fitted[best_name]

    return results, best_name, best_model, X_train, X_test, y_train, y_test, feature_cols

def melt_expert_scores(df, k_range=range(1, 9),
                       code_prefix="专家编码", score_prefix="原始分"):
    parts = []
    for k in k_range:
        code_col = f"{code_prefix}{k}"
        score_col = f"{score_prefix}{k}"
        if code_col in df.columns and score_col in df.columns:
            tmp = df[[code_col, score_col]].copy()
            tmp = tmp.rename(columns={code_col: "专家编码", score_col: "分数"})
            tmp["专家序号"] = k
            parts.append(tmp)

    melted_df = pd.concat(parts, ignore_index=True)
    # 可选：把没分数或没专家编码的行去掉
    melted_df = melted_df.dropna(subset=["专家编码", "分数"])
    return melted_df

def attach_expert_score(df,
                        expert_score_df,
                        expert_code_cols=("专家编码1","专家编码2","专家编码3","专家编码4","专家编码5"),
                        expert_id_col="专家编码",
                        score_col="专业性得分",
                        out_prefix="专业性得分",
                        inplace=False):
    """
    expert_score_df: 仅包含 [专家编码, 专业性得分] 两列
    df: 包含 专家编码1~5 列
    为 df 添加 out_prefix1~out_prefix5 五列（每个专家编码对应的专业性得分）
    """
    if not inplace:
        df = df.copy()

    # 构造映射：专家编码 -> 专业性得分
    tmp = expert_score_df[[expert_id_col, score_col]].copy()
    tmp[expert_id_col] = tmp[expert_id_col].astype(str).str.strip()
    score_map = pd.Series(tmp[score_col].values, index=tmp[expert_id_col]).to_dict()

    # 映射到每个专家编码列
    for i, col in enumerate(expert_code_cols, start=1):
        if col not in df.columns:
            raise KeyError(f"df 中缺少列：{col}")
        out_col = f"{out_prefix}{i}"
        df[out_col] = df[col].apply(lambda x: score_map.get(str(x).strip(), np.nan) if pd.notna(x) else np.nan)

    return df

def add_innovation_index(df,expert_score,
                         std_prefix="标准分",
                         ks=range(1, 6),
                         out_col="创新性指标I",
                         inplace=False):
    """
    对每个样本（每行）的标准分1~标准分8计算创新性指标：
    I = 0.1*(μ/σ) + 0.7*|Skewness| + 0.2*(Max/Q3)

    Parameters
    ----------
    df : pd.DataFrame
    std_prefix : str
        标准分列名前缀，默认“标准分”
    ks : iterable
        标准分编号，默认 1~8
    out_col : str
        输出列名
    inplace : bool
        True: 原地修改；False: 返回新df

    Returns
    -------
    pd.DataFrame
    """
    if not inplace:
        df = df.copy()

    std_cols = [f"{std_prefix}{k}" for k in ks]
    missing = [c for c in std_cols if c not in df.columns]
    if missing:
        raise KeyError(f"缺少这些标准分列，无法计算 {out_col}: {missing}")

    # 转成数值（避免Excel读入为字符串）
    df[std_cols] = df[std_cols].apply(pd.to_numeric, errors="coerce")

    def _calc_row(values):
        s = pd.Series(values).dropna()
        if len(s) == 0:
            return np.nan

        mu = s.mean()
        sigma = s.std(ddof=1)
        skew = s.skew()
        mx = s.max()
        q3 = s.quantile(0.75)

        term1 = mu / sigma if (sigma != 0 and not np.isnan(sigma)) else np.nan
        term3 = mx / q3 if (q3 != 0 and not np.isnan(q3)) else np.nan

        return 0.1 * term1 + 0.7 * abs(skew) + 0.2 * term3

    df[out_col] = df[std_cols].apply(lambda row: _calc_row(row.values), axis=1)
    df = attach_expert_score(
        df,
        expert_score,
        expert_code_cols=("专家编码1","专家编码2","专家编码3","专家编码4","专家编码5"),
        expert_id_col="专家编码",
        score_col="专业性得分",
        out_prefix="专家专业性得分"   # 输出列名：专家专业性得分1~5
    )
    return df

def preprocess_scores(df,
                      dropna_col=None,
                      review_ks=(6, 7, 8),
                      std_prefix="标准分",
                      rev_prefix="复议分",
                      score1_col="第一阶段最终成绩",
                      score1_ks=(1, 2, 3),
                      inplace=False):
    """
    1) 可选：删除 dropna_col 为 NaN 的行
    2) 复议分替代：标准分k = 复议分k（若非空）否则保留原标准分k，k in review_ks
    3) 计算 score1_col = sum(标准分1,2,3)（可自定义 score1_ks）

    返回处理后的 DataFrame
    """
    if not inplace:
        df = df.copy()

    # 1) 删除专家编码6为空的行（如果指定）
    if dropna_col is not None:
        df = df.dropna(subset=[dropna_col]).copy()

    # 2) 复议分替代（6/7/8）
    for k in review_ks:
        std_col = f"{std_prefix}{k}"
        rev_col = f"{rev_prefix}{k}"
        if std_col in df.columns and rev_col in df.columns:
            df[std_col] = df[rev_col].combine_first(df[std_col])

    # 3) 计算第一阶段最终成绩：标准分1+2+3
    need_cols = [f"{std_prefix}{k}" for k in score1_ks]
    for c in need_cols:
        if c not in df.columns:
            raise KeyError(f"缺少列：{c}，无法计算 {score1_col}")

    # 确保可加和（处理字符串数字）
    df[need_cols] = df[need_cols].apply(pd.to_numeric, errors="coerce")
    df[score1_col] = df[need_cols].sum(axis=1)

    return df


def plot_two_fig_compare_enlegend(df,
                                  range1_col="极差",
                                  range2_col="二次评审标准分极差",
                                  score1_col="最终成绩",
                                  score2_col="二次评审最终成绩"):
    df = df.copy().reset_index(drop=True)
    x = range(len(df))

    # Fig A: Range comparison
    plt.figure(figsize=(12, 4.5))
    plt.plot(x, df[range1_col], marker="o", label="Stage 1 Range")
    plt.plot(x, df[range2_col], marker="o", label="Stage 2 Range")
    plt.title("Data1:Range Comparison (Two Stages)")
    plt.xlabel("Sample Index")
    plt.ylabel("Range")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Fig B: Final score comparison
    plt.figure(figsize=(12, 4.5))
    plt.bar(x, df[score1_col], width=0.25, label="Stage 1 Final Score")
    plt.plot(x, df[score2_col], marker="s", linestyle="--", label="Stage 2 Final Score")
    plt.title("Data1:Final Score Comparison (Two Stages)")
    plt.xlabel("Sample Index")
    plt.ylabel("Final Score")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def boxplot_data2_twofig(df,
                         range1_col="极差",
                         range2_col="二次评审标准分极差",
                         score1_col="最终成绩",
                         score2_col="二次评审最终成绩"):
    # Fig A: Range boxplot
    plt.figure(figsize=(7.5, 4.5))
    plt.boxplot(
        [df[range1_col].dropna(), df[range2_col].dropna()],
        labels=["Stage 1 Range", "Stage 2 Range"],
        showfliers=True
    )
    plt.title("Data 2: Range (Boxplot)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Fig B: Final score boxplot
    plt.figure(figsize=(7.5, 4.5))
    plt.boxplot(
        [df[score1_col].dropna(), df[score2_col].dropna()],
        labels=["Stage 1 Final Score", "Stage 2 Final Score"],
        showfliers=True
    )
    plt.title("Data 2: Final Score (Boxplot)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

def calculate_expert_professionalism_score(df):
    """
    计算专家的专业性得分
    参数：
    df : DataFrame
        包含专家编码、打分信息和其他相关数据的DataFrame。
    
    返回：
    grouped_result : DataFrame
        包含每个专家的分数列表、评审作品数量、标准差、极评指标及最终专业性得分的DataFrame。
    """

     #提取所有以 '专家编码' 和 '原始分' 开头的列
    #cols = [col for col in df.columns if '专家编码' in col or '原始分' in col]

    # 将数据进行 melt 操作，展开专家编码和打分信息
    # melted_df = pd.melt(df[cols], id_vars=['专家编码1'], 
    #                     value_vars=[col for col in cols if '原始分' in col], 
    #                     var_name='评分类型', value_name='分数')
    
    melted_df = melt_expert_scores(df, range(1, 9))
    # 按照专家编码分组，将每个专家的打分汇总为列表
    grouped_result = melted_df.groupby('专家编码').agg(
        分数列表=('分数', lambda x: list(x.dropna())),  # 汇总为列表，且去掉缺失值
        分数均值=('分数', 'mean')  # 计算均值
    ).reset_index()

    # 为每个专家的分数列表计算偏度和峰度
    grouped_result['偏度'] = grouped_result['分数列表'].apply(lambda x: skew(x))
    grouped_result['峰度'] = grouped_result['分数列表'].apply(lambda x: kurtosis(x))

    # 计算分数列表长度，并根据长度选择前5个专家
    grouped_result['评审作品数量'] = grouped_result['分数列表'].apply(len)

    # 计算标准差
    grouped_result['标准差'] = grouped_result['分数列表'].apply(lambda x: pd.Series(x).std())

    # 初始化极评指标的计数器
    grouped_result['最高分计数'] = 0
    grouped_result['最低分计数'] = 0

    # 获取所有包含专家编码和原始分的列
    expert_columns = [col for col in df.columns if '专家编码' in col]
    score_columns = [col for col in df.columns if '原始分' in col]

    # 遍历每个作品（样本），计算极评指标
    for i in range(len(df)):
        # 获取该样本所有的分数和对应的专家编码
        scores = df.loc[i, score_columns]
        experts = df.loc[i, expert_columns]

        # 过滤掉 NaN 值的分数和对应的专家
        valid_data = pd.DataFrame({'分数': list(scores), '专家编码': list(experts)}).dropna()

        # 计算最高分和最低分
        max_score = valid_data['分数'].max()
        min_score = valid_data['分数'].min()

        # 找到打出最高分和最低分的专家
        experts_with_max_score = valid_data['专家编码'][valid_data['分数'] == max_score]
        experts_with_min_score = valid_data['专家编码'][valid_data['分数'] == min_score]

        # 对这些专家的最高分计数+1
        grouped_result.loc[grouped_result['专家编码'].isin(experts_with_max_score), '最高分计数'] += 1
        # 对这些专家的最低分计数+1
        grouped_result.loc[grouped_result['专家编码'].isin(experts_with_min_score), '最低分计数'] += 1

    # 计算极评指标
    grouped_result['极评指标'] = abs((grouped_result['最高分计数'] - grouped_result['最低分计数']) / grouped_result['评审作品数量'])

    # 选取需要进行PCA分析的变量
    pca_data = grouped_result[['标准差', '评审作品数量', '极评指标']]

    # 数据标准化到0-1
    scaler = MinMaxScaler()
    pca_data_scaled = scaler.fit_transform(pca_data)

    # 进行PCA分析
    pca = PCA(n_components=3)  # 选择保留所有3个主成分
    pca_result = pca.fit_transform(pca_data_scaled)

    # 计算专家的权重（主成分贡献率）
    weights = pca.explained_variance_ratio_

    # 计算每个专家的最终专业性得分，基于主成分结果和贡献率加权求和
    grouped_result['专业性得分'] = np.dot(pca_result, weights)

    return grouped_result[['专家编码','专业性得分']]


#数据预处理
data2 = data2.drop(columns=["学校编码"]) #去除学校编码
data1 = preprocess_scores(
    data1,
    dropna_col="专家编码6",
    review_ks=(6, 7, 8),
    score1_col="第一次评审最终成绩",      # 你想叫啥列名都行
    score1_ks=(1, 2, 3, 4, 5)
)

data2 = preprocess_scores(
    data2,
    dropna_col="专家编码6",
    review_ks=(6, 7, 8),
    score1_col="第一次评审最终成绩",      # 你想叫啥列名都行
    score1_ks=(1, 2, 3, 4, 5)
)

#画对比图
plot_two_fig_compare_enlegend(data1,
                     range1_col="极差",
                     range2_col="第二次评审标准分极差",
                     score1_col="第一次评审最终成绩",
                     score2_col="最终成绩")

boxplot_data2_twofig(data2,
                     range1_col="极差",
                     range2_col="第二次评审标准分极差",
                     score1_col="第一次评审最终成绩",
                     score2_col="最终成绩")

#建立指标
#数据集1
expert_score1 = calculate_expert_professionalism_score(data1) #计算专家专业性指标
data1 = add_innovation_index(data1,expert_score1) #加入创新性指标和专业性指标

#数据集2
expert_score2 = calculate_expert_professionalism_score(data2) #计算专家专业性指标
data2 = add_innovation_index(data2,expert_score2) #加入创新性指标和专业性指标

#建立极差模型
results, best_name1, best_model1, X_train, X_test, y_train, y_test, feature_cols1 = fit_and_compare_models(data1)
print(results)
print("data1 Best model:", best_name1)

results, best_name2, best_model2, X_train, X_test, y_train, y_test, feature_cols2 = fit_and_compare_models(data2)
print(results)
print("data2 Best model:", best_name2)

#输出文件
out_path = "两个数据集专家专业性指标和作品创新型指标I.xlsx"

# 1) expert_score1 统一成 DataFrame
if isinstance(expert_score1, pd.Series):
    expert_score1_df = expert_score1.reset_index()
    expert_score1_df.columns = ["专家编码", "专业性得分"]
else:
    expert_score1_df = expert_score1.copy()
    

cols = ["创新性指标I"]  # 你也可以加上作品ID等，比如 ["作品编号","创新性指标I"]
# 2) data1（包含“创新性指标I”列）
data1_df = data1[cols].copy()
data2_df = data2[cols].copy()

# 3) 写入同一个 Excel 的两个 sheet
with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    expert_score1_df.to_excel(writer, sheet_name="数据集1专家专业指标", index=False)
    expert_score1_df.to_excel(writer, sheet_name="数据集2专家专业指标", index=False)
    data1_df.to_excel(writer, sheet_name="数据集1创新性指标I", index=False)
    data2_df.to_excel(writer, sheet_name="数据集2创新性指标I", index=False)

print("Saved to:", out_path)


