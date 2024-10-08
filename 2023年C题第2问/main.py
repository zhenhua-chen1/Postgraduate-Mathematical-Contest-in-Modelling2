#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 11:24:52 2024

@author: chenzhenhua
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# 读取数据
df = pd.read_excel('数据1.xlsx', 'Sheet2')

def plot_results(grouped_result):
    """
    绘制专家打分偏度与峰度的散点图，以及每个专家的打分分布图
    """
    plt.rcParams['font.family'] = ['Arial Unicode MS']  # 正常显示中文
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

    # 绘制散点图，用偏度作为x轴，峰度作为y轴
    plt.figure(figsize=(8, 5))
    plt.scatter(grouped_result['偏度'], grouped_result['峰度'], color='blue', marker='x')

    # 标注每个点的专家编码
    for i, row in grouped_result.iterrows():
        plt.text(row['偏度'], row['峰度'], row['专家编码'], fontsize=9)

    plt.title('Scatter Plot of Skewness and Kurtosis for Expert Scores')
    plt.xlabel('Skewness')
    plt.ylabel('Kurtosis')
    plt.grid(True)
    plt.show()

    # 按照评审作品数量排序，选择前5个专家
    top_5_experts = grouped_result.sort_values(by='评审作品数量', ascending=False).head(5)
    experts_to_plot = top_5_experts['专家编码']
    filtered_data = grouped_result[grouped_result['专家编码'].isin(experts_to_plot)]

    # 为前5个专家绘制直方图，并叠加KDE曲线
    for index, row in filtered_data.iterrows():
        plt.figure(figsize=(8, 5))

        # 绘制频数直方图并叠加KDE曲线
        sns.histplot(row['分数列表'], bins=50, kde=True, color='blue', edgecolor='black', stat='count')

        plt.title(f'Score Distribution for Expert {row["专家编码"]}')
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()
        #plt.close()  # 关闭图，防止内存溢出


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
    cols = [col for col in df.columns if '专家编码' in col or '原始分' in col]

    # 将数据进行 melt 操作，展开专家编码和打分信息
    melted_df = pd.melt(df[cols], id_vars=['专家编码'], 
                        value_vars=[col for col in cols if '原始分' in col], 
                        var_name='评分类型', value_name='分数')

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
    plot_results(grouped_result)

    return grouped_result[['专家编码','专业性得分']]

def determine_work_metrics(df):
    """
    确定每个作品的指标，包括原始分、标准分、第二次评审标准分极差（如果存在）、标准分极差和复议标记。
    参数：
    df : DataFrame
        包含作品的原始分、标准分、第二次评审标准分和复议标记的DataFrame。
    
    返回：
    work_metrics_df : DataFrame
        包含每个作品的原始分、标准分、标准分极差、第二次评审标准分极差（如有）、以及复议标记的DataFrame。
    """
    # 初始化存储结果的列表
    result_list = []

    # 获取所有包含专家编码和评分的列
    original_score_columns = [col for col in df.columns if '原始分' in col and not col.startswith('第二次')]
    standard_score_columns = [col for col in df.columns if '标准分' in col and '第二次' not in col]
    reconsideration_columns = [col for col in df.columns if '复议分' in col]

    # 获取第二次评审的相关列
    original_second_review_columns = [col for col in df.columns if '原始分.' in col and int(col.split('.')[1]) >= 5]
    second_review_score_columns = [col for col in df.columns if '标准分.' in col and int(col.split('.')[1]) >= 5]

    # 遍历每个作品（样本）
    for i in range(len(df)):
        # 获取原始分和标准分
        original_scores = df.loc[i, original_score_columns].dropna().tolist()
        standard_scores = df.loc[i, standard_score_columns].dropna().tolist()
        
        # 计算标准分极差
        range_standard = max(standard_scores) - min(standard_scores) if len(standard_scores) > 1 else 0

        # 计算第二次评审的标准分极差
        if len(original_second_review_columns) > 0 and df.loc[i, original_second_review_columns].notna().any():
            # 获取第二次评审的标准分
            second_review_scores = df.loc[i, second_review_score_columns].dropna().tolist()
            range_second_review = max(second_review_scores) - min(second_review_scores) if len(second_review_scores) > 1 else 0
        else:
            range_second_review = None

        # 获取复议标记，如果不为NaN，则设为1
        reconsideration_flag = 1 if df.loc[i, reconsideration_columns].notna().any() else None

        # 将计算结果存储到列表中
        result_list.append({
            '作品ID': i + 1,  # 用作品编号作为ID
            '原始分': np.mean(original_scores),
            '标准分': np.mean(standard_scores),
            '标准分极差': range_standard,
            '第二次评审标准分极差': range_second_review,
            '复议标记': reconsideration_flag
        })

    # 转换为DataFrame
    work_metrics_df = pd.DataFrame(result_list)

    return work_metrics_df



def calculate_standard_deviation_and_professionalism(df, result):
    """
    计算每个作品的标准分的标准差，以及参与专家的专业性得分总和和平均专业性评分。
    参数：
    df : DataFrame
        包含作品的评分和专家信息。
    result : DataFrame
        包含每个专家的编码和专业性得分。

    返回：
    result_df : DataFrame
        包含每个作品的标准分的标准差、专家专业性得分总和和平均值的DataFrame。
    """
    # 获取所有包含专家编码和标准分的列
    expert_columns = [col for col in df.columns if '专家编码' in col]
    standard_score_columns = [col for col in df.columns if '标准分' in col]

    # 初始化结果存储
    result_list = []

    # 遍历每个作品（样本）
    for i in range(len(df)):
        # 获取该样本的所有标准分和专家编码
        standard_scores = df.loc[i, standard_score_columns]
        experts = df.loc[i, expert_columns]

        # 过滤掉 NaN 值的标准分和对应的专家编码
        valid_data = pd.DataFrame({'标准分': list(standard_scores), '专家编码': list(experts)}).dropna()

        # 计算标准分的标准差
        standard_deviation = valid_data['标准分'].std()
        mean = valid_data['标准分'].mean()

        # 获取这些专家的专业性得分
        valid_experts = valid_data['专家编码']
        expert_scores = result[result['专家编码'].isin(valid_experts)]['专业性得分']

        # 计算这些专家的专业性得分总和和平均值
        total_professionalism = expert_scores.sum()
        average_professionalism = expert_scores.mean()

        # 将结果存储
        result_list.append({
            '作品ID': i + 1,  # 用作品编号作为ID
            '标准分标准差': standard_deviation,
            '专家专业性总和': total_professionalism,
            '专家平均专业性评分': average_professionalism,
            '平均分':mean
        })

    # 转换为DataFrame
    result_df = pd.DataFrame(result_list)

    return result_df

def calculate_reasonableness_score(work_metrics_df):
    """
    使用PCA计算权重并计算作品合理性得分。
    参数：
    work_metrics_df : DataFrame
        包含每个作品的原始分均值、标准分均值、标准分极差、第二次评审标准分极差和复议标记的DataFrame。
    
    返回：
    work_metrics_df : DataFrame
        包含作品合理性得分的DataFrame。
    """
    # 将复议标记转换为数值，复议标记为1则更合理，NaN为0
    work_metrics_df['复议标记数值'] = work_metrics_df['复议标记'].apply(lambda x: 1 if x == 1 else 0)

    # 选取需要进行PCA分析的列
    columns_for_pca = ['原始分', '标准分', '标准分极差', '第二次评审标准分极差', '复议标记数值']

    # 数据标准化到0-1范围
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(work_metrics_df[columns_for_pca].fillna(0))

    # 进行PCA分析
    pca = PCA(n_components=5)  # 保留所有5个主成分
    pca.fit(scaled_data)

    # 获取主成分的解释方差贡献率（PCA权重）
    pca_weights = pca.explained_variance_ratio_


    # 计算合理性得分，基于PCA权重
    work_metrics_df['作品合理性得分'] = (
        scaled_data[:, 0] * pca_weights[0] +  # 原始分均值的权重
        scaled_data[:, 1] * pca_weights[1] +  # 标准分均值的权重
        scaled_data[:, 2] * pca_weights[2] +  # 标准分极差的权重
        scaled_data[:, 3] * pca_weights[3] +  # 第二次评审标准分极差的权重
        scaled_data[:, 4] * pca_weights[4]    # 复议标记数值的权重
    )

    return work_metrics_df['作品合理性得分'] 

def calculate_weights(result_df):
    """
    使用PCA计算标准分标准差、专家专业性总和和专家平均专业性评分的权重，并将'平均分'的权重设为0.5。
    参数：
    result_df : DataFrame
        包含每个作品的标准分标准差、专家专业性总和、专家平均专业性评分和平均分的DataFrame。
    
    返回：
    result_df : DataFrame
        加入PCA加权得分的DataFrame。
    """
    # 选取需要进行PCA分析的列（不包含平均分）
    columns_for_pca = ['标准分标准差', '专家专业性总和', '专家平均专业性评分']

    # 数据标准化到0-1范围
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(result_df[columns_for_pca])

    # 进行PCA分析
    pca = PCA(n_components=3)  # 保留这三个特征的主成分
    pca.fit(scaled_data)

    # 获取主成分的解释方差贡献率（PCA权重）
    pca_weights = pca.explained_variance_ratio_


    # 将 0.5 分配给 '平均分'，剩余 0.5 分给 PCA 计算出的权重
    fixed_mean_weight = 0.5
    remaining_weight = 0.5
    
    # 根据 PCA 权重重新分配剩余 0.5
    pca_weights_adjusted = pca_weights * remaining_weight
    pca_weights_adjusted = np.append(pca_weights_adjusted, 0.5)

    # 打印加权得分的公式
    print(f'标准分公式1为: f = {pca_weights_adjusted[0]:.2f}*标准分标准差 + {pca_weights_adjusted[1]:.2f}*专家专业性总和 + {pca_weights_adjusted[2]:.2f}*专家平均专业性评分 + {pca_weights_adjusted[3]:.2f}*平均分')

    
    # 计算加权得分
    result_df['新标准分1'] = (
        result_df['平均分'] * fixed_mean_weight +  # 固定 '平均分' 的权重为 0.5
        scaled_data[:, 0] * pca_weights_adjusted[0] +  # 标准分标准差的权重
        scaled_data[:, 1] * pca_weights_adjusted[1] +  # 专家专业性总和的权重
        scaled_data[:, 2] * pca_weights_adjusted[2]    # 专家平均专业性评分的权重
    )
    
    # 加入作品合理性得分指标
    columns_for_reasonableness = ['标准分标准差', '专家专业性总和', '专家平均专业性评分', '作品合理性得分']
    
     # 数据标准化到0-1范围
    scaled_data_reasonableness = scaler.fit_transform(result_df[columns_for_reasonableness].fillna(0))
    
     # 进行PCA分析
    pca_reasonableness = PCA(n_components=4)  # 保留所有4个特征的主成分
    pca_reasonableness.fit(scaled_data_reasonableness)
    
     # 获取主成分的解释方差贡献率（PCA权重）
    pca_weights_reasonableness = pca_reasonableness.explained_variance_ratio_
    

    
    # 根据 PCA 权重重新分配剩余 0.5
    pca_weights_reasonableness_adjusted = pca_weights_reasonableness * remaining_weight
    pca_weights_reasonableness_adjusted = np.append(pca_weights_reasonableness_adjusted, 0.5)
     # 打印PCA权重信息（可选）
     # 打印加权得分的公式（标准分公式2）
    print(f'标准分公式2为: f = {pca_weights_reasonableness[0]:.2f}*标准分标准差 + {pca_weights_reasonableness[1]:.2f}*专家专业性总和 + {pca_weights_reasonableness[2]:.2f}*专家平均专业性评分 + {pca_weights_reasonableness[3]:.2f}*作品合理性得分+ {pca_weights_adjusted[3]:.2f}*平均分')
    
     # 计算作品合理性得分的加权得分
    result_df['新标准分2'] = (
        result_df['平均分'] * fixed_mean_weight +  # 固定 '平均分' 的权重为 0.5
         scaled_data_reasonableness[:, 0] * pca_weights_reasonableness[0] +  # 标准分标准差的权重
         scaled_data_reasonableness[:, 1] * pca_weights_reasonableness[1] +  # 专家专业性总和的权重
         scaled_data_reasonableness[:, 2] * pca_weights_reasonableness[2] +  # 专家平均专业性评分的权重
         scaled_data_reasonableness[:, 3] * pca_weights_reasonableness[3]    # 作品合理性得分
    )

    return result_df

# 计算专家专业性得分
expert_score = calculate_expert_professionalism_score(df)

# 计算每个作品的标准差、专业性、平均专业性
result_df = calculate_standard_deviation_and_professionalism(df, expert_score)

work_metrics_df = determine_work_metrics(df)

# 调用函数计算作品合理性得分
work_metrics_df_with_score = calculate_reasonableness_score(work_metrics_df)

result_df['作品合理性得分'] = work_metrics_df_with_score

# 调计算各指标公式并输出新标准分
result = calculate_weights(result_df)
# 调用函数并传入数据

# 将结果保存为CSV文件
result.to_excel('指标及新标准分结果.xlsx',index=False)
