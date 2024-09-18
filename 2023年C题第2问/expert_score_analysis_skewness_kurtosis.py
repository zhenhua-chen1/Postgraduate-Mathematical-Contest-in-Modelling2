#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 11:24:52 2024

@author: chenzhenhua
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis

# 读取数据
df = pd.read_excel('数据1.xlsx', 'Sheet2')

# 提取所有以 '专家编码' 和 '原始分' 开头的列
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

plt.rcParams['font.family'] = ['Arial Unicode MS']  # 正常显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

# 绘制散点图，用偏度作为x轴，峰度作为y轴
plt.figure(figsize=(8, 5))
plt.scatter(grouped_result['偏度'], grouped_result['峰度'], color='blue', edgecolor='black', marker='x')

# 标注每个点的专家编码
for i, row in grouped_result.iterrows():
    plt.text(row['偏度'], row['峰度'], row['专家编码'], fontsize=9)

plt.title('打分偏度与峰度的散点图')
plt.xlabel('偏度')
plt.ylabel('峰度')
plt.grid(True)
plt.show()

# 计算分数列表长度，并根据长度选择前5个专家
grouped_result['分数列表长度'] = grouped_result['分数列表'].apply(len)

# 按照分数列表长度排序并选择前5个专家
top_5_experts = grouped_result.sort_values(by='分数列表长度', ascending=False).head(5)

# 过滤出前5个专家的数据
experts_to_plot = top_5_experts['专家编码']
filtered_data = grouped_result[grouped_result['专家编码'].isin(experts_to_plot)]


# 为前5个专家绘制直方图，并叠加KDE曲线
for index, row in filtered_data.iterrows():
    plt.figure(figsize=(8, 5))
    
    # 绘制频数直方图并叠加KDE曲线
    sns.histplot(row['分数列表'], bins=50, kde=True, color='blue', edgecolor='black', stat='count')
    
    plt.title(f'专家编码 {row["专家编码"]} 的打分分布')
    plt.xlabel('分数')
    plt.ylabel('频数')
    plt.grid(True)
    plt.show()