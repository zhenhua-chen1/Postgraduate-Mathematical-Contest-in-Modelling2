#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 19:16:45 2023

@author: chenzhenhua
"""

import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from factor_analyzer.factor_analyzer import calculate_kmo

import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_data(folder_path, csv_file, xls_file):
    """
    预处理数据函数。
    
    参数:
    - folder_path: 文件夹路径，包含.xsl文件
    - csv_file: csv文件路径
    - xls_file: xls文件路径
    
    返回值:
    - final_data: 预处理后的DataFrame
    """
    
    # 获取文件夹中所有的文件
    all_files = os.listdir(folder_path)

    # 过滤出.xls文件并按年份排序
    xls_files = [file for file in all_files if file.endswith('.xls')]
    xls_files.sort(key=lambda x: int(x.split('年')[0]))  # 以年份排序

    # 循环读取每个文件
    all_dataframes = []
    for file in xls_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_excel(file_path, engine='xlrd')
    
        # 若文件是2022年的数据，仅保留4月份之前的数据
        if "2022年" in file:
            df = df[df["月份"] <= 3]
        
        all_dataframes.append(df)

    # 合并所有的dataframes
    combined_df = pd.concat(all_dataframes, ignore_index=True)

    # 读取CSV和XLS文件
    data1 = pd.read_csv(csv_file, encoding='utf-8')
    data2 = pd.read_excel(xls_file, engine='xlrd')

    # 为每个数据集设置年份和月份作为索引
    data1.set_index(['年份', '月份'], inplace=True)
    data2.set_index(['年份', '月份'], inplace=True)
    combined_df.set_index(['年份', '月份'], inplace=True)

    # 使用merge合并data1和data2
    merged_data = pd.merge(data1, data2, left_index=True, right_index=True, how='outer')

    # 再合并combined_df
    final_data = pd.merge(merged_data, combined_df, left_index=True, right_index=True, how='outer')

    # 重置索引以使年和月再次成为列
    final_data.reset_index(inplace=True)

    # 列表中指定要删除的列
    columns_to_drop = [
        "站点号", "海拔高度(m)", "经度(lon)_x", "经度(lon)_y", "纬度(lat)_x", "纬度(lat)_y", "纬度", "经度"
    ]

    # 从final_data中删除这些列
    final_data = final_data.drop(columns=columns_to_drop)
    
    # 删除包含缺失值的列
    final_data = final_data.dropna(axis=1, how='any')
    
    return final_data




# 读取文件
folder_path = "附件8、锡林郭勒盟气候2012-2022"
csv_file = '附件4、土壤蒸发量2012—2022年.csv'
xls_file = '附件3、土壤湿度2022—2012年.xls'
data = preprocess_data(folder_path, csv_file, xls_file)
data_y =  pd.read_excel('内蒙古自治区锡林郭勒盟典型草原不同放牧强度土壤碳氮监测数据集（2012年8月15日-2020年8月15日）.xlsx')
data_y = data_y [data_y['year'] != 2022 ]
# 1. 数据准备
# 假设你的 DataFrame 是 data，首先我们需要去除非数值列
numeric_data = data.select_dtypes(include=['float64', 'int64'])

# 确保没有缺失值（这里简单地使用填充，但实际应用中可能需要更复杂的处理方法）
numeric_data = numeric_data.fillna(numeric_data.mean())

# 首先，排除2022年的数据
numeric_data = numeric_data[numeric_data['年份'] != 2022]

# 然后，删除 '月份' 列
numeric_data = numeric_data.drop(columns=['月份'])
# 然后，根据 '年份' 列对数据进行分组，并计算每个组的平均值
yearly_average = numeric_data.groupby('年份').mean()
numeric_data = numeric_data.drop(columns=['年份'])
data_y = data_y.select_dtypes(include=['float64', 'int64'])
data_y = data_y.groupby('year').mean()
data_y.to_excel('沙漠化指数.xlsx')
numeric_data.to_excel('第四问指标.xlsx')
# 从指标数据集中选取数值型特征进行PCA分析

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(numeric_data)

# 应用PCA
pca = PCA()
X_pca_transformed = pca.fit_transform(X_scaled)

# 查看解释的方差比例
explained_variance_ratio = pca.explained_variance_ratio_

# 累计方差比例
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

# 获取前五个主成分的权重
pc_weights_new = pca.components_[:5]

# 将权重转换为DataFrame以便更好地展示和解释
pc_weights_df_new = pd.DataFrame(pc_weights_new, columns=numeric_data.columns, index=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])
# 输出所有样本的前五个主成分的转换结果
X_new_pca_transformed_df_full = pd.DataFrame(X_pca_transformed[:, :5], columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])

# 准备将数据写入Excel文件，分别放在不同的sheet中
with pd.ExcelWriter('主成分分析结果.xlsx') as writer:
    pc_weights_df_new.to_excel(writer, sheet_name='PC_Weights')
    X_new_pca_transformed_df_full.to_excel(writer, sheet_name='PC_Transformed')

# 设置绘图风格
plt.style.use('seaborn-talk')

# 创建一个图形和轴
fig, ax = plt.subplots(figsize=(14, 10))

# 绘制每个主成分的权重
# 为matplotlib设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 使用SimHei字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
components = pc_weights_df_new.columns
indices = range(len(components))
for i, pc in enumerate(pc_weights_df_new.index):
    weights = pc_weights_df_new.loc[pc]
    ax.bar(indices, weights, width=0.8, label=pc)

# 添加一些图形装饰
ax.set_xticks(indices)
ax.set_xticklabels(components, rotation=90)
ax.set_ylabel('权重')
ax.set_title('PCA 主成分权重图')
ax.legend(title='主成分')

# 显示图形
plt.tight_layout()
plt.show()