#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 10:33:51 2023

@author: chenzhenhua
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


#数据预处理
data1 = pd.read_excel("附件一：325个样本数据.xlsx")
new_header = data1.iloc[0] # 抓取第一行作为标题
data = data1[2:] # 取剩下的数据
data.columns = new_header # 设置新的列标题
# 选择操作变量
data = data.iloc[:, 16:] # 
data1 = data.copy()


#检验异常值
# 计算每列的均值和标准差
mean = data1.mean()
std = data1.std()

# 定义异常值的阈值
upper_bound = mean + 3 * std
lower_bound = mean - 3 * std

# 识别异常值
outliers = (data1 < lower_bound) | (data1 > upper_bound)
threshold = 0.05 * data1.shape[0]  # 如果一个变量中超过5%的数据是异常值，就删除该变量
# 处理异常值
for column in data1.columns:
    if outliers[column].sum() > threshold:  # 假设 threshold 是你设定的异常值数量阈值
        # 异常值太多，删除整个变量
        data1.drop(column, axis=1, inplace=True)
    else:
        # 异常值较少，用平均值替换
        mean_val = data1[column][~outliers[column]].mean()
        data1.loc[outliers[column], column] = mean_val
        
# 设置零值的比例阈值，例如20%
zero_threshold = 0.20

# 对于每个变量，计算零值的比例并删除超过阈值的变量
for column in data1.columns:
    zero_proportion = (data1[column] == 0).sum() / len(data1)
    if zero_proportion > zero_threshold:
        data1.drop(column, axis=1, inplace=True)

# 转换所有列为数值类型，无法转换的设置为 NaN
for column in data1.columns:
    data1[column] = pd.to_numeric(data1[column], errors='coerce')

#标准化数据
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data1)

# 2. 应用PCA降维
pca = PCA() # 或者 pca = PCA(0.95)
pca_data = pca.fit_transform(scaled_data)
explained_variance = pca.explained_variance_ratio_
# 计算累计贡献率
cumulative_variance = np.cumsum(explained_variance)
# 找出累计贡献率达到85%的主成分数目
n_components = np.where(cumulative_variance >= 0.85)[0][0] + 1
# 使用找到的主成分数目重新进行PCA
pca = PCA(n_components=n_components)
pca_data_reduced = pca.fit_transform(scaled_data)

# 可视化结果
#可视化贡献率
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='-', color='b')
plt.title('Scree Plot')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
#plt.show()

#可视化载荷系数
loadings = pca.components_
num_components = loadings.shape[0]
top_features = 5  # 选择每个主成分的前5个载荷
for i in range(num_components):
    component_loadings = loadings[i]
    sorted_indices = np.argsort(np.abs(component_loadings))[::-1][:top_features]
    top_loadings = component_loadings[sorted_indices]
    feature_names = np.array(data1.columns)[sorted_indices]

    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, top_loadings, color='skyblue')
    plt.xlabel('Loadings')
    plt.title(f'Top {top_features} Loadings for Component {i+1}')
    plt.gca().invert_yaxis()
plt.show()


#输出主要降维的操作变量
feature_names = data1.columns
# 创建用于存储数据的DataFrame
components_df = pd.DataFrame(columns=['主成分', '变量', '载荷系数'])

# 对于每个主成分，找出载荷最高的变量
for i, component in enumerate(loadings):
    max_loading_idx = np.argmax(np.abs(component))  # 找到最大载荷的索引
    max_loading_feature = feature_names[max_loading_idx]  # 获取对应的变量名称
    max_loading_value = component[max_loading_idx]  # 获取载荷值

    print(f"Component {i+1}: Highest loading is {max_loading_feature} ({max_loading_value})")
    # 创建一个新的DataFrame来存储当前主成分的信息
    new_row = pd.DataFrame({'主成分': [f'Component {i+1}'],
                            '变量': [max_loading_feature],
                            '载荷系数': [max_loading_value]})

    # 使用concat代替append
    components_df = pd.concat([components_df, new_row], ignore_index=True)

# 输出到Excel文件
components_df.to_excel('降维后的主要操作变量.xlsx', index=False)
