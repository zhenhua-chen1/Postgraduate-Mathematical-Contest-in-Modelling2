#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 17:03:45 2024

@author: chenzhenhua
"""

import numpy as np
import pandas as pd
import os

# 分组并计算差分
def 计算差分(group):
    # 按轮次排序
    group = group.sort_values('轮次排序')
    # 计算“鲜重(g)”和“干重(g)”的差分
    for col in ['鲜重(g)', '干重(g)']:
        group[col + ' 差分'] = group[col].diff()
    return group

# 读取数据
data = pd.read_excel('附近15.xlsx')
grouped_avg = data.groupby(['年份', '轮次', '植物群落功能群', '重复', '处理']).agg({
    '鲜重(g)': 'mean',
    '干重(g)': 'mean',
    '放牧小区Block': 'first'
}).reset_index()

# 映射轮次到有序数值
轮次映射 = {'牧前': 0, '第一轮牧后': 1, '第二轮牧后': 2, '第三轮牧后': 3, '第四轮牧后': 4}
grouped_avg ['轮次排序'] = grouped_avg ['轮次'].map(轮次映射)

# 应用分组和差分计算
result = grouped_avg.groupby(['年份', '植物群落功能群', '重复', '处理']).apply(计算差分)


result = result.dropna(subset=['干重(g) 差分'])

# 按年份拆分DataFrame
dfs = {}
for year in range(2016, 2021):
    dfs[year] = result[result['年份'] == year]

grouped_dfs = {}

for year, df in dfs.items():
    # 重置索引以确保'重复'不是索引的一部分
    df_reset = df.reset_index(drop=True)
    
    # 对每个年份的DataFrame进行分组聚合
    grouped_df = df_reset.groupby(['轮次', '重复', '处理']).agg({
        '鲜重(g)': 'mean',
        '干重(g)': 'mean'
    }).reset_index()
    
    # 将处理后的DataFrame存储在一个新的字典中，保持年份区分
    grouped_dfs[year] = grouped_df



folder_path = '附件8、锡林郭勒盟气候2012-2022'

# 初始化一个空的DataFrame来存储降水量数据
precipitation_data = pd.DataFrame()

# 遍历文件夹中的每个文件
for year in range(2012, 2023):  # 从2012年至2022年
    file_name = f'{year}年.xls'  # 构建文件名
    file_path = os.path.join(folder_path, file_name)  # 构建文件完整路径
    
    # 检查文件是否存在
    if os.path.exists(file_path):
        # 读取文件
        df = pd.read_excel(file_path)
        
        # 假设降水量列名为"降水量"，根据实际情况可能需要调整
        if "降水量(mm)" in df.columns and '月份' in df.columns:
            # 提取降水量数据，并添加到总的DataFrame中
            # 可以根据需要添加额外的标识信息，例如年份
            df['年份'] = year  # 添加年份作为一列
            precipitation_data = pd.concat([precipitation_data, df[['年份', '月份','降水量(mm)']]], ignore_index=True)

#建立优化模型
# 筛选2016～2020年的数据
filtered_data = precipitation_data[(precipitation_data['年份'] >= 2016) & (precipitation_data['年份'] <= 2020)]

# 按年份分组求和
annual_precipitation_sum = filtered_data.groupby('年份')['降水量(mm)'].sum().reset_index()

# 定义修改第一轮"处理"值的为放牧羊的数量/公顷
first_round_mapping = {0: 3, 3: 150, 6: 200, 12: 250}

# 遍历dfs字典中的每个DataFrame
for year, df in dfs.items():
    # 复制DataFrame以避免直接修改原始数据
    df_modified = df.copy()
    
    # 更新第一轮"处理"的值
    for key, value in first_round_mapping.items():
        df_modified.loc[(df_modified['轮次'] == '第一轮牧后') & (df_modified['处理'] == key), '处理'] = value
    
    # 根据轮次顺序累加"处理"值
    rounds = ['第一轮牧后', '第二轮牧后', '第三轮牧后', '第四轮牧后']
    for i in range(1, len(rounds)):
        previous_round = rounds[i - 1]
        current_round = rounds[i]
        
        # 对每个重复的值，累加前一轮的"处理"值
        for repeat in df_modified['重复'].unique():
            previous_value = df_modified.loc[(df_modified['轮次'] == previous_round) & (df_modified['重复'] == repeat), '处理'].sum()
            df_modified.loc[(df_modified['轮次'] == current_round) & (df_modified['重复'] == repeat), '处理'] += previous_value
    
    # 将修改后的DataFrame保存回dfs字典
    dfs[year] = df_modified
    
# 假设目标降水量
target_precipitations = [300, 600, 900, 1200]

# 找到最接近目标降水量的年份
closest_years = {}
for target in target_precipitations:
    closest_year = annual_precipitation_sum.iloc[(annual_precipitation_sum['降水量(mm)'] - target).abs().argsort()[:1]]['年份'].values[0]
    closest_years[target] = closest_year

# 确定与目标降水量最接近的年份并输出推荐的放牧方式
max_dry_weight_samples = {}
for target, year in closest_years.items():
    if year in dfs:
        # 获取对应年份的DataFrame
        df_year = dfs[year]
        # 找到干重最大的样本的索引
        max_dry_weight_index = df_year['干重(g)'].idxmax()
        # 提取这个样本的所有信息
        max_dry_weight_sample = df_year.loc[max_dry_weight_index]
        # 存储结果
        max_dry_weight_samples[year] = max_dry_weight_sample
        best_grazing_method = df_year.loc[max_dry_weight_index, '处理']
        print(f"降水量 {target}mm 最适合的放牧量为：{best_grazing_method}羊/公顷。")