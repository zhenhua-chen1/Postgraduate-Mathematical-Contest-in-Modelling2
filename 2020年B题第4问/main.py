#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 09:56:23 2023

@author: chenzhenhua
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt




def sort_varible(data,column_name):
    # 确保column_name是列表
    if not isinstance(column_name, list):
        column_name = list(column_name)
    temp_df = pd.DataFrame(column_name, columns=['位号'])
    # 使用merge方法进行排序
    sorted_data2 = temp_df.merge(data2, on='位号', how='left')
    
    return sorted_data2
    

# 初始化种群
def initialize_population(size, chromosome_length):
    population = np.random.rand(population_size, chromosome_length)
    return population

#解码
def decode_gene(gene, min_value, max_value, delta_value):
    # 将基因值映射到实际参数空间
    decoded_value = min_value + (max_value - min_value) * gene

    # 调整为变化量的整数倍
    n = round((decoded_value - min_value) / delta_value)
    adjusted_value = min_value + n * delta_value

    # 确保调整后的值在最小值和最大值的范围内
    adjusted_value = np.maximum(min_value, np.minimum(adjusted_value, max_value))
    return adjusted_value

def decode_chromosome(chromosome, min_values, max_values, delta_values):
    # 确保输入的染色体长度、最小值、最大值和变化量列表长度相同
    if not (len(chromosome) == len(min_values) == len(max_values) == len(delta_values)):
        raise ValueError("染色体、最小值、最大值和变化量列表的长度必须相同")

    # 解码整个染色体
    decoded_values = [decode_gene(chromosome[i], min_values[i], max_values[i], delta_values[i]) 
                      for i in range(len(chromosome))]
    return decoded_values

# 定义函数来解码整个种群
def decode_population(population, min_values, max_values, delta_values):
    return np.array([decode_chromosome(chromosome, min_values, max_values, delta_values) 
                     for chromosome in population])

#计算适应度函数
def calculate_fitness(prediction, pre_zin):
    fitness = 0
    penalty1 = 10
    penalty2 = 15

    # 遍历每个样本
    for idx, pred in enumerate(prediction):
        # 对第一列应用条件
        if pred[0] > 5:
            fitness += penalty1

        # 对第二列应用条件
        if pred[1] > pre_zin[idx] * 1.3:
            fitness += penalty2

    return fitness

def calculate_population_fitness(population, model,column_name,pre_zin):
    fitness_scores = []
    for chromosome in population:
        fitness = np.sum(chromosome)/(1e+8)
        # 将染色体转换为21x21矩阵
        matrix = chromosome.reshape(21, 21)
        
        # 转换为DataFrame并添加特征名称
        matrix = pd.DataFrame(matrix, columns = column_name)

        # 使用随机森林模型进行预测
        prediction = model.predict(matrix)
        
        # 计算适应度
        fitness += calculate_fitness(prediction,pre_zin)
        fitness_scores.append(fitness)
        

    return  fitness_scores


#选择
def selection(population, fitness_scores):
    num_select = int(len(population)/2)
    # 计算选择概率
    # 适应度分数越小，其倒数越大，选择概率越高
    selection_probs = 1 / np.array(fitness_scores)
    selection_probs /= selection_probs.sum()

    # 选择新的种群
    selected_indices = np.random.choice(range(len(population)), size=len(population), p=selection_probs)
    new_population = population[selected_indices]
    new_population = new_population [:num_select]

    return new_population

#交叉
def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1))
    child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
    child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
    return child1, child2

#变异
def mutate(chromosome, mutation_rate):
    mutated_chromosome = np.copy(chromosome)
    for i in range(len(chromosome)):
        if np.random.rand() < mutation_rate:
            mutated_chromosome[i] = np.random.uniform(0, 1)  # 假设基因是连续值
    return mutated_chromosome


#读取数据
data = pd.read_excel("附件一：325个样本数据.xlsx", header=1)  # 假设第一行是列标题
data2 = pd.read_excel('主要操作变量的操作信息.xlsx')
column_name = pd.read_excel('降维后的主要操作变量.xlsx')['变量']
data_y = data.iloc[1:, [9,11]] # 取辛烷值作为输出
data_x = data[1:][column_name] # 取上一问的操作变量作为输入
data2 = sort_varible(data2,column_name)
# 计算辛烷值降幅
mean_datay = np.mean(data_y.iloc[:,1])
drop_percentage = (data_y.iloc[:,1] - mean_datay)/mean_datay*100

#drop_percentage_series = pd.Series(drop_percentage)

# 筛选辛烷值降幅大于30%的样本的索引（编号）
filtered_sample_indices = drop_percentage[drop_percentage > 30].index

data_y = data_y.iloc[filtered_sample_indices,:]
data_x = data_x.iloc[filtered_sample_indices,:]

min_values = data2['最小值']
max_values = data2['最大值']
min_values_sample = np.zeros([len(data_x),len(column_name)])
pre_zin = np.zeros(len(data_x))
delta_values = data2['Δ值']
index = data_x.index-1

for i1,i2 in enumerate(index):
    min_values_sample[i1,:] = data.loc[i2,column_name] # 取上一问的操作变量作为输入
    pre_zin[i1] = data.iloc[i2,11]
    
# 将数组转换为一维数组并创建为 Series
min_values = pd.Series(min_values_sample.flatten())
# 将每个元素和变化量重复21次
max_values = pd.Series(np.tile(max_values, 21))
delta_values = pd.Series(np.tile(delta_values, 21))

#建立预测模型
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=42)
# 创建支持向量机回归模型
model = DecisionTreeRegressor(random_state=42)
# 训练模型
model.fit(X_train, y_train)

#调用遗传算法进行优化
# 初始化参数
population_size = 100 # 种群大小
chromosome_length = len(column_name)*len(data_y)  # 染色体长度
max_generations = 50  # 最大代数
mutation_rate = 0.01  # 突变率
crossover_rate = 0.7  # 交叉率

    
# 初始化种群
population = initialize_population(population_size, chromosome_length)
best_fitness_history = []
best_result_history = []

offline_best_fitness = []

for generation in range(max_generations):
    # 打印当前迭代次数
    print("Generation:", generation + 1)  # +1 是因为迭代次数从1开始计数
    # 解码种群
    decoded_population = decode_population(population, list(min_values), list(max_values), list(delta_values))
    #计算适应度
    fitness_scores = calculate_population_fitness(decoded_population, model,column_name,pre_zin)
    
    # 记录最佳适应度和对应的染色体
    best_index = np.argmin(fitness_scores)
    best_fitness_history.append(fitness_scores[best_index])
    best_result_history.append(decoded_population[best_index])
    
    # 更新迄今为止的最佳适应度
    if generation == 0 or fitness_scores[best_index] < offline_best_fitness[-1]:
        offline_best_fitness.append(fitness_scores[best_index])
    else:
        offline_best_fitness.append(offline_best_fitness[-1])
    
    
    new_population = selection(population, fitness_scores)
    
    # 交叉生成子代
    offspring = []
    while len(offspring) < population_size/2:
        indices = np.random.choice(len(new_population), 2, replace=False)
        parent1, parent2 = new_population[indices[0]], new_population[indices[1]]
        if np.random.rand() < crossover_rate:
            child1, child2 = crossover(parent1, parent2)
            offspring.append(child1)  # 只选择第一个子代
    
    # 合并选中的染色体和子代
    new_population = np.concatenate([new_population, offspring])
    
    
    population = new_population
    
    #求适应性函数
    fitness_scores = calculate_population_fitness(decoded_population, model,column_name,pre_zin)
    best_index = np.argmax(fitness_scores)

# 绘制在线图
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(best_fitness_history, label='Online Best Fitness')
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.title('Online Best Fitness over Generations')
plt.legend()

# 绘制离线图
plt.subplot(1, 2, 2)
plt.plot(offline_best_fitness, label='Offline Best Fitness', color='red')
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.title('Offline Best Fitness over Generations')
plt.legend()

# 输出最佳适应度和对应的染色体
print("最佳适应度:", best_fitness_history[0])
best_result =  best_result_history[0].reshape(21, 21)
print("最佳结果:", best_result)

best_result = pd.DataFrame(best_result, columns=column_name)
# 添加 "样本编号" 列
best_result['样本编号'] = index+1
# 首先获取除了 '样本编号' 之外的所有列名
other_columns = best_result.columns.tolist()
other_columns.remove('样本编号')

# 调整列顺序，使 '样本编号' 出现在最前面
best_result = best_result[['样本编号'] + other_columns]

# 保存为 CSV 文件，不包括默认的索引
best_result.to_csv('优化后的操作变量.csv', index=False)














