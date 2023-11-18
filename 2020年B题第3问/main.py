#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 09:53:55 2023

@author: chenzhenhua
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#读取数据
data = pd.read_excel("附件一：325个样本数据.xlsx", header=1)  # 假设第一行是列标题
column_name = pd.read_excel('降维后的主要操作变量.xlsx')['变量']
data_y = data.iloc[2:, 11].ravel()  # 取辛烷值作为输出
data_x = data[2:][column_name] # 取上一问的操作变量作为输入

#预测模型

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=42)

# 初始化模型
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Support Vector Regression": SVR()
}

# 使用同样的模型设置和训练过程

results = pd.DataFrame(columns=['Model', 'MSE', 'RMSE', 'MAE', 'R² Score'])

# 训练和评估模型
for name, model in models.items():
    # 训练模型
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    new_results = pd.DataFrame({'Model': [name],
                                 'MSE': [mse], 
                                 'RMSE': [rmse], 
                                 'MAE': [mae], 
                                 'R² Score': [r2]})
    # 评估
    results = pd.concat([results, new_results], ignore_index=True)
    
    
# 输出结果
print(results)
    
# 找出性能最佳的模型
best_model = results.loc[results['MSE'].idxmin()]
print(f"The best model is {best_model['Model']} with a MSE of {best_model['MSE']}")
    
    