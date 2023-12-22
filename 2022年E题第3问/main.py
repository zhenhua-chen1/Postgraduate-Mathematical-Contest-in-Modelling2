#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 20:59:29 2023

@author: chenzhenhua
"""


import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import f_oneway
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def plot_predictions(y_true, y_pred_transformed):
    plt.figure(figsize=(15, 6))
    plt.plot(y_true, label='True Values', color='blue')
    plt.plot(y_pred_transformed, label='Predicted Values', color='red', linestyle='--')
    plt.title('True Values vs Predicted Values')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()


def lstm_predict(data, n_steps=1):
    # 假设data是一个Pandas DataFrame，并且年份和月份是其前两列
    # 从data中删除年份和月份列
    selected_columns = ['SOC土壤有机碳', 'SIC土壤无机碳', 'STC土壤全碳', '全氮N', '土壤C/N比']
    data = data[selected_columns]
    # 数据预处理
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    # 假设每行是一个时间步，并且所有变量都是连续的时间序列
    # 使用LSTM进行预测时，需要将数据重新形状为(samples, time_steps, features)
    X = scaled_data[:-n_steps]
    y = scaled_data[n_steps:]
    X = X.reshape((X.shape[0], 1, X.shape[1]))

    # 创建LSTM模型
    model = Sequential()
    model.add(LSTM(50, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(y.shape[1]))
    model.compile(optimizer='adam', loss='mae')

    # 训练模型
    model.fit(X, y, epochs=50, verbose=0)

    # 使用模型预测下一个时间点
    x_input = data.values[-n_steps:].reshape((n_steps, 1, data.values.shape[1]))
    yhat = model.predict(x_input, verbose=0)
    yhat = np.array(yhat)
    yhat = scaler.inverse_transform(yhat)
    
    # 计算评价指标
    y_true = scaler.inverse_transform(y)
    y_pred = model.predict(X)
    y_pred_transformed = scaler.inverse_transform(y_pred)
    
    #plot_predictions(y_true, y_pred_transformed)
    
    # mse = mean_squared_error(y_true, y_pred_transformed)
    # mae = mean_absolute_error(y_true, y_pred_transformed)
    # rmse = np.sqrt(mse)
    
    # print("Mean Squared Error (MSE):", mse)
    # print("Mean Absolute Error (MAE):", mae)
    # print("Root Mean Squared Error (RMSE):", rmse)
        
    return  yhat, y_true, y_pred_transformed

data = pd.read_excel('内蒙古自治区锡林郭勒盟典型草原不同放牧强度土壤碳氮监测数据集（2012年8月15日-2020年8月15日）.xlsx')


#相关性分析
# 假设您的原始数据集叫做 data
data['Grazing Intensity'] = data['放牧强度（intensity）']
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
soil_params = ['SOC土壤有机碳', 'SIC土壤无机碳', 'STC土壤全碳', '全氮N', '土壤C/N比']
soil_params_english = ['Soil Organic Carbon (SOC)', 'Soil Inorganic Carbon (SIC) ', 'Total Soil Carbon (STC)', 'Total Nitrogen', 'Soil C/N Ratio']
for i, param in enumerate(soil_params):
    row, col = divmod(i, 3)
    sns.boxplot(x='Grazing Intensity', y=param, data=data, ax=axes[row, col])
    param2 = soil_params_english[i]
    axes[row, col].set_title(f'{param2} vs Grazing Intensity')
    axes[row, col].set_ylabel('')  # 设置空字符串作为y轴标签
    
f_soc,p_soc = f_oneway(*[group["SOC土壤有机碳"].values for name, group in data.groupby("放牧强度（intensity）")])
f_sic,p_sic = f_oneway(*[group["SIC土壤无机碳"].values for name, group in data.groupby("放牧强度（intensity）")])
f_stc,p_stc = f_oneway(*[group["STC土壤全碳"].values for name, group in data.groupby("放牧强度（intensity）")])
f_n,p_n = f_oneway(*[group["全氮N"].values for name, group in data.groupby("放牧强度（intensity）")])
f_cn,p_cn = f_oneway(*[group["土壤C/N比"].values for name, group in data.groupby("放牧强度（intensity）")])
anova_summary = """
ANOVA Results Summary:

- SOC土壤有机碳: F值为{f_soc:.2f}，p值为{p_soc:.2f}
- SIC土壤无机碳: F值为{f_sic:.2f}，p值为{p_sic:.2f}
- STC土壤全碳: F值为{f_stc:.2f}，p值为{p_stc:.2f}
- 全氮N: F值为{f_n:.2f}，p值为{p_n:.2f}
- 土壤C/N比: F值为{f_cn:.2f}，p值为{p_cn:.2f}
""".format(f_soc=f_soc, p_soc=p_soc,
           f_sic=f_sic, p_sic=p_sic,
           f_stc=f_stc, p_stc=p_stc,
           f_n=f_n, p_n=p_n,
           f_cn=f_cn, p_cn=p_cn)
    
print(anova_summary)
# 定义放牧小区列表
plots = ['G17', 'G19', 'G21', 'G6', 'G12', 'G18', 'G8', 'G11', 'G16', 'G9', 'G13', 'G20']

# 初始化一个空字典来存储每个小区的预测结果
predictions = {}

# 对每个放牧小区进行预测
for plot in plots:
    plot_data = data[data['放牧小区（plot）'] == plot]
    y_pred = lstm_predict(plot_data, n_steps=1)
    predictions[plot] = y_pred

# 定义放牧强度和对应的放牧小区
intensity_mapping = {
    'NG': ['G17', 'G19', 'G21'],
    'LGI': ['G6', 'G12', 'G18'],
    'MGI': ['G8', 'G11', 'G16'],
    'HGI': ['G9', 'G13', 'G20']
}

# 初始化一个空的DataFrame
df = pd.DataFrame(columns=['放牧强度', 'Plot放牧小区', 'SOC土壤有机碳', 'SIC土壤无机碳', 'STC土壤全碳', '全N', '土壤C/N比'])

rows = []

# 初始化两个空列表
all_y_true = []
all_y_pred = []

# 填充DataFrame
for intensity, plots in intensity_mapping.items():
    for plot in plots:
        y_pred, y_true, y_pred_transformed = predictions.get(plot, [None, None, None, None, None])  # 获取预测值，如果没有预测值则使用None
        row = {
            '放牧强度': intensity,
            'Plot放牧小区': plot,
            'SOC土壤有机碳': y_pred[0,0],
            'SIC土壤无机碳': y_pred[0,1],
            'STC土壤全碳': y_pred[0,2],
            '全N': y_pred[0,3],
            '土壤C/N比': y_pred[0,4]
        }
        rows.append(row)
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred_transformed)


#评价模型
# 将列表转换为numpy数组
all_y_true = np.array(all_y_true)
all_y_pred = np.array(all_y_pred)

# 计算整体的评价指标
mse = mean_squared_error(all_y_true, all_y_pred)
mae = mean_absolute_error(all_y_true, all_y_pred)
rmse = np.sqrt(mse)

print("Overall Mean Squared Error (MSE):", mse)
print("Overall Mean Absolute Error (MAE):", mae)
print("Overall Root Mean Squared Error (RMSE):", rmse)

#plot_predictions(all_y_true ,all_y_pred)
#输出
df_ture = pd.DataFrame({
    'SOC土壤有机碳': all_y_true[:,0],
    'SIC土壤无机碳': all_y_true[:,1],
    'STC土壤全碳': all_y_true[:,2],
    '全N': all_y_true[:,3],
    '土壤C/N比': all_y_true[:,4]
})

df_pred = pd.DataFrame({
    'SOC土壤有机碳': all_y_pred[:,0],
    'SIC土壤无机碳': all_y_pred[:,1],
    'STC土壤全碳': all_y_pred[:,2],
    '全N': all_y_pred[:,3],
    '土壤C/N比': all_y_pred[:,4]
})
with pd.ExcelWriter('真实值与预测值.xlsx', engine='openpyxl') as writer:
    df_ture.to_excel(writer, sheet_name='真实值', index=False)
    df_pred.to_excel(writer, sheet_name='预测值', index=False)

# 使用pd.concat()添加所有行
df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)

# 显示或保存DataFrame
print(df)
df.to_excel('第三问结果.xlsx', index=False)  # 如果你想保存到Excel文件
