#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 19:32:19 2023

@author: chenzhenhua
"""

import numpy as np
import pandas as pd
import os
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler


def lstm_predict(data, n_steps=1):
    # 假设data是一个Pandas DataFrame，并且年份和月份是其前两列
    # 从data中删除年份和月份列
    data = data.drop(columns=['年份', '月份'])
    
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
    
    return yhat

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
    
all_predictions_df = []

# 循环遍历1到12月份的数据
for month in range(1, 13):
    print(f"预测 {month} 月份...")  # 这里显示进度
    # 提取特定月份的数据
    month_data = data[data['月份'] == month]

    # 根据月份使用适当的LSTM函数进行预测
    if month <= 3:
        # 对于1到3月份，滑动时间窗口一次
        predictions = lstm_predict(month_data,1)
        df = pd.DataFrame({
            '年份': [2023],  # 使用2022年
            '月份': [month],
            '10cm湿度(kg/m2)': [predictions[0, 2]],
            '40cm湿度(kg/m2)': [predictions[0, 3]],
            '100cm湿度(kg/m2)': [predictions[0, 4]],
            '200cm湿度(kg/m2)': [predictions[0, 5]]
        })
    else:
        # 对于其余月份，滑动时间窗口两次
        predictions = lstm_predict(month_data,2)
        df = pd.DataFrame({
            '年份': [2022, 2023],  # 使用2022年和2023年
            '月份': [month, month],
            '10cm湿度(kg/m2)': predictions[:, 2],
            '40cm湿度(kg/m2)': predictions[:, 3],
            '100cm湿度(kg/m2)': predictions[:, 4],
            '200cm湿度(kg/m2)': predictions[:, 5]
        })
    # 将新的DataFrame添加到列表中
    all_predictions_df.append(df)

# 合并所有的预测DataFrame
result = pd.concat(all_predictions_df, ignore_index=True)
result_sorted = result.sort_values(by=['年份', '月份'], ascending=[True, True]).reset_index(drop=True)
print(result)
# 保存为Excel文件
result_sorted.to_excel('问题二结果.xlsx', index=False)


