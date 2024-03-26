#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 14:40:24 2024

@author: chenzhenhua
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso


# 转换为监督学习问题
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # 输入序列 (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # 预测序列 (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # 拼接到一起
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # 丢弃包含NaN的行
    if dropnan:
        agg.dropna(inplace=True)
    return agg


#处理数据
data1 = pd.read_excel('附件14：内蒙古自治区锡林郭勒盟典型草原不同放牧强度土壤碳氮监测数据集（2012年8月15日-2020年8月15日）/内蒙古自治区锡林郭勒盟典型草原不同放牧强度土壤碳氮监测数据集（2012年8月15日-2020年8月15日）.xlsx')
data2 = pd.read_excel('问题二结果.xlsx')
data_NDVI = pd.read_excel('附件6、植被指数-NDVI2012-2022年.xls')
data_NDVI = data_NDVI.drop(columns=['经度(lon)', '纬度(lat)'])
data_2022 =  pd.read_excel('第三问结果.xlsx')
data_precipitation  = pd.read_excel('降水量值.xlsx')
data_train = pd.merge(data_precipitation, data_NDVI, on=['月份', '年份'])
years_of_interest = [2012, 2014, 2016, 2018, 2020]
data_train = data_train[data_train['年份'].isin(years_of_interest)]
data_train = data_train.groupby('年份').sum()
data_train_x = data_train.drop(columns=['月份'])
data_train_y = data1.groupby(['放牧强度（intensity）', 'year']).mean(numeric_only=True)
data_train_y = data_train_y .loc['MGI']

# 确保年份和月份是字符串格式，以便可以将它们合并成日期字符串
data_NDVI['年份'] = data_NDVI['年份'].astype(str)
data_NDVI['月份'] = data_NDVI['月份'].astype(str).str.zfill(2)  # 确保月份是两位数的字符串

# 构造日期字符串（假定每月的第一天），然后转换为datetime
data_NDVI['日期'] = pd.to_datetime(data_NDVI['年份'] + '-' + data_NDVI['月份'] + '-01')

# 将新构造的日期设置为索引
data_NDVI.set_index('日期', inplace=True)


# 过滤出2023年的数据
data_2023 = data2[data2['年份'] == 2023]

# 绘制湿度变化图
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 使用SimHei字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
plt.figure(figsize=(10, 6))
plt.plot(data_2023['月份'], data_2023['10cm湿度(kg/m2)'], marker='o', label='10cm湿度(kg/m2)')
plt.plot(data_2023['月份'], data_2023['40cm湿度(kg/m2)'], marker='s', label='40cm湿度(kg/m2)')
plt.plot(data_2023['月份'], data_2023['100cm湿度(kg/m2)'], marker='^', label='100cm湿度(kg/m2)')
plt.plot(data_2023['月份'], data_2023['200cm湿度(kg/m2)'], marker='x', label='200cm湿度(kg/m2)')

plt.title('2023年各月份不同深度的土壤湿度变化')
plt.xlabel('月份')
plt.ylabel('湿度 (kg/m2)')
plt.xticks(data_2023['月份'])
plt.legend()
plt.grid(True)
plt.show()

    
    
    
# 选择NDVI列并进行缩放
values = data_NDVI['植被指数(NDVI)'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_values = scaler.fit_transform(values)

n_months = 12  # 使用过去12个月的数据
supervised = series_to_supervised(scaled_values, n_months, 1)
supervised_values = supervised.values

# 定义训练数据
n_obs = n_months * 1
train_X, train_y = supervised_values[:, :n_obs], supervised_values[:, -1]
# 重塑为3D形状 [样本, 时间步, 特征]
train_X = train_X.reshape((train_X.shape[0], n_months, 1))

# LSTM模型
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

# 拟合网络
model.fit(train_X, train_y, epochs=50, batch_size=72, verbose=2, shuffle=False)



last_input = scaled_values[-n_months:]  # 使用最后n_input个月的数据作为输入
last_input = last_input.reshape((1, n_months, 1))

# 预测未来9个月的NDVI
future_predictions = []
for _ in range(17):  # 从2022年5月到2023年9月
    ndvi_pred = model.predict(last_input, verbose=0)
    future_predictions.append(ndvi_pred[0][0])
    # 这里假设last_input的形状是 (1, n_input, 1)，所以我们需要ndvi_pred也是一个列向量
    ndvi_pred_reshaped = ndvi_pred.reshape(-1, 1).reshape(1, 1, -1)
    
    # 更新输入数据窗口：移除最早的月份数据，加入最新预测的月份数据
    last_input = np.append(last_input[:, 1:, :], ndvi_pred_reshaped, axis=1)
    
# 反缩放预测结果
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
# 创建一个日期范围
date_range = pd.date_range(start='2022-05', periods=len(future_predictions), freq='M')

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(date_range, future_predictions, label='预测的NDVI', marker='o')
plt.xlabel('日期')
plt.ylabel('NDVI')
plt.title('NDVI预测结果')
plt.legend()
plt.xticks(rotation=45)  # 旋转x轴标签以改善显示
plt.tight_layout()  # 自动调整子图参数, 使之填充整个图像区域
plt.show()



#2023年土壤变化预测
#求出
# 示例数据和草场面积已经按公顷给出
data = {
    '牧户1': {'标准羊单位': [275.83, 272.22, 256.39]},
    '牧户2': {'标准羊单位': [230,210,200]},
    '牧户3': {'标准羊单位': [601.75, 492, 520.75]},
    '牧户4': {'标准羊单位': [245.9, 245.9, 255.7]}
}

#对照（NG， 0羊/天/公顷 ）
#轻度放牧强度（LGI， 1-2羊/天/公顷 ）
#中度放牧强度（MGI，3 -4羊/天/公顷 ）
#重度放牧强度（HGI，5-8羊天/公顷 ）

# 计算每个牧户的平均放牧压力（羊/天/公顷）
for herder, values in data.items():
    # 将标准羊单位的年度数据转换为平均值
    average_su = sum(values['标准羊单位']) / len(values['标准羊单位'])
    # 计算放牧压力
    grazing_pressure = average_su *0.01
    print(f"{herder} 的平均放牧压力: {grazing_pressure:.4f} 羊/天/公顷")
    print('选择MGI强度的放牧方式')
    
# 由于数据量小，这里不再划分训练集和测试集
model = Lasso(alpha=2)
model.fit(data_train_x, data_train_y)

#设置输入数据
pred_x = data_2023[~data_2023['月份'].isin([10, 11, 12])].copy()
future_predictions = future_predictions[-9:]
# 创建一个新的DataFrame，包含最后9个预测值
ndvi_df = pd.DataFrame(future_predictions, columns=['植被指数(NDVI)'])
# 你可以直接将NDVI值分配给这些行的相应列
pred_x .loc[:, '植被指数(NDVI)'] = future_predictions
pred_x = pred_x[['10cm湿度(kg/m2)', '40cm湿度(kg/m2)', '100cm湿度(kg/m2)', '200cm湿度(kg/m2)','植被指数(NDVI)']]

# 预测2023土壤
predicted_soil = model.predict(pred_x)
predicted_soil = pd.DataFrame(predicted_soil,columns = ['SOC土壤有机碳', 'SIC土壤无机碳', 'STC土壤全碳', '全N', '土壤C/N比'])

# 假设predicted_soil已经准备好，索引为月份，列名为各个土壤性质
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]  # 2023年1月到9月
properties = ['SOC土壤有机碳', 'SIC土壤无机碳', 'STC土壤全碳', '全N', '土壤C/N比']  # 土壤性质

# 创建图形和轴对象
fig, ax = plt.subplots(figsize=(10, 6))

# 遍历每个土壤性质，绘制每个性质的预测趋势
for property in properties:
    ax.plot(months, predicted_soil[property], label=property)

# 添加图例
ax.legend()

# 添加图表标题和轴标签
ax.set_title('2023年1-9月土壤性质预测')
ax.set_xlabel('月份')
ax.set_ylabel('预测值')

# 显示图表
plt.xticks(rotation=45)  # 旋转x轴标签，以便更清楚地显示
plt.tight_layout()  # 自动调整子图参数，使之填充整个图表区域
plt.show()



