#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 19:32:19 2023

@author: chenzhenhua
"""

import numpy as np
import pandas as pd
import os
import re
from keras.models import Sequential
from keras.layers import SimpleRNN,LSTM, Dense
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import make_interp_spline


def rf_predict(data, n_steps=1):
    data = data.drop(columns=['年份', '月份'])
    
    train_size = int(len(data) * 0.8)
    train, test = data.iloc[0:train_size, :], data.iloc[train_size:, :]
    
    yhat_df_rf = pd.DataFrame(columns=data.columns)
    y_true_df = test.copy()

    # 创建特征和标签
    X, Y = [], []
    for i in range(len(train) - n_steps):
        X.append(train.iloc[i:i + n_steps].values.flatten())
        Y.append(train.iloc[i + n_steps])
    X, Y = np.array(X), np.array(Y)

    # 训练随机森林模型
    model_rf = RandomForestRegressor(n_estimators=200, random_state=100)
    model_rf.fit(X, Y)

    # 使用训练好的模型进行滑动窗口预测
    inputs = train.iloc[-n_steps:].values.flatten()
    predictions = []

    for _ in range(len(test)):
        prediction = model_rf.predict(inputs.reshape(1, -1))
        predictions.append(prediction[0])
        
        # 更新输入
        inputs = np.append(inputs[len(data.columns):], prediction)

    yhat_df_rf = pd.DataFrame(predictions, columns=data.columns)
    
    # 使用测试数据的最后n_steps个点进行预测
    final_predictions = []
    inputs = test.iloc[-n_steps:].values.flatten()

    for _ in range(n_steps):
        prediction = model_rf.predict(inputs.reshape(1, -1))
        final_predictions.append(prediction[0])
        
        # 更新输入
        inputs = np.append(inputs[len(data.columns):], prediction)

    return np.array(final_predictions), np.array(yhat_df_rf), np.array(y_true_df)

def arima_predict(data, n_steps=1):
    # 假设data是一个Pandas DataFrame，并且年份和月份是其前两列
    data = data.drop(columns=['年份', '月份'])
    
    # 根据n_steps设定order
    if n_steps == 1:
        order = (1,1,1)
    elif n_steps == 2:
        order = (2,1,2)
    else:
        order = (1,1,1)
    
    yhat_df = pd.DataFrame()
    y_true_df = pd.DataFrame()
    y_pred_transformed_df = pd.DataFrame()
    
    for column in data.columns:
        series = data[column]
        
        # 划分数据为训练集和测试集
        train_size = int(len(series) * 0.9)
        train, test = series[0:train_size], series[train_size:]
        
        # 数据预处理
        scaler = MinMaxScaler()
        scaled_train = scaler.fit_transform(train.values.reshape(-1, 1))
        
        # 创建ARIMA模型
        model = ARIMA(scaled_train, order=order)
        model_fit = model.fit()

        # 使用模型预测接下来的n_steps
        yhat = model_fit.forecast(steps=len(test)+n_steps)
        yhat_rescaled = scaler.inverse_transform(yhat.reshape(-1, 1))
        yhat = yhat_rescaled[len(test):]
        yhat_rescaled = yhat_rescaled[0:len(test)]
        #pdb.set_trace()

        
        yhat_df[column] = yhat.ravel()
        y_true_df[column] = test.values.ravel()
        y_pred_transformed_df[column] = yhat_rescaled.ravel()
    #pdb.set_trace() 
    return np.array(yhat_df), np.array(y_true_df), np.array(y_pred_transformed_df)

def evaluate_and_predict(data, predict_method):
    all_predictions_df = []
    all_y_true = []
    all_y_pred = []

    for month in range(1, 13):
        print(f"预测 {month} 月份...")
        month_data = data[data['月份'] == month]

        if month <= 3:
            predictions, y_true, y_pred_transformed = predict_method(month_data, 1)
            #pdb.set_trace()
            df = pd.DataFrame({
                '年份': [2023],
                '月份': [month],
                '10cm湿度(kg/m2)': [predictions[0, 2]],
                '40cm湿度(kg/m2)': [predictions[0, 3]],
                '100cm湿度(kg/m2)': [predictions[0, 4]],
                '200cm湿度(kg/m2)': [predictions[0, 5]]
            })
        else:
            #pdb.set_trace()
            predictions, y_true, y_pred_transformed = predict_method(month_data, 2)
            #pdb.set_trace()
            df = pd.DataFrame({
                '年份': [2022, 2023],
                '月份': [month, month],
                '10cm湿度(kg/m2)': predictions[:, 2],
                '40cm湿度(kg/m2)': predictions[:, 3],
                '100cm湿度(kg/m2)': predictions[:, 4],
                '200cm湿度(kg/m2)': predictions[:, 5]
            })
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred_transformed)
        all_predictions_df.append(df)  
    mse = mean_squared_error(all_y_true, all_y_pred)
    mae = mean_absolute_error(all_y_true, all_y_pred)
    r2_scores = r2_score(all_y_true, all_y_pred)
    rmse = np.sqrt(mse)
    
    # 使用正则表达式提取
    match = re.search(r"<function (\w+)", str(predict_method))
    if match:
        function_name = match.group(1)

    print(function_name+":Mean Squared Error (MSE):", mse)
    print(function_name+":Mean Absolute Error (MAE):", mae)
    print(function_name+":Root Mean Squared Error (RMSE):", rmse)
    print(function_name+":r2_scores:", r2_scores)

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    df_true = pd.DataFrame({
        '10cm湿度(kg/m2)': all_y_true[:, 2],
        '40cm湿度(kg/m2)': all_y_true[:, 3],
        '100cm湿度(kg/m2)': all_y_true[:, 4],
        '200cm湿度(kg/m2)': all_y_true[:, 5]
    })

    df_pred = pd.DataFrame({
        '10cm湿度(kg/m2)': all_y_pred[:, 2],
        '40cm湿度(kg/m2)': all_y_pred[:, 3],
        '100cm湿度(kg/m2)': all_y_pred[:, 4],
        '200cm湿度(kg/m2)': all_y_pred[:, 5]
    })
    df_ture = pd.DataFrame({
          '10cm湿度(kg/m2)': all_y_true[:, 2],
          '40cm湿度(kg/m2)': all_y_true[:, 3],
          '100cm湿度(kg/m2)': all_y_true[:, 4],
          '200cm湿度(kg/m2)': all_y_true[:, 5]
    })

    df_test = pd.DataFrame({
          '10cm湿度(kg/m2)': all_y_pred[:, 2],
          '40cm湿度(kg/m2)': all_y_pred[:, 3],
          '100cm湿度(kg/m2)': all_y_pred[:, 4],
          '200cm湿度(kg/m2)': all_y_pred[:, 5]
    })

    return df_true, df_pred, mse, mae, rmse,r2_scores,df_ture,df_test,all_predictions_df,function_name 

def plot_predictions(y_trues, y_preds_transformed):
    """
    y_trues: DataFrame of true value columns
    y_preds_transformed: DataFrame of predicted value columns
    """
    plt.figure(figsize=(15, 6))
    
    for label in y_trues.columns:
        y_true = y_trues[label].values
        y_pred = y_preds_transformed[label].values
        
        # 检查列是否为空
        if len(y_true) == 0 or len(y_pred) == 0:
            print(f"Skipping {label} due to lack of data.")
            continue
        
        # 定义更密集的x轴数据点，以获得平滑的spline
        xnew = np.linspace(0, len(y_true)-1, len(y_true)*10)
        
        # 使用spline进行插值
        spl_true = make_interp_spline(range(len(y_true)), y_true, k=3)  # k=3 表示cubic spline
        spl_pred = make_interp_spline(range(len(y_pred)), y_pred, k=3)
        
        smooth_true = spl_true(xnew)
        smooth_pred = spl_pred(xnew)
        
        plt.plot(xnew, smooth_true, label=label + ' True Values', color=np.random.rand(3,))
        plt.plot(xnew, smooth_pred, label=label + ' Predicted Values', linestyle='--', color=np.random.rand(3,))
    
    plt.title('True Values vs Predicted Values')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

def rnn_predict(data, n_steps=1):
    # 假设data是一个Pandas DataFrame，并且年份和月份是其前两列
    # 从data中删除年份和月份列
    data = data.drop(columns=['年份', '月份'])
    
    # 数据预处理
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    # 假设每行是一个时间步，并且所有变量都是连续的时间序列
    # 使用RNN进行预测时，需要将数据重新形状为(samples, time_steps, features)
    X = scaled_data[:-n_steps]
    y = scaled_data[n_steps:]
    X = X.reshape((X.shape[0], 1, X.shape[1]))

    # 创建RNN模型
    model = Sequential()
    model.add(SimpleRNN(100, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(y.shape[1]))
    model.compile(optimizer='adam', loss='mae')

    # 训练模型
    model.fit(X, y, epochs=100, verbose=0)

    # 使用模型预测下一个时间点
    x_input = data.values[-n_steps:].reshape((n_steps, 1, data.values.shape[1]))
    yhat = model.predict(x_input, verbose=0)
    yhat = np.array(yhat)
    yhat = scaler.inverse_transform(yhat)
    
    # 计算评价指标
    y_true = scaler.inverse_transform(y)
    y_pred = model.predict(X)
    y_pred_transformed = scaler.inverse_transform(y_pred)
    
    return yhat, y_true, y_pred_transformed

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
    model.add(LSTM(100, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(y.shape[1]))
    model.compile(optimizer='adam', loss='mae')

    # 训练模型
    model.fit(X, y, epochs=100, verbose=0)

    # 使用模型预测下一个时间点
    x_input = data.values[-n_steps:].reshape((n_steps, 1, data.values.shape[1]))
    yhat = model.predict(x_input, verbose=0)
    yhat = np.array(yhat)
    yhat = scaler.inverse_transform(yhat)
    
    # 计算评价指标
    y_true = scaler.inverse_transform(y)
    y_pred = model.predict(X)
    y_pred_transformed = scaler.inverse_transform(y_pred)
    #pdb.set_trace()
    return yhat, y_true, y_pred_transformed

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
    
#调用各种模型预测
df_true1, df_pred1, mse1, mae1, rmse1,r2_scores1,df_ture1,df_test1,all_predictions_df1,function_name1 = evaluate_and_predict(data, lstm_predict)
df_true2, df_pred2, mse2, mae2, rmse2,r2_scores2,df_ture2,df_test2,all_predictions_df2,function_name2 = evaluate_and_predict(data, rnn_predict) 
df_true3, df_pred3, mse3, mae3, rmse3,r2_scores3,df_ture3,df_test3,all_predictions_df3,function_name3 = evaluate_and_predict(data, rf_predict) 
df_true4, df_pred4, mse4, mae4, rmse4,r2_scores4,df_ture4,df_test4,all_predictions_df4,function_name4 = evaluate_and_predict(data, arima_predict)

mse_values = [mse1, mse2, mse3,mse4]
min_index = mse_values.index(min(mse_values))

if  min_index == 0:
    df_true = df_true1
    df_pred = df_pred1
    mse = mse1
    mae = mae1
    rmse = rmse1
    r2_scores = r2_scores1
    df_ture = df_ture1
    df_test = df_test1
    all_predictions_df = all_predictions_df1
    function_name = function_name1
elif min_index == 1:
    df_true = df_true2
    df_pred = df_pred2
    mse = mse2
    mae = mae2
    rmse = rmse2
    r2_scores = r2_scores2
    df_ture = df_ture2
    df_test = df_test2
    all_predictions_df = all_predictions_df2
    function_name = function_name2
elif min_index == 2:
    df_true = df_true3
    df_pred = df_pred3
    mse = mse3
    mae = mae3
    rmse = rmse3
    r2_scores = r2_scores3
    df_ture = df_ture3
    df_test = df_test3
    all_predictions_df = all_predictions_df3
    function_name = function_name3
else:
    df_true = df_true4
    df_pred = df_pred4
    mse = mse4
    mae = mae4
    rmse = rmse4
    r2_scores = r2_scores4
    df_ture = df_ture4
    df_test = df_test4
    all_predictions_df = all_predictions_df4
    function_name = function_name4
        

print('使用'+function_name+"预测")
    
  
with pd.ExcelWriter('真实值与预测值.xlsx', engine='openpyxl') as writer:
        df_ture.to_excel(writer, sheet_name='真实值', index=False)
        df_test.to_excel(writer, sheet_name='预测值', index=False)

labels = ['10cm湿度(kg/m2)','40cm湿度(kg/m2)','100cm湿度(kg/m2)','200cm湿度(kg/m2)']        
plot_predictions(df_ture,  df_test)       
# 合并年份和月份列，以形成一个新的日期列
data['日期'] = pd.to_datetime(data['年份'].astype(str) + '-' + data['月份'].astype(str) + '-01')

# 设置图形大小
plt.figure(figsize=(10, 5))

# 绘制图表
# 为matplotlib设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 使用SimHei字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
plt.plot(data['日期'], data['降水量(mm)'], marker='o')
plt.title('降水量 vs 日期')
plt.xlabel('日期')
plt.ylabel('降水量（mm）')
plt.grid(True)
plt.tight_layout()
plt.show()

# 评价指标列表
metrics = ['mse', 'mae', 'rmse', 'r2_scores']

# 初始化一个空的DataFrame
df_metrics = pd.DataFrame(columns=['function_name', 'mse', 'mae', 'rmse', 'r2_scores'])

# 将每个函数的评价指标添加到DataFrame
for i in range(1, 5):  # 因为您有4个函数
    function_name_var = eval(f'function_name{i}')
    mse_var = eval(f'mse{i}')
    mae_var = eval(f'mae{i}')
    rmse_var = eval(f'rmse{i}')
    r2_scores_var = eval(f'r2_scores{i}')  # 如果r2_scores是一个数字
    # 添加到DataFrame
    df_metrics = df_metrics.append({'function_name': function_name_var, 'mse': mse_var, 'mae': mae_var, 
                                    'rmse': rmse_var, 'r2_scores': r2_scores_var}, ignore_index=True)

# 重设索引
df_metrics.set_index('function_name', inplace=True)

# 输出表格
print(df_metrics)

# 合并所有的预测DataFrame
result = pd.concat(all_predictions_df, ignore_index=True)
result_sorted = result.sort_values(by=['年份', '月份'], ascending=[True, True]).reset_index(drop=True)
# 保存为Excel文件
result_sorted.to_excel('问题二结果.xlsx', index=False)
df_metrics.to_excel('评价指标结果.xlsx')


