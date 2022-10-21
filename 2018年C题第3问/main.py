#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 16:13:06 2022

@author: chenzhenhua
"""
import pandas as pd
import numpy as np
import pmdarima as pm
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def predict_XY(x):
    y=[]
    for i in range(len(x)):
        if ((i+1) % 3) == 0:
            arima = pm.auto_arima(np.array(x.iloc[0:i+2]))
            preds = arima.predict(n_periods=1)
            y.append(preds[0])
    y=pd.Series(y)
    return y



data1 = pd.read_excel("指标总危害性.xlsx")
data2 = pd.read_excel('恐怖袭击空间分布.xlsx')
x1 = data1['危害度']
x2 = data1['事件数量']
x3 = data2['严重性']
x4 = data2['数量']
y1=predict_XY(x1)
y2=predict_XY(x2)
y3=predict_XY(x3)
y4=predict_XY(x4)
y5=list(range(1,13,1))
#转换成dataframe输出
dict1 = {
            '月份': y5,
            '月份危害度': y1,
            '月份数量':y2,
            '空间危害度':y3,
            '空间数量':y4
}
data3=pd.DataFrame(dict1)

writer = pd.ExcelWriter('2018年预测结果.xlsx')
data3.to_excel(writer, index = False)
writer.save()
        




 
