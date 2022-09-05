#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 15:18:14 2022

@author: chenzhenhua
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

def DataGet():
    filename ='附件1 监测点A空气质量预报基础数据.xlsx'
    sheet_name ='监测点A逐小时污染物浓度与气象实测数据'
    data = pd.read_excel(filename, sheet_name,index_col = False)
    pollution = data.iloc[:,2:]
    pollution = pollution.dropna(axis = 0, how = 'all')
    n_pollution = 6
    IaQI=[0,50,100,150,200,300,400,500]#空气质量分指数
    #污染物浓度限值
    limit_pollutant=np.array([[0,50,150,475,800,1600,2100,2620],
                              [0,40,80,180,280,565,750,940],
                              [0,50,150,250,350,420,500,600],
                              [0,35,75,115,150,250,350,500],
                              [0,100,160,215,265,800,5000,5000],
                              [0,50,100,150,200,300,400,500]])

    return data,pollution,IaQI,limit_pollutant,n_pollution

def find_index(Cp,i):
    n_id=8
    for n in range(n_id):
        if Cp>=limit_pollutant[i,n] and Cp<=limit_pollutant[i,n+1]:
            Iaqi_L = IaQI[n]
            Iaqi_H = IaQI[n+1]
            Bp_L = limit_pollutant[i,n]
            Bp_H = limit_pollutant[i,n+1]
    return Iaqi_L,Iaqi_H,Bp_L,Bp_H

print('读取数据')
data,pollution,IaQI,limit_pollutant,n_pollution = DataGet()

sample = pollution.shape[0]
variable = pollution.columns.values
N_variable = pollution.shape[1]
weather_name = variable[6:]

print('开始数据预处理')
#数据预处理
imputer = KNNImputer(n_neighbors = 5)   
pollution = imputer.fit_transform(pollution)
pollution2 = pollution[:,0:6]
#处理-1
a=np.where(np.array(pollution2)<0)[0]
b=np.where(np.array(pollution2)<0)[1]
pollution2[a,b] = 0

Weather =  pollution[:,6:].copy()

#计算AQI
print('开始计算AQI')
AqI=[]
for d in range(sample):
    Iaqi=[]
    for i in range(n_pollution):
        Cp = pollution[d,i]
        Iaqi_L,Iaqi_H,Bp_L,Bp_H= find_index(Cp,i)
        Iaqi.append((Iaqi_H-Iaqi_L)/(Bp_H-Bp_L)*(Cp-Bp_L)+Iaqi_L)
    AqI.append(max(Iaqi))
 

#归一化
scaler = MinMaxScaler(feature_range=(-1, 1))  #将数据归一到0到1，可以根据数据特点归一到-1到1
mydata = scaler.fit_transform(Weather)  #归一化
mydata2 = scaler.fit_transform(pollution)  #归一化

mydata2 = pd.DataFrame(mydata2,columns=['SO2监测浓度(μg/m³)','NO2监测浓度(μg/m³)','PM10监测浓度(μg/m³)','PM2.5监测浓度(μg/m³)',	'O3监测浓度(μg/m³)',	'CO监测浓度(mg/m³)','温度(℃)','湿度(%)','气压(MBar)','风速(m/s)','风向(°)'])
mydata2_corr = mydata2 .corr("pearson")
print('相关系数为：')
print(mydata2 .corr("pearson"))




#调用KMeans聚类算法
print('开始聚类')
clf = KMeans(n_clusters=5)
lab1 = clf.fit(mydata)
lab = lab1.labels_
cluter_center_data=clf.cluster_centers_

#输出结果
#保存AQI
Aqi_mean=[]
for i in range(5):
    pos=np.where(lab==i)[0]
    Aqi_mean.append(np.mean(np.array(AqI)[pos]))
    
#保存5个天气指标
result_dict = {}
for key in range(5):
    key_value=[]
    for i in range(5):
        pos=np.where(lab==i)[0]
        key_value.append(np.mean(Weather[pos,key]))
    result_dict[weather_name[key]] = key_value
result_dict['AQI'] = Aqi_mean
result_dict = pd.DataFrame(result_dict,index=['第一类','第二类','第三类','第四类','第五类'])
print(result_dict)

#输出文件
writer = pd.ExcelWriter('相关系数.xlsx')
mydata2_corr.to_excel(writer)
writer.save()