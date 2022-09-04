#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 21:21:09 2022

@author: chenzhenhua
"""

import pandas as pd
import numpy as np


'''
read the data
'''
def DataGet():
    filename='附件1 监测点A空气质量预报基础数据.xlsx'
    sheet_name='监测点A逐日污染物浓度实测数据'
    data = pd.read_excel(filename, sheet_name,index_col = False)
    data1 = data.iloc[497:501,:]
    day_name = data1['监测日期']
    data = data.iloc[497:501,2:]
    pollu_name = ['SO2','NO2','PM10','PM2.5','O3','CO']
    data = np.array(data)
    n_pollution = 6 #污染物数量
    day = 4 #天数
    IaQI=[0,50,100,150,200,300,400,500]#空气质量分指数
    #污染物浓度限值
    limit_pollutant=np.array([[0,50,150,475,800,1600,2100,2620],
                              [0,40,80,180,280,565,750,940],
                              [0,50,150,250,350,420,500,600],
                              [0,35,75,115,150,250,350,500],
                              [0,100,160,215,265,800,5000,5000],
                              [0,50,100,150,200,300,400,500]])

    return data,n_pollution,IaQI,limit_pollutant,day,pollu_name,day_name

def find_index(Cp,i):
    n_id=8
    for n in range(n_id):
        if Cp>=limit_pollutant[i,n] and Cp<=limit_pollutant[i,n+1]:
            Iaqi_L = IaQI[n]
            Iaqi_H = IaQI[n+1]
            Bp_L = limit_pollutant[i,n]
            Bp_H = limit_pollutant[i,n+1]
    return Iaqi_L,Iaqi_H,Bp_L,Bp_H

#读取数据
data,n_pollution,IaQI,limit_pollutant,day,pollu_name,day_name = DataGet()

AqI=[]
name=[]
#计算AQI并得出污染物
for d in range(day):
    Iaqi=[]
    for i in range(n_pollution):
        Cp = data[d,i]
        Iaqi_L,Iaqi_H,Bp_L,Bp_H= find_index(Cp,i)
        Iaqi.append((Iaqi_H-Iaqi_L)/(Bp_H-Bp_L)*(Cp-Bp_L)+Iaqi_L)
    AqI.append(max(Iaqi))
    name.append(pollu_name[Iaqi.index(max(Iaqi))])
    
#剔除无污染
pos = np.where(np.array(AqI)<=50)[0][0]
name[pos] = '无'

#输出
output = {'监测日期': day_name,
          '地点':['监测点A','监测点A','监测点A','监测点A'],
          'AQI':AqI,
          '首要污染物':name}
output = pd.DataFrame(output)
output  = output .reset_index(drop=True)
print(output.to_string(index=False))