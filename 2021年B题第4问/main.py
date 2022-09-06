#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 15:19:04 2022

@author: chenzhenhua
"""

import pandas as pd
import numpy as np
import math
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

#处理数据
def DataGet():
    filename1 ='附件1 监测点A空气质量预报基础数据.xlsx'
    filename2 ='附件3 监测点A1、A2、A3空气质量预报基础数据.xlsx'
    sheet_name1 = '监测点A逐小时污染物浓度与气象一次预报数据'
    sheet_name2 ='监测点A逐小时污染物浓度与气象实测数据'
    sheet_name3 = '监测点A1逐小时污染物浓度与气象一次预报数据'
    sheet_name4 = '监测点A2逐小时污染物浓度与气象一次预报数据'
    sheet_name5 = '监测点A3逐小时污染物浓度与气象一次预报数据'
    sheet_name6 = '监测点A1逐小时污染物浓度与气象实测数据'
    sheet_name7 = '监测点A2逐小时污染物浓度与气象实测数据'
    sheet_name8 = '监测点A3逐小时污染物浓度与气象实测数据'
    pollu_name = ['SO2','NO2','PM10','PM2.5','O3','CO']
    IaQI=[0,50,100,150,200,300,400,500]#空气质量分指数
    #污染物浓度限值
    limit_pollutant=np.array([[0,50,150,475,800,1600,2100,2620],
                              [0,40,80,180,280,565,750,940],
                              [0,50,150,250,350,420,500,600],
                              [0,35,75,115,150,250,350,500],
                              [0,100,160,215,265,800,5000,5000],
                              [0,50,100,150,200,300,400,500]])
    
    data1 = pd.read_excel(filename1, sheet_name1,index_col = False)
    data2 = pd.read_excel(filename1, sheet_name2,index_col = False)
    data3 = pd.read_excel(filename2, sheet_name3,index_col = False)
    data4 = pd.read_excel(filename2, sheet_name4,index_col = False)
    data5 = pd.read_excel(filename2, sheet_name5,index_col = False)
    data6 = pd.read_excel(filename2, sheet_name6,index_col = False)
    data7 = pd.read_excel(filename2, sheet_name7,index_col = False)
    data8 = pd.read_excel(filename2, sheet_name8,index_col = False)
    
    A = np.array([[0,0]]) 
    A = np.append(A,[[-14.4846, -1.9699]],axis = 0) 
    A = np.append(A,[[-6.6716, 7.5953]],axis = 0) 
    A = np.append(A,[[-3.3543, -5.0138]],axis = 0)  
    return data1,data2,data3,data4,data5,data6,data7,data8,limit_pollutant,IaQI,pollu_name,A


#保留当日测当日的数据
def delte_num(data,n):
    sample1 = data.shape[0]
    data11=[]
    for s in range(sample1):
        if data.iloc[s,0].year+data.iloc[s,0].month+data.iloc[s,0].day == data.iloc[s,1].year+data.iloc[s,1].month+data.iloc[s,1].day:
            data11.append(data.iloc[s,:])
    data11 = pd.DataFrame(data11,index = None)
    data12 = data.iloc[n:,:]
    data_new = pd.concat([data11,data12],ignore_index=True)
    return data_new

def Withdrawal_data(data,n):
    data_once = data[['预测时间','SO2小时平均浓度(μg/m³)','NO2小时平均浓度(μg/m³)','PM10小时平均浓度(μg/m³)','PM2.5小时平均浓度(μg/m³)','O3小时平均浓度(μg/m³)','CO小时平均浓度(mg/m³)','近地2米温度（℃）','湿度（%）','大气压（Kpa）','近地10米风速（m/s）','近地10米风向（°）']].copy()
    data_once.loc[:,'大气压（Kpa）'] = data_once.loc[:,'大气压（Kpa）']*10
    data_x = data_once.iloc[0:n,:].copy()
    data_x.reset_index(drop = True, inplace=True)
    return data_x,data_once

#删除数据使得矩阵数量一样
def delte_num2(data_real,data_x):
    s = 0
    while data_real.shape[0]!= data_x.shape[0]:
        while data_real.iloc[s,0]!= data_x.iloc[s,0]:
             if data_real.iloc[s,0] < data_x.iloc[s,0]:
                 data_real.drop(index = s,inplace = True)
                 data_real = data_real.reset_index(drop=True)                
             else:
                 data_x.drop(index = s,inplace = True)
                 data_x = data_x.reset_index(drop=True)
        s = s+1
        if s > data_real.shape[0]:
            data_x.drop( data_x.index[s:],inplace = True)
        elif s > data_x.shape[0]:
            data_real.drop(data_real.index[s:],inplace = True)
           
    data_real = data_real.reset_index(drop=True) 
    data_x = data_x.reset_index(drop=True)
    return data_real,data_x

#添加数据
def imputer_data(data_real,data_x):
    x = data_x.iloc[:,1:]
    y = data_real.iloc[:,2:]
    imputer = KNNImputer(n_neighbors = 2)   
    x = imputer.fit_transform(x)
    y = imputer.fit_transform(y)
    pos1 = np.where(y<0)[0]
    pos2 = np.where(y<0)[1]
    y[pos1,pos2] = 0
    return x,y

#开始预测
def predict_ABC(x,y,data,n):
    x_train, x_test, y_train, y_test = train_test_split(x, y, 
         random_state=123, train_size=0.8)
    model = RandomForestRegressor(random_state=0)  # 实例化模型RandomForestClassifier
    model.fit(x_train, y_train)  # 在训练集上训练模型
    score = model.score(x_test,y_test)
    # 在测试集上测试模型
    x_input = data.iloc[n:,1:]
    y_output= model.predict(x_input)
    return y_output,score

#合并13-15天的数据
def Merge_13_15days(n,y_output,data_real):
    y_output = pd.DataFrame(y_output[:,0:6],columns=['SO2监测浓度(μg/m³)','NO2监测浓度(μg/m³)',
                                                 'PM10监测浓度(μg/m³)','PM2.5监测浓度(μg/m³)',
                                                 'O3监测浓度(μg/m³)','CO监测浓度(mg/m³)'])
    y_output =pd.concat([data_real.iloc[n:,2:8],y_output],ignore_index=True)
    return  y_output

#求平均值并转化为结果
def get_result(y_output,place):
    Time=y_output.shape[0]
    vaiable=y_output.shape[1]
    time_variable=np.ones([3,vaiable])
    i=0
    for t in range(Time):
        if (t+1)%24== 0:
            for v in range(vaiable):
                time_variable[i,v] = np.mean(y_output.iloc[t+1-24:t,v])
            i = i+1
            
    result = pd.DataFrame(time_variable,index=['2021/7/13','2021/7/14','2021/7/15'],columns=['SO2监测浓度(μg/m³)','NO2监测浓度(μg/m³)',
                                                     'PM10监测浓度(μg/m³)','PM2.5监测浓度(μg/m³)',
                                                     'O3监测浓度(μg/m³)','CO监测浓度(mg/m³)'])
    
    AqI=[]
    name=[]
    day = 3
    n_pollution=6
    #计算AQI并得出污染物
    for d in range(day):
        Iaqi=[]
        for i in range(n_pollution):
            Cp = result.iloc[d,i]
            Iaqi_L,Iaqi_H,Bp_L,Bp_H= find_index(Cp,i)
            Iaqi.append((Iaqi_H-Iaqi_L)/(Bp_H-Bp_L)*(Cp-Bp_L)+Iaqi_L)
        AqI.append(max(Iaqi))
        name.append(pollu_name[Iaqi.index(max(Iaqi))])
       
    #剔除无污染
    pos = np.where(np.array(AqI)<=50)[0]
    for p in pos:
        name[p] = '无'
    
    result['AQI']=AqI#添加数据
    result['首要污染物']=name
    result.insert(0,'地点',[place,place,place])
    return result

#找到AQI和对应的污染物
def find_index(Cp,i):
    n_id=8
    for n in range(n_id):
        if Cp>=limit_pollutant[i,n] and Cp<=limit_pollutant[i,n+1]:
            Iaqi_L = IaQI[n]
            Iaqi_H = IaQI[n+1]
            Bp_L = limit_pollutant[i,n]
            Bp_H = limit_pollutant[i,n+1]
    return Iaqi_L,Iaqi_H,Bp_L,Bp_H


#把A1、A2和A3的权重按比例系数添加到A中成为新的输出
def corrA_A123(dataA_x,dataA1_x,dataA2_x,dataA3_x,d_corr,data2,rate):
    dataA_xt = dataA_x.copy()
    dataA_xt.iloc[:,1:] = dataA_xt.iloc[:,1:]*(1-rate)
    dataA_xt,dataA1_x = delte_num2(dataA_xt,dataA1_x)
    dataA_xt.iloc[:,1:] = dataA_xt.iloc[:,1:]+dataA1_x.iloc[:,1:]*d_corr[0,0]*(rate)
    dataA_xt,dataA2_x = delte_num2(dataA_xt,dataA2_x)
    dataA_xt.iloc[:,1:] = dataA_xt.iloc[:,1:]+dataA2_x.iloc[:,1:]*d_corr[0,1]*(rate)
    dataA_xt,dataA3_x = delte_num2(dataA_xt,dataA3_x)
    dataA_xt.iloc[:,1:] = dataA_xt.iloc[:,1:]+dataA3_x.iloc[:,1:]*d_corr[0,2]*(rate)
    data2 = data2.reset_index(drop=True)
    dataA_real2,dataA_x2=delte_num2(data2,dataA_xt)#删除A
    x2_A,y2_A = imputer_data(dataA_real2,dataA_x2)#处理缺失值
    return x2_A,y2_A,dataA_real2
    


print('读取数据')
data1,data2,data3,data4,data5,data6,data7,data8,limit_pollutant,IaQI,pollu_name,A = DataGet()

print('计算系数')
dij = np.zeros([4,4])
for i in range(4):
    for j in range(4):
        dij[i,j] =  math.sqrt(((A[i,0] - A[j,0]) ** 2) +((A[i,1] - A[j,1]) ** 2))
        dij[j,i] = dij[i,j]
        
d_corr = np.zeros([1,3])
for i in range(3):
    d_all = np.sum(dij[0,:])
    d_corr[0,i-1] = dij[0,i+1]/d_all
    

print('预处理数据')
data1 = delte_num(data1,25368)
data3 = delte_num(data3,25368)
data4 = delte_num(data4,25368)
data5 = delte_num(data5,25368)

#提取需要预测的数据并转换单位
dataA_x,data_onceA = Withdrawal_data(data1,8456)
dataA1_x,data_onceA1 = Withdrawal_data(data3,8456)
dataA2_x,data_onceA2 = Withdrawal_data(data4,8456)
dataA3_x,data_onceA3 = Withdrawal_data(data5,8456)

#删除多余数据
dataA_real,dataA_x=delte_num2(data2,dataA_x)#删除A
dataA1_real,dataA1_x=delte_num2(data6,dataA1_x)#删除B
dataA2_real,dataA2_x=delte_num2(data7,dataA2_x)#删除C
dataA3_real,dataA3_x=delte_num2(data8,dataA3_x)#删除C


#处理缺失值
x_A,y_A = imputer_data(dataA_real,dataA_x)
x_A1,y_A1 = imputer_data(dataA1_real,dataA1_x)
x_A2,y_A2 = imputer_data(dataA2_real,dataA2_x)
x_A3,y_A3 = imputer_data(dataA3_real,dataA3_x)

#添加A与A1、A2和A3的相关性
rate = 0 #学习率
x2_A,y2_A,dataA_real2= corrA_A123(dataA_x,dataA1_x,dataA2_x,dataA3_x,d_corr,data2,rate)




print('开始预测')
yA_output,scoreA=predict_ABC(x_A,y_A,data_onceA,8456)
y2A_output,score2A=predict_ABC(x2_A,y2_A,data_onceA,8456)
yA1_output,scoreA1=predict_ABC(x_A1,y_A1,data_onceA1,8456)
yA2_output,scoreA2=predict_ABC(x_A2,y_A2,data_onceA2,8456)
yA3_output,scoreA3=predict_ABC(x_A3,y_A3,data_onceA3,8456)
print('A的准确率为:'+str(scoreA)) # 输出模型RandomForestClassifier
print('添加A1、A2与A3后的准确率为：'+str(score2A)) # 输出模型RandomForestClassifier
print('A1的准确率为:'+str(scoreA1)) # 输出模型RandomForestClassifier
print('A2的准确率为:'+str(scoreA2)) # 输出模型RandomForestClassifier
print('A3的准确率为:'+str(scoreA3)) # 输出模型RandomForestClassifier

#合并13-15天的数据
yA_output = Merge_13_15days(8445,y2A_output,dataA2_real)
yA1_output = Merge_13_15days(8448,yA1_output,dataA1_real)
yA2_output = Merge_13_15days(8445,yA2_output,dataA2_real)
yA3_output = Merge_13_15days(8363,yA3_output,dataA3_real)

#获得结果
result_A = get_result(yA_output,'监测点A')
result_A1 = get_result(yA1_output,'监测点A1')
result_A2 = get_result(yA2_output,'监测点A2')
result_A3 = get_result(yA3_output,'监测点A3')


result = pd.concat([result_A,result_A1,result_A2,result_A3])
print(result)

#输出文件
writer = pd.ExcelWriter('第四问结果.xlsx')
result.to_excel(writer)
writer.save()

