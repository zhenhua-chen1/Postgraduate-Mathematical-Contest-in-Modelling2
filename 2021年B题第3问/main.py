#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 17:56:49 2022

@author: chenzhenhua
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # 导入绘图库
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

#处理数据
def DataGet():
    filename1 ='附件1 监测点A空气质量预报基础数据.xlsx'
    filename2 ='附件2 监测点B、C空气质量预报基础数据.xlsx'
    sheet_name1 = '监测点A逐小时污染物浓度与气象一次预报数据'
    sheet_name2 ='监测点A逐小时污染物浓度与气象实测数据'
    sheet_name3 = '监测点B逐小时污染物浓度与气象实测数据'
    sheet_name4 = '监测点C逐小时污染物浓度与气象实测数据'
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
    dataB_real = pd.read_excel(filename2, sheet_name3,index_col = False)
    dataC_real = pd.read_excel(filename2, sheet_name4,index_col = False)
    data_real = data1.iloc[0:819,:]
    return data1,data_real,data2,limit_pollutant,IaQI,pollu_name,dataB_real,dataC_real

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

#删除数据使得矩阵数量一样
def delte_num(data_real,data_x):
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
    return data_real,data_x

#添加数据
def imputer_data(data_real,data_x):
    x = data_x.iloc[:,1:]
    y = data_real.iloc[:,2:]
    imputer = KNNImputer(n_neighbors = 2)   
    x = imputer.fit_transform(x)
    y = imputer.fit_transform(y)
    return x,y

#开始预测
def predict_ABC(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, 
         random_state=123, train_size=0.8)
    model = RandomForestRegressor(random_state=0)  # 实例化模型RandomForestClassifier
    model.fit(x_train, y_train)  # 在训练集上训练模型
    score = model.score(x_test,y_test)
    # 在测试集上测试模型
    x_input = data_once.iloc[8456:,1:]
    x_input2 = data_once.iloc[1:8456,1:]
    y_output= model.predict(x_input)
    y_predict = model.predict(x_input2)
    y_predict = y_predict[:,0:6]
    y_real = data_once.iloc[1:8456:,1:7]
    y_real = np.array(y_real)
    return y_output,score,y_predict,y_real

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
    
    
print('读取数据')
data1,data_real,data2,limit_pollutant,IaQI,pollu_name,dataB_real,dataC_real = DataGet()
print('预处理数据')

#数据预处理(当日测当日的)
sample1 = data1.shape[0]
data11=[]
for s in range(sample1):
    if data1.iloc[s,0].year+data1.iloc[s,0].month+data1.iloc[s,0].day == data1.iloc[s,1].year+data1.iloc[s,1].month+data1.iloc[s,1].day:
        data11.append(data1.iloc[s,:])
data11 = pd.DataFrame(data11,index = None)
data12 = data1.iloc[25368:,:]
data_new = pd.concat([data11,data12],ignore_index=True)

#提取需要预测的数据并转换单位
data_once = data_new[['预测时间','SO2小时平均浓度(μg/m³)','NO2小时平均浓度(μg/m³)','PM10小时平均浓度(μg/m³)','PM2.5小时平均浓度(μg/m³)','O3小时平均浓度(μg/m³)','CO小时平均浓度(mg/m³)','近地2米温度（℃）','湿度（%）','大气压（Kpa）','近地10米风速（m/s）','近地10米风向（°）']].copy()
data_once.loc[:,'大气压（Kpa）'] = data_once.loc[:,'大气压（Kpa）']*10
data_x = data_once.iloc[0:8456,:].copy()
data_x.reset_index(drop = True, inplace=True)
data_real = data2.iloc[11036:19432,:].copy()
data_real.reset_index(drop = True, inplace=True)


#删除多余数据
data_real,dataA_x=delte_num(data_real,data_x)#删除A
dataB_real,dataB_x=delte_num(dataB_real,data_x)#删除B
dataC_real,dataC_x=delte_num(dataC_real,data_x)#删除C

#处理缺失值
x_A,y_A= imputer_data(data_real,dataA_x)
x_B,y_B = imputer_data(dataB_real,dataB_x)
x_C,y_C = imputer_data(dataC_real,dataC_x)
data_real.iloc[8316:,2:8]=y_A[8316:,0:6]


print('开始预测')
yA_output,scoreA,y_Ap,yAr=predict_ABC(x_A,y_A)
yB_output,scoreB,y_Bp,yBr=predict_ABC(x_B,y_B)
yC_output,scoreC,y_Cp,yCr=predict_ABC(x_C,y_C)
for i in range(3):
    if i == 0:
        y_p = y_Ap
        y_r = yAr
        name = 'A的预测和真实值对比'
    elif i == 1:
        y_p = y_Bp
        y_r = yBr 
        name = 'B的预测和真实值对比'
    else:
        y_p = y_Cp
        y_r = yCr
        name = 'C的预测和真实值对比'
    plt.subplots(2,3,figsize=(15,6))
    plt.rcParams['font.family'] = ['Arial Unicode MS'] #正常显示中文
    plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
    plt.suptitle(name,fontsize=20)  #设置主标题
    for n in range (6):
        number = n+1
        plt.subplot(2,3,number)
        plt.plot(y_p[:,n],label='预测值')
        plt.plot(y_r[:,n],label='真实值')
        plt.title(pollu_name[n],fontsize=10)
        plt.legend()
        plt.show()

print('A的准确率为:'+str(scoreA)) # 输出模型RandomForestClassifier
print('B的准确率为:'+str(scoreB)) # 输出模型RandomForestClassifier
print('C的准确率为:'+str(scoreC)) # 输出模型RandomForestClassifier

#合并13-15天的数据
yA_output = Merge_13_15days(8316,yA_output,data_real)
yB_output = Merge_13_15days(8430,yB_output,dataB_real)
yC_output = Merge_13_15days(8405,yC_output,dataC_real)

#获得结果
result_A = get_result(yA_output,'监测点A')
result_B = get_result(yB_output,'监测点B')
result_C = get_result(yC_output,'监测点C')
result = pd.concat([result_A,result_B,result_C])
print(result)

#输出文件
writer = pd.ExcelWriter('第三问结果.xlsx')
result.to_excel(writer)
writer.save()
        