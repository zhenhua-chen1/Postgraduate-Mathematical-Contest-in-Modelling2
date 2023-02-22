#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 20:00:33 2023

@author: chenzhenhua
"""

import pandas as pd
import numpy as np

def three_sigmal(data):
    n = 3  # n*sigma
    outlier = []  # 将异常值保存
    ymean = np.mean(data)
    ystd = np.std(data)
    threshold1 = ymean - n * ystd
    threshold2 = ymean + n * ystd
    for i in range(0, len(data)):
        if (data[i] < threshold1) | (data[i] > threshold2):
            outlier.append(i)
        else:
            continue
    return outlier

#对数据预处理后取平均值
def get_mean(data,sample):
    delte_varitime = {}
    #对样本进行预处理
    for column in data:
        temp_data = data[column].values
        #找到缺失值过多的变量
        b = list(temp_data).count(0) #判断0出现次数
        #若0出现次数太多则删除时间点
        if b > 0.2*len(temp_data) and b != len(temp_data):
            pos = np.where(temp_data==0)[0]#找到需要删除的时间节点
            delte_varitime[column]=pos
        #若0出现次数少则取平均值
        elif b!=0 and b != len(temp_data):
            pos = np.where(temp_data==0)[0]#找到需要删除的时间节点
            for ind in pos:
                if ind == 0:
                   temp_data[ind] =  (temp_data[ind+1] + temp_data[ind+2])/2#部分缺失值取平均值
                elif ind == len(temp_data)-1:
                    temp_data[ind] =  (temp_data[ind-1] + temp_data[ind-2])/2#部分缺失值取平均值
                else:    
                    temp_data[ind] = (temp_data[ind-1] + temp_data[ind+1])/2#部分缺失值取平均值
        else:
        #保留异常值
            outlier = three_sigmal(temp_data)
            if outlier:
                delte_varitime[column]=np.array(outlier)
                
    #取平均值
    keys =  delte_varitime.keys()
    newdata = np.ones([1,len(data.keys())])
    newdata = pd.DataFrame(newdata,index =[sample],columns = data.keys())
    for key in data:
        if key in keys:
    #若需处理先删除再取平均值
            temp_data = np.array(data[key]).copy()
            delet_pos = delte_varitime[key]
            temp_data2 = np.delete(temp_data,delet_pos,axis = 0)
            newdata.loc[:,key] = np.mean(temp_data2)
        else:
    #若无需处理直接取平均值
            newdata.loc[:,key] = np.mean(data[key])
    return newdata

if __name__ == "__main__": 
    data = pd.read_excel("附件三：285号和313号样本原始数据.xlsx",sheet_name='操作变量')
    #读取数据285和313
    data_285 = data.iloc[2:42,1:]
    data_313 = data.iloc[43:,1:]
    data_285.index = data.iloc[2:42,0]
    data_285.columns = data.iloc[0,1:]
    data_313.index = data.iloc[43:,0]
    data_313.columns = data.iloc[0,1:]          
    newdata285 = get_mean(data_285,'285')#对285进行预处理
    newdata313 = get_mean(data_313,'313')#对313进行预处理
    
    #输出文件
    writer = pd.ExcelWriter('285号处理后结果.xlsx')
    writer2 = pd.ExcelWriter('313号处理后结果.xlsx')
    newdata285.to_excel(writer)
    newdata313.to_excel(writer2)
    writer.save()
    writer2.save()
            
  