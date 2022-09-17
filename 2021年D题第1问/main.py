#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 14:07:08 2022

@author: chenzhenhua
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor #导入随机森林
import seaborn as sns


def DataGet():
    filename1 = 'Molecular_Descriptor.xlsx'
    filename2 = 'ERα_activity.xlsx'
    sheet_name1 = 'training'
    sheet_name2 = 'test'
    data1_training = pd.read_excel(filename1, sheet_name1,index_col = False)
    data1_test = pd.read_excel(filename1, sheet_name2,index_col = False)
    data_activity_training = pd.read_excel(filename2, sheet_name1,index_col = False)
    data2_output = data_activity_training.iloc[:,2]
    data_activity_test = pd.read_excel(filename2, sheet_name2,index_col = False)
    return data1_training,data1_test,data_activity_training,data_activity_test,data2_output

def three_sigmal(data):
    n = 3  # n*sigma
    ymean = np.mean(data)
    ystd = np.std(data)
    threshold1 = ymean - n * ystd
    threshold2 = ymean + n * ystd
    outlier = []  # 将异常值保存
    for i in range(0, len(data)):
        if (data[i] < threshold1) | (data[i] > threshold2):
            outlier.append(data[i])
        else:
            continue
    return outlier
    

data1_training,data1_test,data_activity_training,data_activity_test,data2_output = DataGet()  
    
#剔除含0值过多的变量
print('开始数据预处理')
print('剔除含0值过多的变量')
n_variable1 = data1_training.shape[1]
num_zero = (data1_training == 0).astype(int).sum(axis=0) #按列统计变量数
pos= np.where(num_zero>1974*0.9)[0]
data1_training.drop(data1_training.columns[pos],axis = 1,inplace = True) #axis参数默认为0 
n_variable = data1_training.shape[1]
print('共剔除'+str(n_variable1-n_variable)+'个变量')

print('剔除异常值过多的变量')
num_len = []
for n in range(1,n_variable):
    outlier = three_sigmal(data1_training[data1_training.columns[n]])  
    if len(outlier)>100:
        num_len.append(len(outlier))
        #data1_training.drop(data1_training.columns[n],axis = 1,inplace = True) #axis参数默认为0
data1_training.drop(data1_training.columns[num_len],axis = 1,inplace = True) #axis参数默认为0 
print('共剔除'+str(len(num_len))+'个变量')

#使用随机森林降维
print('使用随机森林开始降维')
model = RandomForestRegressor(random_state=1, max_depth=10)
data1_training=pd.get_dummies(data1_training)
model.fit(data1_training,data2_output)
features = data1_training.columns
importances = model.feature_importances_
indices = np.argsort(importances[0:30])  # top 20 features
plt.figure(1)
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in reversed(indices)])
plt.xlabel('Relative Importance')
plt.figure(2)
plt.rcParams['font.family'] = ['Arial Unicode MS'] #正常显示中文
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
plt.plot([importances[i] for i in reversed(indices)])
plt.title('特征值')
plt.show()
print('降维后对生物活性最具有显著影响的30个分子描述符为:')
print([features[i] for i in indices])

#计算相关性
print('使用pearson计算30个维度的相关性并输出文件')
features_names = [features[i] for i in indices]
data_ra = data1_training[[features[i] for i in indices]].copy()
data_ra = data_ra.corr("pearson")
pos1 = np.where(np.array((data_ra))>=0.96)[0]
pos2 = np.where(np.array((data_ra))>=0.96)[1]
#输出文件
writer = pd.ExcelWriter('30个分子描述的相关性.xlsx')
data_ra.to_excel(writer)
writer.save()

#删除10个强相关性变量
print('开始删除10个强相关性的变量')
delte = []
for p in range(len(pos1)):
    if pos1[p] < pos2[p]: 
        delte.append(pos2[p])
delte = list(set(delte)) #去重
delte = np.array(delte)[0:10] #删除10个强相关性的变量
delte_names = np.array(features_names)[delte]
data_ra = data_ra.drop(index=delte_names, columns=delte_names)
plt.figure(3)
sns.heatmap(round(data_ra,2), annot = True, linewidths=.5, cmap="YlGnBu") 
plt.title('Correlation between features', fontsize = 30)
plt.show()
print('删除变量为:')
print(delte_names)
features_names = np.delete(np.array(features_names),delte)
print('最后对生物活性最具有显著影响的20个分子描述符为:：')
print(features_names)
        


