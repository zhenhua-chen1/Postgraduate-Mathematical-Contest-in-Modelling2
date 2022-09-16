#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 15:39:04 2022

@author: chenzhenhua
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt  # 导入绘图库
from sklearn.linear_model import LinearRegression #线性拟合回归
from sklearn.neural_network import MLPRegressor  # 多层线性回归
from sklearn.ensemble import RandomForestRegressor # 随机森林
from sklearn.tree import DecisionTreeRegressor

def DataGet():
    name=['nHeavyAtom','nS','naAromAtom','nX','nO','nN','nF','nAtom','ATSp2',
     'ATSm4','ATSm2','ATSm1','ATSm5','ATSc1','ALogp2','ALogP','ATSc4','ATSc5',
     'ATSc2','ATSc3']
    filename1 = 'Molecular_Descriptor.xlsx'
    filename2 = 'ERα_activity.xlsx'
    sheet_name1 = 'training'
    sheet_name2 = 'test'
    data1_training = pd.read_excel(filename1, sheet_name1,index_col = False)
    data1_training = data1_training[name]
    data1_test = pd.read_excel(filename1, sheet_name2,index_col = False)
    data1_test = data1_test[name]
    data_activity_training = pd.read_excel(filename2, sheet_name1,index_col = False)
    data_activity_training = data_activity_training.iloc[:,1:]
    data_activity_test = pd.read_excel(filename2, sheet_name2,index_col = False)
    data_activity_test2 = data_activity_test
    data_activity_test = data_activity_test.iloc[:,1:]
    return name,data1_training,data1_test,data_activity_training,data_activity_test,data_activity_test2

name,data1_training,data1_test,data_activity_training,data_activity_test,data_activity_test2 = DataGet()


print('使用KNN插值')
imputer = KNNImputer(n_neighbors = 2)   
data1_training = imputer.fit_transform(data1_training)

print('选择预测模型')
clf1=LinearRegression()
clf1.fit(data1_training,data_activity_training); 
score1=clf1.score(data1_training,data_activity_training)
print("线性回归预测的准确率：",score1)
clf2=MLPRegressor(hidden_layer_sizes=(10,), random_state=10,learning_rate_init=0.1)
clf2.fit(data1_training,data_activity_training)
score2=clf2.score(data1_training,data_activity_training)
print("神经网络预测的准确率：",score2)
clf3=RandomForestRegressor()
clf3.fit(data1_training,data_activity_training)
score3=clf3.score(data1_training,data_activity_training)
print("随机森林预测的准确率：",score3)
clf4=DecisionTreeRegressor()
clf4.fit(data1_training,data_activity_training)
score4=clf4.score(data1_training,data_activity_training)
print("决策树预测的准确率：",score4)

score=[score1,score2,score3,score4]
indx_score=score.index(max(score))
score=max(score1,score2,score3,score4)
if indx_score==0:
    clf=clf1
    print('选择线性回归预测模型')
elif indx_score==1:
    clf=clf2
    print('神经网络回归预测模型')
elif indx_score==2:
    clf=clf3
    print('选择随机森林预测模型')
else:
    clf=clf4
    print('选择决策数预测模型')

print('开始预测')   
y=clf.predict(data1_test)
data_activity_test2.iloc[:,1:] = y
print('预测结果为：')  
print(data_activity_test2)


index = np.random.randint(0,len(data1_training),size=50)
plt.subplots(2,1)
y_p=clf.predict(data1_training)
y_r=data_activity_training
plt.rcParams['font.family'] = ['Arial Unicode MS'] #正常显示中文
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
plt.suptitle('预测值与真实值对比',fontsize=20)  #设置主标题
name2 = ['IC50_nM',	'pIC50']
flag = 1
for n in range (2):
    number = n+1
    plt.subplot(2,1,number)
    if flag ==1 :
        plt.plot(y_p[index,n],label='预测值')
        plt.plot(np.array(y_r)[index,n],label='真实值')
        plt.title(name2[n],fontsize=10)
        plt.legend()
        flag = 0
    else:
        plt.plot(y_p[index,n])
        plt.plot(np.array(y_r)[index,n])
        plt.title(name2[n],fontsize=10)
    plt.show()
    
    
#输出文件
writer = pd.ExcelWriter('第二问结果.xlsx')
data_activity_test2.to_excel(writer,sheet_name='test')
writer.save()


