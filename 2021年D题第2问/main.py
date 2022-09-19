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
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error #绝对误差
from sklearn.metrics import mean_squared_error #均方误差
from sklearn.metrics import explained_variance_score #解释回归模型方差得分

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

def get_assess(data_activity_training,score,clf,f_name):
    y=clf.predict(data1_training)
    mae = mean_absolute_error(y, data_activity_training)
    mse = mean_squared_error(y, data_activity_training)
    evs = explained_variance_score(y, data_activity_training)
    rmse = np.sqrt(mean_squared_error(y, data_activity_training))
    #print(f_name+"预测的R2：",score)
    #print(f_name+"预测的绝对误差为：",mae)
    #print(f_name+"预测的均方误差为：",mse)
    #print(f_name+"预测的均方根误差为：",rmse)
    #print(f_name+"预测的解释方差模型回归得分为：",evs)
    return y,mae,mse,evs,rmse

def polt_result(clf):
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
    

name,data1_training,data1_test,data_activity_training,data_activity_test,data_activity_test2 = DataGet()


print('使用KNN插值')
imputer = KNNImputer(n_neighbors = 2)   
data1_training = imputer.fit_transform(data1_training)
data_activity_training = imputer.fit_transform(data_activity_training)

#x_train, x_test, y_train, y_test = train_test_split(data1_training, data_activity_training , random_state=123, train_size=0.8)
index = np.random.randint(0,len(data1_training),size=50)
print('选择预测模型并得出评价指标')
dic = {'评价指标\模型':['MAE','MSE','RMSE','EVS','R2']}
#使用线性回归
clf1=LinearRegression()
clf1.fit(data1_training, data_activity_training); 
score1=clf1.score(data1_training, data_activity_training)
y1,mae1,mse1,evs1,rmse1= get_assess(data_activity_training,score1,clf1,'线性回归') #得出评价指标
dic.update({'线性回归':[mae1,mse1,rmse1,evs1,score1]})
polt_result(clf1)

#使用神经网络
clf2=MLPRegressor(hidden_layer_sizes=(10,), random_state=10,learning_rate_init=0.1)
clf2.fit(data1_training, data_activity_training)
score2=clf2.score(data1_training, data_activity_training)
y2,mae2,mse2,evs2,rmse2= get_assess(data_activity_training,score2,clf2,'神经网络')  #得出评价指标
polt_result(clf2)
dic.update({'神经网络':[mae2,mse2,rmse2,evs2,score2]})

#使用随机森林
clf3=RandomForestRegressor()
clf3.fit(data1_training, data_activity_training)
score3=clf3.score(data1_training, data_activity_training)
y3,mae3,mse3,evs3,rmse3 = get_assess(data_activity_training,score3,clf3,'随机森林')  #得出评价指标
polt_result(clf3)
dic.update({'随机森林':[mae3,mse3,rmse3,evs3,score3]})

#使用决策树
clf4=DecisionTreeRegressor()
clf4.fit(data1_training, data_activity_training)
score4=clf4.score(data1_training, data_activity_training)
y4,mae4,mse4,evs4,rmse4 = get_assess(data_activity_training,score4,clf4,'决策树')  #得出评价指标
polt_result(clf4)
dic.update({'决策树':[mae4,mse4,rmse4,evs4,score4]})

Evaluation_indicators = pd.DataFrame(dic)
print(Evaluation_indicators.to_string(index=False))

#选出使用模型
score=[score1,score2,score3,score4]
indx_score=score.index(max(score))
score=max(score1,score2,score3,score4)
if indx_score==0:
    clf=clf1
    y=y1
    print('选择线性回归预测模型')
elif indx_score==1:
    clf=clf2
    y=y2
    print('神经网络回归预测模型')
elif indx_score==2:
    clf=clf3
    y=y3
    print('选择随机森林预测模型')
else:
    clf=clf4
    y=y4
    print('选择决策数预测模型')

print('开始预测')   
y=clf3.predict(data1_test)
data_activity_test2.iloc[:,1:] = y
print('预测结果为：')  
print(data_activity_test2)

    
    
#输出文件
writer = pd.ExcelWriter('第二问结果.xlsx')
writer2 = pd.ExcelWriter('第二问的模型评价指标.xlsx')
data_activity_test2.to_excel(writer,sheet_name='test',index = False)
Evaluation_indicators.to_excel(writer2,sheet_name='指标',index = False)
writer.save()
writer2.save()

