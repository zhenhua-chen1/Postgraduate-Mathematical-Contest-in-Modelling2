#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 14:36:42 2022

@author: chenzhenhua
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC # 支持向量机
from sklearn.ensemble import RandomForestClassifier # 随机森林
from sklearn.tree import DecisionTreeClassifier # 决策树
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt  

def DataGet():
    name=['nHeavyAtom','nS','naAromAtom','nX','nO','nN','nF','nAtom','ATSp2',
     'ATSm4','ATSm2','ATSm1','ATSm5','ATSc1','ALogp2','ALogP','ATSc4','ATSc5',
     'ATSc2','ATSc3']
    name2 = ['Caco-2','CYP3A4','hERG','HOB','MN']
    filename1 = 'Molecular_Descriptor.xlsx'
    filename2 = 'ADMET.xlsx'
    sheet_name1 = 'training'
    sheet_name2 = 'test'
    data1_training = pd.read_excel(filename1, sheet_name1,index_col = False)
    data1_training = data1_training[name]
    data1_test = pd.read_excel(filename1, sheet_name2,index_col = False)
    data1_test = data1_test[name]
    data_y_training = pd.read_excel(filename2, sheet_name1,index_col = False)
    data_y_training = data_y_training.iloc[:,1:]
    data_y_test = pd.read_excel(filename2, sheet_name2,index_col = False)
    data_y2_test = data_y_test
    data_y_test = data_y_test.iloc[:,1:]
    return name,data1_training,data1_test, data_y_training,data_y_test,data_y2_test,name2



name,data1_training,data1_test, data_y_training,data_y_test,data_y2_test,name2 = DataGet()

x_train, x_test, y_train, y_test = train_test_split(data1_training, data_y_training , random_state=123, train_size=0.8)
#开始用支持向量机预测

num = 5 #分类总数
method_num = 3 #方法总数
method = ['SVC','随机森林','决策树']
print('开始预测')
score1 = []
score2 = []
score3 = []
for m in range(method_num):
    if m == 0:
        clf = SVC(probability = True)
        clf1 = clf
    elif m ==1:
        clf = RandomForestClassifier(random_state=0)
        clf2 = clf
    else:
        clf = DecisionTreeClassifier(random_state=0)
        clf3 = clf
    plt.figure()
    plt.rcParams['font.family'] = ['Arial Unicode MS'] #正常显示中文
    plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
    for n in range (num):
        clf.fit(x_train,y_train.iloc[:,n])
        print(method[m]+'方法下'+name2[n]+'的分类准确率为:'+str(clf.score(x_test,y_test.iloc[:,n])))
        if m == 0:
            fpr,tpr, thresholds = roc_curve(y_test.iloc[:,n],clf.decision_function(x_test))    
            plt.plot(fpr,tpr,label=name[n])
            plt.xlabel('FPR')
            plt.ylabel('TPR')
        else:
            fpr,tpr, thresholds = roc_curve(y_test.iloc[:,n],clf.predict_proba(x_test)[:,1])    
            plt.plot(fpr,tpr,label=name[n])
            plt.xlabel('FPR')
            plt.ylabel('TPR')              
    plt.title(method[m]+'的ROC',fontsize=10)
    plt.legend()
    plt.show()
    
print('选择随机森林模型')
for n in range (num):
    clf2.fit(x_train,y_train.iloc[:,n])
    data_y2_test.iloc[:,n+1]=clf2.predict(data1_test)
    
#输出文件
writer = pd.ExcelWriter('第三问结果.xlsx')
data_y2_test.to_excel(writer,sheet_name='test',index = False)
writer.save()





