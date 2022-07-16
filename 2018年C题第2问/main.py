#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 16:35:04 2022

@author: chenzhenhua
"""




import pandas as pd
import numpy as np
from kmodes import kmodes
from sklearn import svm # 支持向量机
from sklearn.ensemble import RandomForestClassifier # 随机森林
from sklearn.tree import DecisionTreeClassifier # 决策树
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt  


def DataGet():
    data = pd.read_excel("附件2.xlsx")
    data=data.fillna(0)
    data_out= pd.read_excel("附件3.xlsx")
    data_out=data_out.fillna(0)
    data1=data[['country','region','specificity','success','attacktype1','targtype1']]
    return data,data1,data_out

#重新排序(根据损害程度排序)
def SortLab(damage,lab):
    b = np.argsort(np.negative(damage))
    pos=[]
    temp_lab=lab.copy()
    for k in range(len(b)):
        pos=np.where(temp_lab==b[k])[0]
        lab[pos]=k+1
    return lab

#选取聚类数量
def getMax_K(data1):
    Se=[]
    K=21
    for k in range(5,K):
        print('k='+str(k)+'时进行聚类')
        km = kmodes.KModes(n_clusters=k)
        km.fit(data1)
        se=km.cost_
        Se.append(se)
    plt.figure(1)
    plt.plot(Se)
    labels=[]
    for l in range(5,K):
        labels.append(str(l))
    plt.xticks(range(0,len(labels)),labels=labels)
    plt.xlabel('Number of clusters')
    plt.ylabel('sse')
    plt.show
    
    delta=[]
    for s in range(len(Se)-1):
        delta.append(Se[s]-Se[s+1])
    K=5+(np.argmax(delta)+1)
    print('选取聚类数量的为：'+str(K))
    return K

print('读取数据')
data,data1,data_out=DataGet()

#确定聚类个数
print('选择kmodes的聚类个数')
K=getMax_K(data1)

print('开始聚类')    
km = kmodes.KModes(n_clusters=K)
clusters = km.fit_predict(data1)

number_list=[]
for k in range(K):
    number_list.append(len(np.where(clusters==k)[0]))  
print('求出各类数量为：')
print(number_list)

#对分类恐怖事件排序并提取前5个
data_num=data[['nkill','propextent','success']]
scaler = MinMaxScaler(feature_range=(0, 1))  #将数据归一到0到1，可以根据数据特点归一到0到1
mydata = scaler.fit_transform(data_num)  #归一化
damage=[]
for n in range(K):
    pos=np.where(clusters==n)[0]
    damage.append(np.sum(np.sum(mydata[pos,:]))/len(pos))
clusters=SortLab(damage,clusters)
pos=np.where(clusters<=5)[0]

#设置模型的训练集和测试集
x=data.iloc[pos,5:]
ran=x.shape[0]
train_num=int(ran*0.75)#训练数量
x=x._get_numeric_data() 
x=x[['country','region','specificity','success','attacktype1','targtype1','nkill','propextent']]
x_train=x.iloc[1:train_num,:]
x_test=x.iloc[train_num:,:]
y=clusters[pos]
y_train=y[1:train_num]
y_test=y[train_num:]
x_input=data_out[list(x_train)]

#开始训练模型
print('开始训练并选择预测模型')
clf1 = svm.SVC(probability = True)
clf1.fit(x_train,y_train)
clf2=RandomForestClassifier(random_state=0)
clf2.fit(x_train,y_train)
clf3=DecisionTreeClassifier(random_state=0)
clf3.fit(x_train,y_train)
score1=clf1.score(x_test,y_test)
print('SVM分类准确率为:'+str(score1))
score2=clf2.score(x_test,y_test)
print('随机森林分类准确率为:'+str(score2))
score3=clf3.score(x_test,y_test)
print('决策树分类准确率为:'+str(score3))
score=[score1,score2,score3]
indx_score=score.index(max(score))
score=max(score1,score2,score3)
if indx_score==0:
    clf=clf1
    print('选择支持向量机预测模型')
elif indx_score==1:
    clf=clf2
    print('选择随机森林预测模型')
else:
    clf=clf3
    print('选择决策数预测模型')
y=clf.predict(x_input)
y_proba=clf.predict_proba(x_input)

#输出
A=y_proba.shape[0]
result=pd.DataFrame(columns=['eventid','1号嫌疑人','2号嫌疑人','3号嫌疑人','4号嫌疑人','5号嫌疑人'])
result['eventid']=data_out['eventid']
for a in range(A):
    temp=np.argsort(-y_proba[a,:])+1
    b=temp.copy()
    b[temp[np.array(range(0,5))]-1]=np.array(range(0,5))+1
    result.iloc[a,1:]=b
print('预测结果为:')
print(result)


