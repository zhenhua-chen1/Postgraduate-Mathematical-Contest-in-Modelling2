#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 20:49:40 2022

@author: chenzhenhua
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
from factor_analyzer import factor_analyzer
import matplotlib.pyplot as plt  # 导入绘图库
import random

def DataGet():
    data = pd.read_excel("附件1.xlsx")
    number_id=data.iloc[:,0]
    height, width = data.shape
    data=data._get_numeric_data() #找到数字列
    data=data.fillna(-9)
    data=data.iloc[:,9:]
    return data,number_id,height

#重新排序(根据数量多少排序)
def SortLab(number_list,lab):
    b = np.argsort(number_list)
    for k in range(len(b)):
        pos=np.where(lab==b[k])[0]
        lab[pos]=-k-1
    lab=abs(lab)
    return lab

def Kmeans_data(data3):
    K = 5 # 分为K类
    #for k in [K]:
    clf = KMeans(n_clusters=K,n_init=10)
    lab1 = clf.fit(data3)
    lab = lab1.labels_
    cluter_center_data=clf.cluster_centers_
    cluter_center_data=pd.DataFrame(cluter_center_data,columns=list_name)
    #计算各分类个数
    number_list=[]
    for k in range(K):
       number_list.append(len(np.where(lab==k)[0]))
    cluter_center_data.loc[:,'number']= number_list # 在最后一列后，插入值
    #调整
    lab=SortLab(number_list,lab) 
    return lab,cluter_center_data

def Kmeans_data2(data3):
    K = 3 # 分为K类
    #for k in [K]:
    clf = KMeans(n_clusters=K,n_init=10)
    lab1 = clf.fit(data3)
    lab = lab1.labels_
    cluter_center_data=clf.cluster_centers_
    cluter_center_data=pd.DataFrame(cluter_center_data,columns=list_name)
    #计算各分类个数
    number_list=[]
    for k in range(K):
       number_list.append(len(np.where(lab==k)[0]))
    cluter_center_data.loc[:,'number']= number_list # 在最后一列后，插入值
    #调整
    lab=SortLab(number_list,lab) 
    return lab,cluter_center_data

       
print('读取数据')
data,number_id,height=DataGet()

# kmo检验 
print(factor_analyzer.calculate_kmo(data)[1])
if factor_analyzer.calculate_kmo(data)[1]>0.6:
    print('通过kmo检验')
    
#归一化
scaler = MinMaxScaler(feature_range=(-1, 1))  #将数据归一到0到1，可以根据数据特点归一到-1到1
mydata = scaler.fit_transform(data)  #归一化

#开始降维
print('开始降维')
pca=PCA(n_components='mle',whiten= True)
data2=pca.fit_transform(mydata)
Contribution_rate=np.cumsum(pca.explained_variance_ratio_)     #计算贡献率
dim=np.where(Contribution_rate>0.85)[0][1] #找到贡献率大于0.85的维度数
components_=pca.components_
list_name=[]
for d in range(dim):
    max_pos=np.where(abs(components_[d])==max(abs(components_[d])))[0][0]
    list_name.append(data.columns[max_pos])

#去重
temp_list=[]
for l in list_name:
    if l not in temp_list and l!='hostkidoutcome':
        temp_list.append(l)
list_name=temp_list

#加入nkill
dict1 ={}
for l in list_name:
    dict1.update({l:data[l]})
dict1.update({'nkill':data['nkill']})    
list_name.append('nkill')
data3=pd.DataFrame(dict1)

#调用KMeans聚类算法
print('开始聚类')
lab,cluter_center_data=Kmeans_data(data3)

 #画聚类图
samples=[random.randrange(0,round(height/2)) for i in range(499)]
samples.append(np.where(lab==1)[0][1])
samples.extend([random.randrange(round(height/2),height) for i in range(500)])
colorStore = ['or', 'og', 'ob', 'oc', 'om', 'oy', 'ok']
fig1=plt.figure()
ax1=fig1.add_subplot(211)
ax2=fig1.add_subplot(212)
ax1.set_ylabel('propextent')
ax1.set_title('the result of clustering')
ax2.set_xlabel('events')
ax2.set_ylabel('nkill')
flag=np.ones(5)
for i in range(len(samples)):
        k=samples[i]
        color = colorStore[lab[k]]
        if flag[lab[k]-1]:
            ax2.plot(i, data3[u'nkill'][k], color,label='the'+str(lab[k])+'class')
            ax1.plot(i, data3[u'propextent'][k], color)
            flag[lab[k]-1]=0
        else:
            ax1.plot(i, data3[u'propextent'][k], color)
            ax2.plot(i, data3[u'nkill'][k], color)
plt.legend()
plt.show()

#输出文件
result=number_id.to_frame()
result.loc[:,'degree']=lab # 在最后一列后，插入值
writer1 = pd.ExcelWriter('案件等级编号.xlsx')
writer2 = pd.ExcelWriter('降维之后各变量均值.xlsx')
sheetname = '数据' 
result.to_excel(writer1, header=None, sheet_name=sheetname, index=False)
cluter_center_data.to_excel(writer2, sheet_name=sheetname, index=False)
writer1.close()
writer2.close()


#输出十大恐怖事件
ten_num=np.where(lab==1)[0]
Cases_name=list(number_id[ten_num])
ten_num2=np.where(lab==2)[0]
temp_number_id=number_id
while len(Cases_name)<10:
    data3 = data3.iloc[ten_num2,:]
    temp_number_id =temp_number_id[ten_num2]
    temp_number_id=temp_number_id.reset_index(drop=True)
    lab2,cluter_center_data2=Kmeans_data2(data3)
    ten_num=np.where(lab2==1)[0]
    Cases_name.extend(temp_number_id[ten_num])
    ten_num2=np.where(lab2==2)[0]
pos=[np.where([Cases_name[x]==number_id])[1][0] for x in range(len(Cases_name))]
data_num=pd.DataFrame(data,index=pos,columns=['nkill','propextent','INT_IDEO','success'])
mydata_num= scaler.fit_transform(data_num)  #归一化
mydata_num=mydata_num.sum(axis=1)
mydata_num2=np.argsort(-mydata_num)
Cases_name=np.array(Cases_name)[mydata_num2]
print('危害程度最高的十大恐怖事件为:')
print(Cases_name[0:10])
