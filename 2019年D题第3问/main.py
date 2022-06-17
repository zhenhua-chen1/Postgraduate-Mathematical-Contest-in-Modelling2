#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 19:52:51 2022

@author: chenzhenhua
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt  # 导入绘图库

#读取数据
def DataGet():
    filenumber = 3 #文件2修改为2，文件3修改为3.
    filename='文件'+str(filenumber)+'.xlsx'
    sheet_name = '文件' + filename[-6]+"的运动学片段"
    data = pd.read_excel(filename, sheet_name, index_col=False)
    return data,filename

#计算各个指标
def IndicatorsGet(data):
    data=np.array(data)
    speed_mean=[]#平均速度
    speed_std=[]#速度标准差
    delta_speed_std=[]#加速度标准差
    speed_travel_mean=[]#平均行驶速度
    Acceleration_mean=[]#平均加速度
    Deceleration_mean=[]#平均减速度
    delta_speed=[]#速度变化量
    rate_Idling=[]#怠速时间比
    rate_Acceleration=[]#加速时间比
    rate_Deceleration=[]#减速时间比
    samples=np.size(data,axis=0)#计算样本数
    
    #算法开始
    for i in range(0,samples):
        temp_Acceleration=[]
        temp_Deceleration=[]
        temp_delta=[]
        speed_temp=data[i,~np.isnan(data[i,:])]
        speed_num=len(speed_temp)
        speed_mean.append(np.mean(speed_temp))#计算平均速度
        speed_travel_mean.append(np.mean(speed_temp[speed_temp>2.7]))#计算平均行驶速度
        Idling_speed=speed_temp[speed_temp<2.7]#计算怠速时间
        rate_Idling.append(len(Idling_speed)/len(speed_temp))#计算怠速时间比
        speed_std.append(np.std(speed_temp))#计算速度标准差
        for s in range(0,speed_num-1):
            if speed_temp[s+1]-speed_temp[s]>0:
                temp_Acceleration.append((speed_temp[s+1]-speed_temp[s])*1000/3600)
            elif speed_temp[s+1]-speed_temp[s]<0:
                temp_Deceleration.append((speed_temp[s+1]-speed_temp[s])*1000/3600)
            temp_delta.append((speed_temp[s+1]-speed_temp[s])*1000/3600)
        delta_speed.append(np.mean(temp_delta))
        Acceleration_mean.append(np.mean(temp_Acceleration))#计算平均加速度
        Deceleration_mean.append(np.mean(temp_Deceleration))#计算平均减速度
        rate_Acceleration.append(len(temp_Acceleration)/len(speed_temp))#计算加速时间比
        rate_Deceleration.append(len(temp_Deceleration)/len(speed_temp))#计算减速时间比
        delta_speed_std.append(np.std(delta_speed))
    
    #转换成dataframe输出
    dict1 = {
            '平均速度': speed_mean,
            '平均行驶速度':speed_travel_mean,
            '平均加速度':Acceleration_mean,
            '平均减速度':Deceleration_mean,
            '怠速时间比':rate_Idling,
            '加速时间比':rate_Acceleration,
            '减速时间比':rate_Deceleration,
            '速度标准差': speed_std,
            '加速度标准差':delta_speed_std
     }
    data1=pd.DataFrame(dict1)
    data1.iloc[:,0:8]=round(data1.iloc[:,0:8],2)
    return data1

#重新排序(低速为0，中速为1，低速为2)
def SortLab(data1,K,lab):
    s=[0]*K
    flag=[1]*K
    for l in range(0,len(lab)):
        if sum(flag)==0:
            break
        else:
            if lab[l]==0 and flag[0]:
                s[0]=data1[u'平均速度'][l]
                flag[0]=0
            elif lab[l]==1 and flag[1]:
                s[1]=data1[u'平均速度'][l]
                flag[1]=0
            elif lab[l]==2 and flag[2]:
                s[2]=data1[u'平均速度'][l]
                flag[2]=0
    idx_max=s.index(max(s))
    idx_min=s.index(min(s))
    for i in range(0,len(s)):
        if i!=idx_max and i!=idx_min:
            idx_middle=i
    idx_0=np.where(lab==idx_min)
    idx_1=np.where(lab==idx_middle)
    idx_2=np.where(lab==idx_max)
    lab[idx_0]=0
    lab[idx_1]=1
    lab[idx_2]=2
    return lab


def MinMax(C):
    min = np.min(C)
    max = np.max(C)
    i=0
    for one in C:
        one = (one-min) / (max-min)
        C[i]=one
        i=i+1
    return C


#输出误差
def GapGet(slit,data1):
    data2=IndicatorsGet([list(slit)])
    data2=np.array(data2)
    data1_mean=np.array(np.mean(data1,axis=0)).reshape((1,9))
    data3=np.append(data1_mean,data2).reshape((2,9))
    ran=np.size(data3,axis=1)#计算样本数
    gap=[0]*ran
    for i in range(0,ran):
        gap[i]=np.std(data3[:,i])
    gap=MinMax(gap)
    gap=sum(gap)
    return gap
    
if __name__ == "__main__":
    # 读取数据
    data, filename = DataGet()  
    data=np.array(data)
    
    #去除首部和尾部不是0的运动学片段
    idx1=data[:,0]==0
    data=data[idx1,:]
    data1=data
    ran=np.size(data1,axis=0)
    data=[]
    data2=[]
    for i in range(0,ran):
        if data1[i,~np.isnan(data1[i,:])][-1]==0:
            data.append(list(data1[i,:]))
            data2.append(list(data1[i,~np.isnan(data1[i,:])]))
            
    print('开始计算各指标')
    #计算各个指标
    data1=IndicatorsGet(data) 
    data=data2
    samples=len(data)
    
    print('输出指标文件')
    print(data1)
    #输出
    writer = pd.ExcelWriter('文件'+filename[-6]+'的指标结果.xlsx')
    sheetname='文件'+filename[-6]+'的指标'
    data1.to_excel(writer, sheet_name = sheetname, index = False)
    writer.save()
    
    print('开始聚类')
    #调用KMeans聚类算法
    score = []
    K = 3 # 分为K类
    for k in [K]:
        clf = KMeans(n_clusters=k)
        lab = clf.fit_predict(data1)
        score.append(silhouette_score(data1, clf.labels_, metric='euclidean'))
    
    #画聚类图
    colorStore = ['or', 'og', 'ob', 'oc', 'om', 'oy', 'ok']
    fig1=plt.figure()
    ax1=fig1.add_subplot(211)
    ax2=fig1.add_subplot(212)
    ax1.set_ylabel('average_speed(km/h)')
    ax1.set_title('the result of clustering')
    ax2.set_xlabel('time(s)')
    ax2.set_ylabel('average_travel(km/h)')
    for i in range(0,samples):
        color = colorStore[lab[i]]
        ax1.plot(i, data1[u'平均速度'][i], color)
        ax2.plot(i, data1[u'平均行驶速度'][i], color)
        plt.show()
        
    #重新排序(低速为0，中速为1，低速为2)
    lab=SortLab(data1,K,lab)
    
    #提取运动学片段
    slit=[]
    for k in range(0,K):
        temp_slit=[]
        while 1:
            ind=np.where(lab==k)[0]
            ind2=np.random.choice(ind)
            ind=np.delete(ind,ind==ind2)
            for t in data[ind2]:
                temp_slit.append(t)
            if 400<len(temp_slit)<430:
                break
            elif len(temp_slit)>430:
                temp_slit=[]
        for t in temp_slit:
            slit.append(t)
    print('所得运动学片段共'+str(len(slit)))   
    
    #画结果图
    plt.figure()
    plt.plot(slit)
    plt.xlabel('time(s)')
    plt.ylabel('speed(km/h)')
    plt.show()
    
    #输出
    print('输出该片段')
    slit=pd.Series(slit)
    writer = pd.ExcelWriter('文件'+filename[-6]+'运动学片段结果.xlsx')
    sheetname='文件'+filename[-6]+'运动学片段'
    slit.to_excel(writer, sheet_name = sheetname, index = False, header=False)
    writer.save()
    
    #输出误差
    gap=GapGet(slit,data1)
    print('误差为：')
    print(gap)
    
    