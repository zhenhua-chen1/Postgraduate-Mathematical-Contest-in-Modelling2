#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 18:29:03 2022

@author: chenzhenhua
"""
import pandas as pd
import time as t
import numpy as np
from sys import exit
from scipy.interpolate import interp1d #倒入插值库


'''
read the data
'''
def DataGet():
    print("Select number of file (1-3):")
    n = None
    try:
        n = int(input())
    except Exception as e:
        print("Invalid number of files.")
        exit(0)
    if not n or n < 1 or n > 3:
        print("Invalid number of files.")
        exit(0)
    filename='文件'+str(n)+'.xlsx'
    sheet_name='原始数据'+filename[-6]
    data = pd.read_excel(filename, sheet_name,index_col = False)
#    data2 = pd.read_excel('文件2.xlsx', sheet_name='原始数据2',index_col = False)
#    data3 = pd.read_excel('文件3.xlsx', sheet_name='原始数据3',index_col = False)
#    data = data1.append(data2,ignore_index=True)
#    data = data.append(data3,ignore_index=True)
    #data= pd.read_excel('文件11.xlsx', sheet_name='原始数据1',index_col = False)
    data_speed = data[u'GPS车速']
    data_time = data[u'时间']
    x_pos=np.array(data[u'经度'])
    y_pos=np.array(data[u'纬度'])
    data_speed = data_speed*1000/3600
    return data,data_time,data_speed,x_pos,y_pos,filename

def TimeGet(time1,time2):
    timeArray1=t.strptime(time1, "%Y/%m/%d %H:%M:%S.000.")
    timeStamp1 = t.mktime(timeArray1)
    timeArray2=t.strptime(time2, "%Y/%m/%d %H:%M:%S.000.")
    timeStamp2 = t.mktime(timeArray2)
    delta_t=timeStamp2-timeStamp1
    return delta_t,timeStamp1

def insert_num(delta_t,speed1,speed2):
    f1=interp1d([0,1],[speed1,speed2],kind='linear')
    x_pred=np.linspace(0,delta_t-1,num=delta_t+1)
    y=f1(x_pred)*3600/1000
    return y
    

n=0 # 计数器
emplison=0.1#精度(小于该值视为匀速)
data,data_time,data_speed,x_pos,y_pos,filename=DataGet()# read the data
time_interval=2
data_new= data.iloc[0:1,:]
outlier = [] #将异常值保存
Acceleration=3.96 #最大加速度
print_time=0

print('算法开始')
for i in range(1, len(data_speed)):
    time1=np.array(data_new[u'时间'])[-1]
    x1_pos=np.array(data_new[u'经度'])[-1]
    y1_pos=np.array(data_new[u'纬度'])[-1]
    print_time1=time1[11:13]
    if int(print_time)!=int(print_time1):
        print('开始'+time1[0:13]+'点的数据处理')
        print_time=print_time1
    speed1=np.array(data_new[u'GPS车速'])[-1]*1000/3600
    
    delta_t,timeStamp1=TimeGet(time1,data_time[i])#求出时间变化量变化量
    speed=data_speed[i]- speed1#求出速度变化量
    if data_speed[i]>2.7 or n<180:#处理怠速时间
            if data_speed[i]>2.7:
                n=0
            else:
                n=n+1
            if int(delta_t)!=1:#处理时间不连续
                #print('缺少 '+data_time[i]+' 之后的时间')
                if int(delta_t)<=time_interval:
                    for j in range(int(delta_t)-1):
                        data_new=data_new.append(data.iloc[i,:],ignore_index=True) 
                        new_timeArray=t.localtime(timeStamp1+(j+1))
                        new_time=t.strftime("%Y/%m/%d %H:%M:%S.000.",new_timeArray)
                        data_new['时间'][-1:]=new_time
                    if abs(speed)>emplison and speed<=Acceleration and speed>=-8:
                        y=insert_num(int(delta_t),speed1,data_speed[i])
                        data_new['GPS车速'][-1:-1-int(delta_t):-1]=y[-2::-1]
                    data_new=data_new.append(data.iloc[i,:],ignore_index=True) 
                else:
                   data_new=data_new.append(data.iloc[i,:],ignore_index=True)
            else: #处理车速毛刺点
                data_new=data_new.append(data.iloc[i,:],ignore_index=True)
                if speed>Acceleration:
                    data_new['GPS车速'][-1:]=speed1+Acceleration
                elif speed<-8:
                    data_new['GPS车速'][-1:]=speed1-8           

'''
output
'''

print('算法结束导入数据'+filename[-6])

writer = pd.ExcelWriter('result'+filename[-6]+'.xlsx')
sheetname='预处理数据'+filename[-6]
data_new.to_excel(writer, sheet_name = sheetname, index = False)
writer.save()
writer.close()
