#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 29 17:48:23 2022

@author: chenzhenhua
"""

import pandas as pd
import time
import matplotlib.pyplot as plt  # 导入绘图库


def DataGet():
    filenumber = 1 #文件2修改为2，文件3修改为3.
    filename='文件'+str(filenumber)+'.xlsx'
    sheet_name = '预处理数据' + filename[-6]
    data = pd.read_excel(filename, sheet_name, index_col=False)
    data_speed = data[u'GPS车速']
    data_time = data[u'时间']
    return data, data_time, data_speed, filename


def TimeGet(time1, time2):
    timeArray1 = time.strptime(time1, "%Y/%m/%d %H:%M:%S.000.")
    timeStamp1 = time.mktime(timeArray1)
    timeArray2 = time.strptime(time2, "%Y/%m/%d %H:%M:%S.000.")
    timeStamp2 = time.mktime(timeArray2)
    delta_t = timeStamp2 - timeStamp1
    return delta_t


print('读取数据')
n = 1
data, data_time, data_speed, filename = DataGet()  # read the data
Clips = []
speed = []
clips = []
Idling = 5  # 最大速度
Max_len = 20  # 最大长度
flag = 1

print('算法开始')
# 选出非怠速时间和连续区域
for i in range(0, len(data_speed)):
    if data_speed[i] > Idling:
        time1 = data_time[i]
        time2 = data_time[i - 1]
        speed1 = data_speed[i]
        speed2 = data_speed[i - 1]
        if flag == 1:
            for j in data_speed[i - 5:i]:
                speed.append(j)
            flag = 0
        delta_t = TimeGet(time2, time1)
        if delta_t <= 5:
            speed.append(data_speed[i])
        else:
            Clips.append(speed)
            speed = [data_speed[i]]
            continue
    else:
        if flag == 0:
            flag = 1
            time3 = data_time[i + 1]
            delta_t = TimeGet(time1, time3)
            if delta_t <= 5:
                for j in data_speed[i + 1:i + 6]:
                    speed.append(j)
            if len(speed) != 0:
                Clips.append(speed)
                speed = []

# 删除长度太小的列表

for i in range(0, len(Clips)):
    if len(Clips[i]) >= Max_len:
        clips.append(Clips[i])
'''
j=0       
for i in range(0, len(Clips)):
    if len(Clips[j])< Max_len:
        Clips.pop(j) 
    else:
        j=j+1
'''

# 找到最大值和最小值
max_len = max(map(len, clips))
min_len = min(map(len, clips))

print('文件' + filename[-6] + '未删除长度太小片段前一共划分的运动学片段数量为：', len(Clips))
print('文件' + filename[-6] + '删除长度太小片段后一共划分的运动学片段数量为：', len(clips))
print('文件' + filename[-6] + '最长运动学片段长度为：', max_len)
print('算法结束导入数据' + filename[-6])

df = pd.DataFrame(clips)

writer = pd.ExcelWriter('result' + filename[-6] + '.xlsx')
sheetname = '文件' + filename[-6] + '的运动学片段'
df.to_excel(writer, header=None, sheet_name=sheetname, index=False)
writer.save()

print('选取某一片段开始画图')
for Lis in clips:
    if Lis[0] == 0 and Lis[-1] == 0:
        plt.plot(Lis)
        plt.xlabel('time(s)')
        plt.ylabel('speed(km/h)')
        plt.show()
        break
