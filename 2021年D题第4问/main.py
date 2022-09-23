#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 13:20:27 2022

@author: chenzhenhua
"""
import pandas as pd
import numpy as np
import math
from ga import * #调用遗传算法包
from sklearn.ensemble import RandomForestRegressor # 随机森林
from sklearn.ensemble import RandomForestClassifier # 随机森林

def DataGet():
    name=['nHeavyAtom','nS','naAromAtom','nX','nO','nN','nF','nAtom','ATSp2',
     'ATSm4','ATSm2','ATSm1','ATSm5','ATSc1','ALogp2','ALogP','ATSc4','ATSc5',
     'ATSc2','ATSc3']
    name2 = ['Caco-2','CYP3A4','hERG','HOB','MN']
    filename1 = 'Molecular_Descriptor.xlsx'
    filename2 = 'ERα_activity.xlsx'
    filename3 = 'ADMET.xlsx'
    sheet_name1 = 'training'
    sheet_name2 = 'test'
    data1_training = pd.read_excel(filename1, sheet_name1,index_col = False)
    data1_training = data1_training[name]
    data1_test = pd.read_excel(filename1, sheet_name2,index_col = False)
    data1_test = data1_test[name]
    data2_training = pd.read_excel(filename2, sheet_name1,index_col = False)
    data2_training = data2_training['pIC50']
    data2_test = pd.read_excel(filename2, sheet_name2,index_col = False)
    data22_test = data2_test
    data22_test = data2_test.iloc[:,1:]
    data3_training = pd.read_excel(filename3, sheet_name1,index_col = False)
    data3_training = data3_training.iloc[:,1:]
    data3_test = pd.read_excel(filename3, sheet_name2,index_col = False)
    sample_num = 50
    return name,data1_training,data1_test, data2_training,data2_test,data22_test,name2,data3_training,data3_test,sample_num

#遗传算法解码
def b2d(b, chrom_length, var_range, div,sample_num):
    rwno = []
    #因为染色体里面有多个变量，所以需要div来分割
    rwno = np.zeros([sample_num,len(div)])
    b = np.reshape(b,[sample_num,int(chrom_length/sample_num)])
    for s in range (sample_num):
        for i in range(len(div)):
            max_value = var_range[1,i]
            min_value = var_range[0,i]
            if i == 0:
                star = 0
                end = div[i]
            else:
                star = div[i-1] + 1
                end = div[i]
            t = 0
            for j in range(star, end): # 分隔参数[1,2,3||4,5,6]
                t += b[s,j] * (math.pow(2, j - star))
            t = t * max_value / (math.pow(2, end - star + 1) - 1) - min_value
            rwno[s,i]=t
    return rwno

#算出适应性函数
def calfitValue(x,clf1,clf2):
    y1 = clf1.predict(x)
    y2 = clf2.predict(x)
    y2[:,2]=1-y2[:,2]
    y2[:,4]=1-y2[:,4]
    fit_value=sum(y1)+sum(sum(y2))
    return fit_value


name,data1_training,data1_test, data2_training,data2_test,data22_test,name2,data3_training,data3_test,sample_num = DataGet()


clf1=RandomForestRegressor()
clf2 = RandomForestClassifier(random_state=0)
clf1.fit(data1_training, data2_training)
clf2.fit(data1_training, data3_training)

var_range = np.zeros([2,len(name)])
for ind,n in enumerate(name):
    var_range[0,ind] = min(data1_test[n])
    var_range[1,ind] = max(data1_test[n])

#调用遗传算法
generation = 10  # 繁衍代数
group_size = 100     # 染色体数量，偶数
chrom_length = 800 *sample_num  # 染色体长度
group = getFisrtGroup(group_size, chrom_length) #求出初始染色体
pc = 0.7            # 交配概率
pm = 0.1            # 变异概率
results = []        # 存储每一代的最优解
best_value = 0

#对染色体进行分割
div= []
div_num = 800/20
for i in range(800):
    if (i+1)% div_num ==0:
        div.append(i)

# 调用遗传算法
for g in range(generation):  
    print('第'+str(g+1)+'次迭代')
    fit_value = []    
    for i in range(len(group)): 
        #解码且求出适应性函数值  
        x = b2d(group[i], chrom_length, var_range, div,sample_num)#这里面可能是多个变量
        f= calfitValue(x,clf1,clf2)
        fit_value.append(f)
    best_individual, best_fit = best(group, fit_value)  # 返回最优基因, 最优适应值
    xx = b2d(best_individual, chrom_length, var_range, div,sample_num)
    if  best_fit>best_value:
        best_value = best_fit
    results.append([g+1, best_fit, best_value,xx ])  #进坐标里
    crossover(group, fit_value, pc) # 交配
    mutation(group, pm) # 变异
rank = sorted(results, key=lambda x:x[1])

#输出适应图
X = []
Y = []
for i in range(generation):
   X.append(i)
   Y.append(results[i][2])
plt.rcParams['font.family'] = ['Arial Unicode MS'] #正常显示中文
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
plt.plot(X, Y)
plt.title('遗传算法迭代图')
plt.show()

#导出结果
var = results[-1][3]
y2 = clf2.predict(var)
y = pd.DataFrame(y2, columns=name2 )
var = pd.DataFrame(var, columns=name )
var_range = np.zeros([2,len(name)])
for ind,n in enumerate(name):
    var_range[0,ind] = min(var[n])
    var_range[1,ind] = max(var[n])
var_range = pd.DataFrame(var_range, index= ['最小值','最大值'],columns=name )

#输出文件
writer = pd.ExcelWriter('第四问结果.xlsx')
writer2 = pd.ExcelWriter('五个ADMET的性质.xlsx')
var_range.to_excel(writer,sheet_name='变化范围')
y.to_excel(writer2,sheet_name='变化范围',index = False)
writer.save()
writer2.save()




