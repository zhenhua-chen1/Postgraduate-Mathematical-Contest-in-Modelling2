#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 14:37:10 2022

@author: chenzhenhua
"""

import matplotlib.pyplot as plt
import random
import math
#计算函数
def f(args):
    return f2(args)
def f1(args):
    return (3 - (math.sin(2*args[0]))**2 - (math.sin(2*args[1]))**2)
def f2(args):
    x = 1
    for i in range(len(args)):
        z = 0
        for j in range(5):
            z += (j+1) * math.cos(((j+1)+1)*args[i]+(j+1))
        x *= z
    return x

#适应函数
def s(x):
    return s2(x)
def s1(x):
    return math.exp(-abs(x-1))
def s2(x):
    return math.exp(-abs(x+187))

# 计算2进制序列代表的数值
'''
解码并计算值
group 染色体
chrom_length 染色体长度
max_value, min_value 上下限
div 分界点
'''
def b2d(b, chrom_length, max_value, min_value, div):
    rwno = []
    #因为染色体里面有多个变量，所以需要div来分割
    for i in range(len(div)):
        if i == 0:
            star = 0
            end = div[i]
        else:
            star = div[i-1] + 1
            end = div[i]
        t = 0
        for j in range(star, end): # 分隔参数[1,2,3||4,5,6]
            t += b[j] * (math.pow(2, j - star))
        t = t * max_value / (math.pow(2, end - star + 1) - 1) - min_value
        rwno.append(t)

    return rwno # 这是一个list

'''
计算当前函数值
group 染色体
chrom_length 染色体长度
max_value，min_value 最大最小值
divid 分割
'''
def calobjValue(group, chrom_length, max_value, min_value, divid):
    obj_value = []
    for i in range(len(group)):      
        x = b2d(group[i], chrom_length, max_value, min_value, divid)#这里面可能是多个变量

        obj_value.append(f(x))
    return obj_value

# 获取适应值
def calfitValue(obj_value):
    fit_value = []
    for i in range(len(obj_value)):
        temp =  s(obj_value[i]) # 调用适应函数计算
        fit_value.append(temp)
    return fit_value

#累计适应值方便计算平均
def sum_fit(fit_value):
    total = 0
    for i in range(len(fit_value)):
        total += fit_value[i]
    return total

# 转轮盘选择法
def selection(group, fit_value):
    newfit_value = [] #[ [[染色体], [锚点]],... ]
    newgroup = [] #[ [父], [母], [父], [母],....]
    # 适应度总和
    total_fit = sum_fit(fit_value)
    # 设置各个的锚点
    t = 0
    for i in range(len(group)):
        t += fit_value[i]/total_fit
        newfit_value.append([group[i], t])
    # 转轮盘选择法
    for i in range(len(newfit_value)):
        parents = len(newfit_value) # 初始化指针
        r = random.random() #指针
        for j in range(len(newfit_value)):#看看指针指到睡了
            if newfit_value[j][1] > r:
                parents = j
                break
        newgroup.append(newfit_value[parents][0])
        
    return newgroup

# 交配
def crossover(group, fit_value, pc):
    parents_group = selection(group, fit_value) #[ [[父], [母]],....]
    group_len = len(parents_group)
    for i in range(0, group_len, 2):
        if(random.random() < pc): # 看看是否要交配
            cpoint = random.randint(0, len(parents_group[0])) # 随机交叉点
            temp1 = []
            temp2 = []
            temp1.extend(parents_group[i][0:cpoint])
            temp1.extend(parents_group[i+1][cpoint:len(parents_group[i])])
            temp2.extend(parents_group[i+1][0:cpoint])
            temp2.extend(parents_group[i][cpoint:len(parents_group[i])])
            group[i] = temp1
            group[i+1] = temp2

# 基因突变
def mutation(group, pm):
    px = len(group)
    py = len(group[0])

    for i in range(px): # 遍历
        if(random.random() < pm):
            mpoint = random.randint(0, py-1) # 取要变异哪个
            if(group[i][mpoint] == 1):
                group[i][mpoint] = 0
            else:
                group[i][mpoint] = 1

'''
找出最优解和最优解的基因编码
group 种群染色去
fit_value 种群适应
'''
def best(group, fit_value):
    px = len(group)
    best_in = group[0]
    best_fit = fit_value[0]
    for i in range(1, px):
        if(fit_value[i] > best_fit):
            best_fit = fit_value[i]
            best_in = group[i]
    #print(best_in)
    return [best_in, best_fit]

'''
创建初代种群
group_size 种群大小
chrom_length 染色体长度
'''
def getFisrtGroup(group_size, chrom_length):
    #print('初代种群：')
    group = []
    for i in range(group_size):
        temp = []
        for j in range(chrom_length):
            temp.append(random.randint(0, 1))
        group.append(temp)
    #print(group)

    return group



if __name__ == '__main__':

    generation = 5000    # 繁衍代数
    group_size = 400     # 染色体数量，偶数
    max_value = 20       # 范围
    min_value = 10       # 偏移修正
    chrom_length = 800   # 染色体长度
    divid = [399, chrom_length-1]    # 输入值分界点， 最后一位必须是染色体长度
    pc = 0.7            # 交配概率
    pm = 0.1            # 变异概率
    results = []        # 存储每一代的最优解
    fit_value = []      # 个体适应度
    points = [] #多个最优解
    #生成初代
    group = getFisrtGroup(group_size, chrom_length)
    
    for i in range(generation):
        if i > 100:
            pm = 0.01
        if i > 1000:
            pm = 0.001
        obj_value = calobjValue(group, chrom_length, max_value, min_value, divid)   # 个体评价
        fit_value = calfitValue(obj_value)  # 获取群体适应值
        best_individual, best_fit = best(group, fit_value)  # 返回最优基因, 最优适应值
    
        xx = b2d(best_individual, chrom_length, max_value, min_value, divid)
        if( abs(f(xx)+186.730909) < 0.000001):#找到最优解
            flag = False
            for p in points:
                if( (abs(xx[0]-p[0]) < 0.1) and (abs(xx[1]-p[1]) < 0.1) ):#剔除重复解
                    flag = True
                    break
            if flag == False:
                print(xx)
                points.append(xx)
    
        results.append([i, best_fit, b2d(best_individual, chrom_length, max_value, min_value, divid), best_individual])  #加进坐标里
        crossover(group, fit_value, pc) # 交配
        mutation(group, pm) # 变异
    
    #results.sort(key=lambda x:x[1])
    
    rank = sorted(results, key=lambda x:x[1])
    #print('\n', rank[-1])
    
    #print(results)
    x = b2d(rank[-1][3], chrom_length, max_value, min_value, divid)
    #最终结果
    print("f(x) = " , f(x) , "x = " , x , " 染色体 = ", rank[-1][3], "  适应值 = ", rank[-1][1], "代数：", rank[-1][0])
    
    #输出适应图
    X = []
    Y = []
    for i in range(generation):
        X.append(i)
        Y.append(results[i][1])
    
    plt.plot(X, Y)
    
    plt.show()