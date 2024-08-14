import numpy as np
import pandas as pd

# 参数
POP_SIZE =  100  # 种群大小
GENES = 375000          # 基因数（问题的维度）
GEN_MAX = 10         # 基因最大值
GEN_MIN = -10        # 基因最小值
MUTATION_RATE = 0.2  # 变异率
CROSSOVER_RATE = 0.9 # 交叉率
MAX_GEN = 10       # 最大迭代次数
L = 20
U = 25

# 目标函数
def objective_function(chromosome):
    decoded = decode(chromosome)
    
    # 计算每份作品被评审的次数
    review_per_paper = decoded.sum(axis=1)
    
    # 计算每位评审专家评审的作品数量
    papers_per_reviewer = decoded.sum(axis=0)
    
    # 约束惩罚
    penalty = 0
    
    
    # 对于评审的作品数量超出范围的，惩罚
    penalty += 100*np.sum(np.clip(papers_per_reviewer - U, 0, None))
    penalty += 100*np.sum(np.clip(L - papers_per_reviewer, 0, None))
    
    # 评价目标函数
    intersection_matrix = np.dot(decoded, decoded.T) - np.eye(decoded.shape[0]) * 5
    objective = np.sum(intersection_matrix)
    # 计算每个作品与每个评审专家对的交集
    sum_per_row = np.sum(intersection_matrix, axis=1) - 5
    return objective - penalty

# 目标函数
def find_value(chromosome):
    decoded = decode(chromosome)
    
    # 计算每份作品被评审的次数
    review_per_paper = decoded.sum(axis=1)
    
    # 计算每位评审专家评审的作品数量
    papers_per_reviewer = decoded.sum(axis=0)
    
    # 约束惩罚
    penalty = 0
    
    
    # 对于评审的作品数量超出范围的，惩罚
    penalty += np.sum(np.clip(papers_per_reviewer - U, 0, None))
    penalty += np.sum(np.clip(L - papers_per_reviewer, 0, None))
    
    # 评价目标函数
    intersection_matrix = np.dot(decoded, decoded.T) - np.eye(decoded.shape[0]) * 5
    objective = np.sum(intersection_matrix)
    # 计算每个作品与每个评审专家对的交集
    sum_per_row = np.sum(intersection_matrix, axis=1) - 5
    return objective,sum_per_row

#解码
def decode(chromosome):
    # 将染色体重塑为3000x125的矩阵
    matrix = chromosome.reshape(3000, 125)

    # 初始化3000x125的全零矩阵
    decoded = np.zeros((3000, 125), dtype=int)

    # 对于每一行，找出5个最大的值的位置，并设置为1
    for i in range(3000):
        ones_positions = matrix[i].argsort()[-5:]  # 选择5个最大的值的位置
        decoded[i, ones_positions] = 1

    return decoded

# 初始化种群
def initialize_population():
    return np.random.uniform(GEN_MIN, GEN_MAX, (POP_SIZE, GENES))


# 选择操作
def selection(population, fitness):
    idx = np.argsort(fitness)[::-1] 
    return population[idx[:POP_SIZE//2]]

# 交叉操作
def crossover(parent1, parent2):
    if np.random.rand() < CROSSOVER_RATE:
        if GENES == 1:
            child1 = (parent1 + parent2) / 2.0
            child2 = (parent1 + parent2) / 2.0
        else:
            crossover_point = np.random.randint(1, GENES)
            child1 = np.hstack((parent1[:crossover_point], parent2[crossover_point:]))
            child2 = np.hstack((parent2[:crossover_point], parent1[crossover_point:]))
        return child1, child2
    return parent1, parent2

# 变异操作
def mutate(child):
    for i in range(GENES):
        if np.random.rand() < MUTATION_RATE:
            child[i] += np.random.uniform(-1, 1)
    return child

# 遗传算法流程
population = initialize_population()

# 在算法开始时初始化全局最佳解和适应度
best_global_solution = None
best_global_fitness = -np.inf  # 假设你在求解的是最大化问题

for generation in range(MAX_GEN):
    fitness = [objective_function(p) for p in population]
    fitness = np.array(fitness)
    
    
    
    # 更新全局最佳解和适应度
    current_best_index = np.argmax(fitness)
    current_best_fitness = fitness[current_best_index]
    if current_best_fitness > best_global_fitness:
        best_global_fitness = current_best_fitness
        best_global_solution = population[current_best_index]
    
    print(f"Generation {generation} - Best Fitness: {np.max(best_global_fitness)}")
    selected_parents = selection(population, fitness)
    children = []

    for i in range(0, POP_SIZE//2, 2):
        parent1 = selected_parents[i]
        parent2 = selected_parents[i + 1]
        child1, child2 = crossover(parent1, parent2)
        children.append(mutate(child1))
        children.append(mutate(child2))

    population = np.vstack((selected_parents, children))

# 最后打印全局最佳解
print(f"Best overall solution found: {best_global_solution}")
print(f"Fitness of best overall solution: {best_global_fitness}")


# 在循环结束后重新计算适应度

objective,sum_per_row = find_value(best_global_solution)
solution = decode(best_global_solution)
print(f"Best solution found: {objective}")
# 输出解码后的解决方案
print("每份作品被评审的次数:")
print(solution.sum(axis=1))

print("\n每位评审专家评审的作品数量:")
print(solution.sum(axis=0))

print("\n每份作品交叉数量:")
print(sum_per_row)

with pd.ExcelWriter('results.xlsx', engine='openpyxl') as writer:
    reviewers = ["作品" + str(i+1) for i in range(solution.shape[0])]
    papers = ["专家" + str(i+1) for i in range(solution.shape[1])]
    # 将每份作品被评审的次数保存到工作表1
    df4 = pd.DataFrame(solution, index=reviewers, columns=papers)
    df4.to_excel(writer, sheet_name='分配表')
    
    # 将每份作品被评审的次数保存到工作表1
    df1 = pd.DataFrame(solution.sum(axis=1), columns=['Number of Reviews'])
    df1.to_excel(writer, sheet_name='每个作品的专家评审人数', index=False)
    
    # 将每位评审专家评审的作品数量保存到工作表2
    df2 = pd.DataFrame(solution.sum(axis=0), columns=['Number of Papers'])
    df2.to_excel(writer, sheet_name='每个评审人数的作品数', index=False)

    # 将每份作品交叉数量保存到工作表3
    intersection_matrix = np.dot(solution, solution.T)
    sum_per_row = np.sum(intersection_matrix, axis=1) - 5  # 你的交叉计算
    df3 = pd.DataFrame(sum_per_row, columns=['每个作品的交叉数量'])
    df3.to_excel(writer, sheet_name='Intersection Counts', index=False)