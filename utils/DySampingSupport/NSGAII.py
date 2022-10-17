import random
import numpy as np
import matplotlib.pyplot as plt
from deap import creator, tools, base, algorithms


def uniform(low, up):
    # 均匀分布生成个体
    return [random.uniform(a, b) for a, b in zip(low, up)]


# ZDT3评价函数,ind长度为2
def ZDT3(ind):
    n = len(ind)
    f1 = ind[0]
    g = 1 + 9 * np.sum(ind[1:]) / (n - 1)
    f2 = g * (1 - np.sqrt(ind[0] / g) - ind[0] / g * np.sin(10 * np.pi * ind[0]))
    return f1, f2


def NSGA2(f):
    # 定义问题
    creator.create('MultiObjMin', base.Fitness, weights=(-1.0, -1.0))
    creator.create('Individual', list, fitness=creator.MultiObjMin)

    pop_size = 100
    NDim = 2
    # 变量下界
    # low = [0] * NDim
    low = [0, 0]
    # 变量上界
    # up = [1] * NDim
    up = [10, 8]

    uniform(low, up)
    # 生成个体
    toolbox = base.Toolbox()
    toolbox.register('Attr_float', uniform, low, up)
    toolbox.register('Individual', tools.initIterate, creator.Individual, toolbox.Attr_float)
    # 生成种群
    toolbox.register('Population', tools.initRepeat, list, toolbox.Individual)
    pop = toolbox.Population(n=pop_size)

    # 注册工具
    toolbox.register('evaluate', f)
    # 锦标赛选择
    toolbox.register('selectGen1', tools.selTournament, tournsize=2)
    # selTournamentDCD(individuals, k)
    toolbox.register('select', tools.selTournamentDCD)
    # tools.cxSimulatedBinaryBounded(ind1, ind2, eta, low, up)
    toolbox.register('mate', tools.cxSimulatedBinaryBounded, eta=20.0, low=low, up=up)
    # 变异 - 多项式变异
    toolbox.register('mutate', tools.mutPolynomialBounded, eta=20.0, low=low, up=up, indpb=1.0 / NDim)

    # 用数据记录算法迭代过程
    # 创建统计信息对象
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register('avg', np.mean)
    stats.register('std', np.std)
    stats.register('min', np.min)
    stats.register('max', np.max)

    # 创建日志对象
    logbook = tools.Logbook()

    # 遗传算法主程序
    # 参数设置
    maxGen = 50
    cxProb = 0.7
    mutateProb = 0.2

    # 遗传算法迭代
    # 第一代
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    record = stats.compile(pop)
    logbook.record(gen=0, **record)

    # 快速非支配排序操作
    fronts = tools.emo.sortNondominated(pop, k=pop_size)
    # 将每个个体的适应度设置为pareto前沿的次序
    for idx, front in enumerate(fronts):
        for ind in front:
            ind.fitness.values = (idx + 1),

    # 创建子代
    offspring = toolbox.selectGen1(pop, pop_size)  # 锦标赛选择
    # algorithms.varAnd进化算法的一部分，仅应用变化部分（交叉和变异）,克隆了个体，因此返回的种群独立于输入种群
    offspring = algorithms.varAnd(offspring, toolbox, cxProb, mutateProb)

    # 第二代之后的迭代
    for gen in range(1, maxGen + 1):
        # 合并父代与子代
        combinedPop = pop + offspring
        # 评价族群-更新新族群的适应度
        fitnesses = map(toolbox.evaluate, combinedPop)
        for ind, fit in zip(combinedPop, fitnesses):
            ind.fitness.values = fit

        # 快速非支配排序
        fronts = tools.emo.sortNondominated(combinedPop, k=pop_size, first_front_only=False)

        # 拥挤距离计算
        for front in fronts:
            tools.emo.assignCrowdingDist(front)

        # 环境选择--精英保留
        pop = []
        for front in fronts:
            pop += front

        # 复制
        pop = toolbox.clone(pop)
        # 基于拥挤度的选择函数用来实现精英保存策略
        pop = tools.selNSGA2(pop, k=pop_size, nd='standard')

        # 创建子代
        offspring = toolbox.select(pop, pop_size)
        offspring = toolbox.clone(offspring)
        offspring = algorithms.varAnd(offspring, toolbox, cxProb, mutateProb)

        # 记录数据-将stats的注册功能应用于pop，并作为字典返回
        record = stats.compile(pop)
        logbook.record(gen=gen, **record)

    # 输出计算过程
    # logbook.header = 'gen', 'avg', 'std', 'min', 'max'
    # print(logbook)

    # 输出最优解
    bestInd = tools.selBest(pop, 1)[0]
    bestFit = bestInd.fitness.values
    print('当前最优解:', bestInd)
    print('对应的函数最小值为:', bestFit)

    # front = tools.emo.sortNondominated(pop, len(pop))[0]
    # # gridPop
    # for ind in front:
    #     plt.plot(ind.fitness.values[0], ind.fitness.values[1], 'r.', ms=2)
    # plt.xlabel('f_1')
    # plt.ylabel('f_2')
    # plt.tight_layout()
    # plt.show()

    return bestInd