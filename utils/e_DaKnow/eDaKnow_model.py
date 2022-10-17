# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 21:51:20 2018

@author: yewenbin
"""

import random
import numpy as np
from smt.utils.e_DaKnow.class_Neu_net import Neu_Net
from smt.utils.e_DaKnow.class_Error import Error
from deap import base, creator
from deap import tools
from smt.utils.e_DaKnow.json_python import process
from knowledge.MonotonicityKnowledge import MonotonicityKnowledge
from data.CsvData import ExcelData
import json
import matplotlib.pyplot as plt
from pylab import mpl


###################################################################

def PLT(plt_avg, address):
    mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    plt_avg = np.array(plt_avg)
    x = plt_avg[:, 0]
    x1 = []

    for i in range(len(x)):
        x1.append(i + 1)

    plt.plot(x1, x, '-', c='k')
    plt.grid(True, linestyle=":", color="k", linewidth="0.6")
    plt.xlabel('迭代代数')
    plt.ylabel('误差值')
    #        plt.annotate('100% pass possible', xy=(30, 0.99), xytext=(40, 0.85),
    #                     arrowprops=dict(arrowstyle="->",facecolor='black'))

    label = ["训练误差"]
    plt.legend(label, loc=0, ncol=1)

    plt.savefig(address)


class GEN(object):

    '''
    def initialize(self, path_data, path_param, path_know):
        self.path_data = path_data
        self.path_param = path_param
        self.path_know = path_know
    '''
    def __init__(self,xt,yt,title,know,iterationTime = 100,initialIndividual = 100,hiddenLayer = 3,cross = 0.3,mutant =0.7):


        self.know = know
        self.xt = xt
        self.yt = yt
        self.title = title
        self.in_type = self.know['input_type']
        self.in_range = self.know['input_range']
        self.out_type =self.know['output_type']

        if self.know['mapping_relation'] == ['单调递增']:
            self.status = True
        else:
            self.status = False
        self.nx = len(xt[0])
        self.ny = len(yt[0])
        self.iterationTime = iterationTime
        self.initialIndividual = initialIndividual
        self.hiddenLayer = hiddenLayer
        self.cross = cross
        self.mutant = mutant


    # 随机值生成函数 【-1，1】
    def fun_random():
        a = random.random()
        return 2 * a - 1

    ########################################################################
    # 评价函数
    def evaluate(individual,xt,yt,title,know,iterationTime = 100,initialIndividual = 100,hiddenLayer = 3,cross = 0.3,mutant =0.7):

        '''
        path_data = "C:\data\测试数据.xlsx"
        path_param = "C:\data\TestParam.json"
        path_know = "C:\data\测试知识.txt"
        '''

        in_type = know['input_type'][0]
        in_range = know['input_range'][0]
        out_type = know['output_type'][0]


        in_min = in_range[0]
        in_max = in_range[1]

        if know['mapping_relation'] == ['单调递增']:
            status = True
        else:
            status = False

        input_param = title[0]
        output_param = title[1]
        in_know_num = input_param.index(in_type)
        out_know_num = output_param.index(out_type)

        nx = len(xt[0])
        ny = len(yt[0])
        # Hid_layer = param[2]

        a = 2 * (np.array(1) + 0) - 1

        # 将individual分配给神经网络的两个连接矩阵
        syn0, syn1 = np.zeros([nx, hiddenLayer]), np.zeros([hiddenLayer, ny])
        for i in range(nx):
            for j in range(hiddenLayer):
                syn0[i][j] = individual[hiddenLayer * i + j]
        for i in range(hiddenLayer):
            for j in range(ny):
                syn1[i][j] = individual[nx * hiddenLayer + ny * i + j]
        # 调用Error函数，得到训练误差error
        er = Error(hiddenLayer, syn0, syn1, xt,yt)
        error = er.error()

        #        #调用a1_Press函数，得到规则一通过率
        test_a = np.random.random((100, nx)) * (in_max - in_min) + in_min
        test_b = test_a.copy()
        for i in range(len(test_a)):
            test_b[i][in_know_num] = test_b[i][in_know_num] + abs(random.gauss(0, 1))

        # 测试组
        test_a_l2_0 = Neu_Net(test_a, syn0, syn1)
        test_a_l2 = test_a_l2_0.Neu_net()
        test_a_l2 = np.array(test_a_l2)

        # 对照组
        test_b_l2_0 = Neu_Net(test_b, syn0, syn1)
        test_b_l2 = test_b_l2_0.Neu_net()
        test_b_l2 = np.array(test_b_l2)

        # test_1 = test_b_l2[:,out_know_num] - test_a_l2[:,out_know_num]
        # pass_possible = np.sum( np.sign(test_1) == a) / 100

        if status == True:
            test_1 = test_b_l2[:, out_know_num] - test_a_l2[:, out_know_num]
        else:
            test_1 = test_a_l2[:, out_know_num] - test_b_l2[:, out_know_num]

        pass_possible = np.sum(np.sign(test_1) == a) / 100

        return error, pass_possible  # 分别是数据的适应度函数和知识的适应度函数

    ##############################################################################
    # 主函数
    def train(self):


        IND_SIZE = self.nx * self.hiddenLayer + self.hiddenLayer * self.ny
        creator.create("FitnessMulti", base.Fitness, weights=(-1, 1))
        creator.create("Individual", list, fitness=creator.FitnessMulti)
        toolbox = base.Toolbox()
        toolbox.register("attribute", GEN.fun_random)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=IND_SIZE)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
        toolbox.register("select", tools.selNSGA2)
        toolbox.register("evaluate", GEN.evaluate)

        pareto_front = tools.ParetoFront()
        logbook = tools.Logbook()
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        NGEN = self.iterationTime
        NPOP = self.initialIndividual
        Mu = self.mutant
        CXPB = self.cross

        logbook.header = "gen", "evals", "std", "min", "avg", "max"
        pop = toolbox.population(n=NPOP)

        # Evaluate the entire population
        # 给每个ind匹配相应的fit
        fitnesses = map(lambda x: GEN.evaluate(x,self.xt,self.yt,self.title,self.know,iterationTime= self.iterationTime,initialIndividual= self.initialIndividual,hiddenLayer= self.hiddenLayer,cross= self.cross,mutant= self.mutant), pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        for i in range(NGEN):
            # print("算法进度：" + str(i/NGEN) )
            print(str((i + 1) / NGEN))

            # Select the next generation individuals
            # Clone the selected individuals
            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))

            # 交叉操作
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            # 变异操作
            for mutant in offspring:
                if random.random() < Mu:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # 只有在ind.fitnes.valid = 0 的情况下才会是的invalid_ind = ind
            # 变异交叉之后，删去了个体的fitness，使得valid为0
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(lambda x: GEN.evaluate(x,self.xt,self.yt,self.title,self.know,iterationTime= self.iterationTime,initialIndividual= self.initialIndividual,hiddenLayer= self.hiddenLayer,cross= self.cross,mutant= self.mutant), pop)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # 选择较优后代
            pop = toolbox.select(pop + offspring, len(pop))

            record = stats.compile(pop)
            logbook.record(gen=i, evals=len(invalid_ind), **record)
            pareto_front.update(pop)
        plt_avg = logbook.select("avg")

        # best_ind
        ccc = []
        for i in range(len(pop)):
            aaa = pop[i]
            bbb = aaa.fitness.values
            ccc.append(bbb)
        ccc = np.array(ccc)
        pop_list = np.lexsort([-1 * ccc[:, 0]])
        best_ind = pop[pop_list[99]]

        #        address = PLT.plt(plt_avg)
        a = 2 * (np.array(1) + 0) - 1

        # 将individual分配给神经网络的两个连接矩阵
        syn0, syn1 = np.zeros([self.nx, self.hiddenLayer]), np.zeros([self.hiddenLayer, self.ny])
        for i in range(self.nx):
            for j in range(self.hiddenLayer):
                syn0[i][j] = best_ind[self.hiddenLayer * i + j]
        for i in range(self.hiddenLayer):
            for j in range(self.ny):
                syn1[i][j] = best_ind[self.nx * self.hiddenLayer + self.ny * i + j]

        self.syn0 = syn0
        self.syn1 = syn1

#      return best_ind, best_ind.fitness.values, pop, plt_avg, Mu, CXPB, NGEN, NPOP, syn0, syn1, self.title


    def predict_value(self,x):

        net = Neu_Net(x, self.syn0, self.syn1)
        y_p = net.Neu_net()

        return y_p



##############################################################################
if __name__ == '__main__':

    path_data = "C:\data\测试数据.xlsx"
    path_param = "C:\data\TestParam.json"
    path_know = "C:\data\测试知识.txt"
    address = 'C:\data\ENN.png'
    file = 'C:\data\ENN.npz'

    #获取知识
    mono = MonotonicityKnowledge(path_know)
    know = mono.readKnowledge()

    #获取训练数据
    data = ExcelData(path_data)
    xt,yt = data.getData()
    title = data.getTitle()

    #训练代理模型
    gen = GEN(xt,yt,title,know)
    gen.train()

    #利用模型预测
    x = np.array([[15,15,45],[10,10,10]])
    y = gen.predict_value(x)
    print(y)





    # np.savez(file, k_a=syn0, k_b=syn1, input=title[0], output=title[1])
    # PLT(plt_avg, address)
    # Back = {}
    # Back['error'] = best_ind.fitness.values[0].tolist()
    # Back['mutant'] = Mu
    # Back['cross'] = CXPB
    # Back['activeFunc'] = 'Sigmoid'
    # Back['iterationTime'] = NGEN
    # Back['populationSize'] = NPOP
    # # Back['hiddenlayers'] = Hid_layer
    #
    # print("_end:" + json.dumps(Back))




