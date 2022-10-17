import random
import numpy as np
from smt.utils.e_DaKnow.class_Neu_net import Neu_Net
from smt.utils.e_DaKnow.class_Error import Error
from deap import base, creator
from deap import tools
from smt.utils.e_DaKnow.json_python import process
from surrogate_models.SurrogateModel import SurrogateModel
from knowledge.MonotonicityKnowledge import MonotonicityKnowledge
from data.CsvData import CsvData
import json
import matplotlib.pyplot as plt
from pylab import mpl
import pickle


class NNbase(SurrogateModel):
    pass


class EDaKnow(NNbase):
    """
    融合知识的进化神经网络
    """

    def _initialize(self):
        self.iterationTime = 100
        self.initialIndividual = 100
        self.hiddenLayer = 3
        self.cross = 0.3
        self.mutant = 0.7

    def __fun_random():
        a = random.random()
        return 2 * a - 1



    # 评价函数
    def __evaluate(individual, dataSet, knowlist,hiddenLayer):

        inputData = dataSet["input"]
        outputData = dataSet["output"]
        input_param = dataSet["title"][0]
        output_param = dataSet["title"][1]

        # in_P, in_B, out_P, out_B, status = process.processKnow(path_know)
        in_P, in_B, out_P, out_B, status = [], [], [], [], []
        for i in knowlist:

            assert i["type"] == "单调型","该算法只能支持单调型的知识"
            in_P.append(i["input_type"])
            in_B.append(i["input_range"])
            out_P.append(i["output_type"])

            if i["mapping_relation"] == ['单调递增']:
                status.append(True)
            elif i["mapping_relation"] == ['单调递减']:
                status.append(False)

        in_know_num = input_param.index(in_P[0][0])
        out_know_num = output_param.index(out_P[0][0])

        len_in = len(inputData[0])
        len_out = len(outputData[0])
        Hid_layer = hiddenLayer

        a = 2 * (np.array(1) + 0) - 1

        # 将individual分配给神经网络的两个连接矩阵
        syn0, syn1 = np.zeros([len_in, Hid_layer]), np.zeros([Hid_layer, len_out])
        for i in range(len_in):
            for j in range(Hid_layer):
                syn0[i][j] = individual[Hid_layer * i + j]
        for i in range(Hid_layer):
            for j in range(len_out):
                syn1[i][j] = individual[len_in * Hid_layer + len_out * i + j]
        # 调用Error函数，得到训练误差error
        er = Error(Hid_layer, syn0, syn1, inputData, outputData)
        error = er.error()

        #        #调用a1_Press函数，得到规则一通过率
        test_a = np.random.random((100, len_in))
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

        test_1 = test_b_l2[:, out_know_num] - test_a_l2[:, out_know_num]
        pass_possible = np.sum(np.sign(test_1) == a) / 100

        return error, pass_possible    # 分别是数据的适应度函数和知识的适应度函数



    def _train(self):
        IND_SIZE = self.nx * self.hiddenLayer + self.hiddenLayer * self.ny
        creator.create("FitnessMulti", base.Fitness, weights=(-1, 1))
        creator.create("Individual", list, fitness=creator.FitnessMulti)
        toolbox = base.Toolbox()
        toolbox.register("attribute", EDaKnow.__fun_random)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=IND_SIZE)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
        toolbox.register("select", tools.selNSGA2)
        toolbox.register("evaluate", EDaKnow.__evaluate)

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
        fitnesses = map(
            lambda x: EDaKnow.__evaluate(x, self.dataSet, self.knowList, hiddenLayer=self.hiddenLayer), pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        for i in range(NGEN):

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
            fitnesses = map(
                lambda x: EDaKnow.__evaluate(x, self.dataSet, self.knowList, hiddenLayer=self.hiddenLayer), pop)
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

    def predict(self, x, cov=False):

        assert cov == False,"该算法不能输出方差信息"
        net = Neu_Net(x, self.syn0, self.syn1)
        y_p = net.Neu_net()

        return y_p




##############################################################################
if __name__ == '__main__':

    from data.CsvData import CsvData
    from knowledge.KnowledgeSet import KnowledgeSet
    from benchmark.sphere import Sphere




    # 获取数据
    # path = r"C:\data\代理模型训练\代理模型训练.csv"
    # d = CsvData(path)
    # dataSet = d.read()
    # print(dataSet)
    # trainSet, testSet = d.divide(0.8)

    ndim = 2
    trainNum = 100
    testNum = 10

    x = np.ones((trainNum, ndim))
    x[:, 0] = np.linspace(-10, 10.0, trainNum)
    x[:, 1] = 0.0

    xt = np.ones((testNum, ndim))
    xt[:, 0] = np.linspace(-10, 10.0, testNum)
    xt[:, 1] = 0.0


    s = Sphere()
    trainSet = s(x)
    testSet = s(xt)

    # plt.plot(x[:, 0], trainSet["output"][:, 0])


    # 获取知识
    k = KnowledgeSet("C:\data\sphereShapeKnowledge.xml", "C:\data\sphereMonoKnowledge1.xml","C:\data\sphereMonoKnowledge2.xml")
    knowList = k.readKnowledge()
    k.visualKnowledge()
    x_test = testSet["input"]

    gpModel = EDaKnow()
    # gpModel = GP()

    gpModel.setData(trainSet)  # 设置数据

    gpModel.setKnowledge(knowList=knowList)  # 设置知识

    gpModel.train()  # 训练


    yp = gpModel.predict(x_test)  # 预测
    print(yp)

    plt.plot(x[:, 0], trainSet["output"][:, 0],label = "真实")
    plt.plot(x_test[:, 0], yp[:, 0],label = "预测")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    b = gpModel.score(testSet, index="RSME")  # 评价代理模型 ，"RSME", "R2", "MAE", "Confidence"
    print(b)

    gpModel.save(r"C:\data\代理模型训练\高斯过程代理模型.pkl")  # 保存模型文件

    # gpModel2 = None  # 加载模型文件
    # filename = r"C:\data\代理模型训练\高斯过程代理模型.pkl"
    # with open(filename, "rb") as f:
    #     gpModel2 = pickle.load(f)
    #
    # p = gpModel2.predict(x_test)
    # print(p)
