from nn import NNbase
import random
import numpy as np
from utils.e_DaKnow.class_Neu_net import Neu_Net
from utils.e_DaKnow.class_Error import Error
from deap import base, creator
from deap import tools
from utils.e_DaKnow.json_python import process
from surrogate_models.SurrogateModel import SurrogateModel
from surrogate_models.gp import getDerivative, randomRange
from knowledge.MonotonicityKnowledge import MonotonicityKnowledge
from data.CsvData import CsvData
import json
import matplotlib.pyplot as plt
from pylab import mpl
import pickle
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
import matplotlib.animation as animation
from matplotlib import cm
import warnings


class EDaKnow(NNbase):
    """
    融合知识的进化神经网络
    """

    def _initialize(self):
        self.iterationTime = 10000
        self.initialIndividual = 100
        self.hiddenLayer = 8
        self.cross = 0.97
        self.mutant = 0.03

    def __fun_random():
        a = random.random()
        return 2 * a - 1

    def knowPoints(self, dataSet, knowList, knowPointsNum):
        """
        使用知识生成知识点
        :return:
        """
        inputTitle = dataSet['title'][0]
        outputTitle = dataSet['title'][1]
        inputRange = dataSet["range"][0]
        outputRange = dataSet["range"][1]
        inputNum = len(inputTitle)
        knowNNInput = []
        knowNNOutput = []

        for i in knowList:

            assert i["type"] == "单调型", "该算法只能支持单调型的知识"
            assert i['output_type'] == outputTitle, "请确保知识和数据的输出变量保持一致"
            assert i['input_type'][0] in inputTitle, "请确保知识和数据的输入变量保持一致"

            knowListNNOutput = []
            knowListNNInput = []

            knowListNNInput = np.array([randomRange(inputRange) for j in range(knowPointsNum)])  # 其他值是任意取的

            knowListNNInput_ = np.linspace(i['input_range'][0][0], i['input_range'][0][1], knowPointsNum)

            # 输入的最后一列表明知识梯度的索引
            inputIndex = inputTitle.index(i['input_type'][0])
            knowListNNInput[:, inputIndex] = knowListNNInput_
            knowListNNInput = np.column_stack((knowListNNInput, [inputIndex] * knowPointsNum))

            if i['mapping_relation'] == ['单调递增']:
                knowListNNOutput = [[1] for j in range(knowPointsNum)]
            elif i['mapping_relation'] == ['单调递减']:
                knowListNNOutput = [[-1] for j in range(knowPointsNum)]

            if i['convar'] != None:
                for item in i['convar']:
                    convarType = item['convar_type']
                    if convarType in inputTitle:
                        convarTypeIndex = inputTitle.index(convarType)
                        convarRangeOrValue = item["convar_RangeOrValue"]
                        if convarRangeOrValue == 'value':
                            convarValue = item["convar_value"]
                            knowListNNInput[:, convarTypeIndex] = convarValue
                        elif convarRangeOrValue == 'range':
                            convar_range = item['convar_range']
                            convar_range_min, convar_range_max = convar_range[0], convar_range[1]
                            for m in range(knowPointsNum):
                                knowListNNInput[m, convarTypeIndex] = random.uniform(convar_range_min,
                                                                                     convar_range_max)

            # 输处的最后一列表明知识梯度的索引
            outputIndex = outputTitle.index(i['output_type'][0])
            knowListNNOutput = np.column_stack((knowListNNOutput, [outputIndex] * knowPointsNum))

            knowNNInput.append(knowListNNInput)
            knowNNOutput.append(np.array(knowListNNOutput))

        return knowNNInput, knowNNOutput

    def knowPass(self, dataSet, knowList, syn0, syn1):

        knowPointsNum = 100  # 每条知识生成100个点用于判断知识的通过率

        knowNNInput, knowNNOutput = EDaKnow().knowPoints(dataSet, knowList, knowPointsNum)

        knowNum = len(knowNNInput)  # 融入知识的数量

        passRate = []  # 记录不同的知识通过的比例

        for i in range(knowNum):

            testA = knowNNInput[i][:, :-1].copy()  # 知识点的输入取值
            inputIndex = int(knowNNInput[i][:, -1][0])  # 知识点的梯度信息输入索引

            gradient = knowNNOutput[i][0][0]  # 知识点的梯度信息，对于单调型知识来说，单调递增是为1，单调递减时为-1

            outputIndex = int(knowNNOutput[i][:, -1][0])  # 知识点的梯度信息输出索引

            testB = testA.copy()  # testA为知识点，testB为知识点在对应梯度上稍大一些的点

            for j in range(len(testA)):
                testB[j][inputIndex] = testB[j][inputIndex] + abs(random.gauss(0, 1))

            # 将testA和testB输入到神经网络之中，获取运算结果
            # 测试组
            nnA = Neu_Net(testA, syn0, syn1)
            testAy = nnA.Neu_net()
            testAy = np.array(testAy)

            # 对照组
            nnB = Neu_Net(testB, syn0, syn1)
            testBy = nnB.Neu_net()
            testBy = np.array(testBy)

            testResult = testBy[:, outputIndex] - testAy[:, outputIndex]
            passPossible = np.sum(np.sign(testResult) == gradient)/ knowPointsNum
            passRate.append(passPossible)

        passRateTotal = np.mean(passRate)

        return passRateTotal

    # 评价函数
    def __evaluate(individual, dataSet, knowList, hiddenLayer):

        inputData = dataSet["input"]
        outputData = dataSet["output"]
        input_param = dataSet["title"][0]
        output_param = dataSet["title"][1]

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

        # 调用a1_Press函数，得到规则一通过率
        pass_possible = EDaKnow().knowPass(dataSet, knowList, syn0, syn1)
        # print(error, pass_possible)

        print(error,pass_possible)
        return error, pass_possible  # 分别是数据的适应度函数和知识的适应度函数

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

            print(i)
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

        assert cov == False, "该算法不能输出方差信息"
        net = Neu_Net(x, self.syn0, self.syn1)
        y_p = net.Neu_net()

        return y_p


##############################################################################
if __name__ == '__main__':
    from data.CsvData import CsvData
    from knowledge.KnowledgeSet import KnowledgeSet
    from benchmark.sphere import Sphere
    from utils.normalization import Normalization

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
    x[:, 1] = np.linspace(-10, 10.0, trainNum)

    xt = np.ones((testNum, ndim))
    xt[:, 0] = np.linspace(-10, 10.0, testNum)
    xt[:, 1] = np.linspace(-10, 10.0, testNum)

    s = Sphere()
    trainSet = s(x)
    testSet = s(xt)

    # plt.plot(x[:, 0], trainSet["output"][:, 0])


    # 获取知识
    k = KnowledgeSet("C:\data\sphereMonoKnowledge1.xml", "C:\data\sphereMonoKnowledge2.xml")
    knowList = k.readKnowledge()

    # k.visualKnowledge()
    x_test = testSet["input"]

    # 对知识和数据进行归一化操作
    n = Normalization()
    trainSet, knowList = n(trainSet, knowList)
    print(knowList)
    xt = n.transform(xt)

    nnModel = EDaKnow()
    # nnModel.iterationTime = 100

    nnModel.setData(trainSet)  # 设置数据

    nnModel.setKnowledge(knowList=knowList)  # 设置知识

    nnModel.train()  # 训练

    yp = nnModel.predict(xt)  # 预测
    print(yp)

    plt.plot(trainSet["input"][:, 0], trainSet["output"][:, 0], label="真实", color="red")
    plt.plot(xt[:, 0], yp[:, 0], label="预测", color='green')
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    # b = nnModel.score(testSet, index="RSME")  # 评价代理模型 ，"RSME", "R2", "MAE", "Confidence"
    # print(b)

    nnModel.save(r"C:\data\代理模型训练\高斯过程代理模型.pkl")  # 保存模型文件

    # gpModel2 = None  # 加载模型文件
    # filename = r"C:\data\代理模型训练\高斯过程代理模型.pkl"
    # with open(filename, "rb") as f:
    #     gpModel2 = pickle.load(f)
    #
    # p = gpModel2.predict(x_test)
    # print(p)
