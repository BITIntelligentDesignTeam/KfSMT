import numpy as np
from math import sqrt
import pickle
from utils.check import checkDataKnow

class SurrogateModel(object):
    """
    代理模型的基类
    """

    def __init__(self):
        self.dataSet = {}
        self.knowList = []
        self.inputTitle = []
        self.outputTitle = []

        self._initialize()
        self.model = None

    def _initialize(self):
        pass

    def setData(self, dataSet):
        self.dataSet = dataSet


        self.inputTitle = self.dataSet["title"][0]          #数据的输入参数列表
        self.outputTitle = self.dataSet["title"][1]         #数据的输出参数列表

        self.nx = len(self.inputTitle)         #输入的维度
        self.ny = len(self.outputTitle)        #输出的维度

    def setKnowledge(self, *args, knowList=None):
        for i in args:
            self.knowList.append(i)

        for i in knowList:
            self.knowList.append(i)

    def train(self):

        # self.dataSet,self.knowList = checkDataKnow(self.dataSet,self.knowList)       #检查知识的数据的输入输出情况是否一致
        self._train()
        print("*" * 75)
        print("模型训练成功！")
        print("*" * 75)

    def _train(self):
        pass

    def predict(self, x, cov=False):
        pass

    def score(self, dataSet, index="RSME", error=0.05):
        """

        :param dataSet:类型为dict。测试用数据集，包含着数据全部信息的dict
        :param index:可选项有“RSME”、“R2”、“Confidence”。评价代理模型时所使用到的指标，“RSME”为均方误差根，“R2”为R-平方，“Confidence”为一定误差水平下的置信度
        :param error:类型为float。相对误差，当index = “Confidence”时，将会返回测试集中小于该相对误差下的置信度
        :return:numpy.array[ny]，在模型的每一个输出维度上，利用测试数据集计算出的所选指标值
        """

        testInput = dataSet["input"]
        testOutput = dataSet["output"]

        preOutput = self.predict(testInput)

        assert index in ["RSME", "R2", "MAE", "Confidence"], "请输入正确的模型评价指标"

        if index == "RSME":
            mse = np.sum((testOutput - preOutput) ** 2) / len(testOutput)
            rmse = sqrt(mse)
            return rmse

        elif index == "R2":
            mse = np.sum((testOutput - preOutput) ** 2) / len(testOutput)
            r2 = 1 - mse / np.var(testOutput)
            return r2

        elif index == "MAE":
            mae = np.sum(np.absolute(testOutput - preOutput)) / len(testOutput)
            return mae

        elif index == "Confidence":

            nt, ny = testOutput.shape
            num_error = 0
            for i in range(nt):
                for j in range(ny):
                    if abs(testOutput[i][j] - preOutput[i][j])/ abs(testOutput[i][j]) <= error:
                        num_error += 1

            con = num_error / nt

            return con

    def save(self, path):

        with open(path, "wb") as f:
            pickle.dump(self, f)

