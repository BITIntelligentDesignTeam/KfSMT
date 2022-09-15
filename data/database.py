'''
数据的基类
'''

import numpy as np
from abc import ABCMeta, abstractmethod
import random
import copy


class Database(object, metaclass=ABCMeta):

    def __init__(self, path):
        self.path = path
        self.title = []
        self.input = []
        self.output = []
        self.range = []
        self.dataDic = {}

    def read(self):
        dataDic = self._read()
        return dataDic

    @abstractmethod
    def _read(self):
        pass

    def divide(self, proportion=0.7):
        """
        按照比例划分训练集和测试集
        :param proportion: float,训练集占所有样本集的比例
        :return: trainSet，testSet: 样本集和测试集
        """
        assert proportion <= 1, "请输入小于1的数"
        pointsNum = len(self.input)
        trainNum = int(proportion * pointsNum)

        testPointsInput = list(self.input)
        testPointsOutput = list(self.output)
        trainPointsInput = []
        trainPointsOutput = []
        for i in range(trainNum):
            rand_index = 0
            num_items = len(testPointsInput)
            rand_index = random.randrange(num_items)
            trainPointsInput.append(testPointsInput[rand_index])
            trainPointsOutput.append(testPointsOutput[rand_index])
            testPointsInput.pop(rand_index)
            testPointsOutput.pop(rand_index)

        trainSet = copy.deepcopy(self.dataDic)
        testSet = copy.deepcopy(self.dataDic)

        trainSet["input"] = np.array(trainPointsInput)
        trainSet["output"] = np.array(trainPointsOutput)

        testSet["input"] = np.array(testPointsInput)
        testSet["output"] = np.array(testPointsOutput)

        return trainSet, testSet
