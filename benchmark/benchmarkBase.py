import numpy as np
import copy
import random
from abc import ABCMeta, abstractmethod
import os
from knowledge.KnowledgeSet import KnowledgeSet
from data.CsvData import CsvData

class Benchmark(object, metaclass=ABCMeta):

    def __init__(self, **kwargs):
        """
        Constructor where values of options can be passed in.

        For the list of options, see the documentation for the problem being used.

        Parameters
        ----------
        **kwargs : named arguments
            Set of options that can be optionally set; each option must have been declared.

        """

        self.ndim = 2
        self.xlimits = np.zeros((self.ndim, 2))
        self.name = ""
        self._initialize()

    def _initialize(self) -> None:
        """
        Implemented by problem to declare options (optional).

        Examples
        --------
        self.options.declare('option_name', default_value, types=(bool, int), desc='description')
        """
        pass

    def __call__(self, x: np.ndarray, proportion=None) -> np.ndarray:
        """
        Evaluate the function.

        Parameters
        ----------
        x : ndarray[n, nx] or ndarray[n]
            Evaluation points where n is the number of evaluation points.
        kx : int or None
            Index of derivative (0-based) to return values with respect to.
            None means return function value rather than derivative.

        Returns
        -------
        ndarray[n, 1]
            Functions values if kx=None or derivative values if kx is an int.
        """

        assert self.ndim == len(x[0]), "输入的维度不正确"

        self.dataSet = self._evaluate(x)

        if proportion == None:
            return self.dataSet
        else:
            assert proportion <= 1, "请输入小于1的数"
            pointsNum = len(self.dataSet["input"])
            trainNum = int(proportion * pointsNum)

            testPointsInput = list(self.dataSet["input"])
            testPointsOutput = list(self.dataSet["output"])
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

            trainSet = copy.deepcopy(self.dataSet)
            testSet = copy.deepcopy(self.dataSet)

            trainSet["input"] = np.array(trainPointsInput)
            trainSet["output"] = np.array(trainPointsOutput)

            testSet["input"] = np.array(testPointsInput)
            testSet["output"] = np.array(testPointsOutput)

            return trainSet, testSet


    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Implemented by surrogate models to evaluate the function.

        Parameters
        ----------
        x : ndarray[n, nx]
            Evaluation points where n is the number of evaluation points.
        kx : int or None
            Index of derivative (0-based) to return values with respect to.
            None means return function value rather than derivative.

        Returns
        -------
        ndarray[n, 1]
            Functions values if kx=None or derivative values if kx is an int.
        """
        raise Exception("This problem has not been implemented correctly")


    def getKnowledge(self,knowType = None,visual = True):
        """

        :param knowType:list,想要得到的知识的类型
        :param visual: bool,是否查看知识的具体类型
        :return:
        """

        path = os.path.dirname(os.path.abspath(__file__))
        path2 = os.path.join(path, self.name)             #函数或者工程案例对应的文件夹
        knowPath = os.path.join(path2, "knowledge")       #存放知识的文件夹

        k = KnowledgeSet(folder=[knowPath],knowType=knowType)
        knowList = k.readKnowledge()

        if visual:
            k.visualKnowledge()

        return knowList

    def getData(self, proportion= None):
        """

        :param proportion: None 或者 float,训练集与测试集的划分
        :return:
        """

        path = os.path.dirname(os.path.abspath(__file__))
        path2 = os.path.join(path, self.name)             #函数或者工程案例对应的文件夹
        dataPath = os.path.join(path2, "data")        #存放数据的文件夹
        csvPath = os.path.join(dataPath, self.name +"data.csv")

        d = CsvData(csvPath)
        dataSet = d.read()

        if proportion:
            trainSet, testSet = d.divide(0.8)
            return trainSet, testSet
        else:
            return dataSet

