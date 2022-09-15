import numpy as np
from smt.sampling_methods import Random
from abc import ABCMeta, abstractmethod
from smt.sampling_methods import LHS
from smt.sampling_methods import FullFactorial
import matplotlib.pyplot as plt
import csv


class SamplingBase(object, metaclass=ABCMeta):
    """
    采样部分的基类
    """

    def __init__(self, xlimts):
        """
        :param xlimts: 采样空间
        """
        self.xTitle = [i for i in xlimts]
        self.xRange = [xlimts[i] for i in self.xTitle]
        self._initialize()

    def _initialize(self):
        pass

    def sample(self, nt, tablePath=None):
        """
        开始采样
        :param nt: 采样点数量
        :param table: 默认为None，不生成采样表，可接受str类型，生成采样表的路径，
        :return: samplingPoints，类型为numpy.ndarray[nt,nx]，nt为采样点数量，nx为采样空间维度
        """
        samplingPoints = self._sample(nt)

        if tablePath != None:
            with open(tablePath, 'w', encoding='utf-8', newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(self.xTitle)
                csv_writer.writerow(["input"] * len(self.xTitle))
                csv_writer.writerow([str(i) for i in self.xRange])
                csv_writer.writerows(samplingPoints)

        return samplingPoints

    @abstractmethod
    def _sample(self, nt):
        pass


class RandomSampling(SamplingBase):
    """
    简单随机采样
    """

    def _sample(self, nt):
        sampling = Random(xlimits=np.array(self.xRange))
        samplingPoints = sampling(nt)

        return samplingPoints


class LatinHypercubeSampling(SamplingBase):
    """
    拉丁超立方采样
    """

    def _initialize(self):
        """
        criterion,类型为str，默认为"c"，可选的范围有["center", "maximin", "centermaximin", "correlation", "c", "m", "cm", "corr",
                                  "ese"]，用于构建 LHS 设计的标准， c、m、cm 和 corr 分别是 center、maximin、centermaximin 和correlation, respectively，
                                  分别为将采样间隔内的点居中、最大化点之间的最小距离，并将点放置在其间隔内的随机位置、最大化点之间的最小距离并在其间隔内将点居中、最小化最大相关系数、
                                  使用增强随机进化算法 (ESE) 优化设计
        """
        self.criterion = "c"

    def _sample(self, nt):
        """
        采样
        :param nt: 采样点数量
        :return:
        """
        assert self.criterion in ["center", "maximin", "centermaximin", "correlation", "c", "m", "cm", "corr",
                                  "ese"], "criterion类型不在所给范围中间"

        sampling = LHS(xlimits=np.array(self.xRange), criterion=self.criterion)
        samplingPoints = sampling(nt)
        return samplingPoints


class FullFactorialSampling(SamplingBase):
    """
    全因素采样
    """

    def _initialize(self):
        """
        clip,类型为bool，默认为"False"，将样本数取整到每个 nx 维度的样本数乘积
        """
        self.clip = False

    def _sample(self, nt):
        sampling = FullFactorial(xlimits=np.array(self.xRange), clip=self.clip)
        samplingPoints = sampling(nt)
        return samplingPoints


if __name__ == "__main__":
    path = r"C:\data\采样表示例.csv"
    xlimts = {"x1": [0, 10], "x2": [0, 20]}
    # s = RandomSampling(xlimts)
    s = LatinHypercubeSampling(xlimts)
    s.criterion = "m"
    # s = FullFactorialSampling(xlimts)
    # s.clip = True
    points = s.sample(50, tablePath=path)
    print(points.shape)

    # 画图展示
    plt.plot(points[:, 0], points[:, 1], "o")
    plt.xlabel(s.xTitle[0])
    plt.ylabel(s.xTitle[1])
    plt.show()
