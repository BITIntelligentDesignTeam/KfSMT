import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy import integrate
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sampling import SamplingBase, RandomSampling

from utils.DySampingSupport.sys_BO import sysBO

from knowledge.SpaceKnowledge import SpaceKnowledge
from CsvData import CsvData


class KnowledgeSamplingBase(SamplingBase):
    """
    融合知识信息进行采样的基类
    """

    def setKnowledge(self, *args, knowledgeList=[]):
        self.knowList = [i for i in args] + knowledgeList
        self._setKnowledge()

    def _setKnowledge(self):
        """
        对于知识具体的应用情况
        """
        pass


class DynamicSampling(KnowledgeSamplingBase):
    """
    动态采样
    """

    def _initialize(self):
        self.gpList = []  # 不同高斯过程组成的list
        self.utility = "ud"
        self.proportion = 0.9
        self.weightList = []

    def setData(self, dataSet):
        """
        设置动态采样需要的数据
        """
        self.x = dataSet["input"]
        self.y = dataSet["output"]
        self.title = dataSet["title"]
        self.dataSet = dataSet

    def updateData(self, dataSet):
        """
        更新动态采样的数据，目前是为了重新设定数据，之后应该要改成更新新的数据
        :param dataSet:
        :return:
        """
        self.x = dataSet["input"]
        self.y = dataSet["output"]
        self.title = dataSet["title"]
        self.dataSet = dataSet


    def _setKnowledge(self):
        '''
        动态采样知识的具体使用方式，这里为从空间型知识中获取权重信息
        :return:
        '''

        # 动态采样仅支持空间型的知识
        for i in self.knowList:
            assert i["type"] == "空间型", "动态采样仅支持空间型的知识"

        # 当前版本仅支持单条知识，后续更新中会加入多条知识
        space = self.knowList[0]

        spaceList = space["space_relation"]
        spaceNum = len(spaceList)

        # weightList是最后的权重列表，包含pram_num, pramName, pramRange, weight
        self.weightList = []

        for i in range(spaceNum):
            input_i = spaceList[i]

            pramName = []
            pramRange = []

            level = input_i['area_level']

            # 复杂程度的分级，当前就只有这两种等级，后续还需要和馥琳姐沟通其他等级的权重

            if level == '较复杂':
                weight = 1.5
            elif level == '很复杂':
                weight = 2.5

            Input_list = input_i['input_type']
            Rang_list = input_i['input_range']

            pramNum = len(Input_list)

            for j in range(len(Input_list)):
                pramName.append(Input_list[j])
                pramRange.append(Rang_list[j])

            self.weightList.append([pramNum, pramName, pramRange, weight])

    def _tranInVariable(self,InVariable):
        """
        将{'框架转速': [0.0, 70.0], '温度': [0.0, 50.0], '压力': [1.0, 5.0]}转化为{'x1': [0.0, 70.0], 'x2': [0.0, 50.0], 'x3': [1.0, 5.0]},方便算法运行
        :param In_Variable:
        :return:
        """

        keyInVariable = [key for key in InVariable]
        newDict = {}
        for i in range(len(InVariable)):
            value = InVariable[keyInVariable[i]]
            newDict["x" + str(i + 1)] = value

        return newDict, keyInVariable


    def _sample(self, nt):

        assert self.utility in ["ud", "ei", "ucb"], "请选择正确的效用函数"

        #采样空间
        xlimts, keyInVariable = self._tranInVariable(self.xlimts)

        # 声明一个全局高斯过程
        # 高斯过程核参数设定——各向异性
        length_scale = []
        for i in range(self.x.shape[1]):
            length_scale.append(1)

        # 训练一个高斯过程
        gp_0 = GaussianProcessRegressor(
            kernel=Matern(nu=1.5, length_scale=length_scale),  # 高斯过程超参数
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=20,
        )

        gp_0.fit(self.x, self.y)

        self.gp = gp_0

        def obj_0(x1, x2):
            """两个变量：贝叶斯优化动态采样的目标函数，输入x,返回y"""
            x = [[x1, x2]]
            y = gp_0.predict(x)
            y = np.squeeze(y)
            return y

        # 知识 ------------------------------------------------------

        def knowledge(x1, x2):

            knowledge_list = self.weightList

            weight = 1

            for i in range(len(knowledge_list)):

                space = knowledge_list[i]

                if space[0] == 2:  # 一定要是2维空间？
                    # if space[1] == ['攻角', '马赫数']:

                    if (space[2][0][0] <= x1 < space[2][0][1]) and (space[2][1][0] <= x2 < space[2][1][1]):
                        if weight == 1:
                            weight = space[3]
                    else:
                        if weight != 1:
                            continue
                        else:
                            weight = 1
            return weight

        # 边界惩罚
        def SK_punishment(x1, x2):

            if x2 >= 0.0:
                p = bound_punish(x1, l=0.9, a=0.4, b1=0, b2=10)
                return p
            else:
                p = 1
                return p

        # temp = SK_fa_punishment(12.15922, 0.54228)
        # print(temp)

        def bound_punish(x_punish, l, a, b1, b2):
            # l1 = 0.7
            # a1 = 0.3
            # b1 = 1
            # b2 = 33
            punish_b1 = puni(x_punish, b1, l, a)
            punish_b2 = puni(x_punish, b2, l, a)

            punish_value = (punish_b1 + punish_b2) / 2
            return punish_value

        def puni(x, b, l1, a1):
            mm = l1 * np.abs(x - b) - a1
            if isinstance(mm, float) or isinstance(mm, int):
                erfc = 2 / np.pi ** 0.5 * (integrate.quad(lambda t: np.exp(-t ** 2), 0, mm)[0])
            else:
                erfc = []
                for i in range(len(mm)):
                    erfc.append(2 / np.pi ** 0.5 * (integrate.quad(lambda t: np.exp(-t ** 2), 0, mm[i])[0]))
                erfc = np.array(erfc)
            return erfc

        # 输入参数和边界定义-----------------------------------------------------------------------------
        # pbounds = {'x1': (0, 10), 'x2': (0, 8)}
        # 参数化为：In_Variable

        # f是目标函数
        optimizer_0 = sysBO(
            f=obj_0,
            pbounds=xlimts,
            verbose=2,
            random_state=None
        )

        # 最终采样的函数
        optimizer_0.maximize(
            n_iter=nt,
            acq=self.utility,  # ud，acq，
            x_train=None,  # 初始数据
            SK=knowledge,  # SK函数的参数必须跟f的参数一致
            punish=SK_punishment
        )

        # 输出------------------------------------------------------------------------------
        a_0 = np.array([list(optimizer_0.space.params[:, 0])]).T

        a_0 = a_0[5:, :]

        ma_0 = np.array([list(optimizer_0.space.params[:, 1])]).T

        ma_0 = ma_0[5:, :]

        result = np.concatenate((a_0, ma_0), axis=1)

        return result

    def score(self, num=100):
        """
        测试当前高斯过程的准确度
        :param num: 生成测试点的数量
        :return: 当前高斯过程的准确度
        """

        xlimts = {"x1": [0, 10], "x2": [0, 20]}
        s = RandomSampling(xlimts)
        x = s.sample(num)
        y, cov = self.gp.predict(x, return_cov=True)
        y = np.squeeze(y)
        cov = np.squeeze(cov)
        cov = cov.flatten()
        cov_mean = cov.mean()

        return cov_mean


if __name__ == "__main__":
    dataPath = r"C:\data\动态采样单目标.csv"
    knowPath = r'C:\data\新空间型知识.txt'

    # 获取数据
    d = CsvData(dataPath)
    dataSet = d.read()
    # print(dataSet)

    # 获取知识
    s = SpaceKnowledge(knowPath)
    space = s.readKnowledge()
    # print(space)

    xlimts = {"攻角": [0, 10], "法向力": [0, 8]}

    dy = DynamicSampling(xlimts)

    dy.utility = "ud"  # 设置效用函数，可选项有“ud”, “ei”, “ucb”
    dy.proportion = 0.8  # 设置训练集和训练集的比例

    dy.setKnowledge(space)
    dy.setData(dataSet)

    points = dy.sample(20, tablePath=r"C:\data\动态采样表示例.csv")
    score = dy.score()
