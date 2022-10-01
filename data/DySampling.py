import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy import integrate
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sampling import SamplingBase, RandomSampling

from utils.DySampingSupport.data_read import data_process, object_label
from utils.DySampingSupport.space_knowledge import sortSpaceKnowledge
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
        pass


class DynamicSampling(KnowledgeSamplingBase):
    """
    动态采样
    """

    def _initialize(self):
        self.gpList = []  # 不同高斯过程组成的list
        self.uinity = "ud"
        self.proportion = 0.1

    def setData(self, dataSet):
        """
        设置动态采样需要的数据
        """
        self.x = dataSet["input"]
        self.y = dataSet["output"]
        self.title = dataSet["title"]

    def gp(self):
        pass

    def _setKnowledge(self):
        pass

    def _sample(self, nt):

        assert self.uinity in ["ud", "ei", "ucb"], "请选择正确的效用函数"

        # 声明一个全局高斯过程
        # 高斯过程核参数设定——各向异性
        length_scale = []
        for i in range(self.x.shape[1]):
            length_scale.append(1)

        gp_0 = GaussianProcessRegressor(
            kernel=Matern(nu=1.5, length_scale=length_scale),  # 全弹法向力使用nu=1.5的效果更好
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

            Param_Know = r'C:\Users\石磊\Desktop\馥琳姐代码\Source\空间型知识1.txt'
            knowledge_list = sortSpaceKnowledge(Param_Know)
            weight = 1

            for i in range(len(knowledge_list)):

                space = knowledge_list[i]

                if space[0] == 2:
                    if space[1] == ['攻角', '马赫数']:

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
        def SK_fa_punishment(x1, x2):
            """全弹法向力边界惩罚"""
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
            pbounds=self.xlimts,
            verbose=2,
            random_state=None
        )

        # 最终采样的函数
        optimizer_0.maximize(
            n_iter=nt,
            acq="ud",  # ud，acq，
            x_train=None,  # 初始数据
            SK=knowledge,  # SK函数的参数必须跟f的参数一致
            punish=SK_fa_punishment
        )

        # 输出------------------------------------------------------------------------------
        a_0 = np.array([list(optimizer_0.space.params[:, 0])]).T

        a_0 = a_0[5:, :]

        ma_0 = np.array([list(optimizer_0.space.params[:, 1])]).T

        ma_0 = ma_0[5:, :]

        phi_0 = np.full(a_0.shape, 0)

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
        y ,cov= self.gp.predict(x, return_cov=True)
        y = np.squeeze(y)
        cov = np.squeeze(cov)
        cov =cov.flatten()
        cov_mean = cov.mean()

        return cov_mean


class DynamicsamplingMulti(KnowledgeSamplingBase):
    """
    动态采样
    """

    def _initialize(self):
        self.gpList = []  # 不同高斯过程组成的list
        self.uinity = "ud"
        self.proportion = 0.1

    def setData(self, dataSet):
        """
        设置动态采样需要的数据
        """
        self.x = dataSet["input"]
        self.y = dataSet["output"]
        self.title = dataSet["title"]

    def gp(self):
        pass

    def _sample(self, nt):
        data1_path = Data_Param['data1']
        data2_path = Data_Param['data2']

        label1 = object_label(data1_path)
        if label1 == 'D_Cy1':
            Data_dcy0, Data_dcy45, Data_dcy90 = data_process(data1_path)
        elif label1 == 'T_Cy1':
            Data_tcy0, Data_tcy45, Data_tcy90 = data_process(data1_path)
        elif label1 == 'D_Xcp1':
            Data_dxcp0, Data_dxcp45, Data_dxcp90 = data_process(data1_path)
        elif label1 == 'T_CA1':
            Data_tca0, Data_tca45, Data_tca90 = data_process(data1_path)

        label2 = object_label(data2_path)
        if label2 == 'D_Cy1':
            Data_dcy0, Data_dcy45, Data_dcy90 = data_process(data2_path)
        elif label2 == 'T_Cy1':
            Data_tcy0, Data_tcy45, Data_tcy90 = data_process(data2_path)
        elif label2 == 'D_Xcp1':
            Data_dxcp0, Data_dxcp45, Data_dxcp90 = data_process(data2_path)
        elif label2 == 'T_CA1':
            Data_tca0, Data_tca45, Data_tca90 = data_process(data2_path)

        if (Data_dcy0 is not None) and (Data_dxcp0 is not None):
            x_0 = Data_dcy0[:, :-1]
            y_01 = Data_dcy0[:, -1]
            y_02 = Data_dxcp0[:, -1]

            x_45 = Data_dcy45[:, :-1]
            y_451 = Data_dcy45[:, -1]
            y_452 = Data_dxcp45[:, -1]

            x_90 = Data_dcy90[:, :-1]
            y_901 = Data_dcy90[:, -1]
            y_902 = Data_dxcp90[:, -1]

            # 目标函数
            global gp_fa_0, gp_fa_45, gp_fa_90
            global gp_yaxin_0, gp_yaxin_45, gp_yaxin_90

            length_scale = []
            for i in range(x_0.shape[1]):
                length_scale.append(1)

            gp_fa_0 = GaussianProcessRegressor(
                kernel=Matern(nu=1.5, length_scale=length_scale),  # 全弹法向力使用nu=1.5的效果更好
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=5,
            )

            gp_fa_45 = GaussianProcessRegressor(
                kernel=Matern(nu=1.5, length_scale=length_scale),  # 全弹法向力使用nu=1.5的效果更好
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=5,
            )

            gp_fa_90 = GaussianProcessRegressor(
                kernel=Matern(nu=1.5, length_scale=length_scale),  # 全弹法向力使用nu=1.5的效果更好
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=5,
            )

            gp_yaxin_0 = GaussianProcessRegressor(
                kernel=Matern(nu=1.5, length_scale=length_scale),  # 全弹法向力使用nu=1.5的效果更好
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=5,
            )

            gp_yaxin_45 = GaussianProcessRegressor(
                kernel=Matern(nu=1.5, length_scale=length_scale),  # 全弹法向力使用nu=1.5的效果更好
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=5,
            )

            gp_yaxin_90 = GaussianProcessRegressor(
                kernel=Matern(nu=1.5, length_scale=length_scale),  # 全弹法向力使用nu=1.5的效果更好
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=5,
            )

            gp_fa_0.fit(x_0, y_01)
            gp_yaxin_0.fit(x_0, y_02)

            gp_fa_45.fit(x_45, y_451)
            gp_yaxin_45.fit(x_45, y_452)

            gp_fa_90.fit(x_90, y_901)
            gp_yaxin_90.fit(x_90, y_902)

            def obj_fa_0(x1, x2):
                """两个变量：贝叶斯优化动态采样的目标函数，输入x,返回y"""
                x = [[x1, x2]]
                y = gp_fa_0.predict(x)
                y = np.squeeze(y)
                return y

            def obj_yaxin_0(x1, x2):
                """两个变量：贝叶斯优化动态采样的目标函数，输入x,返回y"""
                x = [[x1, x2]]
                y = gp_yaxin_0.predict(x)
                y = np.squeeze(y)
                return y

            def obj_fa_45(x1, x2):
                """两个变量：贝叶斯优化动态采样的目标函数，输入x,返回y"""
                x = [[x1, x2]]
                y = gp_fa_45.predict(x)
                y = np.squeeze(y)
                return y

            def obj_yaxin_45(x1, x2):
                """两个变量：贝叶斯优化动态采样的目标函数，输入x,返回y"""
                x = [[x1, x2]]
                y = gp_yaxin_45.predict(x)
                y = np.squeeze(y)
                return y

            def obj_fa_90(x1, x2):
                """两个变量：贝叶斯优化动态采样的目标函数，输入x,返回y"""
                x = [[x1, x2]]
                y = gp_fa_90.predict(x)
                y = np.squeeze(y)
                return y

            def obj_yaxin_90(x1, x2):
                """两个变量：贝叶斯优化动态采样的目标函数，输入x,返回y"""
                x = [[x1, x2]]
                y = gp_yaxin_90.predict(x)
                y = np.squeeze(y)
                return y

            # 知识 ------------------------------------------------------

            know1 = Know_Param['know1']
            know2 = Know_Param['know2']
            # knowledge_list1, knowledge_list2 = sortSpaceKnowledge(r'C:\Users\DELL\Desktop\pythonProject\空间型知识1.txt',
            #                                                       r'C:\Users\DELL\Desktop\pythonProject\空间型知识2.txt')
            knowledge_list1, knowledge_list2 = sortSpaceKnowledge(know1, know2)

            def knowledge1(x1, x2):
                # knowledge_list = sortSpaceKnowledge(r'C:\Users\DELL\Desktop\pythonProject\空间型知识1.txt')
                weight = 1

                for i in range(len(knowledge_list1)):

                    space = knowledge_list1[i]

                    if space[0] == 2:
                        if space[1] == ['攻角', '马赫数']:

                            if (space[2][0][0] <= x1 < space[2][0][1]) and (space[2][1][0] <= x2 < space[2][1][1]):
                                if weight == 1:
                                    weight = space[3]
                            else:
                                if weight != 1:
                                    continue
                                else:
                                    weight = 1

                return weight

            def knowledge2(x1, x2):
                # knowledge_list = sortSpaceKnowledge(r'C:\Users\DELL\Desktop\pythonProject\空间型知识1.txt')
                weight = 1

                for i in range(len(knowledge_list2)):

                    space = knowledge_list2[i]

                    if space[0] == 2:
                        if space[1] == ['攻角', '马赫数']:

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
            def SK_fa_punishment(x1, x2):
                """全弹法向力边界惩罚"""
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

            pbounds = In_Variable

            # main-------------------------------------------------------------------------------
            def num_process(num):
                n = int(num / 3)
                if num % 3 == 1:
                    n1 = n + 1
                    n2 = n
                    n3 = n
                elif num % 3 == 2:
                    n1 = n + 1
                    n2 = n + 1
                    n3 = n
                elif num % 3 == 0:
                    n1 = n
                    n2 = n
                    n3 = n
                return n1, n2, n3

            n1, n2, n3 = num_process(Batch_Size)
            optimizer_0 = BO(
                f1=obj_fa_0,
                f2=obj_yaxin_0,
                pbounds=pbounds,
                verbose=2,
                random_state=1
            )

            optimizer_0.maximize(
                n_iter=n1 + 6,
                acq="ud_multiobj",
                punish=SK_fa_punishment,
                SK1=knowledge1,
                SK2=knowledge2,
                x_train=None  # 初始数据
                # SK函数的参数必须跟f的参数一致
            )
            optimizer_45 = BO(
                f1=obj_fa_45,
                f2=obj_yaxin_45,
                pbounds=pbounds,
                verbose=2,
                random_state=1
            )

            optimizer_45.maximize(
                n_iter=n2 + 6,
                acq="ud_multiobj",
                punish=SK_fa_punishment,
                SK1=knowledge1,
                SK2=knowledge2,
                x_train=None  # 初始数据
                # SK函数的参数必须跟f的参数一致
            )

            optimizer_90 = BO(
                f1=obj_fa_90,
                f2=obj_yaxin_90,
                pbounds=pbounds,
                verbose=2,
                random_state=1
            )

            optimizer_90.maximize(
                n_iter=n3 + 6,
                acq="ud_multiobj",
                punish=SK_fa_punishment,
                SK1=knowledge1,
                SK2=knowledge2,
                x_train=None  # 初始数据
                # SK函数的参数必须跟f的参数一致
            )
            # 输出------------------------------------------------------------------------------
            a_0 = np.array([list(optimizer_0.space.params[:, 0])]).T
            a_45 = np.array([list(optimizer_45.space.params[:, 0])]).T
            a_90 = np.array([list(optimizer_90.space.params[:, 0])]).T

            a_0 = a_0[6:, :]
            a_45 = a_45[6:, :]
            a_90 = a_90[6:, :]

            ma_0 = np.array([list(optimizer_0.space.params[:, 1])]).T
            ma_45 = np.array([list(optimizer_45.space.params[:, 1])]).T
            ma_90 = np.array([list(optimizer_90.space.params[:, 1])]).T

            ma_0 = ma_0[6:, :]
            ma_45 = ma_45[6:, :]
            ma_90 = ma_90[6:, :]

            phi_0 = np.full(a_0.shape, 0)
            phi_45 = np.full(a_45.shape, 45)
            phi_90 = np.full(a_90.shape, 90)

            a_all = np.concatenate((a_0, a_45, a_90), axis=0)
            ma_all = np.concatenate((ma_0, ma_45, ma_90), axis=0)
            phi_all = np.concatenate((phi_0, phi_45, phi_90), axis=0)

            result = np.concatenate((a_all, ma_all, phi_all), axis=1)

            """
            生成的“多目标测试”是本批次采样的结果，可以与追加到原excel中的数据互相验证
            """
            array2excel = {
                'a': list(result[:, 0]),
                'ma': list(result[:, 1]),
                'phi': list(result[:, 2])
            }
            df = pd.DataFrame(data=array2excel)  # 创建DataFrame
            df.to_excel('0228多目标测试.xls')  # 存表，去除原始索引列（0,1,2...）

            """
            此处的读和写是追加到原数据文件的操作
            """
            # 读————————————————————————
            # data1 = pd.read_excel(r'C:\Users\DELL\Desktop\Sample\全弹法向力系数1_3d.xls')
            # data2 = pd.read_excel(r'C:\Users\DELL\Desktop\Sample\全弹法向力系数1_3d.xls')
            data1_path = Data_Param['data1']
            data2_path = Data_Param['data2']

            data1 = pd.read_excel(data1_path)
            data2 = pd.read_excel(data2_path)
            n1 = data1.shape[0]
            n2 = data1.shape[0]

            for i in range(a_all.shape[0]):
                data1.loc[n1 + i] = [np.squeeze(phi_all[i]), np.squeeze(ma_all[i]), np.squeeze(a_all[i]), 1000]  # 追加行
                data2.loc[n2 + i] = [np.squeeze(phi_all[i]), np.squeeze(ma_all[i]), np.squeeze(a_all[i]), 1000]  # 追加行

            # 写—————————————————————————
            DataFrame(data1).to_excel(data1_path, sheet_name='Sheet1', index=False, header=True)
            DataFrame(data2).to_excel(data2_path, sheet_name='Sheet1', index=False, header=True)

            return result

        # Unfinished 此处未完成，是针对加入舵偏角的三个变量的舵偏法向力和舵偏轴向力，还没有写完
        elif Data_tcy0 != 0 and Data_tca0 != 0:
            x_01 = Data_tcy0[:, :-1]
            y_01 = Data_tcy0[:, -1]
            y_02 = Data_tca0[:, -1]

            x_451 = Data_tcy45[:, :-1]
            y_451 = Data_tcy45[:, -1]
            y_452 = Data_tca45[:, -1]

            x_901 = Data_tcy90[:, :-1]
            y_901 = Data_tcy90[:, -1]
            y_902 = Data_tca90[:, -1]

            return Data_tcy0, Data_tcy45, Data_tcy90, Data_tca0, Data_tca45, Data_tca90

        else:
            print("只能读取具有一致性的两个目标")

    def score(self):
        xlimts = {"x1": [0, 10], "x2": [0, 20]}
        s = RandomSampling(xlimts)
        x = s.sample()
        y, cov = self.gp.predict(x, return_cov=True)
        y = np.squeeze(y)
        cov = np.squeeze(cov)
        cov = cov.flatten()
        cov_mean = cov.mean()



if __name__ == "__main__":
    dataPath = r"C:\data\动态采样多目标.csv"
    knowPath = r'C:\Users\石磊\Desktop\馥琳姐代码\Source\空间型知识1.txt'

    # 获取数据
    d = CsvData(dataPath)
    dataSet = d.read()
    # print(dataSet)

    # 获取知识
    s = SpaceKnowledge(knowPath)
    space = s.readKnowledge()
    # print(space)

    xlimts = {"x1": [0, 10], "x2": [0, 8]}

    dy = DynamicSamplingMulti(xlimts)

    dy.uinity = "ei"          # 设置效用函数，可选项有“ud”, “ei”, “ucb”
    dy.proportion = 0.1       # 设置训练集和训练集的比例

    dy.setKnowledge(space)
    dy.setData(dataSet)

    points = dy.sample(20, tablePath=r"C:\data\动态采样表示例.csv")
    score = dy.score()
