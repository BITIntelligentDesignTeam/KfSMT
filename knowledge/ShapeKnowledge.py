from knowledge.MappingBase import MappingBase
import numpy as np
from hausdorff import hausdorff_distance
import random
from deap import base, creator
from deap import tools
from abc import ABCMeta, abstractmethod
import os
from scipy.special import comb
import os
from sklearn.preprocessing import MinMaxScaler


def get_path(folderpath):
    """
    该函数的目的是，从一个文件夹中获取所有文件的名称
    :param folderpath: 文件夹的路径
    :return: 该文件夹中，所有文件d的路径组成的列表
    """
    allpath = []
    dirs = os.listdir(folderpath)
    for a in dirs:
        if os.path.isfile(folderpath + "/" + a):
            a = folderpath + '\\' + a
            allpath.append(a)

    return allpath


def sort(M, i):
    '''
    M:矩阵
    i:列数
        0：第一列
        -1：最后一列
    '''
    idx = np.lexsort([M[:, i]])
    return M[idx, :]


class ShapeKnowledge(MappingBase):

    # 形状型知识

    def __init__(self, path):

        super(ShapeKnowledge, self).__init__(path)

        self.type = '形状型'
        self.input_range = []
        self.output_range = []

    def readKnowledge(self):

        self.input_range = []
        self.output_range = []

        super(ShapeKnowledge, self).readKnowledge()

        # 读取变量范围
        ranges = self.root.getElementsByTagName('points')
        range_ = ranges[0]
        input_min = range_.getAttribute("minX")
        input_max = range_.getAttribute("maxX")
        output_min = range_.getAttribute("minY")
        output_max = range_.getAttribute("maxY")
        input_range = [[float(input_min), float(input_max)]]
        output_range = [[float(output_min), float(output_max)]]

        self.input_range = input_range
        self.knowledge['input_range'] = self.input_range
        self.output_range = output_range
        self.knowledge['output_range'] = self.output_range

        # 读取贝塞尔曲线控制点
        point = self.root.getElementsByTagName('point')

        for i in range(len(point)):
            point_i = point[i]
            points = []
            for i in point_i.firstChild.data.split(','):
                points.append(float(i))

            self.mapping_relation.append(points)

        self.knowledge['mapping_relation'] = self.mapping_relation

        return self.knowledge

    def writeKnowledge(self,
                       input_type: [],
                       output_type: [],
                       input_range: [],
                       output_range: [],
                       mapping_relation: [],
                       convar=[]):

        self.input_type = input_type
        self.output_type = output_type
        self.input_range = input_range
        self.output_range = output_range
        self.mapping_relation = mapping_relation
        self.convar = convar

        super(ShapeKnowledge, self).writeKnowledge()

        nodePoints = self.doc.createElement('points')
        nodePoints.setAttribute('minX', str(self.input_range[0][0]))
        nodePoints.setAttribute('maxX', str(self.input_range[0][1]))
        nodePoints.setAttribute('minY', str(self.output_range[0][0]))
        nodePoints.setAttribute('maxY', str(self.output_range[0][1]))

        for i in self.mapping_relation:
            nodePoint = self.doc.createElement('point')
            nodePoint.appendChild(self.doc.createTextNode(str(i[0]) + ',' + str(i[1])))
            nodePoints.appendChild(nodePoint)

        self.root.appendChild(nodePoints)

        try:
            fp = open(self.path, 'x')
            self.doc.writexml(fp, indent='\t', addindent='\t', newl='\n', encoding="utf-8")
        except FileExistsError:
            print('该知识路径已被创建过,请勿重复创建！')
        else:
            print(str(self.path) + '创建成功！')

    def visualKnowledge(self):

        print("_" * 75)
        print('知识名称:' + self.path)
        print('知识类型:' + self.type)
        print('变量:' + self.input_type[0])
        print('变量范围:' + str(self.input_range[0]))
        print('性能:' + self.output_type[0])
        print('性能范围:' + str(self.output_range[0]))

        num = 1
        for i in self.mapping_relation:
            print('贝塞尔曲线控制点' + str(num) + ':' + str(i))
            num += 1

        super(ShapeKnowledge, self).visualKnowledge()

    def bezierPoints(self, num=100):
        """
        通过贝塞尔曲线拟合的办法，将形状型知识的控制点拟合成曲线，输出拟合曲线上的点
        :param num: 采点的数量
        :return: dataPoints[0]:贝塞尔拟合曲线点的横坐标
                 dataPoints[1]:贝塞尔拟合曲线点的纵坐标
        """

        # 只有形状型知识才能使用该方法

        controlPoints = []
        controlPoints_x = []
        controlPoints_y = []

        for i in self.mapping_relation:
            controlPoints_x.append(i[0])
            controlPoints_y.append(i[1])

        controlPoints.append(controlPoints_x)
        controlPoints.append(controlPoints_y)

        t = np.linspace(0, 1, num)  # t 范围0到1
        le = len(controlPoints[0]) - 1
        le_1 = 0
        b_x, b_y = 0, 0
        for x in controlPoints[0]:
            b_x = b_x + x * (t ** le_1) * ((1 - t) ** le) * comb(len(controlPoints[0]) - 1, le_1)  # comb 组合，perm 排列
            le = le - 1
            le_1 = le_1 + 1
        le = len(controlPoints[0]) - 1
        le_1 = 0
        for y in controlPoints[1]:
            b_y = b_y + y * (t ** le_1) * ((1 - t) ** le) * comb(len(controlPoints[0]) - 1, le_1)
            le = le - 1
            le_1 = le_1 + 1

        dataPoints = np.array([b_x, b_y])

        return dataPoints

    def __bezierFirstGradient(self, inputs, num=100):

        t = np.linspace(0, 1, num)  # t 范围0到1
        n = len(inputs) - 1
        k = 0
        y = 0
        for x in inputs:
            if k == 0:
                y = y - inputs[0] * (1 - t) ** (n - 1) * n
                k += 1
            elif k == (len(inputs) - 1):
                y = y + inputs[-1] * (t ** (k - 1)) * k
                k += 1
            else:
                y = y + x * comb(n, k) * (
                        -((n - k) * (t ** k) * ((1 - t) ** (n - k - 1))) + (1 - t) ** (n - k) * k * t ** (
                        k - 1))  # comb 组合
                k += 1

        return y

    def bezierFirstGradient(self, num=100):
        """
        计算贝塞尔曲线的梯度
        :param num: 采集点的数量
        :return: gradientPoints[0]:贝塞尔拟合曲线点的横坐标
                 gradientPoints[1]:贝塞尔曲线的梯度
        """

        dataPoints = self.bezierPoints(num)
        bx = self.__bezierFirstGradient(dataPoints[0], num)
        by = self.__bezierFirstGradient(dataPoints[1], num)
        t = len(bx)
        by_bx = np.zeros(len(bx))
        for i in range(t):
            by_bx[i] = by[i] / bx[i]
        #     np.save('.\\data\\1_gradient', out)

        gradientPoints = np.array([dataPoints[0], by_bx])

        return gradientPoints


def bezierPoints(points, num=100):
    """
    通过贝塞尔曲线拟合的办法，将形状型知识的控制点拟合成曲线，输出拟合曲线上的点
    :param num: 采点的数量
    :return: dataPoints[0]:贝塞尔拟合曲线点的横坐标
             dataPoints[1]:贝塞尔拟合曲线点的纵坐标
    """

    controlPoints = []
    controlPoints_x = []
    controlPoints_y = []

    for i in points:
        controlPoints_x.append(i[0])
        controlPoints_y.append(i[1])

    controlPoints.append(controlPoints_x)
    controlPoints.append(controlPoints_y)

    t = np.linspace(0, 1, num)  # t 范围0到1
    le = len(controlPoints[0]) - 1
    le_1 = 0
    b_x, b_y = 0, 0
    for x in controlPoints[0]:
        b_x = b_x + x * (t ** le_1) * ((1 - t) ** le) * comb(len(controlPoints[0]) - 1, le_1)  # comb 组合，perm 排列
        le = le - 1
        le_1 = le_1 + 1
    le = len(controlPoints[0]) - 1
    le_1 = 0
    for y in controlPoints[1]:
        b_y = b_y + y * (t ** le_1) * ((1 - t) ** le) * comb(len(controlPoints[0]) - 1, le_1)
        le = le - 1
        le_1 = le_1 + 1

    dataPoints = np.array([b_x, b_y])

    return dataPoints


import copy
import matplotlib as mpl
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
from scipy.special import comb
from sklearn.preprocessing import MinMaxScaler
import math
import matplotlib.patches as mpatches

mpl.rcParams['font.family'] = ['sans-serif']
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False


class KnowledgeSection(object, metaclass=ABCMeta):

    def __init__(self, *args, knowledgelist=[], folder=[]):
        """
        获取知识路径
        :param knowledgelist: 含有知识路径的列表，list
        :param folder: 含有知识的文件夹列表，list
        :param *args:  知识的路径，str
        """

        self.path = []  # 所有知识的路径列表

        for i in args:
            self.path.append(i)

        for i in knowledgelist:
            self.path.append(i)

        for i in folder:
            allpath = get_path(i)
            for j in allpath:
                self.path.append(j)

        self.num = len(self.path)
        self.passlist = []  # 筛选通过的知识的路径列表
        self.remain = []  # 筛选未通过的知识的路径列表
        self.initialize()

    def initialize(self):
        pass

    def select(self, printPicture=True, savePath=None):
        knowPass = self._select()
        self.remain = [i for i in self.path if i not in self.passlist]

        for i in self.remain:
            k = ShapeKnowledge(i)
            know_dic = k.readKnowledge()
            points = k.bezierPoints(num=1000)
            x, y = points[0], points[1]
            plt.plot(x, y, ls="-", color="r", marker=",", lw=1)

        for i in self.passlist:
            k = ShapeKnowledge(i)
            know_dic = k.readKnowledge()
            points = k.bezierPoints(num=1000)
            x, y = points[0], points[1]
            plt.plot(x, y, ls="-", color="g", marker=",", lw=1)

        labels = ['筛选未通过', '筛选通过']
        color = ['red', 'green']

        patches = [mpatches.Patch(color=color[i], label="{:s}".format(labels[i])) for i in range(len(color))]
        ax = plt.gca()
        ax.legend(handles=patches, ncol=8, fontsize=14)  # 生成legend

        k = ShapeKnowledge(self.path[0])
        know_dic = k.readKnowledge()
        plt.xlabel(know_dic['input_type'][0], fontsize=14)
        plt.ylabel(know_dic['output_type'][0], fontsize=14)
        plt.rcParams['font.sans-serif'] = ['SimHei']

        if savePath is not None:
            plt.savefig(savePath)
        if printPicture:
            plt.show()

        return knowPass

    @abstractmethod
    def _select(self):
        pass


class GradientSelection(KnowledgeSection):
    """
    基于梯度一致性的形状型知识筛选
    """

    def __init__(self, xt, yt, *args, knowledgelist=[], folder=[]):
        """
        获取知识路径
        :param folder: 含有知识的文件夹列表，list
        :param *args:  知识的路径，str
        """
        super().__init__(*args, knowledgelist=knowledgelist, folder=folder)
        self.xt = xt
        self.yt = yt
        self.__creatGpr(xt, yt)

    def __creatGpr(self, xt, yt):
        """
        创建高斯过程
        :param x_t: 用于训练的输入数据
        :param y_t: 用于训练的输出数据

        """
        n = len(xt[0])
        length_scale = []
        for i in range(n):
            length_scale.append(1)

        self.gpr = GaussianProcessRegressor(
            kernel=RBF(length_scale=length_scale),  # 数据维度
            alpha=1e-3,
            normalize_y=True,
            n_restarts_optimizer=5,
        )
        self.gpr.fit(xt, yt)

    def __mean1Gradient(self, x1):
        # 单点
        '''
        针对均值函数求导
        Args:
            x1: ndArray，  1个点 (1 x d).导数预测点

        返回:
            均值矩阵 (d x 1).
        '''

        K_XX = self.gpr.kernel_.__call__(self.xt, self.xt)
        a = np.dot(np.linalg.pinv(K_XX), self.yt)

        Xt = []
        for i in range(self.xt.shape[0]):
            Xt.append(np.squeeze(x1 - self.xt[i]))
        Xt = np.array(Xt).T

        K_x1X = -self.gpr.kernel_.__call__(x1, self.xt)
        scale = np.array([self.gpr.kernel_.length_scale]).T
        l = ((scale ** 2) * np.eye(len(scale)))

        grad = np.dot(np.linalg.inv(l), Xt)
        grad = np.dot(grad, K_x1X.T * a)

        return grad

    def __derKnow(self, x_s):
        """
        args:
            x_s: 测试数据集
        return：
            grad_gp: 返回梯度
        """
        grad_gp = []

        for i in range(len(x_s)):
            # 均值
            grad = self.__mean1Gradient(x_s[i])
            grad_gp.append(grad[0, 0])
        return grad_gp

    def __checkGra(self, path, grad_gp):
        """
        args:
            path: 知识路径
        return:
            symbol_gra:符号一致性比例
            mape_gra:  梯度误差值
        """
        gradient = np.squeeze(grad_gp)
        GP_gradient_pre = gradient

        know = ShapeKnowledge(path)
        know.readKnowledge()
        # 贝塞尔拟合点
        x1_check, y1_check = know.bezierPoints()
        # 一阶导数点
        gradient = know.bezierFirstGradient()
        args2_check = np.row_stack((x1_check, gradient[1]))

        ## 导数符号
        # 导数符号判断
        k = [i for i in (GP_gradient_pre * args2_check[1]) if i > 0]
        symbol_gra = len(k) / len(args2_check[1])

        ## 导数误差值:
        mape_gra = np.sqrt(sum((args2_check[1] - GP_gradient_pre) ** 2) / len(GP_gradient_pre))  # rmse 均方根误差
        #     mape_gra=np.mean(abs((args2_check[1]-GP_gradient_pre)/(GP_gradient_pre))) # 平均相对百分比误差
        #     mape_gra=MAPE_pre(args2_check[1],GP_gradient_pre,p)  #平均相对百分比误差——pre
        #     mape_gra=np.mean(abs(args2_check[1]-GP_gradient_pre))  # 平均误差

        return symbol_gra, mape_gra

    def __dataAndKnow(self):

        res_combine = 1 / self.res[:, 0] * (self.res[:, 1])
        sort_know = sort(np.column_stack((res_combine.T, np.linspace(1, self.num, self.num))), 0)

        # 归一化距离
        mm = MinMaxScaler()
        data = mm.fit_transform(sort_know[:, 0].reshape(-1, 1))

        res3 = []
        res2 = []

        res1_select = []
        res2_select = []
        res3_select = []

        std_3 = 3 * np.std(data)
        std_2 = 2 * np.std(data)

        # 判断
        for i in range(len(data)):
            if data[i] > std_2:
                if data[i] > std_3:
                    res3.append(int(sort_know[i, 1]))
                    res2.append(int(sort_know[i, 1]))
                else:
                    res2.append(int(sort_know[i, 1]))
            if data[i] < np.std(data):
                res1_select.append(int(sort_know[i, 1]))
            if data[i] < std_2:
                res2_select.append(int(sort_know[i, 1]))
            if data[i] < std_3:
                res3_select.append(int(sort_know[i, 1]))

        ## 基于1sigma经验筛选
        # 提取索引
        self.passlist = []
        for i in res1_select:
            self.passlist.append(self.path[i - 1])

    def _select(self):
        """
        API函数，用于筛选知识
        :return: 筛选过后的知识路径组成的列表
        """

        # 存放一致性指标结果
        self.res = np.zeros((self.num, 2))

        # 存放拟合曲线信息
        x_valadation_c_ = np.zeros((100, 2))
        x_valadation_c_[:, -1] = np.ones(100) * 0.5

        num = 0
        all_num = len(self.path)
        print("*" * 35 + "开始筛选" + "*" * 35)
        for i in self.path:
            # ------基于贝塞尔公式数据处理------
            ## 贝塞尔拟合点
            know = ShapeKnowledge(i)
            know.readKnowledge()
            x1, y1 = know.bezierPoints()

            ## 经验x1切片放入
            x_valadation_c_[:, 0] = x1
            x_s = copy.deepcopy(x_valadation_c_)

            ## GP 求导
            gradGP = self.__derKnow(x_s)

            ## 误差检验（值+符号）
            symbol_gra, mape_gra = self.__checkGra(i, gradGP)

            ## 梯度值误差，梯度符号一致性误差
            self.res[num, :] = [symbol_gra, mape_gra]

            num += 1
            print(str(100 * num / all_num) + "%")

        # ------基于小样本与知识的一致性度量筛选------
        print("*" * 35 + "筛选完成！" + "*" * 35)
        self.__dataAndKnow()

        knowPass = []
        for i in self.passlist:
            know = ShapeKnowledge(i)
            knowPass.append(know.readKnowledge())

        return knowPass


class HausdorffSelection(KnowledgeSection):
    """
    基于豪斯多夫距离晒选指标的知识筛选方法
    """

    # 单条知识的豪斯多夫距离
    def __hausdorff_single(self, M):
        '''
        M：字典
        '''
        M_back = copy.deepcopy(M)
        k = len(M)
        for i in range(k):
            mid = []
            for j in range(k):
                mid.append(hausdorff_distance(M['{}'.format(i)], M['{}'.format(j)], distance='euclidean'))
            M_back['{}'.format(i)] = np.sum(mid) / (k - 1)
        return M_back

    # 个体间豪斯多夫距离均值排序
    def __mean_sort(self, M):
        d = self.__hausdorff_single(M)
        # 用以存返回的最后结果
        back_M = np.zeros((len(M), 2))
        w = 0
        for k in sorted(d, key=d.__getitem__):
            back_M[w, 0] = int(k)
            back_M[w, 1] = d[k]
            w += 1
        return back_M

    # 基于豪斯多夫距离经验筛选
    def _select(self):
        '''
            args：
                res:      度量结果
                K  :      各知识拟合曲线
                y_predict_DataAndKnow:    预测均值
                err_DataAndKnow：预测方差
                x_tru,y_tru: 真值

        '''

        M_data = {}
        for i in range(self.num):
            know = ShapeKnowledge(self.path[i])
            know.readKnowledge()
            M_data[str(i)] = np.array(know.bezierPoints())

        #  筛选
        sort_know = sort(self.__mean_sort(M_data), -1)

        # 归一化距离
        mm = MinMaxScaler()
        data = mm.fit_transform(sort_know[:, -1].reshape(-1, 1))

        res1 = []
        res3 = []
        res2 = []
        res2_select = []
        res3_select = []

        std_3 = 3 * np.std(data)
        std_2 = 2 * np.std(data)

        # 判断
        for i in range(len(data)):
            if data[i] > std_2:
                if data[i] > std_3:
                    res3.append(int(sort_know[i, 0]))
                    res2.append(int(sort_know[i, 0]))
                else:
                    res2.append(int(sort_know[i, 0]))
            if data[i] < np.std(data):
                res1.append(int(sort_know[i, 0]))
            if data[i] < std_2:
                res2_select.append(int(sort_know[i, 0]))
            if data[i] < std_3:
                res3_select.append(int(sort_know[i, 0]))

        ## 基于1sigma经验筛选
        # 提取索引

        for i in res1:
            self.passlist.append(self.path[i - 1])

        knowPass = []
        for i in self.passlist:
            know = ShapeKnowledge(i)
            knowPass.append(know.readKnowledge())

        return knowPass


class KnowlwdgeFusion(object, metaclass=ABCMeta):

    def __init__(self, *args, knowledgelist=[], folder=[]):
        """
        获取知识路径
        :param knowledgelist: 含有知识路径的列表，list
        :param folder: 含有知识的文件夹列表，list
        :param *args:  知识的路径，str
        """
        self.path = []  # 所有知识的路径列表

        for i in args:
            self.path.append(i)

        for i in knowledgelist:
            self.path.append(i)

        for i in folder:
            allpath = get_path(i)
            for j in allpath:
                self.path.append(j)

        self.num = len(self.path)

    def fuse(self,printPicture=True, savePath=None):
        know_dic = self._fuse()

        for i in self.path:
            k = ShapeKnowledge(i)
            know_dic = k.readKnowledge()
            points = k.bezierPoints(num=1000)
            x, y = points[0], points[1]
            plt.plot(x, y, ls="-", color="r", marker=",", lw=1)

        points = bezierPoints(know_dic["mapping_relation"], num=1000)
        x, y = points[0], points[1]
        plt.plot(x, y, ls="-", color="g", marker=",", lw=4)

        labels = ['融合前', '融合后']
        color = ['red', 'green']

        patches = [mpatches.Patch(color=color[i], label="{:s}".format(labels[i])) for i in range(len(color))]
        ax = plt.gca()
        ax.legend(handles=patches, ncol=8, fontsize=14)  # 生成legend

        k = ShapeKnowledge(self.path[0])
        know_dic = k.readKnowledge()
        plt.xlabel(know_dic['input_type'][0], fontsize=14)
        plt.ylabel(know_dic['output_type'][0], fontsize=14)
        plt.rcParams['font.sans-serif'] = ['SimHei']

        if savePath is not None:
            plt.savefig(savePath)
        if printPicture:
            plt.show()

        return know_dic

    @abstractmethod
    def _fuse(self):
        pass


class FermatPointsFusion(KnowlwdgeFusion):
    """
    利用费马点进行知识的融合
    """

    def __fermat(self, points):
        """
        argmas： points:度量完毕的经验数据切片
        :return: 费马点
        """
        n = points.shape[1]
        # x坐标
        a1 = points[0]
        # y坐标&梯度坐标
        a2 = points[1]
        x = sum(a1) / n
        y = sum(a2) / n
        while True:
            xfenzi = 0
            xfenmu = 0
            yfenzi = 0
            yfenmu = 0
            for i in range(n):
                g = math.sqrt((x - a1[i]) ** 2 + (y - a2[i]) ** 2)
                if g == 0:
                    g = 1e-5
                xfenzi = xfenzi + a1[i] / g
                xfenmu = xfenmu + 1 / g
                yfenzi = yfenzi + a2[i] / g
                yfenmu = yfenmu + 1 / g
            xn = xfenzi / xfenmu
            yn = yfenzi / yfenmu
            if abs(xn - x) < 0.001 and abs(yn - y) < 0.001:
                break
            else:
                x = xn
                y = yn
        return [x, y]

    def _fuse(self):

        M = {}

        for i in range(self.num):
            know = ShapeKnowledge(self.path[i])
            know.readKnowledge()
            M[str(i)] = np.array(know.bezierPoints())

        # 融合策略3 -------费马点拟合-------
        # 数据按观测点切片
        w = M['1'].shape[1]
        gra_fermat = np.zeros((2, w))

        for i in range(w):
            # 按w的切片取出所有的点
            point_mid = np.zeros((2, self.num))

            # 取idx索引
            for j in range(self.num):
                point_mid[:, j] = M[str(j)][:, i]
            gra_fermat[:, i] = self.__fermat(point_mid)

        # 可能需要修改
        know = ShapeKnowledge(self.path[5])
        know_dict = know.readKnowledge()
        contolPoints = []
        for i in range(len(gra_fermat[0])):
            point = []
            point.append(gra_fermat[0][i])
            point.append(gra_fermat[1][i])
            contolPoints.append(point)
        know_dict["mapping_relation"] = contolPoints

        return know_dict


def bezier(args):
    '''
    x:控制点 2*N
    '''
    t = np.linspace(0, 1, 100)  # t 范围0到1
    le = len(args[0]) - 1
    le_1 = 0
    b_x, b_y = 0, 0
    for x in args[0]:
        b_x = b_x + x * (t ** le_1) * ((1 - t) ** le) * comb(len(args[0]) - 1, le_1)  # comb 组合，perm 排列
        le = le - 1
        le_1 = le_1 + 1
    le = len(args[0]) - 1
    le_1 = 0
    for y in args[1]:
        b_y = b_y + y * (t ** le_1) * ((1 - t) ** le) * comb(len(args[0]) - 1, le_1)
        le = le - 1
        le_1 = le_1 + 1
    return b_x, b_y


class GAFusion(KnowlwdgeFusion):
    """
    利用遗传算法进行知识融合
    """

    # 计算个体豪斯多夫距离均值
    def __hausdorff_mean(individual, M):
        '''
        individual:个体
        M：字典
        '''
        M_back = copy.deepcopy(M)
        k = len(M)
        mid = []
        for i in range(k):
            mid.append(hausdorff_distance(individual, M['{}'.format(i)], distance='euclidean'))
        return np.mean(mid)

    # 个体的DNA转码
    def __translate_ind(individual):
        '''
            individual:个体
        '''
        d = int(0.5 * len(individual))
        x_final = individual[:d]
        y_final = individual[d:]

        args_final = np.row_stack((x_final, y_final))

        return args_final

    # 评价函数
    def __evaluate(individual):
        '''

        :param individual:  个体DNA
        :param points:   真值2x100 全局定义

        :return: 豪斯多夫距离均值
        '''
        global points

        # DNA转换
        args = GAFusion.__translate_ind(individual)

        # 贝塞尔转换
        x, y = bezier(args)
        args_tru = np.row_stack((x, y))

        # 计算豪斯多夫距离
        b = GAFusion.__hausdorff_mean(args_tru, points)

        # 这个，要保留
        return b,

    def __Brain_random(self):
        a = random.uniform(-10, 10)
        return a

    def __sort_pop(pop):
        '''
        :param pop:  种群

        :return: 返回将x做一个升序排序的算法，y不变

        '''
        for i in pop:
            i[:10] = np.sort(i[:10])
        return pop

    def __GA(self, M):
        # 定义参数
        # 个体的大小，也就是控制点

        #     IND_SIZE = 20 # 车架
        IND_SIZE = 20  # Brain测试函数

        # weight为-1表示越小越好
        creator.create("FitnessMulti", base.Fitness, weights=(-1,))
        creator.create("Individual", list, fitness=creator.FitnessMulti)
        toolbox = base.Toolbox()
        # fun_random 是我自己定义的个体范围定义方法，范围为-10，10
        toolbox.register("attribute", GAFusion.__Brain_random)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=IND_SIZE)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
        toolbox.register("select", tools.selNSGA2)
        toolbox.register("evaluate", GAFusion.__evaluate)

        # 种群个数
        NPOP = 200
        # CXPB交叉概率 MU变异概率
        CXPB, Mu = 0.6, 0.4

        # 真值导入

        global points

        points = copy.deepcopy(M)

        pareto_front = tools.ParetoFront()
        logbook = tools.Logbook()
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        # 迭代代数
        NGEN = 1000

        # NPOP表示种群中的个体数
        # gen:迭代嗲书  eval：表示评估值 std：表示当前代的标准差， min：最小值
        logbook.header = "gen", "evals", "std", "min", "avg", "max"
        pop = toolbox.population(n=NPOP)
        pop = GAFusion.__sort_pop(pop)

        # Evaluate the entire population

        # 给每个ind匹配相应的fit
        fitnesses = map(toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        print("*" * 35 + "开始融合" + "*" * 35)

        # NGEN 表示代数
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
            invalid_ind = GAFusion.__sort_pop(invalid_ind)
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # 选择较优后代
            pop = toolbox.select(pop + offspring, len(pop))

            record = stats.compile(pop)
            logbook.record(gen=i, evals=len(invalid_ind), **record)
            pareto_front.update(pop)

            # 显示训练进度
            print('{:.2%}'.format((i + 1) / NGEN))

        print("*" * 35 + "融合完成" + "*" * 35)
        plt_avg = logbook.select("avg")

        # best_ind
        # 最小的值挑出来？
        ccc = []
        for i in range(len(pop)):
            aaa = pop[i]
            bbb = aaa.fitness.values
            ccc.append(bbb)
        ccc = np.array(ccc)
        pop_list = np.lexsort([-1 * ccc[:, 0]])
        best_ind = pop[pop_list[99]]

        return best_ind

    def _fuse(self):

        M_data = {}
        for i in range(self.num):
            know = ShapeKnowledge(self.path[i])
            know.readKnowledge()
            M_data[str(i)] = np.array(know.bezierPoints())

        best_ind = self.__GA(M_data)
        # DNA转换
        args_final = GAFusion.__translate_ind(best_ind)

        # 可能需要修改
        know = ShapeKnowledge(self.path[5])
        know_dict = know.readKnowledge()
        contolPoints = []
        for i in range(len(args_final[0])):
            point = []
            point.append(args_final[0][i])
            point.append(args_final[1][i])
            contolPoints.append(point)
        know_dict["mapping_relation"] = contolPoints

        return know_dict


def matyas(x1, x2):
    return 0.26 * (x1 ** 2 + x2 ** 2) - 0.48 * x1 * x2


import matplotlib.pyplot as plt

if __name__ == "__main__":
    know1 = ShapeKnowledge(path="C:\data\形状型知识3.xml")

    know1.writeKnowledge(input_type=['攻角'],
                         output_type=['法向力'],
                         input_range=[[3.0, 10.0]],
                         output_range=[[2.5, 6.0]],
                         mapping_relation=[[1.0, 2.0], [2.1, 3.5], [2.7, 3.8]],
                         convar=[{'convar_type': '马赫数', 'convar_RangeOrValue': 'value', 'convar_value': 4.0},
                                 {'convar_type': '雷诺数', 'convar_RangeOrValue': 'range', 'convar_range': [3.0, 5.0]}])

    know1.visualKnowledge()
    a = know1.readKnowledge()
    print(a)

    # know3 = ShapeKnowledge("C:\data\形状型知识3.xml")
    # c = know3.readKnowledge()
    # know3.visualKnowledge()
    # print(c)
    # a = know3.bezierPoints()
    # print(a)
    # gradient = know3.bezierFirstGradient()
    # print(gradient)

    # x1,x2均匀采样
    x1 = np.linspace(-10, 10, 15)
    x2 = np.linspace(-10, 10, 15)
    # meshgrid 扩展成矩阵
    x1v, x2v = np.meshgrid(x1, x2)
    # flatten 合并
    x_train = np.column_stack((x1v.flatten(), x2v.flatten()))

    y_train = np.array(matyas(x_train[:, 0], x_train[:, 1])) + np.random.randn(225) * 0.5
    y_train = y_train.reshape(-1, 1)

    # select = GradientSelection(xt=x_train, yt=y_train, folder=["C:\data\筛选测试"])
    # select = HausdorffSelection(folder=["C:\data\筛选测试"])
    # passlist = select.select(savePath="C:\data\筛选测试\select_pic.png")
    # print(passlist)

    mix = FermatPointsFusion(folder=["C:\data\筛选测试"])
    # mix = GAFusion(folder=["C:\data\筛选测试"])
    knowNew = mix.fuse(savePath=r"C:\data\fuse_pic.png")
    print(knowNew)

    # # 画图
    # plt.plot(points[0, :], points[1, :])
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.savefig("C:\data\mix.png")
    # plt.show()
    #
