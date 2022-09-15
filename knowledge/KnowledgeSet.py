from knowledge.AttributeKnowledge import AttributeKnowledge
from knowledge.ShapeKnowledge import ShapeKnowledge, GradientSelection, HausdorffSelection, FermatPointsFusion, GAFusion
from knowledge.MonotonicityKnowledge import MonotonicityKnowledge
import xml.dom.minidom
import os
import numpy as np


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


class KnowledgeSet(object):
    """
    外部想要对一个或者多个知识进行读取，查看，融合，筛选等操作的时候，调用此类即可，无需知道知识的具体类型
    """

    def __init__(self, *args, knowType=None, knowInput=None, knowOutput=None, folder=None, knowledgelist=None):
        """
        获取知识的类型
        :param *args : 知识路径
               folder : 装有知识的文件夹路径列表
        """

        # 先确定知识的类型
        if knowledgelist is None:
            knowledgelist = []
        if folder is None:
            folder = []

        self.path = []  # 知识的路径
        self.type = []  # 知识的种类
        self.num = 0  # 知识的总个数

        self.passlist = []  # 通过筛选的知识路径

        for i in folder:
            allpath = get_path(i)
            for j in allpath:
                self.path.append(j)

        for i in args:
            self.path.append(i)

        for i in knowledgelist:
            self.path.append(i)

        self.num = len(self.path)

        # 获取知识的类型信息
        for i in self.path:

            try:
                file_object = open(i, 'r+', encoding="gb2312")
                xmlfile = file_object.read()
                file_object.close()
            except UnicodeDecodeError:
                file_object = open(i, 'r+', encoding="utf-8")

                # 将"gb2312"格式转化为"utf-8"格式
                xmlfile = file_object.read()
                file_object.close()
                xmlfile = xmlfile.replace("utf-8", "gb2312")

            # 读取xml文件
            dom = xml.dom.minidom.parseString(xmlfile)
            root = dom.documentElement
            knowledgeType = root.getAttribute("infoType")

            self.type.append(knowledgeType)

        # 首先进行知识类型的筛选
        passlistType = []
        if knowType is not None:
            typePass = []
            for i in range(self.num):
                if self.type[i] in knowType:
                    passlistType.append(self.path[i])
                    typePass.append(self.type[i])
            self.type = typePass
            self.path = passlistType
            self.num = len(self.path)

        # 进行输入参数的筛选,这里的逻辑是给一个输入参数组成的集合，当该集合为实际知识中输入参数的子集时，判定为筛选通过
        passlistInput = []
        if knowInput is not None:
            know_pass = self.readKnowledge()
            typePass = []
            for i in range(self.num):
                inputKnow = know_pass[i]["input_type"]
                inputIntersection = [x for x in knowInput if x in inputKnow]
                if inputIntersection == knowInput:
                    passlistInput.append(self.path[i])
                    typePass.append(self.type[i])
            self.type = typePass
            self.path = passlistInput
            self.num = len(self.path)

        know_pass = self.readKnowledge()
        # 进行输出参数的筛选,这里的逻辑是给一个输入参数组成的集合，当该集合为实际知识中输入参数的子集时，判定为筛选通过
        passlistOutput = []
        if knowOutput is not None:
            typePass = []
            know_pass = self.readKnowledge()
            for i in range(self.num):
                outputKnow = know_pass[i]["output_type"]
                outputIntersection = [x for x in outputKnow if x in knowOutput]
                if outputIntersection == knowOutput:
                    passlistOutput.append(self.path[i])
                    typePass.append(self.type[i])
            self.type = typePass
            self.path = passlistOutput
            self.num = len(self.path)

    def readKnowledge(self):
        """
        该函数的主要功能是在不知道具体知识是什么类型的情况下，直接读取知识的信息
        :return: 整理好的知识字典，当输入的是单条知识时，返回表征该知识的字典；当输入的是多条知识时，返回多条知识对应的知识字典组成的元组
        """
        know = []
        knowledgeDic = {}
        for i in range(self.num):

            if self.type[i] == '单调型':
                k = MonotonicityKnowledge(self.path[i])
                knowledgeDic = k.readKnowledge()

            if self.type[i] == '属性型':
                k = AttributeKnowledge(self.path[i])
                knowledgeDic = k.readKnowledge()

            if self.type[i] == '形状型':
                k = ShapeKnowledge(self.path[i])
                knowledgeDic = k.readKnowledge()

            know.append(knowledgeDic)

        return know

    def visualKnowledge(self):
        """
        查看知识的具体内容
        """
        for i in range(self.num):
            if self.type[i] == '单调型':
                k = MonotonicityKnowledge(self.path[i])
                know = k.readKnowledge()
                k.visualKnowledge()

            if self.type[i] == '属性型':
                k = AttributeKnowledge(self.path[i])
                know = k.readKnowledge()
                k.visualKnowledge()

            if self.type[i] == '形状型':
                k = ShapeKnowledge(self.path[i])
                know = k.readKnowledge()
                k.visualKnowledge()

    def gradientSelect(self, x_t, y_t, printPicture=True, savePath=None):
        """

        :param method:筛选知识的方法，str，可选的有”GradientConsistency“（梯度一致性）
        :param x_t: 训练数据的输入
        :param y_t: 训练数据的输出
        :return: passlist: 筛选过后的知识路径组成的列表
        """

        knowPass = []

        # 梯度一致性的筛选方法只适用于形状型知识

        # 该筛选方法只适用于形状型知识
        for i in range(self.num):
            assert self.type[i] == '形状型' or "单调型"

        gradient_select = GradientSelection(x_t, y_t, knowledgelist=self.path)
        KnowPass = gradient_select.select(printPicture=printPicture, savePath=savePath)

        self.passlist = gradient_select.passlist

        return KnowPass

    def hausdorffSelect(self, printPicture=True, savePath=None):
        """
        :param method:筛选知识的方法，str，可选的有”GradientConsistency“（梯度一致性）
        :param x_t: 训练数据的输入
        :param y_t: 训练数据的输出
        :return: passlist: 筛选过后的知识路径组成的列表
        """
        knowPass = []

        # 梯度一致性的筛选方法只适用于形状型知识

        # 该筛选方法只适用于形状型知识
        for i in range(self.num):
            assert self.type[i] == '形状型' or "单调型"

        hau_select = HausdorffSelection(knowledgelist=self.path)
        KnowPass = hau_select.select(printPicture=printPicture, savePath=savePath)

        self.passlist = hau_select.passlist

        return KnowPass

    def fermatPointsFuse(self, select=None, x_t=None, y_t=None, printPicture=True, savePath=None):
        """
        知识的融合
        select:“gradient”,“hausdorff”
        :return:
        """

        if select == None:
            knowlist = self.path
        elif select == "gradient":
            self.gradientSelect(x_t, y_t)
            knowlist = self.passlist
        elif select == "hausdorff":
            self.hausdorffSelect()
            knowlist = self.passlist

        for i in range(self.num):
            assert self.type[i] == '形状型' or "单调型"

        mix = FermatPointsFusion(knowledgelist=knowlist)
        knowNew = mix.fuse(printPicture=printPicture, savePath=savePath)

        return knowNew

    def gaFuse(self, select=None, x_t=None, y_t=None, printPicture=True, savePath=None):
        if select == None:
            knowlist = self.path
        elif select == "gradient":
            self.gradientSelect(x_t, y_t)
            knowlist = self.passlist
        elif select == "hausdorff":
            self.hausdorffSelect()
            knowlist = self.passlist

        for i in range(self.num):
            assert self.type[i] == '形状型' or "单调型"

        mix = GAFusion(knowledgelist=knowlist)
        knowNew = mix.fuse(printPicture=printPicture, savePath=savePath)

        return knowNew


def matyas(x1, x2):
    return 0.26 * (x1 ** 2 + x2 ** 2) - 0.48 * x1 * x2


if __name__ == "__main__":
    # k = KnowledgeSet("C:\data\测试1.txt", "C:\data\测试2.txt", "C:\data\测试3.txt", "C:\data\测试4.txt")
    # k = KnowledgeSet("C:\data\测试1.txt", "C:\data\测试2.txt", "C:\data\测试3.txt", "C:\data\测试4.txt", knowType=["单调型"])
    # k = KnowledgeSet("C:\data\测试1.txt", "C:\data\测试2.txt", "C:\data\测试3.txt", "C:\data\测试4.txt",knowInput=["马赫数"])
    # k = KnowledgeSet("C:\data\测试1.txt", "C:\data\测试2.txt", "C:\data\测试3.txt", "C:\data\测试4.txt", knowOutput=["全弹法向力系数"])
    #
    # a = k.readKnowledge()
    # k.visualKnowledge()
    # print(a)

    x1 = np.linspace(-10, 10, 15)
    x2 = np.linspace(-10, 10, 15)
    # meshgrid 扩展成矩阵
    x1v, x2v = np.meshgrid(x1, x2)
    # flatten 合并
    x_train = np.column_stack((x1v.flatten(), x2v.flatten()))

    y_train = np.array(matyas(x_train[:, 0], x_train[:, 1])) + np.random.randn(225) * 0.5
    y_train = y_train.reshape(-1, 1)
    #
    # k = KnowledgeSet(folder=["C:\data\筛选测试"])
    # #knowPass = k.gradientSelect(x_train, y_train)
    # knowPass = k.hausdorffSelect(savePath=r"C:\data\筛选测试\筛选.png")
    # print(knowPass)
    # # # #
    k = KnowledgeSet(folder=["C:\data\筛选测试"])
    # knowNew = k.fermatPointsFuse()
    knowNew = k.fermatPointsFuse(savePath=r"C:\data\筛选测试\先筛选后融合1.png")
    print(knowNew)
