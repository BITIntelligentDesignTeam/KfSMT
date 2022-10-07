from surrogate_models.SurrogateModel import SurrogateModel
import GPy
import matplotlib.pyplot as plt
import numpy as np
import pickle

from knowledge.ShapeKnowledge import bezierPoints

sigma = 0.1


def getDerivative(x: np.ndarray, y: np.ndarray, x0):
    """
        在x0处差分求导
        :param x: np.ndarray[nt, 1]，升序排列
        :param y: np.ndarray[nt, 1]，对应x处的y值
        :param x0: float
        :return: dy0/dx0
        """
    x = list(x)
    y = list(y)

    x.append(x0)
    x.sort()
    x_index = x.index(x0)

    dy0 = y[x_index] - y[x_index - 1]
    dx0 = x[x_index + 1] - x[x_index - 1]

    # 防止导数过大或者过小
    if dy0 > 5:
        dy0 = 5
    elif dy0 < -5:
        dy0 = -5
    elif dy0 > 0 and dy0 < 0.3:
        dy0 = 0.3
    elif dy0 < 0 and dy0 > -0.3:
        dy0 = -0.3

    return dy0 / dx0

def randomRange(rangeList):
    """
    在给定的试验因子范围内生成随机数
    :param rangeList:
    :return:
    """
    rangeList_ = []
    for i in rangeList:
        rangeList_.append(i[0] + (i[1] - i[0]) * np.random.random())
    return rangeList_

class GPBase(SurrogateModel):

    def predict(self, x, cov=False):
        mu,var = self.model.predict_noiseless([x])
        if cov==False:
            return mu
        if cov == True:
            return mu,var

class GP(GPBase):
    """
    普通高斯过程
    """
    def _train(self):
        kernel = GPy.kern.RBF(input_dim=len(self.inputTitle))

        # 设置对应的似然函数
        gauss = GPy.likelihoods.Gaussian()


        m = GPy.models.MultioutputGP(X_list=[self.dataSet['input']], Y_list=[self.dataSet['output']],
                                     kernel_list=[kernel,],
                                     likelihood_list=[gauss])
        m.optimize(messages=0, ipython_notebook=False)


        self.model = m


class GPK(GPBase):
    """
    融合知识的高斯过程
    """

    def _dataKnow(self):
        """
        将知识和数据处理成算法能够接受的形式
        :return:
        """
        inputTitle = self.dataSet['title'][0]
        outputTitle = self.dataSet['title'][1]
        inputRange = self.dataSet["range"][0]
        outputRange = self.dataSet["range"][1]
        inputNum = len(inputTitle)
        knowGpInput = []
        knowGpOutput = []

        for i in self.knowList:

            assert i["type"] == "单调型" or "形状型", "该算法只能支持形状型和单调型的知识"
            assert i['output_type'] == outputTitle, "请确保知识和数据的输出变量保持一致"
            assert i['input_type'][0] in inputTitle, "请确保知识和数据的输入变量保持一致"

            monoNum = 5  # 每条单调型知识的取点数量
            shapeNum = 5  # 每条形状型知识的取点数量
            knowListGpOutput = []
            knowListGpInput = []

            if i["type"] == "单调型":

                knowListGpInput = np.array([randomRange(inputRange) for j in range(monoNum)])  # 其他值是任意取的

                knowListGpInput_ = np.linspace(i['input_range'][0][0], i['input_range'][0][1], monoNum)

                index = inputTitle.index(i['input_type'][0])
                knowListGpInput[:, index] = knowListGpInput_
                knowListGpInput = np.column_stack((knowListGpInput, [index] * monoNum))

                if i['mapping_relation'] == ['单调递增']:
                    knowListGpOutput = [[1] for j in range(monoNum)]
                elif i['mapping_relation'] == ['单调递减']:
                    knowListGpOutput = [[0] for j in range(monoNum)]

                if i['convar'] != None:
                    for item in i['convar']:
                        convarType = item['convar_type']
                        if convarType in inputTitle:
                            convarTypeIndex = inputTitle.index(convarType)
                            convarRangeOrValue = item["convar_RangeOrValue"]
                            if convarRangeOrValue == 'value':
                                convarValue = item["convar_value"]
                                knowListGpInput[:, convarTypeIndex] = convarValue
                            elif convarRangeOrValue == 'range':
                                convar_range = item['convar_range']
                                convar_range_min, convar_range_max = convar_range[0], convar_range[1]
                                for m in range(monoNum):
                                    knowListGpInput[m, convarTypeIndex] = random.uniform(convar_range_min,
                                                                                         convar_range_max)


            elif i["type"] == "形状型":

                knowListGpInput = np.array([randomRange(inputRange) for i in range(shapeNum)])  # 其他值是任意取的

                knowListGpInput_ = np.linspace(i['input_range'][0][0], i['input_range'][0][1], shapeNum)

                index = inputTitle.index(i['input_type'][0])
                knowListGpInput[:, index] = knowListGpInput_
                knowListGpInput = np.column_stack((knowListGpInput, [index] * shapeNum))

                x, y = bezierPoints(i['mapping_relation'])
                for m in range(shapeNum):
                    x0 = knowListGpInput_[m]
                    x0_derivative = getderivative(x, y, x0)
                    knowListGpOutput.append([x0_derivative])

                if i['convar'] != None:
                    for item in i['convar']:
                        convarType = item['convar_type']
                        if convarType in inputTitle:
                            convarTypeIndex = inputTitle.index(convarType)
                            convarRangeOrValue = item["convar_RangeOrValue"]
                            if convarRangeOrValue == 'value':
                                convarValue = item["convar_value"]
                                knowListGpInput[:, convarTypeIndex] = convarValue
                            elif convarRangeOrValue == 'range':
                                convar_range = item['convar_range']
                                convar_range_min, convar_range_max = convar_range[0], convar_range[1]
                                for m in range(shapeNum):
                                    knowListGpInput[m, convarTypeIndex] = random.uniform(convar_range_min,
                                                                                         convar_range_max)

            knowGpInput.append(knowListGpInput)
            knowGpOutput.append(np.array(knowListGpOutput))

        return knowGpInput, knowGpOutput


    def _train(self):

        knowGpInput, knowGpOutput = self._dataKnow()

        kernel = GPy.kern.RBF(input_dim = len(self.inputTitle), variance=1., lengthscale=1.)  # 针对数据的核
        gauss = GPy.likelihoods.Gaussian(variance=sigma ** 2)  # 针对数据的分布

        x_list = [self.dataSet['input']]  # 把原始数据放入输入list第一个
        y_list = [self.dataSet['output']]
        k_list = [kernel]
        l_list = [gauss]

        # 遍历经验list，根据经验的数量和维度设置对应的核、分布
        for x in knowGpInput:
            x_list.append(x[:, :-1])  # 经验的最后一列是维度，不取这一列

            kernel_k = GPy.kern.DiffKern(kernel, x[0, -1])  # 通过最后一列的数值确定核维度

            k_list.append(kernel_k)
            gauss_s = GPy.likelihoods.Gaussian()
            l_list.append(gauss_s)

        for y in knowGpOutput:
            y_list.append(y)

        m = GPy.models.MultioutputGP(X_list=x_list, Y_list=y_list, kernel_list=k_list,
                                     likelihood_list=l_list)
        m.optimize(messages=0, ipython_notebook=False)

        self.model = m


if __name__ == "__main__":

    from data.CsvData import CsvData
    from knowledge.KnowledgeSet import KnowledgeSet

    def y_test(x1,x2,x3):
        y = 3 * np.cos(x1) + (x2 ** 2 / 5 )+ np.sin(x3)
        return y

    N = 10
    x1 = np.linspace(0, 5, N)
    x2 = np.linspace(0, 5, N)
    x3 = np.linspace(0, 5, N)
    x = np.array([x1, x2, x3]).T

    y = []
    for i in range(N):
        y.append(y_test(x1[i],x2[i],x3[i]))

    y = np.asarray(y).reshape(-1, 1)



    #获取数据
    path = r"C:\data\代理模型训练\代理模型训练.csv"
    d = CsvData(path)
    dataSet = d.read()
    print(dataSet)
    trainSet, testSet = d.divide(0.8)

    #获取知识
    k = KnowledgeSet("C:\data\代理模型训练\单调型知识1.txt","C:\data\代理模型训练\单调型知识2.txt" )
    knowList = k.readKnowledge()
    k.visualKnowledge()
    x_test = testSet["input"]


    gpModel = GPK()
    # gpModel = GP()

    gpModel.setData(trainSet)               #设置数据

    gpModel.setKnowledge(knowList=knowList)           #设置知识

    gpModel.train()          #训练

    t = 10
    x1t = np.linspace(0, 5, t)
    x2t = np.linspace(0, 5, t)
    x3t = np.linspace(0, 5, t)
    x_test = np.array([x1t, x2t, x3t]).T

    yp = gpModel.predict(x_test)        #预测

    b = gpModel.score(testSet,index="RSME")
    print(b)

    gpModel.save(r"C:\data\代理模型训练\高斯过程代理模型.pkl")    #保存模型文件



    gpModel2 = None                           #加载模型文件
    filename = r"C:\data\代理模型训练\高斯过程代理模型.pkl"
    with open(filename, "rb") as f:
        gpModel2 = pickle.load(f)

    p = gpModel2.predict(x_test)
    print(p)













