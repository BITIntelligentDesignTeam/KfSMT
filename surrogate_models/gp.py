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
    # dx0 = x[x_index + 1] - x[x_index - 1]
    dx0 = x[x_index ] - x[x_index - 1]

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
            print(mu)
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
        m = GPy.models.MultioutputGP(X_list=[self.dataSet['input']], Y_list=[self.dataSet['output']],kernel_list=[kernel,],
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

            monoNum = 5   # 每条单调型知识的取点数量
            shapeNum = 5   # 每条形状型知识的取点数量
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
                    x0_derivative = getDerivative(x, y, x0)
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
    from benchmark.sphere import Sphere
    from utils.normalization import Normalization
    from sklearn.preprocessing import MinMaxScaler

    ndim = 2
    trainNum = 50
    testNum = 10

    x = np.ones((trainNum, ndim))
    x[:, 0] = np.linspace(-10, 10.0, trainNum)
    x[:, 1] = np.linspace(-10, 10.0, trainNum)

    xt = np.ones((testNum, ndim))
    xt[:, 0] = np.linspace(-10, 10.0, testNum)
    xt[:, 1] = np.linspace(-10, 10.0, testNum)

    s = Sphere()
    trainSet = s(x)           #训练集
    testSet = s(xt)            #测试集


    # 获取知识
    k = KnowledgeSet( "C:\data\sphereMonoKnowledge1.xml","C:\data\sphereShapeKnowledge.xml",
                     "C:\data\sphereMonoKnowledge2.xml")
    knowList = k.readKnowledge()

    #对知识和数据进行归一化操作
    n = Normalization()
    trainSet,knowList = n(trainSet,knowList)
    print(trainSet)
    print(knowList)
    xt = n.transform(xt)


    Model = GPK()

    Model.setData(trainSet)                       #设置数据

    Model.setKnowledge(knowList=knowList)                        #设置知识

    Model.train()          #训练

    yp = Model.predict(xt)        #预测


    ln1 = plt.plot(trainSet["input"][:, 0], trainSet["output"][:, 0], label ="真实",color = "red")
    ln2 = plt.plot(xt[:, 0], yp[:, 0], label ="预测",color = 'green')
    print(xt[:, 0])
    print(yp[:, 0])
    plt.legend()
    plt.show()

    yp = n.inverse(yp)
    #将预测完毕后的数据进行反归一化
    print(yp)
    # b = Model.score(testSet,index="RSME")  #评价代理模型 ，"RSME", "R2", "MAE", "Confidence"

    Model.save(r"C:\data\代理模型训练\代理模型.pkl")    #保存模型文件


    gpModel2 = None                           #加载模型文件
    filename = r"C:\data\代理模型训练\高斯过程代理模型.pkl"
    with open(filename, "rb") as f:
        gpModel2 = pickle.load(f)

    p = gpModel2.predict(x_test)
    print(p)













