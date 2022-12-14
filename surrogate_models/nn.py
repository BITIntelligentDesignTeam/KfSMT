import random
import numpy as np
from utils.e_DaKnow.class_Neu_net import Neu_Net
from utils.e_DaKnow.class_Error import Error
from deap import base, creator
from deap import tools
from utils.e_DaKnow.json_python import process
from surrogate_models.SurrogateModel import SurrogateModel
from surrogate_models.gp import getDerivative, randomRange
from knowledge.MonotonicityKnowledge import MonotonicityKnowledge
from data.CsvData import CsvData
import json
import matplotlib.pyplot as plt
from pylab import mpl
import pickle
import tensorflow as tf
import xlrd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
import matplotlib.animation as animation
from matplotlib import cm
import warnings


class NNbase(SurrogateModel):
    pass




class EBNN(NNbase):
    """

    """

    def _initialize(self):
        self.lr = 0.3
        self.epoch = 3
        self.hidderlayer = 16

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

        train_x = self.dataSet["input"]
        train_y = self.dataSet["output"]

        tool = MinMaxScaler(feature_range=(0, 1))  # 根据需要设置最大最小值，这里设置最大值为1.最小值为0

        train_x = tool.fit_transform(train_x)  # 标准化，注意这里的values是array
        train_y = tool.fit_transform(train_y)

        # 将经验对应的位置的数据提出来，用来做后面的梯度求解
        x_1 = train_x[know_a_1]
        y_1 = train_y[know_a_1]

        x_2 = train_x[know_a_2]
        y_2 = train_y[know_a_2]

        x_train = tf.cast(train_x, tf.float32)
        x_1 = tf.cast(x_1, tf.float32)
        x_2 = tf.cast(x_2, tf.float32)

        # x_test = tf.cast(test_x, tf.float32)
        x_train = tf.Variable(x_train)
        x_1 = tf.Variable(x_1)
        x_2 = tf.Variable(x_2)

        w1 = tf.Variable(tf.random.truncated_normal([3, 16], stddev=0.1, seed=1))
        w2 = tf.Variable(tf.random.truncated_normal([16, 1], stddev=0.1, seed=1))
        # w3 = tf.Variable(tf.random.truncated_normal([8, 1], stddev=0.1, seed=1))
        b1 = tf.Variable(tf.random.truncated_normal([16], stddev=0.1, seed=1))
        b2 = tf.Variable(tf.random.truncated_normal([1], stddev=0.1, seed=1))
        # b3 = tf.Variable(tf.random.truncated_normal([1], stddev=0.1, seed=1))

        lr = self.lr  # 学习率为0.3

        epoch = self.epoch  # 循环500轮

        L = []
        # 训练部分

        for epoch in range(epoch):  # 数据集级别的循环，每个epoch循环一次数据集
            with tf.GradientTape(persistent=True) as tape:  # with结构记录梯度信息
                y1 = tf.matmul(x_train, w1) + b1  # 神经网络乘加运算
                y1 = tf.nn.relu(y1)
                y = tf.matmul(y1, w2) + b2
                # y2 = tf.nn.relu(y2)
                # y = tf.matmul(y2, w3) + b3

                y1 = tf.matmul(x_1, w1) + b1  # 神经网络乘加运算
                y1 = tf.nn.relu(y1)
                y_1_k = tf.matmul(y1, w2) + b2

                y1 = tf.matmul(x_2, w1) + b1
                y1 = tf.nn.relu(y1)
                y_2_k = tf.matmul(y1, w2) + b2
                # y = tf.nn.relu(y)
                x_grad_1 = tape.gradient(y_1_k, x_1)
                x_grad_2 = tape.gradient(y_2_k, x_2)

                # # # 滚转角
                # # phi_grad = x_grad[:, 0]
                # # 攻角 Fa/a
                # a_grad_1 = x_grad_1[:, 1]
                # 马赫数 Fa/ma
                Ma_grad_1 = x_grad_1[:, 2]
                Ma_grad_2 = x_grad_2[:, 2]
                # x_grad_1 = tf.reduce_mean(Ma_grad_1)0.006

                # loss = tf.reduce_mean(tf.square(train_y - y)) + \
                #        4*tf.reduce_mean(tf.square(Ma_grad_1-(3*0.06438698*tf.square(x_1[:,2])-2*0.03838*x_1[:,2]+0.01340464))) + \
                #        tf.reduce_mean(tf.square(Ma_grad_2-(3*(-0.04822684)*tf.square(x_2[:,2])+2*0.11701624*x_2[:,2]- 0.13385506)))
                # loss = tf.reduce_mean(tf.square(train_y - y)) # 采用均方误差损失函数mse = mean(sum(y-out)^2)

                # 0.1747是梯度值，可以从XML里先读取贝塞尔不等式再求导得到。
                loss = tf.reduce_mean(tf.square(train_y - y)) + 0.1 * tf.reduce_mean(tf.square(x_grad_1 - 0.1747))
                if epoch % 100 == 0:
                    #     # print('x_grad:', x_grad)
                    print('Loss:', loss)
                #     L.append(loss.numpy())
                L.append(loss.numpy())
                # print(b1)
                # print(Ma_grad_1)
                # 计算loss对各个参数的梯度
                grads = tape.gradient(loss, [w1, b1, w2, b2])
                # print('grad:', grads)
                # 实现梯度更新 w1 = w1 - lr * w1_grad    b = b - lr * b_grad
                w1.assign_sub(lr * grads[0])  # 参数w1自更新
                b1.assign_sub(lr * grads[1])  # 参数b1自更新
                w2.assign_sub(lr * grads[2])  # 参数w2自更新
                b2.assign_sub(lr * grads[3])  # 参数b2自更新
                # w3.assign_sub(lr * grads[4])  # 参数w2自更新
                # b3.assign_sub(lr * grads[5])  # 参数b2自更新


##############################################################################
if __name__ == '__main__':
    from data.CsvData import CsvData
    from knowledge.KnowledgeSet import KnowledgeSet
    from benchmark.sphere import Sphere
    from utils.normalization import Normalization

    # 获取数据
    # path = r"C:\data\代理模型训练\代理模型训练.csv"
    # d = CsvData(path)
    # dataSet = d.read()
    # print(dataSet)
    # trainSet, testSet = d.divide(0.8)

    ndim = 2
    trainNum = 100
    testNum = 10

    x = np.ones((trainNum, ndim))
    x[:, 0] = np.linspace(-10, 10.0, trainNum)
    x[:, 1] = 0.0

    xt = np.ones((testNum, ndim))
    xt[:, 0] = np.linspace(-10, 10.0, testNum)
    xt[:, 1] = 0.0

    s = Sphere()
    trainSet = s(x)
    testSet = s(xt)

    # plt.plot(x[:, 0], trainSet["output"][:, 0])



    # 获取知识
    k = KnowledgeSet("C:\data\sphereMonoKnowledge1.xml", "C:\data\sphereMonoKnowledge2.xml")
    knowList = k.readKnowledge()
    k.visualKnowledge()
    x_test = testSet["input"]



    nnModel = EBNN()
    # gpModel = BNN()

    nnModel.setData(trainSet)  # 设置数据

    nnModel.setKnowledge(knowList=knowList)  # 设置知识

    nnModel.train()  # 训练

    yp = nnModel.predict(x_test)  # 预测
    print(yp)

    plt.plot(x[:, 0], trainSet["output"][:, 0], label="真实")
    plt.plot(x_test[:, 0], yp[:, 0], label="预测")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    b = nnModel.score(testSet, index="RSME")  # 评价代理模型 ，"RSME", "R2", "MAE", "Confidence"
    print(b)

    nnModel.save(r"C:\data\代理模型训练\高斯过程代理模型.pkl")  # 保存模型文件

    # gpModel2 = None  # 加载模型文件
    # filename = r"C:\data\代理模型训练\高斯过程代理模型.pkl"
    # with open(filename, "rb") as f:
    #     gpModel2 = pickle.load(f)
    #
    # p = gpModel2.predict(x_test)
    # print(p)
