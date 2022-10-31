from utils.check import checkDataKnow
import numpy as np


class Normalization(object):
    """
    进行归一化与反归一化的类
    """

    def __init__(self):
        self.scaling = [[], []]  # 输入和输出每个维度上的缩放比例
        self.minValue = [[], []]  # 输入和输出数据每个维度上的最小值

    def __call__(self, dataSet, knowledgeList, featureRange: list = [0, 1]):
        """
        将数据和知识进行归一化的处理
        :param dataSet: dict,待处理的数据集
        :param knowledgeList: list , 待处理的知识list
        :param knowledgeList: range , 缩放后的范围
        :return: dataSetNormalization,knowledgeListNormalization , 经过归一化后的数据集和知识集
        """

        # 检查知识和数据的输出是否一致
        dataSet, knowledgeList = checkDataKnow(dataSet, knowledgeList)

        # 首先针对数据进行归一化
        inputTitle, outputTitle = dataSet["title"][0], dataSet["title"][1]
        nx, ny = len(inputTitle), len(outputTitle)  # 输入和输出的维度
        inputData, outputData = dataSet["input"], dataSet["output"]
        nt = len(inputData)  # 数据点的个数
        targetMinValue, targetMaxValue = featureRange[0], featureRange[1]
        self.targetMinValue, self.targetMaxValue = targetMinValue, targetMaxValue

        # 针对数据输入的归一化
        inputDataNew = np.ones((nt, nx))
        for i in range(nx):
            x_i = inputData[:, i]
            maxValue, minValue = np.max(x_i), np.min(x_i)
            self.minValue[0].append(minValue)
            k = (targetMaxValue - targetMinValue) / (maxValue - minValue)  # 缩放系数
            self.scaling[0].append(k)
            x_i = np.array([k * (i - minValue) + targetMinValue for i in x_i])
            inputDataNew[:, i] = x_i

        # 针对数据输出的归一化
        outputDataNew = np.ones((nt, ny))
        for i in range(ny):
            y_i = outputData[:, i]
            maxValue, minValue = np.max(y_i), np.min(y_i)
            self.minValue[1].append(minValue)
            k = (targetMaxValue - targetMinValue) / (maxValue - minValue)  # 缩放系数
            self.scaling[1].append(k)
            y_i = np.array([k * (i - minValue) + targetMinValue for i in y_i])
            outputDataNew[:, i] = y_i

        dataSet["input"] = inputDataNew
        dataSet["output"] = outputDataNew
        dataSet["range"] = [[featureRange for i in range(nx)], [featureRange for i in range(ny)]]

        # 接下来对知识进行归一化
        typeList = [i["type"] for i in knowledgeList]
        nk = len(typeList)  # 知识的数量

        for i in range(nk):

            # 针对单调型知识的处理
            if typeList[i] == "单调型":
                # 对输入范围的归一化
                input_range = knowledgeList[i]["input_range"]
                inputIndex = inputTitle.index(knowledgeList[i]["input_type"][0])
                k = self.scaling[0][inputIndex]
                minValue = self.minValue[0][inputIndex]
                input_rangeNew = []
                for j in input_range:
                    input_rangeNew.append(
                        [k * (j[0] - minValue) + targetMinValue, k * (j[1] - minValue) + targetMinValue])
                knowledgeList[i]["input_range"] = input_rangeNew

            # 针对形状型知识的处理
            elif typeList[i] == "形状型":

                # 对输入和输出范围的归一化
                input_range, output_range = knowledgeList[i]["input_range"], knowledgeList[i]["output_range"]
                mapping_relation = knowledgeList[i]["mapping_relation"]

                inputIndex, outputIndex = inputTitle.index(knowledgeList[i]["input_type"][0]), \
                                          outputTitle.index(knowledgeList[i]["output_type"][0])
                k_x, k_y = self.scaling[0][inputIndex], self.scaling[1][outputIndex]
                minValue_x, minValue_y = self.minValue[0][inputIndex], self.minValue[1][outputIndex]

                input_rangeNew = []
                for j in input_range:
                    input_rangeNew.append(
                        [k_x * (j[0] - minValue_x) + targetMinValue, k_x * (j[1] - minValue_x) + targetMinValue])
                knowledgeList[i]["input_range"] = input_rangeNew

                output_rangeNew = []
                for j in output_range:
                    input_rangeNew.append(
                        [k_y * (j[0] - minValue_y) + targetMinValue, k_y * (j[1] - minValue_y) + targetMinValue])
                knowledgeList[i]["output_range"] = output_rangeNew

                # 针对于贝塞尔控制点的归一化
                mapping_relationNew = []
                for j in mapping_relationNew:
                    mapping_relationNew.append(
                        [k_x * (j[0] - minValue_x) + targetMinValue, k_y * (j[1] - minValue_y) + targetMinValue])

            # 针对协边量的处理
            if knowledgeList[i]["convar"]:
                pass

        return dataSet, knowledgeList

    def transform(self, x: np.ndarray):
        """
        将待预测用的数据进行归一化
        :param x: 待预测用的数据
        :return: 归一化完成后的数据
        """

        nt, nx = x.shape
        assert nx == len(self.scaling[0]), "数据的维度不正确"
        xNew = np.ones((nt, nx))

        for i in range(nt):
            for j in range(nx):
                k = self.scaling[0][j]
                minValue = self.minValue[0][j]
                xNew[i][j] = k * (x[i][j] - minValue) + self.targetMinValue

        return xNew

    def inverse(self, y: np.ndarray):
        """
        将模型预测完毕后的数据进行反归一化
        :param y: 模型预测完毕后的数据
        :return: 反归一化的数据
        """

        nt, ny = y.shape
        assert ny == len(self.scaling[1]), "数据的维度不正确"
        yNew = np.ones((nt, ny))

        for i in range(nt):
            for j in range(ny):
                k = self.scaling[1][j]
                minValue = self.minValue [1][j]
                yNew[i][j] = (y[i][j] - self.targetMinValue)/k + minValue

        return yNew

if __name__ == "__main__":

    dataSet = {"input": np.array([[1, 2, 3], [2, 5, 1], [10, 10, 10]]), "output": np.array([[1], [10], [25]]),
               "range": [[[1, 5], [1, 5], [1, 5]], [1, 5]], "title": [["x0", "x1", "x2"], ["y"]]}
    knowlist = [
        {'type': '单调型', 'input_type': ['x0'], 'output_type': ['y'], 'convar': None, 'input_range': [[0.0, 10.0]],
         'mapping_relation': ['单调递增']},
        {'type': '形状型', 'input_type': ['x0'], 'output_type': ['y'], 'convar': None, 'input_range': [[-10.0, 10.0]],
         'output_range': [[0.0, 100.0]],
         'mapping_relation': [[-9.91, 102.4], [-8.06, 20.8], [0.28, -84.8], [7.05, -3.2], [9.97, 105.6]]}]

    n = Normalization()
    dataSet, knowledgeList = n(dataSet, knowlist)
    print(dataSet)
    print(knowlist)

    xt = np.array([[2, 5, 1]])
    y_p = np.array([[1]])
    xtNew = n.transform(xt)
    y_P = n.inverse(y_p)
    print(xtNew)
    print(y_P)
