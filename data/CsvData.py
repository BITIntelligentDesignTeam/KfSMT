import numpy as np
import pandas as pd
import csv
from data.database import Database


def rangeTransform(rangeStr):
    """
    将字符串类型的范围转换为对应的list或者tuple
    :param rangeStr: 字符串类型的范围
    :return: 范围应的list或者tuple
    """
    rangeStr_ = rangeStr[1:-1]
    result = rangeStr_.split(",")
    if rangeStr[0] == "[" and rangeStr[-1] == "]":
        result = [float(i) for i in result]

    elif rangeStr[0] == "(" and rangeStr[-1] == ")":
        result = tuple([float(i) for i in result])

    return result



class CsvData(Database):
    # 读取规定格式的csv文件

    def _read(self):
        with open(self.path, "rt") as csvfile:
            reader = csv.reader(csvfile)
            file = [row for row in reader]

        parameter = file[0]
        inputOrOutput = file[1]
        parameterRange = file[2]
        points = file[3:]
        pointsNum = len(points)
        points_ = []

        for j in range(pointsNum):
            point = [float(x) for x in points[j]]
            points_.append(point)
        points = np.array(points_)

        assert len(parameter) == len(inputOrOutput) == len(parameterRange), "请按照规定格式补充完表格"

        parameterNum = len(parameter)

        titleInput = []
        titleOutput = []
        rangeInput = []
        rangeOutput = []
        inputIndex = []
        outputIndex = []

        for i in range(parameterNum):

            if inputOrOutput[i] == "input":
                titleInput.append(parameter[i])
                rangeInput.append(list(rangeTransform(parameterRange[i])))    #后面还要改，现在是全是list
                inputIndex.append(i)

            elif inputOrOutput[i] == "output":
                titleOutput.append(parameter[i])
                rangeOutput.append(list(rangeTransform(parameterRange[i])))        #后面还要改，现在是全是list
                outputIndex.append(i)

        self.range = [rangeInput, rangeOutput]
        self.title = [titleInput, titleOutput]

        self.input = np.ones((pointsNum, len(inputIndex)))
        self.output = np.ones((pointsNum, len(outputIndex)))

        for i in range(pointsNum):
            for j in range(len(inputIndex)):
                self.input[i][j] = points[i][inputIndex[j]]
            for j in range(len(outputIndex)):
                self.output[i][j] = points[i][outputIndex[j]]

        self.dataDic["input"] = self.input
        self.dataDic["output"] = self.output
        self.dataDic["title"] = self.title
        self.dataDic["range"] = self.range

        return self.dataDic


if __name__ == "__main__":
    path = r"C:\data\动态采样.csv"
    d = CsvData(path)
    dataSet = d.read()
    input = dataSet["input"]
    print(input)
    out = dataSet["output"]
    a  =[input,out]
    print(a)

    print(dataSet)
    trainSet, testSet = d.divide(0.8)
    print(trainSet["input"].shape)
    print(testSet["input"].shape)
