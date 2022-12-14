import numpy as np

from benchmark.benchmarkBase import Benchmark
import matplotlib.pyplot as plt


class Easom(Benchmark):

    def _initialize(self):
        self.name ="easom"

        self.xlimits[:, 0] = -100
        self.xlimits[:, 1] = 100

        self.A = 10

    def _evaluate(self, x):
        assert self.ndim == 2, "该函数的输入只能支持二维"
        dataSet = {}

        inputTitle = []
        outputTitle = ["y"]
        inputRange = []
        outputRange = [0, 1]

        for i in range(self.ndim):
            inputRange.append([0, 5])
            inputTitle.append('x' + str(i))

        dataSet["title"] = [inputTitle, outputTitle]
        dataSet["range"] = [inputRange, outputRange]
        dataSet["input"] = x

        ne, nx = x.shape

        y = np.zeros((ne, 1))
        a = -np.cos(x[:, 0]) * np.cos(x[:, 1])
        b = (x[:, 0] - np.pi)**2 + (x[:, 1] - np.pi)**2
        c = np.exp(-b)
        easom = -a * c

        y[:, 0] = easom.T

        dataSet["output"] = y

        return dataSet


if __name__ == '__main__':
    num = 100
    ndim = 2
    x = np.ones((num, ndim))

    x[:, 0] = np.linspace(0, 5, num)
    x[:, 1] = np.linspace(0, 5, num)

    s = Easom()
    dataSet = s(x)
    print(dataSet)

    #2维图
    plt.plot(dataSet["input"][:, 0], dataSet["output"][:, 0])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()



    #3维图
    from mpl_toolkits.mplot3d import Axes3D
    # 作图的x轴定义
    x1_plot = np.linspace(0, 5, 100)
    x2_plot = np.linspace(0, 5, 100)
    # 转换格式
    x1_plot, x2_plot = np.meshgrid(x1_plot, x2_plot)

    print("开始绘画")
    # 图片定义
    fig = plt.figure()
    ax2 = Axes3D(fig)

    # 画曲面图的z轴定义为
    z_plot = np.zeros((x1_plot.shape[0], x1_plot.shape[1]))
    for i in range(x1_plot.shape[0]):
        for j in range(x1_plot.shape[1]):
            t = Easom()
            dataSet = t(np.array([[x1_plot[i][j], x2_plot[i][j]]]))
            z_plot[i][j] = np.squeeze(dataSet["output"])

    # 画出曲面
    ax2.plot_surface(x1_plot, x2_plot, z_plot, rstride=1, cstride=1, cmap='rainbow')
    ax2.set_xlabel("x0")
    ax2.set_ylabel("x1")
    ax2.set_zlabel("y")
    plt.show()

