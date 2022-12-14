import numpy as np

from benchmark.benchmarkBase import Benchmark
import matplotlib.pyplot as plt


class Rastrigin(Benchmark):

    def _initialize(self):

        self.name = "rastrigin"
        self.xlimits[:, 0] = -5.12
        self.xlimits[:, 1] = 5.12

        self.A = 10

    def _evaluate(self, x):
        dataSet = {}

        inputTitle = []
        outputTitle = ["y"]
        inputRange = []
        outputRange = [0, 100 * self.ndim]

        for i in range(self.ndim):
            inputRange.append([-5.12, 5.12])
            inputTitle.append('x' + str(i))

        dataSet["title"] = [inputTitle, outputTitle]
        dataSet["range"] = [inputRange, outputRange]
        dataSet["input"] = x

        ne, nx = x.shape

        A = self.A
        y = np.zeros((ne, 1))
        rastrigin = A * nx + np.sum(x ** 2 - A * np.cos(2*np.pi),1)
        y[:, 0] = rastrigin.T

        dataSet["output"] = y

        return dataSet


if __name__ == '__main__':
    # num = 100
    # ndim = 2
    # x = np.ones((num, ndim))
    #
    # x[:, 0] = np.linspace(-5.12, 5.12, num)
    # x[:, 1] = np.linspace(-5.12, 5.12, num)
    # s = Rastrigin()
    # dataSet = s(x)
    # plt.plot(dataSet["input"][:, 0], dataSet["output"][:, 0])
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.show()
    # print(dataSet)

    # 3维图
    from mpl_toolkits.mplot3d import Axes3D

    # 作图的x轴定义
    x1_plot = np.linspace(-5, 5, 100)
    x2_plot = np.linspace(-5, 5, 100)
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
            t = Rastrigin()
            dataSet = t(np.array([[x1_plot[i][j], x2_plot[i][j]]]))
            z_plot[i][j] = np.squeeze(dataSet["output"])

    # 画出曲面
    ax2.plot_surface(x1_plot, x2_plot, z_plot, rstride=1, cstride=1, cmap='rainbow')
    ax2.set_xlabel("x0")
    ax2.set_ylabel("x1")
    ax2.set_zlabel("y")
    plt.show()
