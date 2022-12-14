import numpy as np

from benchmark.benchmarkBase import Benchmark
import matplotlib.pyplot as plt


class Branin(Benchmark):

    def _initialize(self):

        self.name = "branin"

        self.xlimits[0, 0] = -5
        self.xlimits[0, 1] = 10.0

        self.xlimits[1, 0] = 0
        self.xlimits[1, 1] = 15.0

    def _evaluate(self, x):
        """
        Arguments
        ---------
        x : ndarray[ne, nx]
            Evaluation points.
        kx : int or None
            Index of derivative (0-based) to return values with respect to.
            None means return function value rather than derivative.

        Returns
        -------
        ndarray[ne, 1]
            Functions values if kx=None or derivative values if kx is an int.
        """

        assert self.ndim == 2, "该函数的输入只能支持二维"

        dataSet = {}

        inputTitle = []
        outputTitle = ["y"]
        inputRange = []
        outputRange = [0, 309]

        for i in range(self.ndim):
            inputRange.append([0, 10])
            inputTitle.append('x' + str(i))

        dataSet["title"] = [inputTitle, outputTitle]
        dataSet["range"] = [inputRange, outputRange]
        dataSet["input"] = x

        ne, nx = x.shape

        y = np.zeros((ne, 1))
        b = 5.1 / (4.0 * (np.pi) ** 2)
        c = 5.0 / np.pi
        t = 1.0 / (8.0 * np.pi)
        u = x[:, 1] - b * x[:, 0] ** 2 + c * x[:, 0] - 6
        r = 10.0 * (1.0 - t) * np.cos(x[:, 0]) + 10
        y[:, 0] = u ** 2 + r


        dataSet["output"] = y

        return dataSet

if __name__ == '__main__':

    num = 100
    ndim = 2
    x = np.ones((num, ndim))

    x[:, 0] = np.linspace(-5, 10.0, num)
    x[:, 1] = np.linspace(0, 15.0, num)
    s = Branin()
    dataSet = s(x)
    plt.plot(dataSet["input"][:, 0], dataSet["output"][:, 0])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    print(dataSet)
    print(s(np.array([[-5,0]])))
