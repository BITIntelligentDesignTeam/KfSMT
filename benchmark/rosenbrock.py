import numpy as np

from benchmark.benchmarkBase import Benchmark
import matplotlib.pyplot as plt

class Rosenbrock(Benchmark):

    def _initialize(self):

        self.xlimits[:, 0] = -2
        self.xlimits[:, 1] = 2

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

        for ix in range(nx - 1):
            y[:, 0] += (
                    100.0 * (x[:, ix + 1] - x[:, ix] ** 2) ** 2 + (1 - x[:, ix]) ** 2
            )


        dataSet["output"] = y

        return dataSet

if __name__ == '__main__':
    num = 100
    ndim = 2
    x = np.ones((num, ndim))

    x[:, 0] = np.linspace(-2, 2, num)
    x[:, 1] = np.linspace(-2, 2, num)
    s = Rosenbrock()
    dataSet = s(x)
    plt.plot(dataSet["input"][:, 0], dataSet["output"][:, 0])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    print(dataSet)
    print(s(np.array([[-5, 0]])))
