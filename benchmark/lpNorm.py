import numpy as np

from benchmark.benchmarkBase import Benchmark
import matplotlib.pyplot as plt


class LpNorm(Benchmark):

    def _initialize(self):

        self.xlimits[:, 0] = -1
        self.xlimits[:, 1] = 1

        self.p = 2

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

        inputTitle=[]
        outputTitle=["y"]
        inputRange = []
        outputRange = [0,100*self.ndim]

        for i in range(self.ndim):
            inputRange.append([0,10])
            inputTitle.append('x'+str(i))

        dataSet["title"] = [inputTitle,outputTitle]
        dataSet["range"] = [inputRange,outputRange]
        dataSet["input"] = x

        ne, nx = x.shape

        p = self.p
        assert p > 0
        y = np.zeros((ne, 1))
        lp_norm = np.sum(np.abs(x) ** p, axis=-1) ** (1.0 / p)
        y[:, 0] = lp_norm

        dataSet["output"] = y


        return dataSet


if __name__ == '__main__':

    num = 100
    ndim = 2
    x = np.ones((num, ndim))

    x[:, 0] = np.linspace(-10, 10.0, num)
    x[:, 1] = np.linspace(-10, 10.0, num)
    s = LpNorm()
    dataSet = s(x)
    plt.plot(dataSet["input"][:, 0], dataSet["output"][:, 0])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    print(dataSet)