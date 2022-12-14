import numpy as np
from benchmark.benchmarkBase import Benchmark

class CMG(Benchmark):

    def _initialize(self):

        self.ndim = 3
        self.xlimits = np.zeros((self.ndim, 2))

        self.xlimits[0, 0] = -20
        self.xlimits[0, 1] = 70

        self.xlimits[1, 0] = 0
        self.xlimits[1, 1] = 6

        self.xlimits[2, 0] = 0
        self.xlimits[2, 1] = 5


        self.name = "CMG"


if __name__ == '__main__':
    c = CMG()
    CMGdata1,CMGdata2= c.getData(proportion=0.7)
    print(CMGdata1)
    print(CMGdata2)

    print(CMGdata1["input"].shape)
    print(CMGdata2["input"].shape)

    knowlist = c.getKnowledge()
    print(knowlist)


