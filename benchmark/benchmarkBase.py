import numpy as np
import copy
import random
from abc import ABCMeta, abstractmethod

class Benchmark(object, metaclass=ABCMeta):
    def __init__(self, **kwargs):
        """
        Constructor where values of options can be passed in.

        For the list of options, see the documentation for the problem being used.

        Parameters
        ----------
        **kwargs : named arguments
            Set of options that can be optionally set; each option must have been declared.

        Examples
        --------
        >>> from smt.problems import Sphere
        >>> prob = Sphere(ndim=3)
        """

        self.ndim = 2
        self.xlimits = np.zeros((self.ndim, 2))
        self._initialize()

    def _initialize(self) -> None:
        """
        Implemented by problem to declare options (optional).

        Examples
        --------
        self.options.declare('option_name', default_value, types=(bool, int), desc='description')
        """
        pass


    def __call__(self, x: np.ndarray,proportion = None ) -> np.ndarray:
        """
        Evaluate the function.

        Parameters
        ----------
        x : ndarray[n, nx] or ndarray[n]
            Evaluation points where n is the number of evaluation points.
        kx : int or None
            Index of derivative (0-based) to return values with respect to.
            None means return function value rather than derivative.

        Returns
        -------
        ndarray[n, 1]
            Functions values if kx=None or derivative values if kx is an int.
        """


        self.dataSet = self._evaluate(x)

        if proportion == None:
            return self.dataSet
        else:
            assert proportion <= 1, "请输入小于1的数"
            pointsNum = len(self.dataSet["input"])
            trainNum = int(proportion * pointsNum)

            testPointsInput = list(self.dataSet["input"])
            testPointsOutput = list(self.dataSet["output"])
            trainPointsInput = []
            trainPointsOutput = []
            for i in range(trainNum):
                rand_index = 0
                num_items = len(testPointsInput)
                rand_index = random.randrange(num_items)
                trainPointsInput.append(testPointsInput[rand_index])
                trainPointsOutput.append(testPointsOutput[rand_index])
                testPointsInput.pop(rand_index)
                testPointsOutput.pop(rand_index)

            trainSet = copy.deepcopy(self.dataSet)
            testSet = copy.deepcopy(self.dataSet)

            trainSet["input"] = np.array(trainPointsInput)
            trainSet["output"] = np.array(trainPointsOutput)

            testSet["input"] = np.array(testPointsInput)
            testSet["output"] = np.array(testPointsOutput)

            return trainSet, testSet


    @abstractmethod
    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Implemented by surrogate models to evaluate the function.

        Parameters
        ----------
        x : ndarray[n, nx]
            Evaluation points where n is the number of evaluation points.
        kx : int or None
            Index of derivative (0-based) to return values with respect to.
            None means return function value rather than derivative.

        Returns
        -------
        ndarray[n, 1]
            Functions values if kx=None or derivative values if kx is an int.
        """
        raise Exception("This problem has not been implemented correctly")