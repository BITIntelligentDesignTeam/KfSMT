import warnings
import numpy as np
from scipy.optimize import minimize
from bayes_opt.util import UtilityFunction


def acq_max(ac, gp, y_max, bounds, random_state, n_warmup=10000, n_iter=10):
    """
    A function to find the maximum of the acquisition function

    It uses a combination of random sampling (cheap) and the 'L-BFGS-B'
    optimization method. First by sampling `n_warmup` (1e5) points at random,
    and then running L-BFGS-B from `n_iter` (250) random starting points.

    Parameters
    ----------
    :param ac:
        The acquisition function object that return its point-wise value.

    :param gp:
        A gaussian process fitted to the relevant data.

    :param y_max:
        The current maximum known value of the target function.

    :param bounds:
        The variables bounds to limit the search of the acq max.

    :param random_state:
        instance of np.RandomState random number generator

    :param n_warmup:
        number of times to randomly sample the aquisition function

    :param n_iter:
        number of times to run scipy.minimize

    Returns
    -------
    :return: x_max, The arg max of the acquisition function.
    """

    # Warm up with random points
    x_tries = random_state.uniform(bounds[:, 0], bounds[:, 1],
                                   size=(n_warmup, bounds.shape[0]))
    ys = ac(x_tries, gp=gp, y_max=y_max)
    x_max = x_tries[ys.argmax()]
    max_acq = ys.max()

    # Explore the parameter space more throughly
    x_seeds = random_state.uniform(bounds[:, 0], bounds[:, 1],
                                   size=(n_iter, bounds.shape[0]))
    for x_try in x_seeds:
        # Find the minimum of minus the acquisition function
        res = minimize(lambda x: -ac(x.reshape(1, -1), gp=gp, y_max=y_max),
                       x_try.reshape(1, -1),
                       bounds=bounds,
                       method="L-BFGS-B")

        # See if success
        if not res.success:
            continue

        # Store it if better than previous minimum(maximum).
        if max_acq is None or -res.fun[0] >= max_acq:
            x_max = res.x
            max_acq = -res.fun[0]


    # ____________________简化______________________________
    x_max = np.around(x_max, 1)
    # ____________________简化______________________________

    # Clip output to make sure it lies within the bounds. Due to floating
    # point technicalities this is not always the case.
    return np.clip(x_max, bounds[:, 0], bounds[:, 1])


class sysUtil(UtilityFunction):
    """
    An object to compute the acquisition functions.
    """

    def __init__(self, pbounds, iteration, kinds, SK=1, punish=1):

        self.iteration = iteration
        # self.kappa = kappa
        # self._kappa_decay = kappa_decay
        # self._kappa_decay_delay = kappa_decay_delay

        # *****加入空间型知识的函数，输入点，输出采集函数权重，没有空间知识则缺省为1
        self.SK_func = SK
        self.punish_f = punish
        self.pbounds = pbounds
        # **

        self._iters_counter = 0

        if kinds not in ['ucb', 'ei', 'poi', 'ud']:
            print("必须有'ucb', 'ei', 'poi', 'ud'之一")

        else:
            self.kind = kinds

    def update_params(self):
        self._iters_counter += 1

        # if self._kappa_decay < 1 and self._iters_counter > self._kappa_decay_delay:
        #     self.kappa *= self._kappa_decay

    def utility(self, x, gp, y_max):


        # 没有加入惩罚
        if self.punish_f == 1:
            # ******空间型知识
            if self.SK_func == 1:  # 没有输入SK-func,SK-func缺省为1，此时权重为1，对采集函数没有影响
                keys = sorted(self.pbounds)
                weight = []
                for i in range(x.shape[0]):
                    params = dict(zip(keys, x[i, :]))
                    weight.append(self.SK_func(**params))
                weight = np.array(weight)
                return weight * self._ud(x, gp)

            else:
                keys = sorted(self.pbounds)
                weight = []
                for i in range(x.shape[0]):
                    params = dict(zip(keys, x[i, :]))
                    weight.append(self.SK_func(**params))
                weight = np.array(weight)
                return weight * self._ud(x, gp)
        # 加入惩罚
        else:
            if self.SK_func == 1:  # 没有输入SK-func,SK-func缺省为1，此时权重为1，对采集函数没有影响
                keys = sorted(self.pbounds)
                pu = []
                for i in range(x.shape[0]):
                    params = dict(zip(keys, x[i, :]))
                    p = self.punish_f(**params)
                    if p is None:
                        p = 1
                    pu.append(p)
                pu = np.array(pu)
                return pu * self._ud(x, gp)

            else:
                keys = sorted(self.pbounds)
                pu = []
                weight = []
                for i in range(x.shape[0]):
                    params = dict(zip(keys, x[i, :]))
                    p = self.punish_f(**params)
                    if p is None:
                        p = 1
                    pu.append(p)

                    weight.append(self.SK_func(**params))
                weight = np.array(weight)
                pu = np.array(pu)
                return pu * weight * self._ud(x, gp)

        # **

    # ******降低不确定性的采集函数ud
    @staticmethod
    def _ud(x, gp):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        return std

    # **


def ensure_rng(random_state=None):
    """
    Creates a random number generator based on an optional seed.  This can be
    an integer or another random state for a seeded rng, or None for an
    unseeded rng.
    """
    if random_state is None:
        random_state = np.random.RandomState()
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    else:
        assert isinstance(random_state, np.random.RandomState)
    return random_state
