import warnings
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from Support.NSGAII import NSGA2
from bayes_opt.util import UtilityFunction


def acq_max1(ac, gp_obj1, gp_obj2):
    func = lambda ind: ac(list(ind), gp_obj1=gp_obj1, gp_obj2=gp_obj2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
    x_max = NSGA2(func)

    # ____________________简化
    x_max = np.around(x_max, 1)
    # ____________________简化

    return x_max


def acq_max(ac, gp_obj1, gp_obj2, bounds, random_state, y_max=1.0, n_warmup=10000, n_iter=10, iteration=1):
    # Warm up with random points
    x_tries = random_state.uniform(bounds[:, 0], bounds[:, 1],
                                   size=(n_warmup, bounds.shape[0]))
    ys = ac(x_tries, gp_obj1=gp_obj1, gp_obj2=gp_obj2, iteration=iteration)
    x_max = x_tries[ys.argmax()]
    max_acq = ys.max()

    # Explore the parameter space more throughly
    x_seeds = random_state.uniform(bounds[:, 0], bounds[:, 1],
                                   size=(n_iter, bounds.shape[0]))
    for x_try in x_seeds:
        # Find the minimum of minus the acquisition function
        res = minimize(lambda x: -ac(x.reshape(1, -1), gp_obj1=gp_obj1, gp_obj2=gp_obj2,
                                     iteration=iteration),
                       x_try.reshape(1, -1),
                       bounds=bounds,
                       method="L-BFGS-B")  # 采集函数作为参数传到acq_max函数里，这里的ac就是采集函数，使用有限梯度下降进行了采集函数的最大化，

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


class MOUtil(UtilityFunction):
    """
    An object to compute the acquisition functions.
    """

    def __init__(self, kinds, xi, pbounds, SK1=1, SK2=1, punish=1):

        # super().__init__(kinds, kappa, xi, kappa_decay, kappa_decay_delay)
        # self.kappa = kappa
        # self._kappa_decay = kappa_decay
        # self._kappa_decay_delay = kappa_decay_delay

        # *****加入空间型知识的函数，输入点，输出采集函数权重，没有空间知识则缺省为1
        self.SK_func1 = SK1
        self.SK_func2 = SK2
        self.punish = punish
        self.pbounds = pbounds
        # **

        self.xi = xi

        self._iters_counter = 0

        if kinds not in ['ucb', 'ei', 'poi', 'ud', 'ud_multiobj', 'multi_nsga2']:
            print("必须有'ucb', 'ei', 'poi', 'ud', 'ud_multiobj', 'multi_nsga2'之一")
        else:
            self.kind = kinds

    def update_params(self):
        self._iters_counter += 1

        # if self._kappa_decay < 1 and self._iters_counter > self._kappa_decay_delay:
        #     self.kappa *= self._kappa_decay

    # def utility(self, x, gp_obj1, gp_obj2, gp_obj3, y_max, iteration):
    def utility(self, x, gp_obj1, gp_obj2, iteration):
        # ******空间型知识
        # if self.SK_func == 1:
        #     weight = 1  # 没有输入SK-func,SK-func缺省为1，此时权重为1，对采集函数没有影响
        if self.kind == 'multi_nsga2':
            return self._multi_nsga2(x, gp_obj1, gp_obj2)
        if self.kind == 'ud_multiobj':  # 降低不确定性的采集函数
            return self.ud_multiobj(x, gp_obj1, gp_obj2, iteration)
        # **

    # ******降低不确定性的采集函数ud
    @staticmethod
    def _ud(x, gp):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)
        return std

    def ud_multiobj(self, x, gp_obj1, gp_obj2, iteration):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std_obj1 = gp_obj1.predict(x, return_std=True)
            mean, std_obj2 = gp_obj2.predict(x, return_std=True)

            if iteration <= 30:
                w_obj1 = 0.5
                w_obj2 = 0.3

            elif iteration < 80:
                w_obj1 = 0.5
                w_obj2 = 0.2

            else:
                w_obj1 = 0.2
                w_obj2 = 0.2

            x1 = x[:, 0]
            x2 = x[:, 1]

            new_k1 = []
            for i in range(x1.shape[0]):
                k1 = self.SK_func1(x1[i], x2[i])
                new_k1.append(float(w_obj1 * k1 * std_obj1[i]))

                # punish1 = float(self.punish(x1[i], x2[i]))
                # new_k1 = new_k1 * punish1

            new_k2 = []
            for j in range(x2.shape[0]):
                k2 = self.SK_func2(x1[i], x2[i])
                new_k2.append(float(w_obj2 * k2 * std_obj2[i]))

            new_k1 = np.array(new_k1)
            new_k2 = np.array(new_k2)

        '''应该归一化，还没有归一化'''
        return new_k1 + new_k2

    def _multi_nsga2(self, x, gp_obj1, gp_obj2):

        x = np.array([x]).reshape(-1, 1).T
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x1 = np.squeeze(x[:, 0])
            x2 = np.squeeze(x[:, 1])
            new_k1 = float(self.SK_func1(x1, x2))
            # punish1 = float(self.punish(x1, x2))
            # new_k1 = know1 * punish1
            new_k2 = float(self.SK_func2(x1, x2))

            std_obj1 = float(gp_obj1.predict(x, return_std=True)[1])
            std_obj2 = float(gp_obj2.predict(x, return_std=True)[1])

            f1_ac = float(new_k1 * std_obj1)
            f2_ac = new_k2 * std_obj2

        return -1 * f1_ac, -1 * f2_ac


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
