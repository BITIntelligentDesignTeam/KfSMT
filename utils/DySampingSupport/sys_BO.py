from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor

from bayes_opt import BayesianOptimization
from bayes_opt.bayesian_optimization import Queue
from bayes_opt.target_space import TargetSpace
from bayes_opt.event import Events, DEFAULT_EVENTS
from bayes_opt.logger import _get_default_logger

from .sys_util import sysUtil, acq_max, ensure_rng

import numpy as np
import time
import warnings


class sysBO(BayesianOptimization):
    def __init__(self, f, pbounds, random_state=None, verbose=2,
                 bounds_transformer=None):
        """"""

        self._random_state = ensure_rng(random_state)

        # 数据结构，包含要优化的函数，域的边界，以及到目前为止我们所做的评估的记录
        self._space = TargetSpace(f, pbounds, random_state)

        # queue
        self._queue = Queue()

        # *****迭代过程中的误差列表
        self._wu = []
        self._f = f
        # **

        # *****pbounds需要传入util，为了融合空间知识
        self._pbounds = pbounds
        # **

        # *****每次迭代的时间
        self._iter_time = []
        # **

        # 各向异性length-scale随着变量个数自动生成初始值
        length = []
        for i in range(pbounds.__len__()):
            length.append(1)

        # Internal GP regressor
        self._gp = GaussianProcessRegressor(
            kernel=Matern(nu=1.5, length_scale=length),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=10,
            random_state=None
        )

        self._verbose = verbose
        self._bounds_transformer = bounds_transformer
        if self._bounds_transformer:
            self._bounds_transformer.initialize(self._space)

        super(BayesianOptimization, self).__init__(events=DEFAULT_EVENTS)

    # 是否在控制台显示采样日志，若有该函数就不显示，若注释掉就显示采样细节
    # def _prime_subscriptions(self):
    #     return 0

    def suggest(self, utility_function):
        """Most promissing point to probe next"""
        if len(self._space) == 0:
            return self._space.array_to_params(self._space.random_sample())

        # Sklearn's GP throws a large number of warnings at times, but
        # we don't really need to see them here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._gp.fit(self._space.params, self._space.target)

        # Finding argmax of the acquisition function.
        suggestion = acq_max(
            ac=utility_function.utility,
            gp=self._gp,
            y_max=self._space.target.max(),
            bounds=self._space.bounds,
            random_state=self._random_state
        )

        return self._space.array_to_params(suggestion)

    def prime_queue(self, init_points, x_train):
        """Make sure there's something in the queue at the very beginning."""
        if self._queue.empty and self._space.empty:
            init_points = max(init_points, 1)

        # ******可以只指定初始点数量init_points，也可以指定初始点x_train
        if x_train == []:
            for _ in range(init_points):
                self._queue.add(self._space.random_sample())
        else:
            for i in range(len(x_train)):
                self._queue.add(np.array(x_train[i]))
        # ***

    def maximize(self,
                 init_points=5,
                 n_iter=25,
                 acq='ucb',
                 kappa=2.576,
                 kappa_decay=1,
                 kappa_decay_delay=0,
                 xi=0.0,
                 # ******
                 SK=1,
                 punish=1,
                 x_train=None,
                 # **
                 **gp_params):
        """Mazimize your function"""
        if x_train is None:
            x_train = []

        self._prime_subscriptions()
        self.dispatch(Events.OPTIMIZATION_START)
        self.prime_queue(init_points, x_train)
        self.set_gp_params(**gp_params)

        self.iteration = 0

        util = sysUtil(pbounds=self._pbounds, iteration=self.iteration, kinds=acq, SK=SK, punish=punish)


        while not self._queue.empty or self.iteration < n_iter:
            # ******每次迭代的初始时间
            t0 = time.time()
            # **
            try:
                x_probe = next(self._queue)
            except StopIteration:
                util.update_params()
                x_probe = self.suggest(util)
                self.iteration += 1

            self.probe(x_probe, lazy=False)
            # ******每次迭代的时间间隔
            self._iter_time.append(time.time() - t0)
            # **

            if self._bounds_transformer:
                self.set_bounds(
                    self._bounds_transformer.transform(self._space))

        self.dispatch(Events.OPTIMIZATION_END)

    def set_gp_params(self, **params):
        self._gp.set_params(**params)

    @property
    def wu(self):
        return self._wu