from bayes_opt import BayesianOptimization
from bayes_opt.bayesian_optimization import Queue
from bayes_opt.target_space import TargetSpace
from bayes_opt.event import Events, DEFAULT_EVENTS

from Support.modi_util_MO import MOUtil, ensure_rng, acq_max

from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor

import time
import warnings


warnings.filterwarnings("ignore")


# 是否在控制台显示采样日志，若有该函数就不显示，若注释掉就显示采样细节
# def _prime_subscriptions(self):
#     return 0

'''计算误差'''


# def error(iteration, gp, f):
#     a_n = 60
#     ma_n = 20
#     a = np.linspace(1, 33, a_n)
#     ma = np.linspace(0, 10, ma_n)
#     uncer = []
#     err_all1 = 0
#
#     for l in range(a_n):
#         for k in range(ma_n):
#             y_gp, std = gp.predict([[a[l], ma[k]]], return_std=True)
#             y_gp = float(y_gp)
#             std = float(std)
#             y = f(a[l], ma[k])
#             # 整体误差
#             # if np.abs(y) < 0.01:
#             err1 = np.squeeze((y_gp - y) ** 2)
#             err_all1 = err_all1 + err1
#             # else:
#             #     err = np.squeeze(np.abs(y_gp - y) / np.abs(y))
#             # if err < 0.05:
#             #     n = n + 1
#             # if err <= 0.04:
#             #     n1 = n1 + 1
#             # if err <= 0.03:
#             #     n2 = n2 + 1
#             # if err <= 0.02:
#             #     n3 = n3 + 1
#             # wu.append(err)
#             uncer.append(std)
#     wu1 = (err_all1 / (a_n * ma_n)) ** 0.5
#     # confi95 = n / (a_n * ma_n)
#     # confi96 = n1 / (a_n * ma_n)
#     # confi97 = n2 / (a_n * ma_n)
#     # confi98 = n3 / (a_n * ma_n)
#
#     # return [iteration, np.float(np.mean(wu)), confi95, confi96, confi97, confi98,
#     #         np.float(np.mean(uncer))]
#     return [iteration, np.float(wu1), np.float(np.mean(uncer))]


# ----------------------------------------------------


class BO(BayesianOptimization):
    def __init__(self, f1, f2, pbounds, random_state=None, verbose=2,
                 bounds_transformer=None):
        """"""
        self._random_state = ensure_rng(random_state)

        # Data structure containing the function to be optimized, the bounds of
        # its domain, and a record of the evaluations we have done so far
        # 也可以使用一个TargetSpace实例

        # 实例化TargetSpace类，叫做_space，用于对搜索空间按的操作
        self._space = TargetSpace(f1, pbounds, random_state)  # 应付零碎的space其他地方
        self._space_obj1 = TargetSpace(f1, pbounds, random_state)
        self._space_obj2 = TargetSpace(f2, pbounds, random_state)
        # self._space_obj3 = TargetSpace(f3, pbounds, random_state)

        # queue
        self._queue = Queue()  # 初始训练点队列

        # *****迭代过程中的误差列表
        self._wu = []
        self._wu_obj1 = []
        self._wu_obj2 = []
        # self._wu_obj3 = []
        self._f1 = f1
        self._f2 = f2
        # self._f3 = f3
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
        self._gp_obj1 = GaussianProcessRegressor(
            kernel=Matern(nu=1.5, length_scale=length),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=10,
            random_state=self._random_state,
        )
        self._gp_obj2 = GaussianProcessRegressor(
            kernel=Matern(nu=1.5, length_scale=length),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=10,
            random_state=self._random_state,
        )


        self._verbose = verbose
        self._bounds_transformer = bounds_transformer
        if self._bounds_transformer:
            self._bounds_transformer.initialize(self._space)

        super(BayesianOptimization, self).__init__(events=DEFAULT_EVENTS)

    @property
    def space(self):
        return self._space

    @property
    def space_obj1(self):
        return self._space_obj1

    @property
    def space_obj2(self):
        return self._space_obj2

    # @property
    # def space_obj3(self):
    #     return self._space_obj3

    @property
    def max(self):
        return self._space.max()  # 计算y_max,多目标采样暂时不需要改

    @property
    def res(self):
        return self._space.res()

    def register(self, params, target):
        """Expect observation with known target"""
        self._space.register(params, target)
        self.dispatch(Events.OPTIMIZATION_STEP)

    def probe(self, params, lazy=True):
        """Probe target of x"""
        if lazy:
            self._queue.add(params)
        else:
            self._space.probe(params)
            self._space_obj1.probe(params)
            self._space_obj2.probe(params)
            # self._space_obj3.probe(params)
            self.dispatch(Events.OPTIMIZATION_STEP)

    def suggest(self, utility_function):
        """Most promissing point to probe next"""
        if len(self._space) == 0:
            '''随机生成初始点'''
            return self._space.array_to_params(self._space.random_sample())

        # Sklearn's GP throws a large number of warnings at times, but
        # we don't really need to see them here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._gp_obj1.fit(self._space_obj1.params, self._space_obj1.target)
            self._gp_obj2.fit(self._space_obj2.params, self._space_obj2.target)
            # self._gp_obj3.fit(self._space_obj3.params, self._space_obj3.target)
            # self._gp.fit(self._space.params, self._space.target)

            # 迭代后计算误差
            # x1 = np.arange(1.0, 33.5, 0.5)  # 作图
            # x2 = np.arange(0.0, 10.2, 0.2)
            # if self.iteration > 0 and self.iteration % 1 == 0:
            #     '''两个变量，是攻角和马赫数'''
            #     # 全弹法向力误差
            #     self._wu_obj1.append(error(iteration=self.iteration, gp=self._gp_obj1, f=self._f1))
            #     self._wu_obj2.append(error(iteration=self.iteration, gp=self._gp_obj2, f=self._f2))
            #     # self._wu_obj3.append(error(iteration=self.iteration, gp=self._gp_obj3, f=self._f3))
            #
            #     if (self.iteration - 5) % 20 == 0:
            #         plot_gp(self._gp_obj1, x1, x2, self._space_obj1.params, self._space_obj1.target, title='fa')
            #         plot_gp(self._gp_obj2, x1, x2, self._space_obj2.params, self._space_obj2.target, title='yaxin')
            #         # plot_gp(self._gp_obj3, x1, x2, self._space_obj3.params, self._space_obj3.target)
            #         plot_gp_std(self._gp_obj1, x1, x2, title='fa_std')
            #         plot_gp_std(self._gp_obj2, x1, x2, title='yaxin_std')
                    # plot_gp_std(self._gp_obj3, x1, x2)

        # Finding argmax of the acquisition function.
        suggestion = acq_max(
            ac=utility_function.utility,  # 采集函数
            gp_obj1=self._gp_obj1,
            gp_obj2=self._gp_obj2,
            bounds=self._space_obj1.bounds,
            random_state=self._random_state,
            iteration=self.iteration
        )
        # suggestion = acq_max1(ac=utility_function.utility, gp_obj1=self._gp_obj1, gp_obj2=self._gp_obj2)
        return self._space.array_to_params(suggestion)

    def prime_queue(self, init_points, x_train):
        """Make sure there's something in the queue at the very beginning."""
        if self._queue.empty and self._space.empty:
            init_points = max(init_points, 1)

        # ******可以只指定初始点数量init_points，也可以指定初始点x_train
        # if x_train == []:
        #     for _ in range(init_points):
        #         self._queue.add(self._space.random_sample())
        # else:
        #     for i in range(len(x_train)):
        #         self._queue.add(np.array(x_train[i]))
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
                 punish=1,
                 SK1=1,
                 SK2=1,
                 x_train=None,
                 # **
                 **gp_params):
        # maximize()函数是贝叶斯优化过程的主循环

        """Mazimize your function"""
        if x_train is None:
            x_train = []
        self._prime_subscriptions()
        self.dispatch(Events.OPTIMIZATION_START)
        self.prime_queue(init_points, x_train)  # 初始训练点函数：x_train是自己指定的初始点，如果没有指定，根据init_points函数随机生成
        self.set_gp_params(**gp_params)
        self.SK1 = SK1
        self.SK2 = SK2
        self.punish = punish

        util = MOUtil(kinds=acq,
                      # kappa=kappa,
                      xi=xi,
                      # kappa_decay=kappa_decay,
                      # kappa_decay_delay=kappa_decay_delay,
                      # ******
                      SK1=SK1,
                      SK2=SK2,
                      punish=punish,
                      pbounds=self._pbounds,
                      # **
                      )
        self.iteration = 0

        while not self._queue.empty or self.iteration < n_iter:
            # ******每次迭代的初始时间
            t0 = time.time()
            # **
            try:
                x_probe = next(self._queue)  # 根据_queue初始点队列采样
            except StopIteration:
                # ----------------------------
                # '''大于50次迭代，再进行边界惩罚'''
                # if self.iteration < 1500:
                #     util.SK_func = 1
                # else:
                #     util.SK_func = self.SK
                # ----------------------------
                util.update_params()
                x_probe = self.suggest(util)  # 使用suggest()函数确定新采样点

                self.iteration += 1

            self.probe(x_probe, lazy=False)  # 根据新采样点采样
            # ******每次迭代的时间间隔
            self._iter_time.append(time.time() - t0)
            # **

            if self._bounds_transformer:
                self.set_bounds(
                    self._bounds_transformer.transform(self._space))

        self.dispatch(Events.OPTIMIZATION_END)

    def set_bounds(self, new_bounds):
        """
        A method that allows changing the lower and upper searching bounds

        Parameters
        ----------
        new_bounds : dict
            A dictionary with the parameter name and its new bounds
        """
        self._space.set_bounds(new_bounds)

    def set_gp_params(self, **params):
        self._gp_obj1.set_params(**params)
        self._gp_obj2.set_params(**params)
        # self._gp_obj3.set_params(**params)
