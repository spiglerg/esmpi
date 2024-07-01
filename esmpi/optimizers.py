"""
Adapted from https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/optimizers.py
"""

import numpy as np


class Optimizer(object):
    """
    Base class for all optimizers.

    Arguments:
        network: The neural network to be optimized.
    """
    def __init__(self, n_params):
        self.n_params = n_params
        self.t = 0

    def update(self, current_parameters, globalg):
        self.t += 1
        step = self._compute_step(globalg)
        return current_parameters+step

    def _compute_step(self, globalg):
        raise NotImplementedError


class SGD(Optimizer):
    """
    Simple SGD optimizer with momentum, compatible with ES-MPI
    """
    def __init__(self, n_params, lr, momentum=0.9):
        Optimizer.__init__(self, n_params)
        self.v = np.zeros(self.n_params, dtype=np.float32)
        self.lr, self.momentum = lr, momentum

    def _compute_step(self, globalg):
        self.v = self.momentum * self.v + (1. - self.momentum) * globalg
        step = -self.lr * self.v
        return step


class Adam(Optimizer):
    """
    Simple Adam optimizer with momentum, compatible with ES-MPI
    """
    def __init__(self, n_params, lr, beta1=0.9, beta2=0.999, epsilon=1e-08):
        Optimizer.__init__(self, n_params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = np.zeros(self.n_params, dtype=np.float32)
        self.v = np.zeros(self.n_params, dtype=np.float32)

    def _compute_step(self, globalg):
        a = self.lr * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
        self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
        step = -a * self.m / (np.sqrt(self.v) + self.epsilon)
        return step
    