"""Optimization module"""
import needle as ndl
import numpy as np
import math


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for p in self.params:
            grad = self.u.get(p, 0) * self.momentum + (1 - self.momentum) * (p.grad.data + self.weight_decay * p.data)
            grad = ndl.Tensor(grad, dtype=p.dtype, required_grad=False)
            self.u[p] = grad
            p.data -= self.lr * grad
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t = self.t + 1
        for p in self.params:
            grad = p.grad.data + self.weight_decay * p.data
            mt = self.beta1 * self.m.get(p, 0) + (1 - self.beta1) * grad
            self.m[p] = mt
            vt = self.beta2 * self.v.get(p, 0) + (1 - self.beta2) * grad * grad
            self.v[p] = vt
            beta1t = 1 - self.beta1 ** self.t
            beta2t = 1 - self.beta2 ** self.t
            mt_corr = mt / beta1t
            vt_corr = vt / beta2t
            update = self.lr * mt_corr / (vt_corr ** 0.5 + self.eps)
            update = ndl.Tensor(update, dtype=p.dtype, required_grad=False)
            p.data -= update
        ### END YOUR SOLUTION
