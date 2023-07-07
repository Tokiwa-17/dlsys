"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []




class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.add_bias = bias

        ### BEGIN YOUR SOLUTION
        self.weight = init.kaiming_uniform(self.in_features, self.out_features, "relu")
        self.bias = Tensor(init.kaiming_uniform(self.out_features, 1).numpy().reshape((1, self.out_features)))
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        x_out = X @ self.weight
        if self.add_bias:
            return x_out + self.bias.broadcast_to(x_out.shape)
        return x_out
        ### END YOUR SOLUTION



class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        return ops.reshape(X, (X.shape[0], -1))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        out = x
        for module in self.modules:
            out = module(out)
        return out
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        batch_size, num_classes = logits.shape
        return 1 / batch_size * (ops.logsumexp(logits, axes=1) - (logits * init.one_hot(logits.shape[-1], y)).sum(axes=1)).sum()
        ### END YOUR SOLUTION



class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(self.dim))
        self.bias = Parameter(init.zeros(self.dim))
        self.running_mean = Parameter(init.zeros(self.dim))
        self.running_var = Parameter(init.ones(self.dim))
        ### END YOUR SOLUTION


    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size = x.shape[0]
        if self.training:
            new_avg = x.sum(axes=0) / x.shape[0]
            broadcast_avg = ops.broadcast_to(ops.reshape(new_avg, (1, -1)), x.shape)
            new_var = ((x - broadcast_avg) ** 2).sum(axes=0) / x.shape[0]
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * new_avg
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * new_var
            broadcast_var = ops.broadcast_to(ops.reshape(new_var, (1, -1)), x.shape)
            broadcast_weight = ops.broadcast_to(ops.reshape(self.weight, (1, -1)), x.shape)
            broadcast_bias = ops.broadcast_to(ops.reshape(self.bias, (1, -1)), x.shape)
            return broadcast_weight * (x - broadcast_avg) / ((broadcast_var + self.eps) ** 0.5) + broadcast_bias
        else:
            broadcast_weight = ops.broadcast_to(ops.reshape(self.weight, (1, -1)), x.shape)
            broadcast_bias = ops.broadcast_to(ops.reshape(self.bias, (1, -1)), x.shape)
            broadcast_avg = ops.broadcast_to(ops.reshape(self.running_mean, (1, -1), x.shape))
            broadcast_var = ops.broadcast_to(ops.reshape(self.running_var, (1, -1), x.shape))
            return broadcast_weight * (x - broadcast_avg) / ((broadcast_var + self.eps) ** 0.5) + broadcast_bias
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim))
        self.bias = Parameter(init.zeros(dim))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        avg = ops.broadcast_to(ops.reshape(x.sum(axes=1) / x.shape[-1], (x.shape[0], 1)), x.shape)
        var = ops.broadcast_to(ops.reshape(((x - avg) ** 2).sum(axes=1) / x.shape[1], (x.shape[0], 1)), x.shape)
        return ops.broadcast_to(self.weight, x.shape) * (x - avg) / ((var + self.eps) ** 0.5) + ops.broadcast_to(self.bias, x.shape)
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION



