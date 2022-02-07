import json
import math
import numpy as np
from .weights import XavierInitializer, HeNormalInitializer


class ActivationFunction:

    default_weight_init = XavierInitializer

    @staticmethod
    def calc(x, y=None):
        return x

    @staticmethod
    def derivative(x, y=None):
        return np.ones(x.shape)


class AFRelu(ActivationFunction):

    default_weight_init = HeNormalInitializer

    @staticmethod
    def calc(x, y=None):
        return (x > 0) * x

    @staticmethod
    def derivative(x, y=None):
        return (x > 0) * 1


class AFLeakyRelu(ActivationFunction):

    default_weight_init = HeNormalInitializer
    alpha = 0.001

    @staticmethod
    def calc(x, y=None):
        return (x > 0) * x + (x <= 0) * AFLeakyRelu.alpha * x

    @staticmethod
    def derivative(x, y=None):
        return (x > 0) * 1 + (x <= 0) * AFLeakyRelu.alpha


class AFArcTan(ActivationFunction):

    @staticmethod
    def calc(x, y=None):
        return 2.0 * math.atan(x)/math.pi

    @staticmethod
    def derivative(x, y=None):
        return 2.0 / (math.pi * (1.0 + x*x))


class AFTanh(ActivationFunction):

    @staticmethod
    def calc(x, y=None):
        pz = np.exp(x)
        nz = np.exp(-x)
        return (pz - nz)/(pz + nz)

    @staticmethod
    def derivative(x, y=None):
        return 1 - y * y


class AFSigmoid(ActivationFunction):

    @staticmethod
    def calc(x, y=None):
        px = x >= 0
        pz = np.logical_and(x < 0, np.exp(x))
        nz = np.logical_and(px, np.exp(-x))
        divisor = 1 + (nz + pz)
        dividend = np.logical_and(px, np.ones(
            x.shape)) + pz
        return dividend / divisor

    @staticmethod
    def derivative(x, y=None):
        return y * (1 - y)


class AFSoftmax(ActivationFunction):

    @staticmethod
    def calc(x, y=None):
        shift_x = x - np.max(x)
        exp = np.exp(shift_x)
        return exp / (np.sum(exp) / x.shape[0])

    @staticmethod
    def derivative(x, y=None):
        X = AFSoftmax.calc(x)
        X = X[:, np.newaxis, :]
        m = np.array([p * np.identity(p.shape[1]) - p.T @ p for p in X])
        return m


class ActivationFunctions:

    Tanh = 'tanh'
    Relu = 'relu'
    ArcTan = 'arctan'
    Softmax = 'softmax'
    Sigmoid = 'sigmoid'
    LeakyRelu = 'leaky-relu'

    available = {
        'tanh': AFTanh,
        'relu': AFRelu,
        'arctan': AFArcTan,
        'softmax': AFSoftmax,
        'sigmoid': AFSigmoid,
        'leaky-relu': AFLeakyRelu,
        '__default__': ActivationFunction
    }

    @staticmethod
    def get(name):
        return ActivationFunctions.available.get(name, '__default__')
