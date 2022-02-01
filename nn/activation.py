import math
import numpy as np


class ActivationFunction:

    @staticmethod
    def calc(x):
        return x

    @staticmethod
    def derivative(x):
        return np.ones(x.shape)


class AFRelu(ActivationFunction):

    @staticmethod
    def calc(x):
        return (x > 0) * x

    @staticmethod
    def derivative(x):
        return (x > 0) * 1


class AFLeakyRelu(ActivationFunction):

    alpha = 0.001

    @staticmethod
    def calc(x):
        return (x > 0) * x + (x <= 0) * AFLeakyRelu.alpha * x

    @staticmethod
    def derivative(x):
        return (x > 0) * 1 + (x <= 0) * AFLeakyRelu.alpha


class AFArcTan(ActivationFunction):

    @staticmethod
    def calc(x):
        return 2.0 * math.atan(x)/math.pi

    @staticmethod
    def derivative(x):
        return 2.0 / (math.pi * (1.0 + x*x))


class AFSigmoid(ActivationFunction):

    @staticmethod
    def calc(x):
        px = x >= 0
        pz = np.logical_and(x < 0, np.exp(x))
        nz = np.logical_and(px, np.exp(-x))
        divisor = 1 + (nz + pz)
        dividend = np.logical_and(px, np.ones(
            x.shape)) + pz
        return dividend / divisor

    @staticmethod
    def derivative(x):
        return AFSigmoid.calc(x) * (1 - AFSigmoid.calc(x))


class AFSoftmax(ActivationFunction):

    @staticmethod
    def calc(x):
        shift_x = x - np.max(x)
        exp = np.exp(shift_x)
        return exp / np.sum(exp)

    @staticmethod
    def derivative(x):
        p = AFSoftmax.calc(x)
        pT = AFSoftmax.calc(x.T)
        return p @ np.identity(p.shape[0]) - pT @ p


class ActivationFunctions:

    Relu = 'relu'
    ArcTan = 'arctan'
    Softmax = 'softmax'
    Sigmoid = 'sigmoid'
    LeakyRelu = 'leaky-relu'

    available = {
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
