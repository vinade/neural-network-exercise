from numpy.random import default_rng
import numpy as np
from math import sqrt

rng = default_rng()


class Initializer:
    @staticmethod
    def create_weights(weights_shape):
        return np.ones(weights_shape)


class NormalInitializer(Initializer):
    @staticmethod
    def create_weights(weights_shape):
        return rng.standard_normal(weights_shape)


class HeNormalInitializer(Initializer):
    @staticmethod
    def create_weights(weights_shape):
        wvar = 2 / weights_shape[0]
        upper = sqrt(wvar)
        lower = -upper

        return lower + rng.standard_normal(weights_shape) * (upper - lower)


class XavierInitializer(Initializer):
    @staticmethod
    def create_weights(weights_shape):
        print(weights_shape)
        wvar = 2 / (weights_shape[0] + weights_shape[1])
        upper = sqrt(wvar)
        lower = -upper

        return lower + rng.standard_normal(weights_shape) * (upper - lower)


class WeightInitializers:
    available = {
        '__default__': XavierInitializer,
        'ones': Initializer,
        'normal': NormalInitializer,
        'xavier': XavierInitializer,
        'he-normal': HeNormalInitializer,
    }

    @staticmethod
    def get(name):
        return WeightInitializers.available.get(name, '__default__')
