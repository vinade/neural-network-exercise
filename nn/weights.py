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


class XavierInitializer(Initializer):
    @staticmethod
    def create_weights(weights_shape):
        sqrt_n = sqrt(weights_shape[0])
        upper = 1 / sqrt_n
        lower = -upper

        return lower + rng.standard_normal(weights_shape) * (upper - lower)


class WeightInitializers:
    available = {
        '__default__': XavierInitializer,
        'ones': Initializer,
        'normal': NormalInitializer,
        'xavier': XavierInitializer,
    }

    @staticmethod
    def get(name):
        return WeightInitializers.available.get(name, '__default__')
