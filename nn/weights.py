from numpy.random import default_rng
import numpy as np

rng = default_rng()

class Initializer:
    @staticmethod
    def create_weights(weights_shape):
        return np.ones(weights_shape)

class NormalInitializer(Initializer):
    @staticmethod
    def create_weights(weights_shape):
        return rng.standard_normal(weights_shape)

class WeightInitializers:
    available = {
        '__default__': Initializer,
        'ones': Initializer,
        'normal': NormalInitializer
    }

    @staticmethod
    def get(name):        
        return WeightInitializers.available.get(name, '__default__')

