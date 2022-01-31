import numpy as np
from .activation import ActivationFunctions as AF


epsilon = 1e-08


class LossFunction:

    output_af = None

    @staticmethod
    def derivative(y, Y):
        return np.ones(y.shape)

    @staticmethod
    def calc(y, Y):
        return np.sum(Y - y) / Y.size

    @staticmethod
    def calc_per_output(y, Y):
        return Y - y


class LossMSE(LossFunction):

    @staticmethod
    def derivative(y, Y):
        return (Y - y) * -2

    @staticmethod
    def calc(y, Y):
        se = (Y - y) ** 2
        return np.sum(se) / Y.size

    @staticmethod
    def calc_per_output(y, Y):
        return (Y - y) ** 2


class LossBinaryCrossEntropy(LossFunction):

    output_af = AF.Sigmoid

    @staticmethod
    def derivative(y, Y):
        complement_y = (1 - y)
        left_part = (Y * complement_y)
        right_part = (y * (1 - Y))
        divisor = (y * complement_y) + epsilon
        return (left_part - right_part)/divisor

    @staticmethod
    def calc(y, Y):
        loss = Y * np.log(y + epsilon) + (1 - Y) * np.log(1 - y + epsilon)
        return - np.sum(loss) / Y.size

    @staticmethod
    def calc_per_output(y, Y):
        return (Y - y) ** 2

    @staticmethod
    def sigmoid(y):
        return y

    @staticmethod
    def sigmoid_prime(y):
        return y


class LossCategoricalCrossEntropy(LossFunction):

    output_af = AF.Softmax

    @staticmethod
    def derivative(y, Y):
        return - y / (Y + epsilon)

    @staticmethod
    def calc(y, Y):
        loss = Y * np.log(y + epsilon)
        return - np.sum(loss) / Y.size

    @staticmethod
    def calc_per_output(y, Y):
        return - Y * np.log(y + epsilon)


class LossFunctions:

    MSE = 'mse'
    CategoricalCrossEntropy = 'categorical-cross-entropy'
    BinaryCrossEntropy = 'binary-cross-entropy'

    available = {
        'mse': LossMSE,
        'categorical-cross-entropy': LossCategoricalCrossEntropy,
        'binary-cross-entropy': LossBinaryCrossEntropy,
        '__default__': LossFunction
    }

    @staticmethod
    def get(name):
        return LossFunctions.available.get(name, '__default__')
