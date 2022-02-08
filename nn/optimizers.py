import math
import numpy as np


class Optimizer:

    name = 'dummy'

    def __init__(self, lr=0.001):
        self.post = False
        self.lr = lr

    def update_weights(self, model):
        return

    def set(self, data):
        self.lr = data.get('learn_rate', 0.001)

    def get(self):
        return {
            'learn_rate': self.lr
        }


class AdamOptimizer(Optimizer):

    name = 'adam'

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08):
        self.post = True
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.epoch = 0

        self.momentum = None
        self.rms = None

    def set_momentum(self, model):
        self.momentum = []
        self.rms = []

        for layer in model.layers:
            if layer.gradients is None:
                self.momentum.append(np.array([]))
                self.rms.append(np.array([]))
                continue

            self.momentum.append(np.zeros(layer.gradients.shape))
            self.rms.append(np.zeros(layer.gradients.shape))

    def update_weights(self, model, l2_rate=None):

        if self.momentum is None:
            self.set_momentum(model)

        self.epoch += 1
        beta1_epoch = self.beta1 ** self.epoch
        beta2_epoch = self.beta2 ** self.epoch
        model_length = len(model.layers)
        for i in range(model_length):

            layer = model.layers[i]

            if layer.gradients is None:
                continue

            self.momentum[i] = self.beta1 * self.momentum[i] + \
                (1 - self.beta1) * layer.gradients

            self.rms[i] = self.beta2 * self.rms[i] + \
                (1 - self.beta2) * layer.gradients * layer.gradients

            momentum_hat = self.momentum[i] / (1 - beta1_epoch)
            rms_hat = self.rms[i] / (1 - beta2_epoch)
            delta = momentum_hat / (np.sqrt(rms_hat) + self.epsilon)

            if l2_rate is not None:
                delta = delta + l2_rate * layer.weights

            layer.weights = layer.weights - self.lr * delta

        return

    def set(self, data):
        self.lr = data.get('learn_rate', 0.001)
        self.beta1 = data.get('beta1', 0.9)
        self.beta2 = data.get('beta2', 0.999)
        self.epsilon = data.get('epsilon', 1e-08)
        _momentum = data.get('momentum', [])
        self.momentum = [np.array(item) for item in _momentum]
        _rms = data.get('rms', [])
        self.rms = [np.array(item) for item in _rms]

    def get(self):
        return {
            'learn_rate': self.lr,
            'momentum': [item.tolist() for item in self.momentum],
            'rms': [item.tolist() for item in self.rms],
            'beta1': self.beta1,
            'beta2': self.beta2,
            'epsilon': self.epsilon,
            'epoch': self.epoch
        }


class Optimizers:
    available = {
        '__default__': AdamOptimizer,
        'dummy': Optimizer,
        'adam': AdamOptimizer
    }

    @staticmethod
    def get(name):
        return Optimizers.available.get(name, '__default__')
