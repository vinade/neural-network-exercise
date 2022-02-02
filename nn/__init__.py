import json
import numpy as np
from .activation import ActivationFunctions, ActivationFunction
from .loss import LossFunctions
from .optimizers import Optimizers
from .weights import WeightInitializers, Initializer
from .backpropagation import BackPropagation

DEFAULT_LOSS_FUNCTION = 'mse'
DEFAULT_ACTIVATION_FUNCTION = None


class Layer:

    TYPE_INPUT = 1
    TYPE_OUTPUT = 2

    def __init__(self, n, activation_function=None, weights=None, layer_type=None, weights_initializer=None):
        self.size = n
        self.weights = None
        self.layer_type = layer_type
        self.activation_function = activation_function
        self.af = None
        self.output = None
        self.output_prime = None
        self.delta = None
        self.gradients = None
        self.input_data = None

        if self.activation_function:
            self.af = ActivationFunctions.get(self.activation_function)

        self.set_weights_initializer(weights_initializer)

        if weights:
            self.update_weights(weights)

    def set_weights_initializer(self, weights_initializer=None):
        if weights_initializer is not None:
            if isinstance(weights_initializer, str):
                weights_initializer = WeightInitializers.get(
                    weights_initializer)
            elif isinstance(weights_initializer, Initializer):
                weights_initializer = weights_initializer
            else:
                weights_initializer = None

        if weights_initializer is None:
            activation_f = self.af or ActivationFunction
            weights_initializer = activation_f.default_weight_init

        self.weights_initializer = weights_initializer

    def update_weights(self, weights):
        self.weights = np.array(weights)

    def execute(self, input_data, train=False):

        if self.layer_type == Layer.TYPE_INPUT:
            if train:
                self.input_data = input_data
            return input_data

        modified_input = np.c_[input_data, np.ones(input_data.shape[0])]
        batch_A = np.matmul(modified_input, self.weights)

        if train:
            self.input_data = modified_input

        if not self.af:

            if train:
                self.output = batch_A
                self.output_prime = np.ones(batch_A.shape)

            return batch_A

        batch_Z = self.af.calc(batch_A)

        if train:
            self.output = batch_Z
            self.output_prime = self.af.derivative(batch_A, batch_Z)

        return batch_Z

    def __str__(self):
        weights_shape = self.weights.shape if self.weights is not None else None
        return f'''
-----------------------------------------------------------------------
neurons: {self.size}\tactivation: {self.activation_function}\t{weights_shape}
'''


class Model:

    def __init__(self, input_size, output_size, layers=[], loss_function='mse', optimizer='adam'):
        self.input_size = input_size
        self.output_size = output_size
        self.layers = layers
        self.loss_function = loss_function
        self.built = False
        self.lf = LossFunctions.get(self.loss_function)

        if isinstance(optimizer, str):
            self.optimizer = Optimizers.get(optimizer)()
        else:
            self.optimizer = optimizer

    def update_optimizer(self, optmizer_data):
        if optmizer_data:
            self.optimizer.set(optmizer_data)

    def save(self, filepath):

        json_data = {
            'input_size': self.input_size,
            'output_size': self.output_size,
            'loss_function': self.loss_function,
            'optimizer': {
                'name': self.optimizer.name,
                'data': self.optimizer.get()
            },
            'layers': []
        }

        for layer in self.layers:
            json_data['layers'].append({
                'type': layer.layer_type,
                'size': layer.size,
                'activation_function': layer.activation_function,
                'weights': layer.weights.tolist() if layer.weights is not None else None
            })

        with open(filepath, 'w') as f:
            json.dump(json_data, f, sort_keys=True, indent=4)

    def build(self):

        if self.built:
            return

        input_layer = Layer(self.input_size, layer_type=Layer.TYPE_INPUT)
        output_layer = Layer(
            self.output_size, activation_function=self.lf.output_af, layer_type=Layer.TYPE_OUTPUT)
        self.layers = [input_layer] + self.layers + [output_layer]

        previous_size = 0
        for layer in self.layers:

            if previous_size:
                weight_array_shape = (previous_size + 1, layer.size)
                layer.weights = layer.weights_initializer.create_weights(
                    weight_array_shape)

            previous_size = layer.size

        for layer in self.layers:
            print(layer)

        self.built = True

    @staticmethod
    def normalize_to_batch(data):
        data_shape = data.shape
        if len(data_shape) == 1:
            data = np.array([data])
        return data

    def fast_forward(self, input_data):
        data = self.normalize_to_batch(input_data)
        for layer in self.layers:
            data = layer.execute(data)
        return data

    def calc_loss(self, input_data, label_output_data):
        label_output_data = self.normalize_to_batch(label_output_data)
        output_data = self.fast_forward(input_data)
        return self.lf.calc(output_data, label_output_data)

    def train(self, input_data, label_output_data, iterations=1, l2_rate=None):
        for i in range(iterations):
            BackPropagation.train(
                self, input_data, label_output_data, l2_rate=None)

    @staticmethod
    def load(filepath):
        ''' load a model '''
        f = open(filepath)
        file_data = json.load(f)

        layers = []
        layers_data = file_data.get('layers', [])
        for layer_data in layers_data:
            layer_type = layer_data.get('type')
            size = layer_data.get('size')
            activation_function = layer_data.get(
                'activation_function', DEFAULT_ACTIVATION_FUNCTION)
            weights = layer_data.get('weights', [])
            layer = Layer(size, activation_function, weights, layer_type)
            layers.append(layer)

        input_size = file_data.get('input_size')
        output_size = file_data.get('output_size')
        loss_function = file_data.get('loss_function', DEFAULT_LOSS_FUNCTION)

        optimizer = file_data.get('optimizer', {})
        optimizer_name = optimizer.get('name', 'adam')
        optimizer_data = optimizer.get('data', {})

        model = Model(input_size, output_size, layers,
                      loss_function, optimizer_name)
        model.update_optimizer(optimizer_data)

        for layer in model.layers:
            print(layer)

        return model
