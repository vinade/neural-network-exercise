import numpy as np


class BackPropagation:

    @staticmethod
    def clear_temp_data(model):
        for layer in model.layers:
            layer.output = None
            layer.output_prime = None
            layer.delta = None
            layer.gradients = None
            layer.input_data = None

    @staticmethod
    def fast_forward(model, input_data, label_output_data):

        input_data = model.normalize_to_batch(input_data)
        label_output_data = model.normalize_to_batch(label_output_data)

        data = input_data
        for layer in model.layers:
            data = layer.execute(data, train=True)

        output_layer = model.layers[-1]
        output_layer.delta = model.lf.derivative(
            output_layer.output, label_output_data)

    @staticmethod
    def update_weights(lr, layer, l2_rate=None):
        delta_w = layer.gradients

        if delta_w is not None:
            delta_w += l2_rate * layer.weights

        layer.weights = layer.weights + lr * delta_w

    @staticmethod
    def optmize(model, l2_rate=None):

        lr = model.optimizer.lr
        next_layer = None
        for layer in reversed(model.layers):

            if layer.weights is None:
                continue

            if layer.delta is not None:
                layer.gradients = (layer.input_data.T @
                                   layer.delta) / layer.input_data.shape[0]

                if not model.optimizer.post:
                    BackPropagation.update_weights(lr, layer, l2_rate)

                next_layer = layer
                continue

            delta = next_layer.delta @ next_layer.weights.T[:, :-1]

            if layer.output_prime is not None:
                delta = layer.output_prime * delta

            layer.delta = delta
            layer.gradients = (layer.input_data.T @
                               layer.delta) / layer.input_data.shape[0]

            if not model.optimizer.post:
                BackPropagation.update_weights(lr, layer, l2_rate)

            next_layer = layer

    @staticmethod
    def train(model, input_data, label_output_data, l2_rate=None):

        BackPropagation.fast_forward(model, input_data, label_output_data)

        BackPropagation.optmize(model, l2_rate)
        if model.optimizer.post:
            model.optimizer.update_weights(model, l2_rate)

        BackPropagation.clear_temp_data(model)
