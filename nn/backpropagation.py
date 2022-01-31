import numpy as np


class BackPropagation:

    @staticmethod
    def clear_temp_data(model):
        model.output = None
        model.output_prime = None
        model.delta = None
        model.gradients = None
        model.input_data = None

    @staticmethod
    def fast_forward(model, input_data, label_output_data):

        input_data = model.normalize_to_batch(input_data)
        label_output_data = model.normalize_to_batch(label_output_data)

        data = input_data
        for layer in model.layers:
            data = layer.execute(data, train=True)

        output_layer = model.layers[-1]
        output_layer.delta = model.lf.derivative(data, label_output_data)

        if output_layer.af:
            delta = output_layer.delta
            for i in range(delta.shape[0]):
                delta[i] = np.matmul(output_layer.output_prime[i], delta[i])

    @staticmethod
    def update_weights(model):

        lr = model.optimizer.lr
        next_layer = None
        for layer in reversed(model.layers):

            if layer.weights is None:
                continue

            if layer.delta is not None:
                # REVER ESSA MULTIPLICACAO, shape incompatíveis
                layer.gradients = np.matmul(
                    layer.input_data.T, layer.delta) / layer.input_data.shape[0]

                if not model.optimizer.post:
                    layer.weights = layer.weights - lr * layer.gradients

                next_layer = layer
                continue

            delta = np.matmul(next_layer.delta, next_layer.weights.T[:, :-1])
            # delta = np.matmul(next_layer.delta, next_layer.weights.T[:, 1:])

            if layer.output_prime is not None:
                for i in range(delta.shape[0]):
                    delta[i] = np.matmul(layer.output_prime[i], delta[i])
                # delta = layer.output_prime * delta

            layer.delta = delta
            # REVER ESSA MULTIPLICACAO, shape incompatíveis
            layer.gradients = np.matmul(
                layer.input_data.T, layer.delta) / layer.input_data.shape[0]

            if not model.optimizer.post:
                layer.weights = layer.weights + lr * layer.gradients

            next_layer = layer

    @staticmethod
    def train(model, input_data, label_output_data):

        BackPropagation.fast_forward(model, input_data, label_output_data)

        BackPropagation.update_weights(model)
        if model.optimizer.post:
            model.optimizer.update_weights(model)

        BackPropagation.clear_temp_data(model)
