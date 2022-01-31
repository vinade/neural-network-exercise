import numpy as np
from nn import Model, Layer
from nn.activation import ActivationFunctions as AF
from nn.loss import LossFunctions as LF
from numpy.random import default_rng
rng = default_rng()


def categorize_labels(labels, categories):
    normalized_labels = np.zeros((labels.size, categories))

    for i in range(labels.size):
        normalized_labels[i][int(labels[i])] = 1
    return normalized_labels


def select_batch(data, batch_size):
    samples = rng.choice(data, batch_size)
    train_data = samples[:, 1:]
    labels = samples[:, 0]
    labels = categorize_labels(labels, 10)
    return train_data, labels


def load_mnist():
    data_path = "data/mnist/"
    mnist_data = np.loadtxt(data_path + "train.csv", delimiter=",")
    train_data = mnist_data[:, 1:]

    mu = train_data.mean(axis=0)
    std = train_data.mean(axis=0)
    np.place(std, std == 0, 1)
    mnist_data[:, 1:] = (train_data - mu) / std

    return mnist_data


print("carregando mnist...")
image_size = 28
image_pixels = image_size * image_size
no_of_different_labels = 10
mnist_data = load_mnist()

print("criando modelo...")
layers = [
    Layer(32, AF.Sigmoid),
]

model1 = Model(
    image_pixels, no_of_different_labels, layers, loss_function=LF.CategoricalCrossEntropy, optimizer='adam')
model1.build()

print("Treino iniciado:")
batch_size = 32
for i in range(50):
    input_data, label_output_data = select_batch(mnist_data, batch_size)
    model1.train(input_data, label_output_data, 100)
    output_data = model1.calc_loss(input_data, label_output_data)
    print(f'Erro: {output_data}')

# model1.save('./tmp/model1_adam.json')
