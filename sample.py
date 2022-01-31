import numpy as np
from nn import Model, Layer
from nn.activation import ActivationFunctions as AF
from nn.loss import LossFunctions as LF

layers = [
    Layer(4, AF.Relu),
    Layer(3, AF.Relu),
    Layer(5, AF.LeakyRelu),
]

model1 = Model(
    3, 1, layers, loss_function=LF.BinaryCrossEntropy, optimizer='adam')
model1.build()

input_data = np.array([[2, 4, 6], [3, 9, 27], [22, 26, 4], [5, 25, 125]])
label_output_data = np.array([[1], [0], [1], [0]])

for i in range(50):
    model1.train(input_data, label_output_data, 100)
    output_data = model1.calc_loss(input_data, label_output_data)
    print(f'Erro: {output_data}')

model1.save('./tmp/model1_adam.json')
