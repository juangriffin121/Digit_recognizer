import numpy as np
from train import load_red
import costos
import matplotlib.pyplot as plt
import copy
import preprocess
import random

# from GraphNet import graph_net

# test = preprocess.preprocess_my_testing_data()
# instance = random.choice(test)
# image = instance["input"]
# image = image.astype(float)


def gauss(x):
    return np.exp(-(x**2))


inp = np.zeros((1, 28, 28))
for i in range(inp.shape[1]):
    for j in range(inp.shape[2]):
        inp[0][i][j] = gauss((i - inp.shape[1] / 2) / 7) * gauss(
            (j - inp.shape[2] / 2) / 5
        )

inp *= np.random.uniform(0.0, 255.0, size=(1, 28, 28))


def DeepDream(
    red, output_correcto, iteraciones=50, dt=10000, input_=None, Temp=0.00001
):
    if input_ is not None:
        pass
    else:
        input_shape = (1, 28, 28)
        # input_ = np.random.uniform(0.0, 10.0, size=input_shape)
        input_ = np.zeros(input_shape) + 100
    plt.imshow(input_[0])
    plt.savefig(f"./dreams/input{digit}")
    for i in range(iteraciones):
        output = red.forward(input_)
        error = costos.softmax_cross_entropy(output, output_correcto)
        grad_output = costos.dev_softmax_cross_entropy(output, output_correcto)
        grad_input = red.backward(grad_output, 0.0)
        grad_input += Temp * input_ * np.random.randn(*input_.shape)
        print(error)
        print(np.max(grad_input) * dt)
        print(np.min(grad_input) * dt)
        input_ -= grad_input * dt
        print(i, np.argmax(output_correcto), np.argmax(output))
    #    graph_net(red, input_)
    return input_


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


path = input("nombre de la red")

red = load_red(path)
print(red)

for digit in range(10):
    image = copy.deepcopy(inp)
    output_correcto = np.zeros((10, 1))
    output_correcto[digit] = 1

    dream = DeepDream(red, output_correcto, input_=image)

    im = sigmoid(dream[0] / np.max(dream[0]))

    plt.imshow(im)
    plt.savefig(f"./dreams/dream{digit}")
