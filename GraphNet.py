from analisis_red import check_conv
from preprocess import preprocess_my_testing_data
import random
from train import load_red
from convolucional import ConvolucionalNoBias
import numpy as np
from grapher import graph_tensor

test = preprocess_my_testing_data()
instance = random.choice(test)
output = instance["output"]
instance = instance["input"]
path = input("nombre de la red")
red = load_red(path)
print(red)


def graph_net(red, input_):
    counter = 0
    graph_tensor(input_, filepath=f"./NetVisualization/input{counter}")
    for layer in red:
        # if type(layer) == ConvolucionalNoBias:
        # check_conv(layer, input_, save=True, number=counter)
        # counter += 1
        counter += 1
        input_ = layer.forward(input_)
        graph_tensor(input_, filepath=f"./NetVisualization/input{counter}")


print(np.argmax(output), np.argmax(red.forward(instance)))

graph_net(red, instance)
