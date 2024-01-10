import matplotlib.pyplot as plt
import train
from densa import Densa
from Flatten import Flatten
from grapher import graph_tensor

# Only to be called on Network with only fully conected layer ((784,1)->(10,1))
path = input("nombre de la red")
red = train.load_red(path)

# densa = red[1]
# for i, peso in enumerate(densa.pesos):
#    plt.imshow(peso.reshape((28, 28)))
#    plt.savefig(f"./pesos/peso{i}")


def GraphBoundaryDense(red, BoundaryPosition):
    dense = red[BoundaryPosition]
    flatten = red[BoundaryPosition - 1]

    if type(flatten) != Flatten:
        raise ValueError("the layer before boundary isnt a Flatten layer")

    if type(dense) != Densa:
        raise ValueError("the layer at boundary isnt a Dense layer")

    pesos = dense.pesos
    flattened_input_size, output_size = pesos.shape
    input_shape = flatten.input_shape
    for i, peso in enumerate(pesos):
        im = peso.reshape(input_shape)
        graph_tensor(im, f"./pesos/peso{i}")


pos = int(input("posicion de la densa limite"))

GraphBoundaryDense(red, pos)
