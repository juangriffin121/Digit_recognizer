import os

import matplotlib.pyplot as plt
import numpy as np

import train
from densa import Densa, DensaNoBias
from Flatten import Flatten

# Only to be called on Network with only fully conected layer ((784,1)->(10,1))
path = input("nombre de la red")
red = train.load_red(path)
print(red)


def GraphBoundaryDense(red, BoundaryPosition, filepath="./pesos/pesos"):
    dense = red[BoundaryPosition]
    flatten = red[BoundaryPosition - 1]

    if type(flatten) != Flatten:
        raise ValueError("the layer before boundary isnt a Flatten layer")

    if type(dense) not in (Densa, DensaNoBias):
        raise ValueError("the layer at boundary isnt a Dense layer")

    pesos = dense.pesos
    n = len(pesos)

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # One row of the weight matrix per subplot, all stacked in a single column
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(3, 3 * n))
    for i, (peso, ax) in enumerate(zip(pesos, axes)):
        # Diverging colormap centred on 0: red = positive weight, blue = negative.
        # Symmetric limits per image so 0 is always the neutral (white) colour.
        lim = np.max(np.abs(peso))
        ax.imshow(peso.reshape((28, 28)), cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_title(f"digit {i}", fontsize=10)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)
    print(f"guardado en {filepath}.png")


pos = int(input("posicion de la densa limite"))

GraphBoundaryDense(red, pos)
