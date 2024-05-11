import preprocess
import train
import numpy as np
import matplotlib.pyplot as plt
from auto_encoder_utils import mean_extra_grad_func, std_extra_grad_func, mul, dev_mul
from auto_encoder_utils import combined_loss, dev_combined_loss
from graph_in_terminal import graph

my_train = preprocess.preprocess_my_data()

path = input("nombre de la red?\n")

autoencoder = train.load_red(path)

inp = my_train[23]["input"]
image = autoencoder.forward(inp)[0]  # poner el otro [0] solo en autoencoder_pred
graph(image)
plt.imshow(image)
plt.savefig("out")

print()

graph(inp[0])
plt.imshow(inp[0])
plt.savefig("in")
