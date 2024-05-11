import numpy as np
from neuralNetwork import NeuralNetwork
from densa import Densa
from branchedNetwork import BranchedNetwork
from activaciones import Sigmoid, Leacky_Relu
from convolucional import ConvolucionalNoBias, Convolucional
from max_pooling import Max_Pooling
from reshape import Reshape
from Flatten import Flatten
import preprocess
import train
from addLoss import AddLoss
from upSampling import UpSampling
from sample import Sample
from activacion import Activacion
from costos import ecm, dev_ecm
from layerNorm import LayerNorm
from auto_encoder_utils import mean_extra_grad_func, std_extra_grad_func, mul, dev_mul

input_shape = (1, 28, 28)
latent_size = 20
beta = 0.0001

encoder = NeuralNetwork(
    input_shape,
    [
        ConvolucionalNoBias(3, 2),
        Leacky_Relu(),
        Max_Pooling(),
        ConvolucionalNoBias(3, 4),
        Leacky_Relu(),
        Max_Pooling(),
        Flatten(),
        Densa(30),
    ],
)

encoder.initialize()

decoder = NeuralNetwork(
    (latent_size, 1),
    [
        Densa(30),
        Leacky_Relu(),
        Densa(2 * 16 * 16),
        LayerNorm(),
        Leacky_Relu(),
        Reshape((2, 16, 16)),
        UpSampling(),
        ConvolucionalNoBias(3, 2),
        Leacky_Relu(),
        Convolucional(3, 1),
        Sigmoid(),
        Activacion(mul, dev_mul),
    ],
)

get_mean = NeuralNetwork(
    encoder.output_shape(), [Densa(latent_size), AddLoss(mean_extra_grad_func, beta)]
)

get_std = NeuralNetwork(
    encoder.output_shape(), [Densa(latent_size), AddLoss(std_extra_grad_func, beta)]
)

auto_encoder = BranchedNetwork(
    input_shape, [encoder, [get_mean, get_std], Sample(), decoder]
)

auto_encoder.initialize()
path = "autoencoder4"
# train.save_red(path, auto_encoder)

mnist_train = preprocess.preprocess_data(100)
my_train = preprocess.preprocess_my_data()

print("done with data")

# Train = my_train
Train = train.mix(mnist_train, my_train)

for dato in Train:
    dato["output"] = dato["input"]

iteraciones = 10  # int(input("cuantas iteraciones?"))
loss = (ecm, dev_ecm)
auto_encoder.train(Train, loss, iteraciones, dt=0.0001)
auto_encoder.save(path)
# train.EntrenarYGuardar(path, Train, iteraciones, loss=(ecm, dev_ecm), dt=0.0001)

# print(auto_encoder)
