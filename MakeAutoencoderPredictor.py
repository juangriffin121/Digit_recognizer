import preprocess
import train
import numpy as np
from auto_encoder_utils import mean_extra_grad_func, std_extra_grad_func, mul, dev_mul
from neuralNetwork import NeuralNetwork
from branchedNetwork import BranchedNetwork
from densa import Densa
from activaciones import Leacky_Relu
from auto_encoder_utils import combined_loss, dev_combined_loss

input_shape = (1, 28, 28)

num = int(input("cuantos datos por digito"))

# mnist_train = preprocess.preprocess_data(num)
my_train = preprocess.preprocess_my_data()

print("done with data")

# Train = train.mix(mnist_train, my_train)
Train = my_train
for dato in Train:
    dato["output"] = dato["input"], dato["output"]
path = input("nombre de la red del autoencoder a usar")

iteraciones = int(input("cuantas iteraciones?"))

auto_encoder = train.load_red(path)

encoder, [get_mean, get_std], sample, decoder = list(auto_encoder)

latent_shape = decoder.input_shape

predictor = NeuralNetwork(latent_shape, [Densa(10), Leacky_Relu(), Densa(10)])

auto_encoder_predictor = BranchedNetwork(
    input_shape,
    [
        encoder,
        [get_mean, get_std],
        sample,
        [decoder, predictor],
    ],
)

auto_encoder_predictor.initialize()

print(auto_encoder_predictor)

save_path = input("nombre para guardar la red del autoencoder_predictor")

train.save_red(save_path, auto_encoder_predictor)
train.EntrenarYGuardar(
    save_path,
    Train,
    iteraciones,
    loss=(combined_loss, dev_combined_loss),
    dt=0.0001,
)
