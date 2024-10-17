from neuralNetwork import NeuralNetwork
from activaciones import Sigmoid, Leacky_Relu, Softmax
from convolucional import Convolucional, ConvolucionalNoBias
from densa import Densa
from max_pooling import Max_Pooling
from grapher import graph_tensor
from reshape import Reshape
from Flatten import Flatten
import gpt_preprocess
import train

input_shape = (1, 28, 28)


red = NeuralNetwork(
    input_shape,
    [
        ConvolucionalNoBias(3, 2),
        Leacky_Relu(),
        Max_Pooling(),
        ConvolucionalNoBias(3, 5),
        Leacky_Relu(),
        Max_Pooling(),
        Flatten(),
        Densa(10),
    ],
)

# red = NeuralNetwork(
#     input_shape,
#     [
#         Flatten(),
#         Densa(100),
#         Leacky_Relu(),
#         Densa(10),
#     ],
# )

red.initialize()
path = "red5.pickle"
train.save_red(path, red)

mnist_train = gpt_preprocess.preprocess_data(200)
my_train = gpt_preprocess.preprocess_my_data()

print("done with data")

Train = train.mix(mnist_train, my_train)
# Test = train.mix(mnist_test, my_test)
iteraciones = int(input("cuantas iteraciones?"))

train.EntrenarYGuardar(path, Train, iteraciones, dt=0.0001)

print(red)
