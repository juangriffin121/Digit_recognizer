from neuralNetwork import NeuralNetwork
from activaciones import Sigmoid, Leacky_Relu, Softmax
from convolucional import Convolucional, ConvolucionalNoBias
from densa import Densa
from max_pooling import Max_Pooling
from grapher import graph_tensor
from reshape import Reshape
from Flatten import Flatten
import preprocess
import train

red = train.load_red("red3.pickle")

mnist_train = preprocess.preprocess_data(100)
my_train = preprocess.preprocess_my_data()

print("done with data")

Train = train.mix(mnist_train, my_train)
# Test = train.mix(mnist_test, my_test)

path = input("nombre de la red")

iteraciones = int(input("cuantas iteraciones?"))

train.EntrenarYGuardar(path, Train, iteraciones, dt=0.0005)

print(red)
