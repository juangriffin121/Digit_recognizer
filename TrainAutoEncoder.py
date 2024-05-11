import preprocess
import train
import numpy as np
from costos import ecm, dev_ecm
from auto_encoder_utils import mean_extra_grad_func, std_extra_grad_func, mul, dev_mul

num = int(input("cuantos datos por digito"))

path = input("nombre de la red")

net = train.load_red(path)

iteraciones = int(input("cuantas iteraciones?"))

mnist_train = preprocess.preprocess_data(num)
my_train = preprocess.preprocess_my_data()

print("done with data")

Train = train.mix(mnist_train, my_train)
# Train = my_train
for dato in Train:
    dato["output"] = dato["input"]
loss = ecm, dev_ecm

net.train(Train, loss, iteraciones, 0.0005)
net.save(path)

# train.EntrenarYGuardar(path, Train, iteraciones, loss=(ecm, dev_ecm), dt=0.0005)
