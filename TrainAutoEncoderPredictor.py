import preprocess
import train
import numpy as np
from costos import ecm, dev_ecm
from auto_encoder_utils import mean_extra_grad_func, std_extra_grad_func, mul, dev_mul
from auto_encoder_utils import combined_loss, dev_combined_loss

num = int(input("cuantos datos por digito"))

mnist_train = preprocess.preprocess_data(num)
my_train = preprocess.preprocess_my_data()

print("done with data")

Train = train.mix(mnist_train, my_train)
#Train = my_train
for dato in Train:
    dato["output"] = dato["input"], dato["output"]
path = input("nombre de la red")

iteraciones = int(input("cuantas iteraciones?"))

train.EntrenarYGuardar(
    path,
    Train,
    iteraciones,
    loss=(combined_loss, dev_combined_loss),
    dt=0.0001,
)
