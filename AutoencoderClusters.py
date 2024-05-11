from train import load_red
import numpy as np
import matplotlib.pyplot as plt
import preprocess
import train

path = input("nombre de la red\n")
autoencoder = load_red(path)
encoder = autoencoder[0]
get_mean = autoencoder[1][0]

num = int(input("cuantos datos por digito"))

mnist_train = preprocess.preprocess_data(num)
my_train = preprocess.preprocess_my_data()

print("done with data")

Train = train.mix(mnist_train, my_train)
# Train = my_train

colors = [
    "blue",
    "grey",
    "red",
    "cyan",
    "magenta",
    "yellow",
    "black",
    "white",
    "green",
    "indigo",
    "saddlebrown",
]


for dato in Train:
    digit = np.argmax(dato["output"])
    latent_vector = get_mean.forward(encoder.forward(dato["input"])).reshape(
        2,
    )
    print(latent_vector)
    print(digit)
    plt.plot(latent_vector[0], latent_vector[1], ".", color=colors[digit], alpha=0.5)

plt.savefig("Clusters")
