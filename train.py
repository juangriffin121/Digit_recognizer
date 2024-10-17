from costos import (
    cross_entropy,
    dev_cross_entropy,
    softmax_cross_entropy,
    dev_softmax_cross_entropy,
)
import numpy as np
import pickle


def respuesta(red, Input):
    output = red.forward(Input)
    return output


def iteracion(red, datos, loss, dt):
    error = 0
    costo, dev_costo = loss
    correctos = 0
    i = 0
    for dato in datos:
        if not i % 100:
            print(i)
        i += 1
        Input = dato["input"]
        output_correcto = dato["output"]
        output = respuesta(red, Input)
        error += costo(output, output_correcto)
        if np.argmax(output) == np.argmax(
            # if np.argmax(output[1]) == np.argmax(
            output_correcto
            # output_correcto[1]
        ):  # poner el [1] solo si es autoencoder_pred
            correctos += 1
        # print("out")
        # print(output)
        grad_error = dev_costo(output, output_correcto)
        # print("grad")
        # print(grad_error)
        red.backward(grad_error, dt)
    # error = 1
    return error / len(datos), correctos


def entrenar(
    red,
    datos,
    num_iteraciones,
    loss,
    dt=0.2,
):
    for i in range(num_iteraciones):
        error, correctos = iteracion(red, datos, loss, dt)
        print(i, error, correctos, f"{int(correctos/len(datos)*100)}%")  # error


def save_red(filepath, red):
    with open(filepath, "wb") as f:
        pickle.dump(red, f, pickle.HIGHEST_PROTOCOL)
    print(f"Red guardada correctamente en archivo {filepath}")


def load_red(filepath):
    with open(filepath, "rb") as f:
        red = pickle.load(f)
    return red


def EntrenarYGuardar(
    filepath,
    datos,
    num_iteraciones,
    loss=(softmax_cross_entropy, dev_softmax_cross_entropy),
    dt=0.2,
):
    red = load_red(filepath)
    entrenar(red, datos, num_iteraciones, loss, dt)
    save_red(filepath, red)


def mix(list1, list2):
    union = list1 + list2
    np.random.shuffle(union)
    return union
