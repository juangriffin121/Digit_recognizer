import numpy as np
from train import respuesta
from grapher import graph_tensor
import random


def correctos(red, datos):
    corr = 0
    counter = 0
    for dato in datos:
        counter += 1
        if not counter % 100:
            print(counter)
            print(100 * corr / counter)
            print()
        Input = dato["input"]
        output_correcto = dato["output"]
        output = respuesta(red, Input)
        if np.argmax(output) == np.argmax(output_correcto):
            corr += 1
    return corr


def distribucion_incorrectos(red, datos):
    incorrectos = {}
    for dato in datos:
        Input = dato["input"]
        output_correcto = dato["output"]
        output = respuesta(red, Input)
        digit = np.argmax(output_correcto)
        if np.argmax(output) != digit:
            try:
                incorrectos[f"digit_class: {digit}"] += 1
            except:
                incorrectos[f"digit_class: {digit}"] = 1
    return incorrectos


def datos_incorrectos(red, datos):
    incorrectos = {}
    for dato in datos:
        Input = dato["input"]
        output_correcto = dato["output"]
        output = respuesta(red, Input)
        digit = np.argmax(output_correcto)
        if np.argmax(output) != digit:
            try:
                incorrectos[f"digit_class: {digit}"].append(dato)
            except:
                incorrectos[f"digit_class: {digit}"] = [dato]
    return incorrectos


def confusion_matrix(red, datos):
    incorrectos_matriz = np.zeros((10, 10))
    for dato in datos:
        Input = dato["input"]
        output_correcto = dato["output"]
        output = respuesta(red, Input)
        digit = np.argmax(output_correcto)
        predicted_digit = np.argmax(output)
        if predicted_digit != digit:
            incorrectos_matriz[digit][predicted_digit] += 1
    return incorrectos_matriz


def check_conv(conv_layer, input_, save=False, number=0):
    #print(input_.shape)
    if save:
        graph_tensor(input_, filepath=f"./NetVisualization/input{number}")
        filtros = conv_layer.filtros
        shape = filtros.shape
        #print("filtros", shape)
        graph_tensor(
            filtros,  # .reshape(shape[0], shape[2], shape[3]),
            filepath=f"./NetVisualization/filtros{number}",
        )
        # graph_tensor(conv_layer.sesgos)
        filtered = conv_layer.forward(input_)
        #print("output", filtered.shape)
        graph_tensor(filtered, filepath=f"./NetVisualization/output{number}")
    else:
        graph_tensor(input_[0])
        filtros = conv_layer.filtros
        shape = filtros.shape
        graph_tensor(filtros.reshape(shape[0], shape[2], shape[3]))
        # graph_tensor(conv_layer.sesgos)
        filtered = conv_layer.forward(input_)
        graph_tensor(filtered)
