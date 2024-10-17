import numpy as np
import pandas as pd
import random


def get_digit(num):
    return np.eye(10)[num].reshape((10, 1))


def process_sample(sample):
    digit = sample[0]
    digit_vector = get_digit(digit)
    pixels = np.array(sample[1:]).reshape((1, 28, 28))
    return {"input": pixels, "output": digit_vector}


def preprocess_generic_data(file_path, num_datos=None):
    df = pd.read_csv(file_path)

    Datos = {}

    for _, sample in df.iterrows():
        processed_sample = process_sample(sample)

        digit_class = f"digit_class: {sample[0]}"
        if digit_class not in Datos:
            Datos[digit_class] = []
        Datos[digit_class].append(processed_sample)

    datos = []
    for digit_class, digit_datos in Datos.items():
        if num_datos:
            datos += random.sample(digit_datos, num_datos)
        else:
            datos += digit_datos

    random.shuffle(datos)
    return datos


def preprocess_data(num_datos=None):
    return preprocess_generic_data("./data/train.csv", num_datos)


def preprocess_testing_data():
    return preprocess_generic_data("./data/test.csv")


def preprocess_my_data(num_datos=None):
    return preprocess_generic_data("./data/my_data.csv", num_datos)


def preprocess_my_testing_data():
    return preprocess_generic_data("./data/my_test_data.csv")
