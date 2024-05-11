import numpy as np


def accuracy(output, output_correcto):
    if np.argmax(output) == np.argmax(output_correcto):
        return 1
    return 0


def accuracy_output_list(output, output_correcto, predictor_position=1):
    if np.argmax(output[predictor_position]) == np.argmax(
        output_correcto[predictor_position]
    ):
        return 1
    return 0
