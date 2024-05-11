import numpy as np
from activaciones import Softmax

softmax = Softmax.softmax


def ecm(output, output_correcto):
    return np.sum((output_correcto - output) ** 2) / (
        len(output_correcto) * 255 * 28 * 28
    )


#  (output_corr > output) -> (increment output -> decrease error)
def dev_ecm(output, output_correcto):
    return -2 * (output_correcto - output) / (len(output_correcto) * 255 * 28 * 28)


def cross_entropy(output, output_correcto):
    return -np.sum(output_correcto * np.log(output))


def dev_cross_entropy(output, output_correcto):
    return -output_correcto / output


def softmax_cross_entropy(output, output_correcto):
    return -np.sum(output_correcto * np.log(softmax(output)))


def dev_softmax_cross_entropy(output, output_correcto):
    return -(output_correcto - softmax(output))  # (-output_correcto/output)
