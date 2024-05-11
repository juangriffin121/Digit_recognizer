import numpy as np
from costos import softmax_cross_entropy, dev_softmax_cross_entropy, ecm, dev_ecm

alpha = 100


def mean_extra_grad_func(x):
    return x


def std_extra_grad_func(x):
    return (np.exp(x) - 1) / 2


def mul(x):
    return 255 * x


def dev_mul(x):
    return 255


def combined_loss(output, output_correcto):
    image, digit = output
    image_correcto, digit_correcto = output_correcto
    return ecm(image, image_correcto) + alpha * softmax_cross_entropy(
        digit, digit_correcto
    )


def dev_combined_loss(output, output_correcto):
    image, digit = output
    image_correcto, digit_correcto = output_correcto
    grad_im = dev_ecm(image, image_correcto)
    grad_dig = alpha * dev_softmax_cross_entropy(digit, digit_correcto)
    return grad_im, grad_dig
