from capa import Capa
import numpy as np


def up_sampling_func(image):
    shape = image.shape
    up_sampled_image = np.zeros((2 * shape[0], 2 * shape[1]))
    for i in range(shape[0]):
        for j in range(shape[1]):
            (
                up_sampled_image[2 * i, 2 * j],
                up_sampled_image[2 * i + 1, 2 * j],
                up_sampled_image[2 * i, 2 * j + 1],
                up_sampled_image[2 * i + 1, 2 * j + 1],
            ) = (image[i, j],) * 4
    return up_sampled_image


class UpSampling(Capa):
    def initialize(self):
        if hasattr(self, "input_shape"):
            image_size = self.input_shape[1]
            output_size = image_size * 2
            self.forma_output = (self.input_shape[0], output_size, output_size)
        else:
            raise ValueError(
                "Input shape not set. Call set_input_shape before initializing."
            )

    def forward(self, Input):
        self.input = Input
        output = []
        for image in self.input:
            up_sampled_image = up_sampling_func(image)
            output.append(up_sampled_image)
        return np.array(output)

    def backward(self, grad_output, dt):
        shape = self.input_shape
        grad_input = np.zeros(shape)
        for k in range(shape[0]):
            for i in range(shape[1]):
                for j in range(shape[2]):
                    grad_input[k, i, j] = (
                        grad_output[k, 2 * i, 2 * j]
                        + grad_output[k, 2 * i + 1, 2 * j]
                        + grad_output[k, 2 * i, 2 * j + 1]
                        + grad_output[k, 2 * i + 1, 2 * j + 1]
                    )
        return grad_input

    def output_shape(self):
        return self.forma_output
