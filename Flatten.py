import numpy as np
from capa import Capa


class Flatten(Capa):
    def initialize(self):
        if hasattr(self, "input_shape"):
            self.forma_output = (np.prod(self.input_shape), 1)
        else:
            raise ValueError(
                "Input shape not set. Call set_input_shape before initializing."
            )

    def forward(self, Input):
        return Input.reshape(self.forma_output)

    def backward(self, grad_output, dt):
        return grad_output.reshape(self.input_shape)

    def output_shape(self):
        return self.forma_output
