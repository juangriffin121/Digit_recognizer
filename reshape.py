import numpy as np
from capa import Capa


class Reshape(Capa):
    def __init__(self, forma_output):
        super().__init__()
        self.forma_output = forma_output

    def initialize(self):
        pass

    def forward(self, Input):
        return Input.reshape(self.forma_output)

    def backward(self, grad_output, dt):
        return grad_output.reshape(self.input_shape)

    def output_shape(self):
        return self.forma_output
