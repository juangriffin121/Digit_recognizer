from capa import Capa
import numpy as np
from scipy import signal


class Densa(Capa):
    def __init__(self, output_size, weight_var=0.1, bias_var=0.1):
        super().__init__()
        self.output_size = output_size
        self.forma_output = (self.output_size, 1)
        self.weight_var = weight_var
        self.bias_var = bias_var

    def initialize(self):
        if hasattr(self, "input_shape"):
            input_size = self.input_shape[0]
            self.pesos = self.weight_var * np.random.randn(self.output_size, input_size)
            self.sesgos = self.bias_var * np.random.randn(self.output_size, 1)
        else:
            raise ValueError(
                "Input shape not set. Call set_input_shape before initializing."
            )

    def forward(self, Input):
        self.input = Input
        return self.pesos @ self.input + self.sesgos

    def backward(self, grad_output, dt):
        if not self.frozen:
            grad_pesos = grad_output @ self.input.T
            grad_sesgos = grad_output
            self.pesos -= grad_pesos * dt
            self.sesgos -= grad_sesgos * dt
        grad_input = self.pesos.T @ grad_output
        return grad_input

    def output_shape(self):
        return self.forma_output
