from capa import Capa
import numpy as np


class Sample(Capa):
    def initialize(self):
        if hasattr(self, "input_shape"):
            self.forma_output = self.input_shape[0]
        else:
            raise ValueError(
                "Input shape not set. Call set_input_shape before initializing."
            )

    def forward(self, input_):
        mean, log_var = input_
        self.input_ = input_
        self.std = np.exp(log_var / 2)
        self.epsilon = np.random.randn(*self.forma_output)
        return mean + self.std * self.epsilon

    def backward(self, grad_output, dt):
        grad_input = [grad_output, grad_output * self.epsilon * self.std / 2] 
        return grad_input

    def output_shape(self):
        return self.forma_output
