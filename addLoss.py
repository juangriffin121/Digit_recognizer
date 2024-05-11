from capa import Capa


class AddLoss(Capa):
    def __init__(self, extra_grad_function, beta):
        super().__init__()
        self.extra_grad_function = extra_grad_function
        self.beta = beta

    def initialize(self):
        if hasattr(self, "input_shape"):
            self.forma_output = self.input_shape
        else:
            raise ValueError(
                "Input shape not set. Call set_input_shape before initializing."
            )

    def forward(self, input_):
        self.input_ = input_
        return input_

    def backward(self, grad_output, dt):
        return grad_output + self.beta * self.extra_grad_function(self.input_)

    def output_shape(self):
        return self.forma_output
