class Capa:
    def __init__(self):
        self.frozen = False

    def forward(self, input_data):
        raise NotImplementedError("forward method must be implemented in each layer")

    def backward(self, grad_output, dt):
        raise NotImplementedError("backward method must be implemented in each layer")

    def initialize(self):
        raise NotImplementedError("initialize method must be implemented in each layer")

    def set_input_shape(self, input_shape):
        self.input_shape = input_shape

    def output_shape(self):
        raise NotImplementedError(
            "output_shape method must be implemented in each layer"
        )

    def __str__(self):
        return f"{self.__class__.__name__}\n\t{self.input_shape}->{self.output_shape()}"

    def __call__(self, *prev_layers):
        self.prev_layers = prev_layers
        return self

    def freeze(self):
        self.frozen = True

    def unfreeze(self):
        self.frozen = False
