class Capa:
    def forward(self, input_data):
        raise NotImplementedError(
            "forward method must be implemented in each layer")

    def backward(self, grad_output, dt):
        raise NotImplementedError(
            "backward method must be implemented in each layer")
        
    def initialize(self):
        raise NotImplementedError(
            "initialize method must be implemented in each layer")

    def set_input_shape(self, input_shape):
        self.input_shape = input_shape

    def output_shape(self):
        raise NotImplementedError(
            "output_shape method must be implemented in each layer")
