class NeuralNetwork:
    def __init__(self, input_shape, layers):
        self.layers = layers
        self.input_shape = input_shape

    def initialize(self):
        for layer in self.layers:
            layer.set_input_shape(self.input_shape)
            layer.initialize()
            self.input_shape = layer.output_shape()

    def forward(self, input_data):
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, grad_output, dt):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output, dt)
