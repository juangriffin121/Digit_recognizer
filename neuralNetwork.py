import numpy as np


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
        max = np.max(output)
        # print("shape", output.shape, "max", max, "min", np.min(output))

        for layer in self.layers:
            output = layer.forward(output)
            # print(layer)
            max_ratio = np.max(output) / max
            max = np.max(output)
            if np.isnan(max_ratio):
                raise ValueError("something is nan")
            # print(
            #   "shape",
            #   output.shape,
            #   "max",
            #   max,
            #   "min",
            #   np.min(output),
            #   "ratio",
            #   max_ratio,
            # )
            # print()
        return output

    def backward(self, grad_output, dt):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output, dt)
        return grad_output

    def __getitem__(self, index):
        return self.layers[index]

    def __len__(self):
        return len(self.layers)

    def __str__(self):
        txt = ""
        for layer in self.layers:
            layer_name = str(layer).split(".")[1].split(" ")[0]
            txt += f"{layer_name}\n\t{layer.input_shape} -> {layer.output_shape()}\n"
        return txt
