import numpy as np
import pickle
import copy


class NeuralNetwork:
    def __init__(self, input_shape, layers):
        self.layers = layers
        self.input_shape = input_shape

    def initialize(self):
        input_shape = self.input_shape
        for layer in self.layers:
            layer.set_input_shape(input_shape)
            layer.initialize()
            input_shape = layer.output_shape()
        self.forma_output = input_shape

    def forward(self, input_data):
        output = input_data
        max = np.max(output)
        for layer in self.layers:
            output = layer.forward(output)
            max = np.max(output)
            if np.isnan(max):
                raise ValueError(f"something is nan in layer {str(layer)}")
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
            txt += f"{str(layer)}\n"
        return txt

    def set_input_shape(self, input_shape):
        if hasattr(self, "input_shape"):
            if input_shape != self.input_shape:
                print("Changing the input shape")
        self.input_shape = input_shape

    def output_shape(self):
        return self.forma_output

    def __call__(self, *prev_modules):
        self.prev_modules = prev_modules
        return self

    def freeze(self):
        for layer in self.layers:
            layer.freeze()

    def unfreeze(self):
        for layer in self.layers:
            layer.unfreeze()

    def save(self, path):
        with open(f"{path}.pickle", "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        print(f"Red guardada correctamente en archivo {path}.pickle")

    def train(self, datos, loss, num_iteraciones, dt):
        prev_error = np.inf
        costo, dev_costo = loss
        correctos = 0
        for j in range(num_iteraciones):
            prev_dict = copy.deepcopy(self.__dict__)
            error = 0
            for i, dato in enumerate(datos):
                if not i % 100:
                    print(i)
                input_ = dato["input"]
                output_correcto = dato["output"]
                output = self.forward(input_)
                error += costo(output, output_correcto)
                if np.argmax(output) == np.argmax(output_correcto):
                    correctos += 1
                grad_error = dev_costo(output, output_correcto)
                self.backward(grad_error, dt)
            print(j, error, correctos, f"{int(correctos/len(datos)*100)}%")
            if error > prev_error:
                self.__dict__ = prev_dict
                error = prev_error
                print("reverting to previous version")
            prev_error = error
