import numpy as np
import pickle
import copy
import random

class BranchedNetwork:
    def __init__(self, input_shape, modules):
        self.modules = modules
        self.input_shape = input_shape

    def initialize(self):
        input_shape = self.input_shape
        for module in self.modules:
            if isinstance(module, list):
                output_shapes = []
                for sub_module in module:
                    sub_module.set_input_shape(input_shape)
                    sub_module.initialize()
                    output_shapes.append(sub_module.output_shape())
                input_shape = output_shapes
            else:
                module.set_input_shape(input_shape)
                module.initialize()
                input_shape = module.output_shape()
        self.forma_output = input_shape

    def forward(self, input):
        max_ = np.max(input)
        for module in self.modules:
            if isinstance(module, list):
                output = []
                for sub_module in module:
                    output.append(sub_module.forward(input))
                maxs = []
                for comp in output:
                    maxs.append(np.max(comp))
                max_ = max(maxs)
            else:
                output = module.forward(input) 
                max_ = np.max(output)
            input = output
        return output

    def backward(self, grad_output, dt):
        for module in reversed(self.modules):
            if isinstance(module, list):
                grad_input = 0
                for sub_module, sub_grad in zip(module, grad_output):
                    grad_input += sub_module.backward(sub_grad, dt)
            else:
                grad_input = module.backward(grad_output, dt)
            grad_output = grad_input
        return grad_input

    def __str__(self):
        txt = "\n"
        for module in self.modules:
            if isinstance(module, list):
                txt += "[\n"
                for sub_module in module:
                    txt += f"\t{str(sub_module).replace("\n","\n\t")}"
                    txt += "--------"
                    txt += "\n"
                txt += "]"
            else:
                txt += f"{str(module)}\n"
            txt += "\n"
        return txt

    def __getitem__(self, index):
        return self.modules[index]

    def __len__(self):
        return len(self.modules)

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
        for module in self.modules:
            module.freeze()

    def unfreeze(self):
        for module in self.modules:
            module.unfreeze()

    def save(self, path):
        with open(f"{path}.pickle", "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        print(f"Red guardada correctamente en archivo {path}.pickle")

    def train(self, datos, loss, num_iteraciones, dt, metric_func = None):
        costo, dev_costo = loss
        prev_error = np.inf
        if metric_func is not None:
            metric = 0
        for j in range(num_iteraciones):
            prev_dict = copy.deepcopy(self.__dict__)
            error = 0
            for i, dato in enumerate(datos):
                if not i % 100:
                    print(i, error, len(datos)*error//(i + 1))
                    if np.isnan(error):
                        print("breaking")
                        break
                input_ = dato["input"]
                output_correcto = dato["output"]
                output = self.forward(input_)
                error += costo(output, output_correcto)
                if metric_func is not None:
                    metric += metric_func(output, output_correcto)
                grad_error = dev_costo(output, output_correcto)
                self.backward(grad_error, dt)
            message = f"{j} {error}"
            if metric_func is not None:
                message += f"{metric}, {int(metric/len(datos)*100)}%)"
            print(message)
            if error > prev_error or np.isnan(error):
                self.__dict__ = prev_dict
                error = prev_error
                dt /= 2
                random.shuffle(datos)
                print(f"reverting to previous version of error: {error}, dt = {dt}")
            prev_error = error
