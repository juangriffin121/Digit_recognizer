from capa import Capa
import numpy as np

class LayerNorm(Capa):
  def __init__(self):
    self.gamma = np.random.randn
    self.beta = np.random.randn
    
  def initialize(self):
    if hasattr(self, 'input_shape'):
      self.forma_output = self.input_shape
    else:
      raise ValueError("Input shape not set. Call set_input_shape before initializing.")
    
  def forward(self,Input):
    mean = np.mean(Input)
    std = np.std(Input)
    norm = (Input - mean)/std
    output = self.gamma*norm + self.beta

    self.input = Input
    self.norm = norm

    return output
  
  def backward(self,grad_output,dt):
    Input = self.Input
    norm = self.norm
    N = Input.size

    grad_gamma = grad_output.T@norm
    grad_beta = np.sum(grad_output)
    grad_input = self.gamma*(np.identity(N) - 1/N(norm@norm.T + 1))@grad_output

    self.gamma -= grad_gamma*dt
    self.beta -= grad_beta*dt
    return grad_input
  
  def output_shape(self):
    return self.forma_output