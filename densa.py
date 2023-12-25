from capa import Capa
import numpy as np
from scipy import signal

class Densa(Capa):
  def __init__(self, output_size):
    self.output_size = output_size
    self.forma_output = (self.output_size,1)
  
  def initialize(self):
    if hasattr(self, 'input_shape'):
      self.pesos = np.random.randn(self.output_size,self.input_size)
      self.sesgos = np.random.randn(self.output_size,1)
    else:
      raise ValueError("Input shape not set. Call set_input_shape before initializing.")

  def forward(self,Input):
    self.input = Input
    return self.pesos @ self.input + self.sesgos
  
  def backward(self,grad_output,dt):
    grad_pesos = grad_output @ self.input.T
    grad_sesgos = grad_output
    grad_input = self.pesos.T @ grad_output
    self.pesos -= grad_pesos*dt
    self.sesgos -= grad_sesgos*dt
    return grad_input
  
  def output_shape(self):
    return self.forma_output