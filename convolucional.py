from capa import Capa
import numpy as np
from scipy import signal

class Convolucional(Capa):
  def __init__(self,tamano_filtro,num_filtros):
    self.tamano_filtro = tamano_filtro
    self.num_filtros = num_filtros
  
  def initialize(self):
    if hasattr(self, 'input_shape'):
      profundidad_in,altura_in,ancho_in = self.input_shape
      
      self.forma_input = self.input_shape
      self.forma_filtro = (self.num_filtros,profundidad_in,self.tamano_filtro,self.tamano_filtro)
      self.forma_output = (self.num_filtros , altura_in - self.tamano_filtro + 1 , ancho_in - self.tamano_filtro + 1 )
      self.filtros = np.random.randn(*self.forma_filtro)
      self.sesgos = np.random.randn(*self.forma_output)
      
    else:
      raise ValueError("Input shape not set. Call set_input_shape before initializing.")
  
  def forward(self,Input):
    self.input = Input
    output = np.zeros(self.forma_output)
    for n,filtro3d in enumerate(self.filtros):
      for m,filtro in enumerate(filtro3d):
        output[n] += signal.correlate2d(self.input[m],filtro,mode = 'valid') + self.sesgos[n]
    return output
  
  def backward(self,grad_output,dt):
    self.grad_sesgos = grad_output
    self.grad_filtros = np.zeros(self.forma_filtro)
    grad_input = np.zeros(self.forma_input)
    for n,filtro3d in enumerate(self.filtros):
      for m,filtro in enumerate(filtro3d):
        self.grad_filtros[n][m] = signal.correlate2d(self.input[m],grad_output[n],mode = 'valid')
        grad_input[m] += signal.convolve2d(grad_output[n],filtro,'full')
    self.sesgos -= dt*self.grad_sesgos
    self.filtros -= dt*self.grad_filtros
    return grad_input
  
  def output_shape(self):
    return self.forma_output