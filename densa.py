from capa import Capa
import numpy as np
from scipy import signal

#hay que asegurar que dos capas conectadas tengan el output_size de una igual al de la otra
#tests

class Densa(Capa):
  def __init__(self,output_size,input_size):
    self.pesos = np.random.randn(output_size,input_size)
    self.sesgos = np.random.randn(output_size,1)
  def forward(self,Input):
    self.input = Input
    return np.dot(self.pesos,self.input) + self.sesgos
  def backward(self,grad_output,dt):
    grad_pesos = grad_output @ self.input
    grad_sesgos = grad_output
    grad_input = self.pesos.T @ grad_output
    self.pesos -= grad_pesos*dt
    self.sesgos -= grad_sesgos*dt
    return grad_input
  

