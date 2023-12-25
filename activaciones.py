import numpy as np 
from capa import Capa
from activacion import Activacion

def sigmoid(x):
  return 1/(1+np.exp(-x))
def dev_sigmoid(x):
  y = sigmoid(x)
  return y*(1-y)

def relu(x):
  if x>0:
    return x
  else:
    return 0
def dev_relu(x):
  if x>0:
    return 1
  else:
    return 0

def leacky_relu(x):
  return x*(x>0) +.1*x*(x<0)
def dev_leacky_relu(x):
  return (x>0) +.1*(x<0)

def tanh(x):
  return np.tanh(x)
def dev_tanh(x):
  return 1 - np.tanh(x) ** 2

class Sigmoid(Activacion):
  def __init__(self,forma_input):
    Activacion.__init__(self,sigmoid,dev_sigmoid,forma_input)

class Relu(Activacion):
  def __init__(self,forma_input):
    Activacion.__init__(self,relu,dev_relu)
    
class Leacky_Relu(Activacion):
  def __init__(self,forma_input):
    Activacion.__init__(self,leacky_relu,dev_leacky_relu)
    
class Tanh(Activacion):
  def __init__(self,forma_input):
    Activacion.__init__(self,tanh, dev_tanh)

#Hereda de capa y no de activacion porque una neurona depende de todas las anteriores no de una sola

class Softmax(Capa):
  def __init__(self):
    pass
  
  def initialize(self):
    if hasattr(self, 'input_shape'):
      self.forma_output = self.input_shape
    else:
      raise ValueError("Input shape not set. Call set_input_shape before initializing.")
  
  def forward(self,Input):
    vector = np.exp(Input - np.max(Input))
    self.output = vector/(np.sum(vector))
    return self.output
  
  def backward(self, grad_output, dt):
    n = len(self.output)
    grad_input = ((np.identity(n) - self.output.T)*self.output)@grad_output
    return grad_input
  
  def output_shape(self):
    return self.forma_output

