import numpy as np 
from capa import Capa
from activacion import Activacion

class Sigmoid(Activacion):
  def __init__(self):
    def sigmoid(x):
      return 1/(1+np.exp(-x))
    def dev_sigmoid(x):
      return np.exp(x)/((1+np.exp(x))**2)
    Activacion.__init__(self,sigmoid,dev_sigmoid)

class Relu(Activacion):
  def __init__(self):
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
    Activacion.__init__(self,relu,dev_relu)
    
class Leacky_Relu(Activacion):
  def __init__(self):
    def leacky_relu(x):
      if x>0:
        return x
      else:
        return .1*x
    def dev_leacky_relu(x):
      if x>0:
        return 1
      else:
        return .1
    Activacion.__init__(self,leacky_relu,dev_leacky_relu)
    
class Tanh(Activacion):
  def __init__(self):
    def tanh(x):
      return np.tanh(x)
    def dev_tanh(x):
      return 1 - np.tanh(x) ** 2
    Activacion.__init__(self,tanh, dev_tanh)

#LEERLA

class Softmax(Capa):
  def __init__(self):
    pass
  def forward(self,Input):
    pass
  def backward(self, grad_output, dt):
    pass

#to delete

s = Sigmoid()
s.forward(np.array([1,2,3]))
print(s.backward(np.array([2,2,2]),.1))