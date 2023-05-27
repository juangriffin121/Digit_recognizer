import numpy as np 
from capa import Capa

class Activacion(Capa):
  def __init__(self,act_fun,dev_act_fun):
    self.act_fun = act_fun
    self.dev_act_fun = dev_act_fun
  def forward(self,Input):
    self.input = Input
    return self.act_fun(self.input)
  def backward(self,grad_output,dt):
    return np.multiply(grad_output,self.dev_act_fun(self.input))