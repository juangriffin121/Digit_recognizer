from abc import ABC, abstractmethod

import numpy as np 

from capa import Capa

class Activacion(Capa):
  def __init__(self,act_fun,dev_act_fun):
    self.act_fun = act_fun
    self.dev_act_fun = dev_act_fun
    
  def initialize(self):
    if hasattr(self, 'input_shape'):
      self.forma_output = self.input_shape
    else:
      raise ValueError("Input shape not set. Call set_input_shape before initializing.")
  
  def forward(self,Input):
    self.input = Input
    return self.act_fun(self.input)
  
  def backward(self,grad_output,dt):
    return grad_output*self.dev_act_fun(self.input)
  
  def output_shape(self):
    return self.forma_output
