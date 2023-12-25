from capa import Capa
import numpy as np

def max_pooling_func(Input,tamano_ventana):
  i=0
  output = []
  position_matrix = []
  while i<len(Input):
    j=0
    vector = []
    position_vector = []
    while j<len(Input[0]):
      pool = Input[i:i+tamano_ventana,j:j+tamano_ventana]
      val = np.max(pool)
      position = np.unravel_index(np.argmax(pool, axis=None), pool.shape)
      position = np.array(position) + np.array([i,j])
      j += tamano_ventana
      vector.append(val)
      position_vector.append(position)
    output.append(vector)
    position_matrix.append(position_vector)
    i += tamano_ventana
  return np.array(output),position_matrix

class Max_Pooling(Capa):
  def __init__(self, tamano_ventana = 2):
    self.tamano_ventana = tamano_ventana
    
  def initialize(self):
    if hasattr(self, 'input_shape'):
      image_size = self.input_shape[1]
      output_size = int(image_size/self.tamano_ventana)
      self.forma_output = (self.input_shape[0],output_size,output_size)
    else:
      raise ValueError("Input shape not set. Call set_input_shape before initializing.")
  
  def forward(self,Input):
    self.input = Input
    output = []
    position_matrix_set = []
    for image in self.input:
      pooled_image,position_matrix = max_pooling_func(image, self.tamano_ventana)
      output.append(pooled_image)
      position_matrix_set.append(position_matrix)
    self.position_matrix_set = position_matrix_set
    return np.array(output)
  
  def backward(self,grad_output,dt):
    grad_input = np.zeros(self.input.shape)
    for n in range(grad_output.shape[0]):
      for i in range(grad_output.shape[1]):
        for j in range(grad_output.shape[2]):
          position = self.position_matrix_set[n][i][j]
          grad_input[n][position[0]][position[1]] += grad_output[n][i][j]
    return grad_input
  
  def output_shape(self):
    return self.forma_output