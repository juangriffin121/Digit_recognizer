import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

def color(x):
  return (.5,.5,.5,(np.tanh(x)+1.)/2)

def graph_image(matrix,ax):
  for i,column in enumerate(matrix):
    for j,val in enumerate(column):
      ax.add_patch(Rectangle((i+.05,j+.05),.95,.95,facecolor=color(val)))
  ax.plot((0,len(matrix)),(0,len(matrix[0])),alpha = 0)
  
def get_axes(shape):
  if len(shape) == 2:
    fig,ax_set = plt.subplots(figsize = (shape[0],shape[1]))
  if len(shape) == 3:
    fig,ax_set = plt.subplots(nrows = shape[0],figsize = (shape[1],shape[2]*shape[0]))
  if len(shape) == 4:
    fig,ax_set = plt.subplots(nrows = shape[0],ncols = shape[1],figsize = (shape[1]*shape[3],shape[2]*shape[0]))
  return ax_set
  
def graph_tensor(tensor):
  shape = tensor.shape
  dimensions = len(shape)
  ax_set = get_axes(shape)
  if dimensions == 2:
    ax = ax_set
    graph_image(tensor,ax)
  if dimensions == 3:
    axes = ax_set
    for matrix,ax in zip(tensor,axes):
      graph_image(matrix,ax)
  if dimensions == 4:
    axes = ax_set
    for subtensor,ax_column in zip(tensor,axes):
      for matrix,ax in zip(subtensor,ax_column):
        graph_image(matrix,ax)