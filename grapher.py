import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

def graph_image(matrix,ax,MAX = None,MIN = None):
  ax.pcolormesh(255-(matrix[::1][::-1]),cmap = 'Greys',vmax = MAX,vmin = MIN)
  
def get_axes(shape):
  if len(shape) == 2:
    fig,ax_set = plt.subplots(figsize = (shape[1],shape[0]))
  if len(shape) == 3:
    fig,ax_set = plt.subplots(nrows = shape[0],figsize = (shape[1],shape[2]*shape[0]))
  if len(shape) == 4:
    fig,ax_set = plt.subplots(nrows = shape[0],ncols = shape[1],figsize = (shape[1]*shape[3],shape[2]*shape[0]))
  return ax_set
  
def graph_tensor(tensor):
  MAX = np.max(tensor)
  MIN = np.min(tensor)
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
      try:
        for matrix,ax in zip(subtensor,ax_column):
          graph_image(matrix,ax,MAX = MAX,MIN = MIN)
      except:
        graph_image(subtensor[0],ax_column,MAX = MAX,MIN = MIN)