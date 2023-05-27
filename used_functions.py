import numpy as np

def get_3_image_from_rgb(image):
  R = np.zeros((len(image),len(image[0])))
  G = np.zeros((len(image),len(image[0])))
  B = np.zeros((len(image),len(image[0])))
  for i,column in enumerate(image):
    for j,pixel in enumerate(column):
      R[i][j] = pixel[0]
      G[i][j] = pixel[1]
      B[i][j] = pixel[2]
  return np.array([R,G,B])
