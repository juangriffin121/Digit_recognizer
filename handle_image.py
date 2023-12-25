import numpy as np
from PIL import Image
from used_functions import get_3_image_from_rgb


def get_im(path):
  image = Image.open(path)
  image = np.asarray(image)
  image = get_3_image_from_rgb(image)
  image = (image[0] + image[1] + image[2])/3
  image = image.astype(int)
  return image

def show_im(array):
  im = Image.fromarray(array)
  im.show()
  
def save_im(array,name):
  im = Image.fromarray(array)
  im.save(f'./images/{name}')
  
def save_images(datos):
  digits = np.zeros((10))
  for dato in datos: #random.sample(datos)
      im = dato['input'][0]
      im = im.astype(np.uint8)
      digit = np.argmax(dato['output'])
      print(digit)
      digits[digit] += 1
      amount = int(digits[digit])
      name = f'{digit}_number_{amount}.bmp'
      save_im(im,name)
      
def resize28(array):
  return np.pad(array,4)
