from train import respuesta,load_red
from PIL import Image
import numpy as np
from used_functions import get_3_image_from_rgb
import os
import time
from scipy.ndimage import gaussian_filter
from handle_image import show_im,save_im
import cv
clear = lambda: os.system('clear')

def add_data(is_right,main_prediction,image):
  if not is_right:
    correct_digit = input('cual era el numero correcto?')
  else:
    correct_digit = main_prediction
  correct_digit
  image = image.reshape((28,28))
  txt = ''
  txt += f'{correct_digit}'
  for row in image:
      for val in row:
          txt += f',{val}'
  txt += '\n'
  if np.random.uniform() > 0.2:
    print(f'guardando la imagen de su {correct_digit} en el dataset train')
    with open('./data/my_data.csv','a') as f:
        f.write(txt)
  else:
    print(f'guardando la imagen de su {correct_digit} en el dataset test')
    with open('./data/my_test_data.csv','a') as f:
        f.write(txt)

def predict(red,correctos,totales):
  clear()
  print('Bienvenido a Wkiki 3.0')
  print('Dibuje en paint en una imagen de 200 x 200 pixeles con blanco sobre negro el digito que quiere que Dajij dWkikikant prediga')
  image = get_image()
  print('imagen cargada')
  time.sleep(1)
  prediction_vector = respuesta(red, image)
  main_prediction = np.argmax(prediction_vector)
  clear()
  print(f'Wkikikant piensa que escribiste un {main_prediction}')
  totales += 1
  is_right = handle_feedback(prediction_vector)
  correctos += is_right
  print(f'wkiki viene acertando {correctos}/{totales} osea un {correctos/totales*100}%')
  save_data = boolean(input('Quiere guardar este dato para que wwkiki aprendda?(s/n)'),'Quiere guardar este dato para que wwkiki aprendda?(s/n)')
  if save_data: 
    add_data(is_right,main_prediction,image)
  time.sleep(2)
  predict(red,correctos,totales)

def get_image():
  image_path = input('Guarde el archivo como .bmp 256 color y sueltelo sobre la consola')
  clear()
  image_path = handle_path(image_path)
  image = Image.open(image_path)
  im = np.asarray(image)
  if im.shape == (20,20):
    image = im
    image = np.pad(image,4)
    image = gaussian_filter(image,.4)
  elif im.shape != (28,28):
    image.thumbnail((20,20),Image.LANCZOS)
    image = np.asarray(image)
    image = gaussian_filter(image,.4)
    image = np.pad(image,4)
  image = np.array([image])
  if image.shape == (1,28,28):
    save_im(image[0],'num.bmp')
    return image
  print('la imagen debe ser un .bmp de 28 x 28 o 20 x 20 intente de nuevo')
  get_image()

def handle_path(path):
  fixedpath = ''
  for letter in path:
    if letter == '"':
      pass
    else:
      if letter != '\\':
        fixedpath += letter
      else:
        fixedpath += '/'
  file = '/mnt/c' + fixedpath[2:]
  return file
  
def boolean(txt,question):
  if txt == 's':
    return True
  elif txt == 'n':
    return False
  print('Escribi una s o una n nomas imbecil')
  return boolean(input(question),question)
  
def handle_feedback(prediction_vector):
  wkiki_is_right = boolean(input('Acerto?(s/n)'),'Acerto?(s/n)')
  if wkiki_is_right:
    clear()
    print('Vamoooo')
    return 1
  else:
    clear()
    print('perdon wkiki esta haciendo lo mejor que puede :c')
    print('si sirve de algo estas son las probabilidades que le daba wkiki a cada digito')
    for digit,prob in enumerate(prediction_vector):
      print(f'{digit}: {prob[0]*100}%')
    time.sleep(3)
    return 0

red = load_red('red.pickle')
def main():
  try:
    predict(red,0,0)
  except Exception as e:
    print(e.message)
    again =  boolean(input('Algo salio mal quiere intentar de nuevo?(s/n)'),'Algo salio mal quiere intentar de nuevo?(s/n)')
    if again:
      main()
    else:
      print('wkiki les desea buenos dias')
      time.sleep(1)

main()
