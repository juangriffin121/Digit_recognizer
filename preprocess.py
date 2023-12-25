import pandas as pd
import numpy as np
import random

def get_digit(num):
  if num in range(0,10):  
    arr = [0,0,0,0,0,0,0,0,0,0]
    arr[num] = 1
  return arr

def preprocess_data(num_datos = None):
  Datos = {}
  with open('./data/train.csv') as f:
    df = pd.read_csv(f)
  df = df.transpose()
  for sample_num in df:
    sample = df[sample_num]
    digit = sample[0]
    digit_vector = np.array(get_digit(digit))
    digit_vector = np.reshape(digit_vector,(10,1))
    pixels = []
    for value in sample[1:]:
      pixels.append([value])
    pixels = np.array(pixels)
    image = np.reshape(pixels,(1,28,28))
    try:
      Datos[f'digit_class: {digit}'].append({'input':image,'output':digit_vector})
    except:
      Datos[f'digit_class: {digit}'] = [{'input':image,'output':np.array(digit_vector)}]
  datos = []
  for digit_class in Datos:
    digit_datos = Datos[digit_class]
    if num_datos:
      datos += random.sample(digit_datos,num_datos) 
    else:
      datos += digit_datos
  random.shuffle(datos)
  return datos

def preprocess_testing_data():
  with open('./data/test.csv') as f:
    df = pd.read_csv(f)
  df = df.transpose()
  datos = []
  for sample_num in df:
    sample = df[sample_num]
    digit = sample[0]
    digit_vector = np.array(get_digit(digit))
    digit_vector = np.reshape(digit_vector,(10,1))
    pixels = []
    for value in sample[1:]:
      pixels.append([value])
    pixels = np.array(pixels)
    image = np.reshape(pixels,(1,28,28))
    datos.append({'input':image,'output':digit_vector})
  return datos

def preprocess_my_data(num_datos = None):
  Datos = {}
  with open('./data/my_data.csv') as f:
    df = pd.read_csv(f)
  df = df.transpose()
  for sample_num in df:
    sample = df[sample_num]
    digit = sample[0]
    digit_vector = np.array(get_digit(digit))
    digit_vector = np.reshape(digit_vector,(10,1))
    pixels = []
    for value in sample[1:]:
      pixels.append([value])
    pixels = np.array(pixels)
    image = np.reshape(pixels,(1,28,28))
    try:
      Datos[f'digit_class: {digit}'].append({'input':image,'output':digit_vector})
    except:
      Datos[f'digit_class: {digit}'] = [{'input':image,'output':np.array(digit_vector)}]
  datos = []
  for digit_class in Datos:
    digit_datos = Datos[digit_class]
    if num_datos:
      datos += random.sample(digit_datos,num_datos) 
    else:
      datos += digit_datos
  random.shuffle(datos)
  return datos

def preprocess_my_testing_data():
  with open('./data/my_test_data.csv') as f:
    df = pd.read_csv(f)
  df = df.transpose()
  datos = []
  for sample_num in df:
    sample = df[sample_num]
    digit = sample[0]
    digit_vector = np.array(get_digit(digit))
    digit_vector = np.reshape(digit_vector,(10,1))
    pixels = []
    for value in sample[1:]:
      pixels.append([value])
    pixels = np.array(pixels)
    image = np.reshape(pixels,(1,28,28))
    datos.append({'input':image,'output':digit_vector})
  return datos
  