import numpy as np

def ecm(output,output_correcto):
  return np.sum((output_correcto-output)**2)/len(output_correcto)

def dev_ecm(output,output_correcto):
  return 2*(output_correcto-output)/len(output_correcto)

def cross_entropy(output,output_correcto):
  return -np.sum(output_correcto*np.log(output))

def dev_cross_entropy(output,output_correcto):
  return (-output_correcto/output)