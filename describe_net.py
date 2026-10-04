import train
import preprocess
from analisis_red import correctos, confusion_matrix

name = input("nombre de la red (without .pickle)\n")
path = name + ".pickle"
red = train.load_red(path)
print(red)
print(red.Test)
