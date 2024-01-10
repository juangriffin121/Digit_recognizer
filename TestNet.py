import train
import preprocess
from analisis_red import correctos, confusion_matrix

path = input("nombre de la red")

red = train.load_red(path)
mnist_test = preprocess.preprocess_testing_data()
my_test = preprocess.preprocess_my_testing_data()
test = train.mix(my_test, mnist_test)

corr = correctos(red, test)
print(red)
print(f"wkiki saco {corr} bien de {len(test)}, porcentage = {100*corr/len(test)}")
setattr(red, "Test", 100 * corr / len(test))
train.save_red(path, red)
print()
#print(confusion_matrix(red, test))
