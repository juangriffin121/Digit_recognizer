import train

path = input("nombre de la red")
red = train.load_red(path)
print(red)
print(red.__dict__)
