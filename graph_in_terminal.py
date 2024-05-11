import train
import preprocess
import random

# path = input("nombre?\n")
# red = train.load_red(path)

# data = preprocess.preprocess_my_data()

# digit = random.choice(data)["input"][0]

with open("./ascii") as f:
    txt = f.read()
    chars = txt.split("  ")
chars = chars[:-1]


# print(chars)
def graph(array):
    num_chars = len(chars)
    txt = ""
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            pixel = round(array[i][j])
            luminosity_pos = pixel * num_chars // 255
            txt += f"{chars[luminosity_pos] }"
        txt += "\n"
    print(txt)
    return txt
