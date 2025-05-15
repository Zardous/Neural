import numpy as np
import time
import scipy.io
import matplotlib.pyplot as plt

print("Loading EMNIST")
T0 = time.time()
data = scipy.io.loadmat("data/emnist-mnist.mat")
print(f"Loaded EMNIST in {round(time.time() - T0,2)}s")
SetSize = data["dataset"][0][0][0][0][0][0].shape[0]
# images:
# print(data["dataset"][0][0][0][0][0][0])

# labels:
# print(data["dataset"][0][0][0][0][0][1])

# mapping:
# print(data["dataset"][0][0][2])
# print(chr(data["dataset"][0][0][2][MapIndex][-1]))


def Sample(Index=np.random.randint(0, SetSize)):
    image = data["dataset"][0][0][0][0][0][0][Index].reshape(784,1) / 255
    shift = np.random.randint(-3,3,(1,2))
    image = np.roll(image, shift=shift[0,0]+28*shift[0,1], axis=0)
    image += np.random.randn(*image.shape)/15

    MapIndex = data["dataset"][0][0][0][0][0][1][Index][0]

    Label = chr(data["dataset"][0][0][2][MapIndex][-1])

    Target = np.zeros((10, 1))
    Target[MapIndex] = 1
    return image, Target, Label, MapIndex

def ShowImage(Index=np.random.randint(0, SetSize)):
    image, Target, Label, MapIndex = Sample(Index)
    plt.imshow(image.reshape(28, 28).T, cmap="Greys")
    plt.title(f"Index: {Index}, Label: {Label}")
    plt.show()