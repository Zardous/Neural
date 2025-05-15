import numpy as np
import time
import scipy.io
import matplotlib.pyplot as plt

print("Loading EMNIST")
T0 = time.time()
data = scipy.io.loadmat("data/emnist-byclass.mat")
print(f"Loaded EMNIST in {round(time.time() - T0,2)}s")
SetSize = data["dataset"][0][0][0][0][0][0].shape[0]

OutputSize = data["dataset"][0][0][2].shape[0]
InputSize = data["dataset"][0][0][0][0][0][0].shape[1]

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

    Target = np.zeros((62, 1))
    Target[MapIndex] = 1
    return image, Target, Label, MapIndex

def ShowImage(Index=np.random.randint(0, SetSize)):
    image, Target, Label, MapIndex = Sample(Index)
    plt.imshow(image.reshape(28, 28).T, cmap="Greys")
    plt.title(f"Index: {Index}, Label: {Label}")
    plt.show()

# mapping:
# 0: 0
# 1: 1
# 2: 2
# 3: 3
# 4: 4
# 5: 5
# 6: 6
# 7: 7
# 8: 8
# 9: 9
# 10: A
# 11: B
# 12: C
# 13: D
# 14: E
# 15: F
# 16: G
# 17: H
# 18: I
# 19: J
# 20: K
# 21: L
# 22: M
# 23: N
# 24: O
# 25: P
# 26: Q
# 27: R
# 28: S
# 29: T
# 30: U
# 31: V
# 32: W
# 33: X
# 34: Y
# 35: Z
# 36: a
# 37: b
# 38: c
# 39: d
# 40: e
# 41: f
# 42: g
# 43: h
# 44: i
# 45: j
# 46: k
# 47: l
# 48: m
# 49: n
# 50: o
# 51: p
# 52: q
# 53: r
# 54: s
# 55: t
# 56: u
# 57: v
# 58: w
# 59: x
# 60: y
# 61: z