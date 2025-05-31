import numpy as np
import time
import scipy.io
import pygame as pg
import os
# import matplotlib.pyplot as plt

# setName = 'mnist' # Choose from downloaded datasets; mnist or byclass
# print(f'Loading {setName.upper()} dataset')
# T0 = time.time()
# data = scipy.io.loadmat(f'data/emnist-{setName}.mat')
# print(f'Loaded {setName.upper()} in {round(time.time() - T0,2)}s')
# SetSize = data['dataset'][0][0][0][0][0][0].shape[0]
# OutputSize = data['dataset'][0][0][2].shape[0]
# InputSize = data['dataset'][0][0][0][0][0][0].shape[1]

# images:
# print(data['dataset'][0][0][0][0][0][0])

# labels:
# print(data['dataset'][0][0][0][0][0][1])

# mapping:
# print(data['dataset'][0][0][2])
# print(chr(data['dataset'][0][0][2][MapIndex][-1]))

SetName = 'Signs'
SetSize = 0
T0 = time.time()
root = 'data/Training'
size = 30
InputSize = size**2 *3
OutputSize = len(os.listdir(f'{root}'))

# def Sample(Index=np.random.randint(0, SetSize)):
#     image = data['dataset'][0][0][0][0][0][0][Index].reshape(784,1) / 255
#     shift = np.random.randint(-3,3,(1,2))
#     image = np.roll(image, shift=shift[0,0]+28*shift[0,1], axis=0)
#     image += np.random.randn(*image.shape)/15

#     MapIndex = data['dataset'][0][0][0][0][0][1][Index][0]

#     Label = chr(data['dataset'][0][0][2][MapIndex][-1])

#     Target = np.zeros((OutputSize, 1))
#     Target[MapIndex] = 1
#     return image, Target, Label, MapIndex

def storeAll(size):
    for f in range(len(os.listdir('data/Training'))):
        set = []
        folder = os.listdir('data/Training')[f]
        for i in range(len(os.listdir(f'data/Training/{folder}'))):
            image = os.listdir(f'data/Training/{folder}')[i]
            path = f'data/Training/{folder}/{image}'
            if path[-4:] == '.ppm':
                image = pg.image.load(f'data/Training/{folder}/{image}')
                image = np.transpose(pg.surfarray.array3d(image), (1,0,2))
                image = Resample(image, (size,size))
                set += [image]
        np.savez_compressed(f'data/NPZs{size}px/{folder}',[w for w in set])

def loadAllSigns(path = 'data/NPZs40px'):
    dataset= []
    for set in os.listdir(path):
        dataset+= [np.load(f'{path}/{set}'), set]
    return dataset

def Resample(imageArray, targetSize=(30, 30)):
    hOriginal, wOriginal, _ = imageArray.shape
    hNew, wNew = targetSize
    resized = np.zeros((hNew, wNew, 3), dtype=imageArray.dtype)

    # Compute scaling ratios
    row_scale = hOriginal / hNew
    col_scale = wOriginal / wNew

    for i in range(hNew):
        for j in range(wNew):
            src_i = int(i * row_scale)
            src_j = int(j * col_scale)
            resized[i, j] = imageArray[src_i, src_j]
    return resized

def SampleSign(f, i):
    folder = os.listdir('data/Training')[f]
    image = os.listdir(f'data/Training/{folder}')[i]
    path = f'data/Training/{folder}/{image}'

    image = loadSignImage(path, size) / 255
    shift = np.random.randint(-3,3,(1,2))
    # image = np.roll(image, shift=shift[0,0], axis=0)
    # image = np.roll(image, shift= shift[0,1], axis=1)
    # image += np.random.randn(*image.shape)/15

    MapIndex = f
    Label = MapIndex + 1

    Target = np.zeros((OutputSize, 1))
    Target[MapIndex] = 1
    return image, Target, Label, MapIndex

def nextPath(f,i):
    folder = os.listdir('data/Training')[f]
    image = os.listdir(f'data/Training/{folder}')[i]
    found = False
    while not found:
        i+=1
        if i == len(os.listdir(f'{root}/{folder}')):
            i=0
            f+=1
        if f == len(os.listdir(f'{root}')):
            break
        folder = os.listdir('data/Training')[f]
        image = os.listdir(f'data/Training/{folder}')[i]
        path = f'data/Training/{folder}/{image}'
        if path[-4:] == '.ppm':
            found =True
    return f, i, path

def randomPath():
    found = False
    while not found:
        f = np.random.randint(0,len(os.listdir('data/Training')))
        folder = os.listdir('data/Training')[f]
        i = np.random.randint(0,len(os.listdir(f'data/Training/{folder}')))
        image = os.listdir(f'data/Training/{folder}')[i]
        path = f'data/Training/{folder}/{image}'
        if path[-4:] == '.ppm':
            found =True
    return f, i, path

def loadSignImage(path, size):
    image = pg.image.load(path)
    image = np.transpose(pg.surfarray.array3d(image), (1,0,2))
    image = Resample(image, (size,size))
    return image

# No longer used
# def ShowImage(Index=np.random.randint(0, SetSize)):
#     image, Target, Label, MapIndex = Sample(Index)
#     plt.imshow(image.reshape(28, 28).T, cmap='Greys')
#     plt.title(f'Index: {Index}, Label: {Label}')
#     plt.show()

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