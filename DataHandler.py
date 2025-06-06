'''
This is a library meant to simplify the handling of any data in the environment.

I used it to convert all 60 thousand images from .ppm to one big .npz per class of road sign (164 .npzs).

Furthermore, it loads these signs from the .npzs to one array. Two loading methods are supported: balanced and unbalanced.
 - Balanced loading, the 'default', outputs a list of 164 lists; one list for each class with all corresponding signs.
   In truth, the actual balancing is done after loading. This type of dataset just simplifies that process.
 - Unbalanced loading creates one big list that contains lists of an image and its class name, one after the other on the same level.
Both have benefits and drawbacks: Balanced loading is used for training, unbalanced loading is used for testing.

From either list, functions are defined to sample parts of the dataset and output the image, the targeted output of the model, the name of the class and the index of the class.

A function is defined to restore a matrix back to an RGB image.

And lastly, a previously trained model can be loaded from a specific folder.
'''

import numpy as np
import pygame as pg
import os
from Activators import *

def storeAllSigns(size, root = 'data/Testing'):
    for f in range(len(os.listdir(root))):
        set = []
        folder = os.listdir(root)[f]
        for i in range(len(os.listdir(f'{root}/{folder}'))):
            image = os.listdir(f'{root}/{folder}')[i]
            path = f'{root}/{folder}/{image}'
            if path[-4:] == '.ppm':
                image = pg.image.load(f'{root}/{folder}/{image}')
                image = np.transpose(pg.surfarray.array3d(image), (1,0,2))
                image = Resample(image, (size,size))
                set += [image]
        np.savez_compressed(f'data/TestingNPZ/{folder}',[w for w in set])
        # np.savez_compressed(f'data/NPZs{size}px/{folder}',[w for w in set])

def loadAllSignsUnbalanced(path = 'datasets/Testing'):
    dataset= []
    for set in os.listdir(path):
        for img in np.load(f'{path}/{set}')['arr_0']:
            dataset+= [[img, set[:3]]]
        # print(f'Loaded {set}')
    return dataset

def loadAllSigns(path = 'datasets/Training'):
    dataset= []
    count = 0
    for i, set in enumerate(os.listdir(path)):
        dataset += [[]]
        for img in np.load(f'{path}/{set}')['arr_0']:
            dataset[-1]+= [img]
            count += 1
        # print(f'Loaded {set}')
    return dataset, count

def createPopset(dataset):
    popset =[]
    for i in range(len(dataset)):
        popset += [i]
    return popset

def Resample(imageArray, targetSize=(40, 40)):
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

def SampleSignUnbalanced(dataset, popset, OutputSize):
    index = popset.pop(np.random.randint(len(popset)))
    image, Label = dataset[index]

    # shift = np.random.randint(-3,3,(1,2))
    # image = np.roll(image, shift=shift[0,0], axis=0)
    # image = np.roll(image, shift= shift[0,1], axis=1)
    # image += np.random.randn(*image.shape)/15

    MapIndex = int(Label) -1

    Target = np.zeros((OutputSize, 1))
    Target[MapIndex] = 1
    return image, Target, Label, MapIndex

def SampleSign(dataset, popset, OutputSize):
    index = popset.pop(np.random.randint(len(popset)))
    image = dataset[index][np.random.randint(len(dataset[index]))]

    # shift = np.random.randint(-3,3,(1,2))
    # image = np.roll(image, shift=shift[0,0], axis=0)
    # image = np.roll(image, shift= shift[0,1], axis=1)
    # image += np.random.randn(*image.shape)/15

    MapIndex = index
    Label = str(MapIndex + 1)

    Target = np.zeros((OutputSize, 1))
    Target[MapIndex] = 1
    return image, Target, Label, MapIndex

def RestoreRGB(inp,size):
    inp = inp.squeeze()
    reshaped = inp.reshape((size, size, 3))  
    return reshaped  

def getInputfolder(default = 'Output'):
    inputFolder = input('Enter folder to load model: ')
    if inputFolder == '':
        inputFolder = default
        print(f'Defaulting input folder to: \'{inputFolder}\'')
    else:
        print(f'Input folder set to: \'{inputFolder}\'')
    return inputFolder

def loadModel(inputFolder):
    print('Loading model parameters')
    ModelParameters = np.load(f'{inputFolder}/ModelParameters.npz')
    HiddenLayersSizes = ModelParameters['HiddenLayersSizes']
    InputSize = ModelParameters['InputSize']
    OutputSize = ModelParameters['OutputSize']
    epochs = ModelParameters['epochs']
    act = eval(str(ModelParameters['act'])) 
    actout = eval(str(ModelParameters['actout']))
    ddsAct = eval("dds"+str(ModelParameters['act']))
    ddsActout = eval("dds"+str(ModelParameters['actout']))

    weights = []
    biases = []

    print('initialising weights and biases')
    for i in range(len(HiddenLayersSizes)+1):
        print(f'Loading weights and biases for layer {i}')
        weights.append(np.load(f'{inputFolder}/weights[{i}].npz')['weights'])
        biases.append(np.load(f'{inputFolder}/biases[{i}].npz')['biases'])
    print('Ready')
    return weights, biases, HiddenLayersSizes, InputSize, OutputSize, epochs, act, actout, ddsAct, ddsActout