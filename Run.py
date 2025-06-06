import numpy as np
import matplotlib.pyplot as plt
import time
from DataHandler import *
from Activators import *
from TrainSigns import ForwardPass

def benchmark(dataset):
    # Initialise counters
    index = 1
    confidenceTotal = 0
    correct = np.zeros((164,1))
    tested = np.zeros((164,1))

    # Store starting time of epoch for iteration time calculation
    epochStart = time.time() 
    popset = createPopset(dataset)

    # Testing loop
    while len(popset) > 0:
        image, _, _, MapIndex = SampleSignUnbalanced(dataset, popset, OutputSize)
        inp = image.flatten() / 255
        inp.shape += (1,)

        activations, _ = ForwardPass(inp, weights, biases)

        PredictIndex = np.argmax(activations[-1])
        if MapIndex == PredictIndex:
            correct[MapIndex] += 1
        tested[MapIndex] += 1
        accuracy = np.sum(correct) / index
        confidenceTotal += np.max(activations[-1])
        confidence = confidenceTotal / index
        if index%100==0 or len(popset)==0:
            print(f'Index: {index}/{len(dataset)}, Accuracy: {100*accuracy:.2f}%, Confidence: {100*confidence:.2f}%, Iteration time: {round(1000*(time.time()-epochStart)/(index),2)}ms Compute time: {int(np.floor((time.time()-epochStart)/60))}m{round((time.time()-epochStart)%60)}s        ', end='\r')
        index += 1
    return correct, tested

if __name__ == '__main__': #only run if this file is run directly, not when imported

    # Get input folder
    inputFolder = input('Enter folder to load model: ')
    if inputFolder == '':
        inputFolder = 'Trained model'
        print(f'Defaulting input folder to: \'{inputFolder}\'')
    else:
        print(f'Input folder set to: \'{inputFolder}\'')
    
    # Load the configuration of the input model
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
    eta = ModelParameters['eta']

    # Load the weights and biases
    weights = []
    biases = []
    print('initialising weights and biases')
    for i in range(len(HiddenLayersSizes)+1):
        print(f'Loading weights and biases for layer {i}')
        weights.append(np.load(f'{inputFolder}/weights[{i}].npz')['weights'])
        biases.append(np.load(f'{inputFolder}/biases[{i}].npz')['biases'])
    print('Ready')

    print('Loading signs')
    dataset = loadAllSignsUnbalanced('datasets/Testing')
    print('Loaded')

    correct, tested = benchmark(dataset)
    print()

    # Uncomment this to print the accuracy per class
    # print(np.where(tested==0, -1, correct / tested))