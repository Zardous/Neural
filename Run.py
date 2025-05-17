import numpy as np
import matplotlib.pyplot as plt
from DataHandler import *
from Activators import *
# from Train import ForwardPass
from Train import trainFraction

def SampleShow(index):
    input = Sample(index)[0]
    Label = Sample(index)[2]
    activations = ForwardPass(input)[0]
    PredictIndex = np.argmax(activations[-1])
    PredictChar = chr(data['dataset'][0][0][2][PredictIndex][-1])
    confidence = activations[-1][PredictIndex][0]
    plt.imshow(input.reshape(28, 28).T, cmap='Greys')
    plt.title(f'Predicted: {PredictChar} ({round(confidence*100,1)}%), Actual: {Label},')
    plt.xticks([])
    plt.yticks([])
    plt.show()

# TODO: Import this function from Train.py
def ForwardPass(inp): #copy of the function in Train.py
    activations = []
    stimuli = []
    for i in range(len(biases)):
        if i == 0:
            z = weights[i] @ inp + biases[i]
            stimuli.append(z)
            activations.append(act(z))
        elif i < len(biases) - 1:
            z = weights[i] @ activations[i-1] + biases[i]
            stimuli.append(z)
            activations.append(act(z))
        else:
            z = weights[i] @ activations[i-1] + biases[i]
            stimuli.append(z)
            activations.append(actout(z))
    return activations, stimuli

def benchmark():
    correct = 0
    count = 0
    Tstart = time.time()
    global incorrect
    incorrect = []
    for index in range(round(SetSize * trainFraction), SetSize):
            count += 1
            image, Target, Label, MapIndex = Sample(index)
            inp = image

            activations, stimuli = ForwardPass(inp)

            PredictIndex = np.argmax(activations[-1]) #Find the index that has the highest predicted probability
            PredictChar = chr(data['dataset'][0][0][2][PredictIndex][-1]) #Map the index to a character

            if PredictChar == Label:
                correct += 1
            else:
                incorrect += [index] #Store indexes that were predicted incorrectly
            
            accuracy = correct / count

            print(f'Index: {index}/{SetSize}, Accuracy: {round(100*accuracy,1)}%, Iteration time: {round(1000*(time.time()-Tstart)/(index+1),2)}ms Compute time: {int(np.floor((time.time()-Tstart)/60))}m{round((time.time()-Tstart)%60)}s        ', end='\r')

if __name__ == '__main__': #only run if this file is run directly, not when imported
    inputFolder = input('Enter folder to load model: ')
    if inputFolder == '':
        inputFolder = 'Output'
        print(f'Defaulting input folder to: \'{inputFolder}\'')
    else:
        print(f'Input folder set to: \'{inputFolder}\'')
    
    print('Loading model parameters')
    ModelParameters = np.load(f'{inputFolder}/ModelParameters.npz')
    HiddenLayersSizes = ModelParameters['HiddenLayersSizes']
    InputSize = ModelParameters['InputSize']
    OutputSize = ModelParameters['OutputSize']
    epochs = ModelParameters['epochs']
    act = actFunctionNames[str(ModelParameters['act'])]
    actout = actFunctionNames[str(ModelParameters['actout'])]
    ddsAct = actFunctionDerivatives[act]
    ddsActout = actFunctionDerivatives[actout]

    weights = []
    biases = []

    print('initialising weights and biases')
    for i in range(len(HiddenLayersSizes)+1):
        print(f'Loading weights and biases for layer {i}')
        weights.append(np.load(f'{inputFolder}/weights[{i}].npz')['weights'])
        biases.append(np.load(f'{inputFolder}/biases[{i}].npz')['biases'])
    print('Ready')

    # Sample and show an image from the dataset
    while True:
        index = input('Enter index to show image: ')
        if index == '':
            SampleShow(np.random.randint(trainFraction*SetSize, SetSize))
        elif index == 'q':
            break
        else:
            try:
                SampleShow(int(index))
            except:
                print('Invalid index')
                continue