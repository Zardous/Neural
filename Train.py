from DataHandler import *
from Activators import *
import os

#row of matrix is array of weights on the neuron

#Choose activation functions
act = Asinh
ddsAct = actFunctionDerivatives[act]
actout = SoftMax
ddsActout = actFunctionDerivatives[actout]

weightsInitRange =0.1
weightsMedian = 0.0
biasesInitRange = 0.0
biasesMedian = 0.0

HiddenLayersSizes = (80,40) #TODO: Figure out how to choose layers, maybe train an AI

epochs = 1

eta = 0.01 #learning rate
etaDecay = 0.7 #learning rate decay

targetAccuracy = 0.95 #Targeted accuracy

trainFraction = 0.9 #fraction of the dataset to use for training

def ForwardPass(inp):
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

#example: weights[0] is weights for inputs to the first hidden layer

if __name__ == '__main__': #only run this if this file is run directly, not when imported
    print('Training model')

    #Randomly initialize every layer's weights and biases in a list, taking into account whether or not there are hidden layers
    weights = []
    if len(HiddenLayersSizes) > 0:
        weights.append(np.random.randn(HiddenLayersSizes[0], InputSize) * weightsInitRange + weightsMedian)
        for i in range(1, len(HiddenLayersSizes)):
            weights.append(np.random.randn(HiddenLayersSizes[i], HiddenLayersSizes[i-1]) * weightsInitRange + weightsMedian)
        weights.append(np.random.randn(OutputSize, HiddenLayersSizes[-1]) * weightsInitRange + weightsMedian)
    elif len(HiddenLayersSizes) == 0:
        weights.append(np.random.randn(OutputSize, InputSize) * weightsInitRange + weightsMedian)
    else:
        raise ValueError('Invalid number of hidden layers')
    
    biases = []
    for i in range(0, len(HiddenLayersSizes)):
        biases.append(np.random.randn(HiddenLayersSizes[i], 1) * biasesInitRange + biasesMedian)
    biases.append(np.random.randn(OutputSize, 1) * biasesInitRange + biasesMedian)
    print('Initialised weights and biases')

    # Train the model
    Tstart = time.time()
    epoch = 1
    accuracy = 0
    results = np.zeros((5000,1))

    outputFolder = input('Enter folder to store output: ')
    if outputFolder == '':
        outputFolder = 'Output'
        print(f'Defaulting output folder to: \'{outputFolder}\'')
    else:
        print(f'Output folder set to: \'{outputFolder}\'')

    while epoch <= epochs or accuracy < targetAccuracy:
        correct = 0
        incorrect = []
        epochStart = time.time() # Store starting time of epoch for iteration time calculation
        for index in range(round(SetSize * trainFraction)):
            image, Target, Label, MapIndex = Sample(index)
            inp = image

            activations, stimuli = ForwardPass(inp)

            PredictIndex = np.argmax(activations[-1]) #Find the index that has the highest predicted probability
            PredictChar = chr(data['dataset'][0][0][2][PredictIndex][-1]) #Map the index to a character

            if PredictChar != Label:
                incorrect += [index] #Store indexes that were predicted incorrectly
            
            results[index % (results.shape[0])] = 1 if PredictChar == Label else 0 #If predicted correctly, set an entry to 1, else 0
            accuracy = np.sum(results) / min( results.shape[0] , index + 1 + (epoch-1) * SetSize * trainFraction)

            if index%100==0:
                print(f'Index: {index}/{round(SetSize*trainFraction)}, epoch: {epoch}/{epochs}, Accuracy: {round(100*accuracy,1)}%, Iteration time: {round(1000*(time.time()-epochStart)/(index+1),2)}ms Compute time: {int(np.floor((time.time()-Tstart)/60))}m{round((time.time()-Tstart)%60)}s        ', end='\r')

            #backward pass

            # cost = np.square(Target - activations[-1]) #Calculate cost

            for i in range(len(HiddenLayersSizes),-1,-1):
                if i == len(HiddenLayersSizes): #output layer
                    gradC = activations[i] - Target #gradient of the cost function
                    delta = gradC * ddsActout(stimuli[i])
                else:
                    delta = weights[i+1].T @ delta * ddsAct(stimuli[i])
                if i != 0:
                    weights[i] += - eta * delta @ activations[i-1].T
                else: # If the first hidden layer accesses the previous layer, give the input
                    # check if this part is correct when there is no hidden layer
                    weights[i] += - eta * delta @ inp.T    
                biases[i] += - eta * delta
        
        eta *= etaDecay
        epoch += 1
    print('\nFinished training')

    # If the output folder doesn't exist, create it
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)

    # Save weights
    for w in range(len(weights)):
        np.savez(f'{outputFolder}/weights[{w}].npz', weights=weights[w])
    print(f'Saved weights')

    # Save biases
    for b in range(len(biases)):
        np.savez(f'{outputFolder}/biases[{b}].npz', biases=biases[b])
    print(f'Saved biases')

    # Save model parameters
    np.savez(f'{outputFolder}/ModelParameters.npz', 
             HiddenLayersSizes=HiddenLayersSizes, 
             epochs=epoch, 
             InputSize=InputSize, 
             OutputSize=OutputSize, 
             SetSize=SetSize, 
             act=act.__name__, 
             actout=actout.__name__, 
             accuracy=accuracy,
             )

    layers = ''
    layers += f'{InputSize}-'
    for i in range(len(HiddenLayersSizes)):
        layers += f'{HiddenLayersSizes[i]}-'
    layers += f'{OutputSize}'

    Summary = f'{layers}, {setName.upper()}, {round(100*accuracy,1)}%, {act.__name__}, {actout.__name__}'
    with open(f'{outputFolder}/{Summary}', 'w') as f:
        f.write('')

    print('Saved model parameters')
