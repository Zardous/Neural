from DataHandler import *
from Activators import *
import os
from UI import *

#row of matrix is array of weights on the neuron

#Choose activation functions
act = Tanh
actout = SoftMax
ddsAct = eval("dds"+act.__name__)
ddsActout = eval("dds"+actout.__name__)

weightsInitRange = 0.1
weightsMedian = 0.0
biasesInitRange = 0.0
biasesMedian = 0.0

HiddenLayersSizes = (5,) #TODO: Figure out how to choose layers, maybe train an AI
batchSize = 20

epochs = 4

etaStart = 0.1 #learning rate
etaDecay = 0.7 #learning rate decay

targetAccuracy = 0.95 #Targeted accuracy

trainFraction = 0.9 #fraction of the dataset to use for training

def ForwardPass(inp, weights, biases):
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

def randomiseWB(InputSize, OutputSize, HiddenLayersSizes, weightsInitRange, weightsMedian, biasesInitRange, biasesMedian):
    global weights, deltaWeights, biases, deltaBiases
    weights = []
    deltaWeights = []
    if len(HiddenLayersSizes) > 0:
        weights.append((np.random.rand(HiddenLayersSizes[0], InputSize)-1/2*recentre) * weightsInitRange + weightsMedian)
        deltaWeights.append(np.zeros((HiddenLayersSizes[0], InputSize)))
        for i in range(1, len(HiddenLayersSizes)):
            weights.append((np.random.rand(HiddenLayersSizes[i], HiddenLayersSizes[i-1])-1/2*recentre) * weightsInitRange + weightsMedian)
            deltaWeights.append(np.zeros((HiddenLayersSizes[i], HiddenLayersSizes[i-1])))
        weights.append((np.random.rand(OutputSize, HiddenLayersSizes[-1])-1/2*recentre) * weightsInitRange + weightsMedian)
        deltaWeights.append(np.zeros((OutputSize, HiddenLayersSizes[-1])))
    elif len(HiddenLayersSizes) == 0:
        weights.append((np.random.rand(OutputSize, InputSize)-1/2*recentre) * weightsInitRange + weightsMedian)
        deltaWeights.append(np.zeros((OutputSize, InputSize)))
    else:
        raise ValueError('Invalid number of hidden layers')

    biases = []
    deltaBiases = []
    for i in range(0, len(HiddenLayersSizes)):
        biases.append((np.random.rand(HiddenLayersSizes[i], 1)-1/2*recentre) * biasesInitRange + biasesMedian)
        deltaBiases.append(np.zeros((HiddenLayersSizes[i], 1)))
    biases.append((np.random.rand(OutputSize, 1)-1/2*recentre) * biasesInitRange + biasesMedian)
    deltaBiases.append(np.zeros((OutputSize, 1)))
    print('Randomised weights and biases')

def saveModel(outputFolder, weights, biases, HiddenLayersSizes, epoch, InputSize, OutputSize, SetSize, act, actout, accuracy, setName):

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

def Divisor(x):
    if x in (0,1):
        return x

    for i in range(round(x**0.5), x // 2+1):
        if x % i == 0:
            return int(i)
        return 1

if __name__ == '__main__': #only run this if this file is run directly, not when imported
    pg.init()
    # Screen dimensions
    windowSizePercentage = 0.95
    displayInfo = pg.display.Info()
    windowWidth = int(displayInfo.current_w * windowSizePercentage)
    windowHeight = int(displayInfo.current_h * windowSizePercentage)
    pg.display.set_caption('Perceptron')
    window = pg.display.set_mode((windowWidth, windowHeight), pg.RESIZABLE)
    resetButton = Button(window, (10,10), 'Reset', 50)
    saveButton = Button(window, (10, 60), 'Save', 1)

    print('Training model')
    recentre = 1

    #Randomly initialize every layer's weights and biases in a list, taking into account whether or not there are hidden layers
    randomiseWB(InputSize, OutputSize, HiddenLayersSizes, weightsInitRange, weightsMedian, biasesInitRange, biasesMedian)

    # Train the model
    Tstart = time.time()
    epoch = 1
    eta = etaStart
    accuracy = 0
    results = np.zeros((5000,1))

    # outputFolder = input('Enter folder to store output: ')
    outputFolder = 'Output'
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

            activations, stimuli = ForwardPass(inp, weights, biases)

            PredictIndex = np.argmax(activations[-1]) #Find the index that has the highest predicted probability
            PredictChar = chr(data['dataset'][0][0][2][PredictIndex][-1]) #Map the index to a character

            if PredictChar != Label:
                incorrect += [index] #Store indexes that were predicted incorrectly
            
            results[index % (results.shape[0])] = 1 if PredictChar == Label else 0 #If predicted correctly, set an entry to 1, else 0
            accuracy = np.sum(results) / min( results.shape[0] , index + 1 + (epoch-1) * SetSize * trainFraction)

            # ==== Draw frame ====#
            if index%100==0:
                window.fill((10,10,10))
                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        quit()
                    resetButton.handleEvent(event)
                    saveButton.handleEvent(event)

                accuracyBar = Header(window, (window.get_width()/2,10), f'Accuracy: {accuracy*100:.1f}%')
                accuracyBar.pixelCoords = window.get_width()/2-accuracyBar.Dimensions[0]/2, accuracyBar.pixelCoords[1]
                accuracyBar.draw()
                resetButton.draw()
                if resetButton.state == True:
                    eta = etaStart
                    randomiseWB(InputSize, OutputSize, HiddenLayersSizes, weightsInitRange, weightsMedian, biasesInitRange, biasesMedian)
                    resetButton.state = False
                saveButton.draw()
                if saveButton.state == True:
                    saveModel(outputFolder, weights, biases, HiddenLayersSizes, epoch, InputSize, OutputSize, SetSize, act, actout, accuracy, setName)
                    saveButton.state = False

                pix = 5
                y=100
                dx = 40
                dy=0
                for layer in range(len(biases)):
                    x = dx
                    y = y + layer*(dy)
                    for i in range(len(biases[layer])):
                        v = weights[layer][i]
                        max_abs = np.max(np.abs(v))
                        norm_v = v/max_abs
                        norm_v = norm_v.reshape(Divisor(norm_v.shape[0]), norm_v.shape[0]//Divisor(norm_v.shape[0]))
                        plotArray(window, (x,y), norm_v*255).draw(pix)
                        x+= (28*pix+dx)
                        if x+(28*pix)>window.get_width():
                            x=dx
                            y += (dy*pix+5)
                        dy = norm_v.shape[0]


                pg.display.flip()

                print(f'Index: {index}/{round(SetSize*trainFraction)}, epoch: {epoch}/{epochs}, Accuracy: {round(100*accuracy,1)}%, Iteration time: {round(1000*(time.time()-epochStart)/(index+1),2)}ms Compute time: {int(np.floor((time.time()-Tstart)/60))}m{round((time.time()-Tstart)%60)}s        ', end='\r')

            #backward pass
            for i in range(len(HiddenLayersSizes),-1,-1):
                if i == len(HiddenLayersSizes): #output layer
                    gradC = activations[i] - Target #gradient of the cost function
                    delta = gradC * ddsActout(stimuli[i])
                else:
                    delta = weights[i+1].T @ delta * ddsAct(stimuli[i])
                if i != 0:
                    # weights[i] += - eta * delta @ activations[i-1].T
                    deltaWeights[i] += - eta * delta @ activations[i-1].T 
                else: # If the first hidden layer accesses the previous layer, give the input
                    # weights[i] += - eta * delta @ inp.T    
                    deltaWeights[i] += - eta * delta @ inp.T 
                # biases[i] += - eta * delta
                deltaBiases[i] += - eta * delta 

            if index%batchSize == 0:
                for i in range(len(weights)):
                    weights[i] += deltaWeights[i]/batchSize
                    biases[i] += deltaBiases[i]/batchSize
                for layer in deltaWeights:
                    layer *= 0
                for layer in deltaBiases:
                    layer *= 0
        
        eta *= etaDecay
        epoch += 1
    print('\nFinished training')
    saveModel(outputFolder, weights, biases, HiddenLayersSizes, epoch, InputSize, OutputSize, SetSize, act, actout, accuracy, setName)
