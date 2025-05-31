from DataHandler import *
from Activators import *
import time
import os
from UI import *

#row of matrix is array of weights on the neuron

size = 40

SetName = 'Signs'
T0 = time.time()
root = 'data/Training'
InputSize = size**2 *3
OutputSize = len(os.listdir(f'{root}'))

#Choose activation functions
act = SiLU
actout = SoftMax
ddsAct = eval("dds"+act.__name__)
ddsActout = eval("dds"+actout.__name__)

weightsInitRange = 0.1
weightsMedian = 0.0
biasesInitRange = 0.0
biasesMedian = 0.0

HiddenLayersSizes = (128,64) 
batchSize = 48

epochs = 4

etaStart = 0.2 #learning rate at start
etaDecay = 0.09 #learning rate decay

targetAccuracy = 0.95 #Targeted accuracy

trainFraction = 0.9 #fraction of the dataset to use for training

drawFrequency = 500

def ForwardPass():
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

def saveModel(outputFolder, weights, biases, HiddenLayersSizes, epoch, InputSize, OutputSize, SetSize, act, actout, accuracy, SetName):

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

    Summary = f'{layers}, {SetName.upper()}, {round(100*accuracy,1)}%, {act.__name__}, {actout.__name__}'
    with open(f'{outputFolder}/{Summary}', 'w') as f:
        f.write('')

    print('Saved model parameters')

def Divisor(x):
    if x in (0,1):
        return x

    d=0
    i=1
    while i < round(x**0.5)+1:
        if i>=d and x%i==0:
            d=i
        i+=1
    return d

def screenAct(x):
    return np.maximum(Sigmoid(2.5*(x-0.5))*(1-np.exp(-3*x))+Sigmoid(2.5*(0.5-x))*x/3,0)

def backpropagation(gradC, write=True):
    for i in range(len(HiddenLayersSizes),-1,-1):
        if i == len(HiddenLayersSizes): #output layer
            delta = gradC * ddsActout(stimuli[i])
        else:
            delta = weights[i+1].T @ delta * ddsAct(stimuli[i])
        if write:
            deltaWeights[i] += - eta * delta @ (inp if i == 0 else activations[i-1]).T
            deltaBiases[i] += - eta * delta 

    deltaSaliency = weights[0].T @ delta
    return deltaWeights, deltaBiases, deltaSaliency
    
if __name__ == '__main__':
    if True: # UI initialisation
        pg.init()
        pg.key.set_repeat(500,50)

        # Screen
        windowSizePercentage = 0.75
        displayInfo = pg.display.Info()
        windowWidth = int(displayInfo.current_w * windowSizePercentage)
        windowHeight = int(displayInfo.current_h * windowSizePercentage)
        pg.display.set_caption('Perceptron')
        window = pg.display.set_mode((windowWidth, windowHeight), pg.RESIZABLE)

        resetButton = Button(window, (10,10), 'Reset', 50)
        saveButton = Button(window, (10, 60), 'Save', 1)
        vDef = txtInput(window,(resetButton.Dimensions[0]+20, 10), 'Display command')
        vDef.text = 'v = v'
        vDef.output = vDef.text
        outputFolderBox = txtInput(window, (vDef.pixelCoords[0], vDef.pixelCoords[1]+50), 'Output folder')
        outputFolderBox.text = 'Output'
        outputFolderBox.output = outputFolderBox.text
        terminal = txtInput(window, (0,0), 'Terminal')
        accuracyBar = Header(window, (window.get_width()/2,10), 'Loading')

        pix = 1.5
        windowColour = 20

    print('Loading all training signs')
    dataset = loadAllSigns('data/NPZs40px')
    SetSize = len(dataset)

    print('Training model')
    recentre = 1

    #Randomly initialize every layer's weights and biases in a list, taking into account whether or not there are hidden layers
    randomiseWB(InputSize, OutputSize, HiddenLayersSizes, weightsInitRange, weightsMedian, biasesInitRange, biasesMedian)

    if True: # Internal parameters initialisation
        Tstart = time.time()
        epoch = 1
        eta = etaStart
        accuracy = 0
        SaliencyDepth = len(weights)
        results = np.zeros((5000,1))
        confidenceList = np.zeros((5000,1))
        saliency = np.zeros((OutputSize,InputSize))

    # outputFolder = input('Enter folder to store output: ')
    outputFolder = outputFolderBox.output
    if outputFolder == '':
        outputFolder = 'Output'
        print(f'Defaulting output folder to: \'{outputFolder}\'')
    else:
        print(f'Output folder set to: \'{outputFolder}\'')

    while epoch <= epochs or accuracy < targetAccuracy:
        index = 0
        correct = 0
        incorrect = []
        epochStart = time.time() # Store starting time of epoch for iteration time calculation
        popset = createPopset(dataset)
        while len(popset) > 0:
            eta = etaStart * np.exp(-etaDecay * epoch)

            image, Target, Label, MapIndex = SampleSign(dataset, popset, OutputSize)
            inp = image.flatten()
            inp.shape += (1,)

            activations, stimuli = ForwardPass()

            PredictIndex = np.argmax(activations[-1]) #Find the index that has the highest predicted probability
            if PredictIndex != MapIndex:
                incorrect += [Label] #Store indeces that were predicted incorrectly
            results[index % (results.shape[0])] = 1 if PredictIndex == MapIndex else 0 #If predicted correctly, set an entry to 1, else 0
            accuracy = np.sum(results) / min(results.shape[0] , index + 1 + (epoch-1) * SetSize)
            confidenceList[index % (results.shape[0])] = np.max(activations[-1])
            confidence = np.sum(confidenceList) / min(confidenceList.shape[0] , index + 1 + (epoch-1) * SetSize)

            deltaWeights, deltaBiases, _ = backpropagation(activations[-1] - Target)

            # ==== Draw frame ====#
            if index%drawFrequency==0:
                window.fill((windowColour, windowColour, windowColour))

                # Maps
                x=0
                y=125
                dx = 30
                dy=0
                if False:
                    actf = act if 0 < len(HiddenLayersSizes) else actout
                    saliency = actf(weights[0] + biases[0])
                    for i in range(1, SaliencyDepth):
                        actf = act if i < len(HiddenLayersSizes) else actout
                        saliency = actf(weights[i] @ saliency + biases[i])
                
                if True: # Draw Saliency maps
                    x += dx
                    y += (dy+dx)
                    i=0
                    for row in saliency:
                        oneHot = np.zeros((OutputSize,1))
                        oneHot[i]=1
                        inp = row
                        inp.shape += (1,)
                        activations, stimuli = ForwardPass()
                        v = inp + eta * backpropagation(oneHot, False)[2]
                        # v = Sigmoid(4*v-2)
                        v = screenAct(v)
                        v = v.squeeze()
                        saliency[i]=v
                        v = RestoreRGB(v,size)
                        try:
                            exec(vDef.output)
                        except Exception as e:
                            print(e)
                            vDef.active = True
                        plotArray(window, (x,y), v.clip(0,1)*255).draw(pix)
                        x+= (v.shape[0] * pix + dx)
                        dy = v.shape[0]*pix
                        if x+(v.shape[0] * pix) > window.get_width() - dx:
                            x=dx
                            y += (dy+dx)
                        i+=1

                active = True
                while active: # Handle and possibly show UI
                    for event in pg.event.get():
                        if event.type == pg.QUIT:
                            quit()
                        resetButton.handleEvent(event)
                        saveButton.handleEvent(event)
                        vDef.handleEvent(event)
                        terminal.handleEvent(event)
                        outputFolderBox.handleEvent(event)
                    # UI
                    accuracyBar.pixelCoords = window.get_width()/2-accuracyBar.Dimensions[0]/2, accuracyBar.pixelCoords[1]
                    accuracyBar.header = f'Accuracy: {accuracy*100:.1f}%, Confidence: {confidence*100:.1f}%'
                    accuracyBar.draw()
                    terminal.pixelCoords = window.get_width()-10-terminal.Dimensions[0], 10
                    try:
                        exec(terminal.output)
                    except Exception as e:
                        print(e)
                        terminal.active = True
                    terminal.draw()
                    vDef.draw()
                    resetButton.draw()
                    if resetButton.state == True:
                        eta = etaStart
                        randomiseWB(InputSize, OutputSize, HiddenLayersSizes, weightsInitRange, weightsMedian, biasesInitRange, biasesMedian)
                        resetButton.state = False
                    saveButton.draw()
                    if saveButton.state == True:
                        outputFolder = outputFolderBox.output
                        if outputFolder == '':
                            outputFolder = 'Output'
                            print(f'Defaulting output folder to: \'{outputFolder}\'')
                        else:
                            print(f'Output folder set to: \'{outputFolder}\'')
                        saveModel(outputFolder, weights, biases, HiddenLayersSizes, epoch, InputSize, OutputSize, SetSize, act, actout, accuracy, SetName)
                        saveButton.state = False
                    outputFolderBox.draw()
                    active = terminal.active or vDef.active or outputFolderBox.active
                    if active:
                        pg.display.flip()

                # rest = RestoreRGB(inp,size)*255
                # plotArray(window, (50,500), rest).draw(5)
                pg.display.flip()
                print(f'Index: {index}/{round(SetSize*trainFraction)}, epoch: {epoch}/{epochs}, Accuracy: {round(100*accuracy,1)}%, Iteration time: {round(1000*(time.time()-epochStart)/(index+1),2)}ms Compute time: {int(np.floor((time.time()-Tstart)/60))}m{round((time.time()-Tstart)%60)}s        ', end='\r')

            if index%batchSize == 0 or len(popset)==0:
                for i in range(len(weights)):
                    weights[i] += deltaWeights[i]/batchSize
                    biases[i] += deltaBiases[i]/batchSize
                for layer in deltaWeights:
                    layer *= 0
                for layer in deltaBiases:
                    layer *= 0
            index +=1
        epoch += 1
    
    print('\nFinished training')
    saveModel(outputFolder, weights, biases, HiddenLayersSizes, epoch, InputSize, OutputSize, SetSize, act, actout, accuracy, SetName)


                # for layer in range(len(biases)):
                #     x = dx
                #     y += (dy+dx)
                #     for i in range(len(biases[layer])):
                #         v = weights[layer][i]
                #         # v = v / np.max(np.abs(v))
                #         v = v.reshape(Divisor(v.shape[0]), v.shape[0]//Divisor(v.shape[0]))
                #         if v.size == 3 * size**2:
                #             v = RestoreRGB(v,size)
                #             try:
                #                 exec(vDef.output)
                #             except Exception as e:
                #                 print(e)
                #                 vDef.active = True
                #         plotArray(window, (x,y), v*255).draw(pix)
                #         x+= (v.shape[0] * pix + dx)
                #         dy = v.shape[0]*pix
                #         if x+(v.shape[0] * pix) > window.get_width() - dx and i+1 != len(biases[layer]):
                #             x=dx
                #             y += (dy+dx)