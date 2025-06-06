'''
This is the main code of my project; it trains an AI to classify images.
The code has a lot of functionality which I've tried to document as good as I can in the code itself, it is too much for a docstring.
When running this file, a window will appear. The images are saliency maps that the code optimises gradually: For every type of sign,
the AI displays what type of image maximally activates the class. Essentially, the AI learns itself what the 164 different signs look
like and shows the user what is has learned so far. Due to the nature of the activation function used on the maps, the images naturally
tend to black. This means that unimportant pixels slowly fade to black over time, and only pixels that the AI continues to find important
retain their colour.

The UI also features buttons to reset the model or to save it to the disk in the specified folder. It allows the user to change the
display logic: by default, every saliency map (referred to internally as v) is normalised. Another option is to type 
v = v/np.max(saliency), to normalise the images w.r.t. their global maximum. This can be interesting to show which images are brighter than
others, indicating that the model has a higher confidence in what this type of sign looks like.

There is also a terminal in the top right which is executed with every update of the screen. This can be used to change many of the
models parameters at runtime (for instance, learning rate can be lowered by running etaStart = 0.01)

With the default parameters as specified below, the UI will feel very sluggish. This was done intentionally to balance performance:
not a lot of time is spent updating the screen or training the saliency maps, so that more time is spent tuning the model. If needed,
the drawFrequency can be lowered to update the screen more often.
'''

from DataHandler import *
from Activators import *
from Range import *
from UI import *
import time
import os

# Specify the dimensions of the images in the dataset
size = 40

# Name of the set the model trains on, for clarity only
SetName = 'Signs'

# Load existing model or create new
LoadExisting = True

# Choose activation functions
act = SiLU # For all layers but the last
actout = SoftMax # For the last (output) layer

# Automatically load the derivatives of the above activation functions
ddsAct = eval("dds"+act.__name__)
ddsActout = eval("dds"+actout.__name__)

# Function to use for the range of initial values of the weights
Init = KaimingN

biasesInitRange = 0.0
biasesMedian = 0.0

HiddenLayersSizes = (256,128) 

batchSize = 164//2

epochs = 100 # This is a minimum used for training

etaStart = 0.1 #learning rate at start
etaDecay = 0.0001 #learning rate decay

targetAccuracy = 0.98 #Targeted accuracy

drawFrequency = 300

# Initialisation for some model parameters
T0 = time.time()
root = 'data/Training'
InputSize = size**2 *3
OutputSize = len(os.listdir(f'{root}'))

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

def randomiseWB(InputSize, OutputSize, HiddenLayersSizes, biasesInitRange, biasesMedian):
    recentre = 1
    global weights, deltaWeights, biases, deltaBiases
    weights = []
    deltaWeights = []
    if len(HiddenLayersSizes) > 0:
        weights.append(Init(InputSize, HiddenLayersSizes[0], HiddenLayersSizes[1]))
        deltaWeights.append(np.zeros((HiddenLayersSizes[0], InputSize)))
        for i in range(1, len(HiddenLayersSizes)):
            weights.append(Init(HiddenLayersSizes[i-1], HiddenLayersSizes[i], HiddenLayersSizes[i+1] if i+1<len(HiddenLayersSizes) else OutputSize))
            deltaWeights.append(np.zeros((HiddenLayersSizes[i], HiddenLayersSizes[i-1])))
        weights.append(Init(HiddenLayersSizes[-1], OutputSize, 0))
        deltaWeights.append(np.zeros((OutputSize, HiddenLayersSizes[-1])))
    elif len(HiddenLayersSizes) == 0:
        weights.append(Init(InputSize, OutputSize, 0))
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
             eta=eta
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
    
def createDeltaWB(InputSize, HiddenLayersSizes, OutputSize):
        deltaWeights = []
        if len(HiddenLayersSizes) > 0:
            deltaWeights.append(np.zeros((HiddenLayersSizes[0], InputSize)))
            for i in range(1, len(HiddenLayersSizes)):
                deltaWeights.append(np.zeros((HiddenLayersSizes[i], HiddenLayersSizes[i-1])))
            deltaWeights.append(np.zeros((OutputSize, HiddenLayersSizes[-1])))
        elif len(HiddenLayersSizes) == 0:
            deltaWeights.append(np.zeros((OutputSize, InputSize)))
        else:
            raise ValueError('Invalid number of hidden layers')
        deltaBiases = []
        for i in range(0, len(HiddenLayersSizes)):
            deltaBiases.append(np.zeros((HiddenLayersSizes[i], 1)))
        deltaBiases.append(np.zeros((OutputSize, 1)))
        print('Created matrix for delta- weights and biases')
        return deltaWeights, deltaBiases
    
if __name__ == '__main__':
    # UI initialisation (I use 'if True:' to collapse this code in the editor; it is never False)
    if True: 
        pg.init()
        pg.key.set_repeat(500,33)

        # Screen
        windowSizePercentage = 0.75
        displayInfo = pg.display.Info()
        windowWidth = int(displayInfo.current_w * windowSizePercentage)
        windowHeight = int(displayInfo.current_h * windowSizePercentage)
        pg.display.set_caption('Perceptron')
        window = pg.display.set_mode((windowWidth, windowHeight), pg.RESIZABLE)

        # UI
        resetButton = Button(window, (10,10), 'Reset', 50)
        saveButton = Button(window, (10, 60), 'Save', 1)
        vDef = txtInput(window,(resetButton.Dimensions[0]+20, 10), 'Display command')
        vDef.text = 'v = v/np.max(v)'
        vDef.output = vDef.text
        outputFolderBox = txtInput(window, (vDef.pixelCoords[0], vDef.pixelCoords[1]+50), 'Output folder')
        outputFolderBox.text = 'Output'
        outputFolderBox.output = outputFolderBox.text
        terminal = txtInput(window, (0,0), 'Terminal')
        accuracyBar = Header(window, (window.get_width()/2,10), 'Loading')

        pix = 1.5 # Scale of the matrices
        windowColour = 20 # Background colour

    print('Loading all training signs')
    dataset, SetSize = loadAllSigns('datasets/Training')
    print(f'Loaded {SetSize} signs')

    # Internal parameters initialisation 
    if True: 
        Tstart = time.time()
        epoch = 1
        eta = etaStart
        accuracy = 0
        results = np.zeros((5000,1))
        confidenceList = np.zeros((5000,1))
        saliency = np.zeros((OutputSize,InputSize))

    # Initialisation of weights and biases
    if LoadExisting:
        inputFolder = getInputfolder('Trained model')
        weights, biases, HiddenLayersSizes, InputSize, OutputSize, epochsDone, act, actout, ddsAct, ddsActout = loadModel(inputFolder)
        deltaWeights, deltaBiases = createDeltaWB(InputSize, HiddenLayersSizes, OutputSize)
        epoch = epochsDone + 1
    else:
        randomiseWB(InputSize, OutputSize, HiddenLayersSizes, biasesInitRange, biasesMedian)
    print('Initialised weights and biases')


    # outputFolder = input('Enter folder to store output: ')
    outputFolder = outputFolderBox.output
    if outputFolder == '':
        outputFolder = 'Output'
        print(f'Defaulting output folder to: \'{outputFolder}\'')
    else:
        print(f'Output folder set to: \'{outputFolder}\'')
    
    print(f'Learning rate after training on {SetSize} images (full dataset): {np.exp(-etaDecay*SetSize/len(dataset)) *100:.2f}% of start rate')
    print('Training model')
    index =0
    while epoch <= epochs or accuracy < targetAccuracy:
        epochStart = time.time() # Store starting time of epoch for iteration time calculation
        popset = createPopset(dataset) # create a set that the model uses to pick a random image and not pick it again until the next epoch
        while len(popset) > 0:
            eta = etaStart * np.exp(-etaDecay * epoch) # Update learning rate

            image, Target, Label, MapIndex = SampleSign(dataset, popset, OutputSize) # Retrieve image with its data
            inp = image.flatten() / 255
            inp.shape += (1,)

            activations, stimuli = ForwardPass(inp, weights, biases) # Process the image

            PredictIndex = np.argmax(activations[-1]) #Find the output index that has the highest predicted probability
            results[index % (results.shape[0])] = 1 if PredictIndex == MapIndex else 0 #If predicted correctly, set an entry to 1, else 0
            accuracy = np.sum(results) / min(results.shape[0] , index + 1 + (epoch-1) * len(dataset))
            confidenceList[index % (results.shape[0])] = np.max(activations[-1])
            confidence = np.sum(confidenceList) / min(confidenceList.shape[0] , index + 1 + (epoch-1) * len(dataset))

            deltaWeights, deltaBiases, _ = backpropagation(activations[-1] - Target) # Update the model

            # ==== Draw frame ====#
            if index%drawFrequency==0:
                window.fill((windowColour, windowColour, windowColour))

                # Maps
                x=0
                y=100
                dx = 30
                dy=0
                if True: # Train and draw saliency maps, can be disabled for performance
                    x += dx
                    y += (dy+dx)
                    for i, row in enumerate(saliency):
                        oneHot = np.zeros((OutputSize,1))
                        oneHot[i]=1
                        inp = row
                        inp.shape += (1,)
                        activations, stimuli = ForwardPass(inp, weights, biases)
                        v = inp + eta * backpropagation(oneHot, False)[2]
                        # v = Sigmoid(4*v-2) # Alternative for screenAct, this will tend the image to gray instead of black
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

                active = True
                while active: # Handle and possibly show UI

                    # Get updates
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
                    accuracyBar.header = f'Accuracy: {accuracy*100:.2f}%, Confidence: {confidence*100:.2f}%'
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
                        randomiseWB(InputSize, OutputSize, HiddenLayersSizes, biasesInitRange, biasesMedian)
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

                # plotArray(window, (50,500), image).draw(5) # Uncomment to draw the last image that was trained on
                pg.display.flip()
                print(f'Index: {index}/{round(SetSize)}, epoch: {epoch}/{epochs}, Accuracy: {round(100*accuracy,2)}%, Learning rate: {eta}, Iteration time: {round(1000*(time.time()-epochStart)/(index+1),2)}ms Compute time: {int(np.floor((time.time()-Tstart)/60))}m{round((time.time()-Tstart)%60)}s        ', end='\r')

            # Update the weights and biases after training a batch or when the epoch is over
            if (index%batchSize == 0 and index!=0) or len(popset)==0:
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


# This code can be swapped in to visualise layer weights instead of saliency maps
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