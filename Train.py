from DataHandler import *
from Activators import *

#row of matrix is array of weights on the neuron

#Choose activation functions
act = Sigmoid
ddsAct = actFunctionDerivatives[act]
actout = Sigmoid
ddsActout = actFunctionDerivatives[actout]

HiddenLayersSizes = (80,80)

epochs = 3

eta = 0.05

targetAccuracy = 0.95

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

def SampleShow(index):
    input = Sample(index)[0]
    Label = Sample(index)[2]
    activations = ForwardPass(input)[0]
    PredictIndex = np.argmax(activations[-1])
    PredictChar = chr(data["dataset"][0][0][2][PredictIndex][-1])
    confidence = activations[-1][PredictIndex][0]
    plt.imshow(input.reshape(28, 28).T, cmap="Greys")
    plt.title(f"Predicted: {PredictChar}({round(confidence*100,1)}%), Actual: {Label},")
    plt.show()

#w[0] is weights for inputs to the first hidden layer

#Initialize every layer's weight in a list, taking into account wether or not there are hidden layers
initrange =0.05
weights = []
if len(HiddenLayersSizes) > 0:
    weights.append(np.random.randn(HiddenLayersSizes[0], InputSize)*initrange)
    for i in range(1, len(HiddenLayersSizes)):
        weights.append(np.random.randn(HiddenLayersSizes[i], HiddenLayersSizes[i-1])*initrange)
    weights.append(np.random.randn(OutputSize, HiddenLayersSizes[-1])*initrange)
elif len(HiddenLayersSizes) == 0:
    weights.append(np.random.randn(OutputSize, InputSize)*initrange)
else:
    raise ValueError("Invalid number of hidden layers")

biases = []
for i in range(0, len(HiddenLayersSizes)):
    biases.append(np.random.randn(HiddenLayersSizes[i], 1)*initrange)
biases.append(np.random.randn(OutputSize, 1)*initrange)


epoch = 1
accuracy = 0
results = np.zeros((1000,1))
while epoch <= epochs or accuracy < targetAccuracy:
    correct = 0
    incorrect = []
    for index in range(SetSize): #SetSize
        image, Target, Label, MapIndex = Sample(index)
        inp = image

        activations, stimuli = ForwardPass(inp)

        PredictIndex = np.argmax(activations[-1])
        PredictChar = chr(data["dataset"][0][0][2][PredictIndex][-1])
        output = activations[-1]

        if int(PredictIndex) == int(MapIndex):
            correct += 1
        else:
            incorrect += [index]
        if index % 1000 == 0:
            correct = 0
        
        results[index%1000] = 1 if int(PredictIndex) == int(MapIndex) else 0
        accuracy = np.sum(results) / 1000

        print("\033[K\r", end="") #clear line
        print(f"Correct: {"Yes" if PredictChar==Label else "No "} Index: {index}, epoch: {epoch}/{epochs}, Accuracy: {round(100*accuracy,1)}% ", end="")
        # print(weights[-1][0][0], end="")

        #backward pass
        cost = np.square(Target - activations[-1])

        for i in range(len(HiddenLayersSizes),-1,-1):
            if i == len(HiddenLayersSizes): #output layer
                gradC = activations[i] - Target #gradient of the cost function
                delta = gradC * ddsActout(stimuli[i])
            else:
                delta = weights[i+1].T @ delta * ddsAct(stimuli[i])
            if i != 0:
                weights[i] += - eta * delta @ activations[i-1].T
            else:
                weights[i] += - eta * delta @ inp.T    
            biases[i] += - eta * delta
    eta *= 0.9
    epoch += 1
print("\nFinished training")

for w in range(len(weights)):
    np.savez(f"weights[{w}].npz", weights=weights[w])
print(f"Saved weights")

for b in range(len(biases)):
    np.savez(f"biases[{b}].npz", biases=biases[b])
print(f"Saved biases")

np.savez("ModelParameters.npz", HiddenLayersSizes=HiddenLayersSizes, epochs=epoch, InputSize=InputSize, OutputSize=OutputSize, SetSize=SetSize, act=act, actout=actout)
print("Saved model parameters")
