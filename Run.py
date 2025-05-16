import numpy as np
import time
import scipy.io
import matplotlib.pyplot as plt
from DataHandler import *
from Activators import *

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
    plt.title(f"Predicted: {PredictChar} ({round(confidence*100,1)}%), Actual: {Label},")
    plt.xticks([])
    plt.yticks([])
    plt.show()

print("Loading model parameters")
folder = "AllChar"
ModelParameters = np.load(f"{folder}/ModelParameters.npz")
HiddenLayersSizes = ModelParameters["HiddenLayersSizes"]
InputSize = ModelParameters["InputSize"]
OutputSize = ModelParameters["OutputSize"]
epochs = ModelParameters["epochs"]
# act = ModelParameters["act"]
# actout = ModelParameters["actout"]
act = Sigmoid
actout = Sigmoid
ddsAct = actFunctionDerivatives[act]
ddsActout = actFunctionDerivatives[actout]


weights = []
biases = []

print("initialising weights and biases")
for i in range(len(HiddenLayersSizes)+1):
    print(f"Loading weights and biases for layer {i}")
    weights.append(np.load(f"{folder}/weights[{i}].npz")["weights"])
    biases.append(np.load(f"{folder}/biases[{i}].npz")["biases"])

print("Ready")

while True:
    index = input("Enter index to show image: ")
    if index == "":
        SampleShow(np.random.randint(0, SetSize))
    elif index == "q":
        break
    else:
        try:
            SampleShow(int(index))
        except:
            print("Invalid index")
            continue