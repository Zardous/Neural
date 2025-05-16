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
    PredictChar = chr(data["dataset"][0][0][2][PredictIndex][-1])
    confidence = activations[-1][PredictIndex][0]
    plt.imshow(input.reshape(28, 28).T, cmap="Greys")
    plt.title(f"Predicted: {PredictChar} ({round(confidence*100,1)}%), Actual: {Label},")
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

if __name__ == "__main__": #only run if this file is run directly, not when imported
    inputFolder = input('Enter folder to load model: ')
    if inputFolder == "":
        inputFolder = "Output"
        print(f"Defaulting input folder to: \"{inputFolder}\"")
    else:
        print(f"Input folder set to: \"{inputFolder}\"")
    
    print("Loading model parameters")
    ModelParameters = np.load(f"{inputFolder}/ModelParameters.npz")
    HiddenLayersSizes = ModelParameters["HiddenLayersSizes"]
    InputSize = ModelParameters["InputSize"]
    OutputSize = ModelParameters["OutputSize"]
    epochs = ModelParameters["epochs"]
    act = Sigmoid #ModelParameters["act"]
    actout = Sigmoid #ModelParameters["actout"]
    ddsAct = actFunctionDerivatives[act]
    ddsActout = actFunctionDerivatives[actout]

    weights = []
    biases = []

    print("initialising weights and biases")
    for i in range(len(HiddenLayersSizes)+1):
        print(f"Loading weights and biases for layer {i}")
        weights.append(np.load(f"{inputFolder}/weights[{i}].npz")["weights"])
        biases.append(np.load(f"{inputFolder}/biases[{i}].npz")["biases"])
    print("Ready")

    while True:
        index = input("Enter index to show image: ")
        if index == "":
            SampleShow(np.random.randint(trainFraction*SetSize, SetSize))
        elif index == "q":
            break
        else:
            try:
                SampleShow(int(index))
            except:
                print("Invalid index")
                continue