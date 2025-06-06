'''
This library defines two functions that can be used to initialise the weights, it takes the amount of neurons in the previous layer into 
account for stability. KaimingN uses a normal distribution, KaimingU uses a flat and bounded distribution.
'''

import numpy as np

def KaimingN(IN, LOCAL, OUT): # Asymmetric functions
    return np.random.randn(LOCAL,IN) * np.sqrt(2/IN)

def KaimingU(IN, LOCAL, OUT): # Asymmetric functions
    return (np.random.rand(LOCAL, IN)-0.5)*2 * np.sqrt(6/IN)