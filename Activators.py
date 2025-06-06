'''
This is a library that defines many common activation functions and their derivatives.
'''

import numpy as np

def Sigmoid(s):
    return 1 / (1 + np.exp(-s))

def ddsSigmoid(s):
    return np.exp(-s)/((1 + np.exp(-s))**2)

def SiLU(s):
    return s/(1+np.exp(-s))

def ddsSiLU(s):
    return (1+np.exp(-s)+s*np.exp(-s))/(1+np.exp(-s))**2

def ReLU(s):
    return np.maximum(0, s)

def ddsReLU(s):
    return np.where(s >= 0, 1, 0)

def LeakyReLU(s, alpha=0.05):
    return np.where(s >= 0, s, alpha*s)

def ddsLeakyReLU(s, alpha=0.05):
    return np.where(s >= 0, 1, alpha)

def SoftPlus(s):
    return np.log1p(np.exp(s))

def ddsSoftPlus(s):
    return np.exp(s)/(1 + np.exp(s))

def Tanh(s):
    return np.tanh(s)

def ddsTanh(s):
    return 1 - np.tanh(s)**2

def Asinh(s):
    return np.arcsinh(s)

def ddsAsinh(s):
    return 1/(s**2+1)**0.5

def SoftMax(s):
    e = np.exp(s - np.max(s))
    return e / np.sum(e)

def ddsSoftMax(s):
    return 1

def Gaussian(s):
    return np.exp(-s**2)

def ddsGaussian(s):
    return -2*s*np.exp(-s**2)

def Exp(s):
    return np.exp(s)

def ddsExp(s):
    return np.exp(s)