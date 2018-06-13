# Alex Yuan, June 2018
import numpy as np
# ACTIVATION FUNCTIONS

def relu(x):
    return np.maximum(x,0)
def relu_prime(x):
    return (x > 0) * 1.0

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_prime(x):
    return sig(x) * (1 - sig(x))

def leaky_relu(x, slope=0.01):
    return (x < 0) * x * slope + (x >= 0) * x
def leaky_relu_prime(x, slope=0.01):
    return (x < 0) * slope + (x >= 0) * 1
