#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

"""The following activation functions are used in forward propagation."""
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def relu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    return np.exp(Z)/sum(np.exp(Z))

"""The following derivatives for activation functions are used back propagation."""
def sigmoid_back(dA, Z):
    S = sigmoid(Z)
    dS = S * (1 - S)
    dZ = dA * dS     # chain rule
    return dZ

def relu_back(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def softmax_back(dA, Z):
    # need to implement
    return None


# In[ ]:




