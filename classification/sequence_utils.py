#!/usr/bin/env python
# coding: utf-8

# In[37]:


# python modules
import numpy as np

# my modules
import activation_utils


# In[14]:


"""
This method takes a list of numbers. The format of the list is as follows:
[n_x, n_h1, n_h2, ..., n_o]
where 
n_x = number of features in the input
n_h1, n_h2, ... = number of nodes in first, second, etc. hidden layers
n_o = number of output nodes.

For example:
If layers = [4, 6, 3, 1], then it has:
number of features in the input = 4
number of nodes in the first hidden layer = 6
number of nodes in the second hidden layer = 3
number of nodes in the output layer = 1
"""
def initialize_parameters(layers):
    L = len(layers)
    W = []
    b = []

    for l in range(0, L - 1):
        w_l = np.random.randn(layers[l + 1], layers[l]) * 0.01
        b_l = np.zeros((layers[l + 1], 1))
        W.append(w_l)
        b.append(b_l) 

    return W, b


# In[15]:


def forward_prop(X, W, b):
    """
    We use a cache to store necessary intermediate metrics for each layers.
    In this cache, for each layer, we store the corresponding input (AL - 1), weight(WL),
    bias(bL) and linear_forward(zL).
    
    We store this intermediate metrics, because we will need these for evaulating derivates
    during back-propagation.
    """
    caches = []
    L = len(W)
    for l in range(L):
        if l == 0:
            input = X
        else:
            input = a
        z = np.dot(W[l], input) + b[l]
        if l < L - 1:
            a = relu(z)
        else:
            a = sigmoid(z)
        """
        Here we are creating a cache for the current layer
        and appending to the caches.
        """
        cache = (input, W[l], b[l], z)
        caches.append(cache)
    return a, caches


# In[16]:


def compute_cost(AL, Y):
    m = AL.shape[1]
    # will have to check for why the commented cost function was returning (10, 10)
#     return (-1 / m) * (np.dot(Y, np.log(AL.T)) + (np.dot(1 - Y, np.log(1 - AL.T))))
    cost = -(1 / m) * np.sum((Y * np.log(AL) + (1 - Y) * np.log(1 - AL)))
    cost=np.squeeze(cost)
    return cost


# In[17]:


def linear_back(A_prev, W, b, dZ):
    m = A_prev.shape[1]
    
    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T, dZ)
    
    return dA_prev, dW, db


# In[18]:


"""
During back propagation, we are going to calculate the derivatives (dA_prev, dW, db and dZ),
which will help us to improve our parameters (W and b).
"""
def back_prop(AL, Y, caches):
    L = len(caches)  # number of layers
    m = AL.shape[1]
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    grads = []

    for l in reversed(range(L)):
        A_prev, W, b, Z = caches[l]
        if l == L-1:
            dZ = sigmoid_back(dAL, Z)
        else:
            dZ = relu_back(dA_prev, Z)

        dA_prev, dW, db = linear_back(A_prev, W, b, dZ)
        grad = (dA_prev, dW, db)
        """
        Because we are moving backward, we would need to store the gradient descent (GD)
        for each layers properly. That is, even if we moving from L->L-1->...->1,
        the grads (list containing GDs) in 1->...->L-1->L order.
        (This approach will help us, while updating the weight(W) and bias(b) parameters
        for next iteration.)
        
        Therefore, instead of append, I have used insert(0, grad) below.
        """
        #grads.append(grad)
        grads.insert(0, grad)
    return grads


# In[19]:


def update_parameters(W, b, grads, learning_rate):
    L = len(grads)
    for l in range(L):
        dA_prev, dW, db = grads[l]
        W[l] = W[l] - learning_rate * dW
        b[l] = b[l] - learning_rate * db
    return W, b

