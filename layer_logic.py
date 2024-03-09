# buisness logic layer

from abc import ABC, abstractmethod
from typing import Any
import numpy as np

# strategy pattern for layers
class Layer(ABC): 
    """ abstract strategy for layers """
    def __init__(self):
        self.input = None
        self.output = None

    @abstractmethod
    def forward(self, input : np.array) -> np.array:
        pass

    def backward(self, grad_output : np.array, learning_rate : float ) -> np.array:
        pass

class Dense(Layer): 
    """ concrete strategy for dense layers

    #### forward method:
    input: input (column) vector x : np.array
    returns: output (column) vector y : np.array
    formula: y = W x + b, where b = bias and W = weight matrix

    #### backward method:
    updates the parameters (weights matrix and bias) by gradient descent dW = - alpha grad E(W) (E = error / loss function)
    input: grad E(y) : np.array, learning rate alpha : float
    returns: grad E(x) : np.array 
    formulae: 
    * grad E(W) = grad E(y) * x^T
    * grad E(b) = grad E(y)
    * grad E(x) = W^t grad E(y)
     
    """
    def __init__(self, input_size : int, output_size : int):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size,1)

    def forward(self, input : np.array) -> np.array:
        self.input = input
        return np.dot(self.weights,self.input) + self.bias
     
    def backward(self, grad_output : np.array, learning_rate : float) -> np.array:
        self.alpha = learning_rate
        self.grad_E = grad_output
        self.grad_weights = np.dot(self.grad_E, self.input.T)
        

        self.input_gradient =  np.dot(self.weights.T, grad_output) # compute input gradient 
        self.weights -= self.alpha * self.grad_weights # update weights
        self.bias -= self.alpha * self.grad_E # update bias
        
        return self.input_gradient

# strategy pattern for activation functions 
class Activation(Layer): # abstract activation layer
    """ abstract strategy for activation layer """
    def __init__(self, activation, dactivation):
        self.activation = activation # need to pass functions, e.g. lambda functions
        self.dactivation = dactivation

    def forward(self, input : np.array) -> np.array:
        self.input = input
        return self.activation(self.input)

    def backward(self, grad_output : np.array, learning_rate : float) -> np.array:
        self.grad_E = grad_output
        return np.multiply(self.grad_E,self.dactivation(self.input))
    
class Tanh(Activation): # concrete activation layer
     """ concrete strategy for activation layer 
     
     implements tanh(x) as activation function
     """
     def __init__(self):
        tanh = lambda x : np.tanh(x)
        dtanh = lambda x : 1 - np.tanh(x)**2
        super().__init__(tanh,dtanh)

class Sigmoid(Activation): # concrete activation layer
    """ concrete strategy for activation layer 
    
    implements sigmoid function s(x) = 1 / (1 + exp(-x)) as activation function
    """
    def __init__(self):
        sigmoid = lambda x : 1 / (1 + np.exp(-x))
        dsigmoid = lambda x : sigmoid(x)*(1 - sigmoid(x))
        super().__init__(sigmoid, dsigmoid)
    
    