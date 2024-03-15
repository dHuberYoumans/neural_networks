# buisness logic layer

from abc import ABC, abstractmethod
from typing import Any
import numpy as np
from numpy.core.multiarray import array as array
from scipy import signal
import binary_decimal_representation as btd


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
    
class one_dim_convolution(Layer):
    ''' concrete strategy for 1d convolution layer '''
    def __init__(self,input_length : int ,kernel_length : int, depth : int):
        self.input_length = input_length
        self.kernel_length = kernel_length
        self.depth = depth
        self.kernel = np.random.rand(self.depth,self.kernel_length)
        self.bias = np.random.rand(self.depth, self.input_length - self.kernel_length+1)

    def forward(self,input : np.array) -> np.array:  
        self.input = input  
        self.output = np.array([self.bias[i] + signal.correlate(input,self.kernel[i],'valid') for i in range(self.depth)])
        return self.output

    def backward(self, grad_output: np.array, learning_rate: float) -> np.array:
        # compute bias gradient 
        self.grad_bias = grad_output
        # compute kernel gradient
        self.grad_kernel = np.array([signal.correlate(self.input, grad_output[i],'valid') for i in range(self.depth)])
        # compute input gradient
        self.grad_input = sum(np.array([signal.correlate(grad_output[i],self.kernel[i],'valid') for i in range(self.depth)]))
        # update parameters
        self.kernel += - learning_rate*self.grad_kernel
        self.bias += - learning_rate*self.grad_bias

        return self.grad_input

class Reshape(Layer):
    def __init__(self,input_shape,output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input : np.array) -> np.array:
        return input.reshape(self.output_shape)
    
    def backward(self, grad_output : np.array, learning_rate : float) -> np.array:
        return grad_output.reshape(self.input_shape)
    
class BinaryToDecimal(Layer): 
    def __init__(self,input_length,kernel_length): 
        self.input_length = input_length
        self.kernel_length = kernel_length
        

    def forward(self, input: np.array) -> np.array: 
        self.input = input
        self.to_return = []
        for i in range(self.input.shape[0]):
            self.to_return.append(btd.signed_binary_to_decimal(self.input[i]))

        self.to_return = np.array(self.to_return)
        self.to_return = self.to_return.reshape(1,len(self.to_return))


        return np.array(self.to_return)
    
    def backward(self,  grad_output : np.array, learning_rate : float ) -> np.array:
        self.to_return = []
        self.grad_output = [int(np.ceil(x)) for x in grad_output[0]]
        
        for i in range(len(self.grad_output)):
            self.to_return.append(btd.signed_decimal_to_binary_list(self.grad_output[i],self.input_length-self.kernel_length+1))

        self.to_return = np.array(self.to_return)
        
        return np.array(self.to_return)
    
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
    
    