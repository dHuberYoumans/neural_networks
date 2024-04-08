# buisness logic layer

from abc import ABC, abstractmethod
from typing import Any
import numpy as np
from numpy.core.multiarray import array as array
from scipy import signal


# strategy pattern for layers
class Layer(ABC): 
    """ abstract strategy for layers """
    def __init__(self):
        self.input = None
        self.output = None

    @abstractmethod
    def forward(self, input : np.array) -> np.array:
        pass

    @abstractmethod
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
    ''' concrete strategy for 1d convolution layer 
    forward method: takes column vectors as input, reshapes them into row vectors and correlates them with kernels
    backward method: ... 
    '''
    def __init__(self,input_length : int ,kernel_length : int, depth : int):
        self.input_length = input_length
        self.kernel_length = kernel_length
        self.depth = depth
        self.kernel = np.random.rand(self.depth,self.kernel_length,1)
        self.bias = np.random.rand(self.depth, self.input_length - self.kernel_length+1,1)
        

    def forward(self,input : np.array) -> np.array: 
        self.input = input
        self.output = np.array([(self.bias[i] + signal.correlate(self.input,self.kernel[i],'valid')) for i in range(self.depth)])
        return self.output

    def backward(self, grad_output: np.array, learning_rate: float) -> np.array:
        grad_kernel = np.zeros((self.depth,self.kernel_length,1))
        grad_input = np.zeros((self.input_length,1))
        # compute bias gradient 
        grad_bias = grad_output

        # compute kernel gradient
        for i in range(self.depth):
            grad_kernel += signal.correlate(self.input, grad_output[i],'valid')

        # compute input gradient
        for i in range(self.depth):
            grad_input += signal.correlate(grad_output[i],self.kernel[i],'full')

        # update parameters
        self.kernel -= learning_rate * grad_kernel
        self.bias -= learning_rate * grad_bias

        return grad_input

    def get_grad_kernel(self):
        return self.grad_kernel
    
    def get_kernel(self):
        return self.kernel

class two_dim_convolution(Layer):
    ''' concrete strategy for 2d convolution layer 
    input_shape = (width, height, channel) of image.
    '''
    def __init__(self,input_shape : tuple[int,int,int], ksize: int, depth : int):
        self.input_shape = input_shape
        self.input_height, self.input_width, self.input_channels = input_shape # dim input img, e.g. X = pixel pos x RGB

        self.ksize = ksize # kernel = square matrix
        self.kchannels = self.input_channels # e.g. RGB 
        self.depth = depth # depth of layer
        self.kernel_shape = (self.depth,self.ksize,self.ksize,self.kchannels)
        self.output_shape = (self.depth, self.input_height - self.ksize + 1, self.input_width - self.ksize + 1)

        self.kernel = np.random.rand(*self.kernel_shape)
        self.bias = np.random.rand(*self.output_shape)
        

    def forward(self,input : np.array) -> np.array: 
        self.input = input
        self.output = np.copy(self.bias)

        for i in range(self.depth):
            for J in range(self.input_channels):
                self.output[i] += signal.correlate2d(self.input[:,:,J], self.kernel[i,:,:,J],'valid')

        return self.output

    def backward(self, grad_output : np.array, learning_rate : float) -> np.array:
        grad_kernel = np.zeros(self.kernel_shape) 
        grad_input = np.zeros(self.input_shape)

        # compute bias gradient 
        grad_bias = grad_output

        # compute kernel gradient
        for i in range(self.depth):
            for J in range(self.input_channels):
                grad_kernel[i,:,:,J] = signal.correlate2d(self.input[:,:,J],grad_output[i],'valid')

        # compute input gradient
        for J in range(self.input_channels):
            for i in range(self.depth):
                grad_input[:,:,J] += signal.correlate2d(grad_output[i],self.kernel[i,:,:,J],'full')

        # update parameters
        self.kernel -= learning_rate * grad_kernel
        self.bias -= learning_rate * grad_bias

        return grad_input
    
    def get_kernel(self):
        return self.kernel

class Reshape(Layer):
    def __init__(self,input_shape,output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input : np.array) -> np.array:
        return input.reshape(self.output_shape)
    
    def backward(self, grad_output : np.array, learning_rate : float) -> np.array:
        return grad_output.reshape(self.input_shape)
       
class SoftMax(Layer):
    '''
    input is column vector (output of dense layer) x.

    Jacobian of softmax: $Jac = diag(s(x)) - s(x)s^t(x)$.

    backpropagation: dI = Jac * dE where dE is column vector.
    '''
    def __init__(self):
        self.input = None
       
    def softmax(self,x):
        return np.exp(x - np.max(x)) / np.sum(np.exp(x-np.max(x)))

    def forward(self, input : np.array) -> np.array:
        self.input = input
        return self.softmax(self.input)
    
    def dsoftmax(self,x): # Jacobian softmax
            s = self.softmax(x)
            return np.diag(s.ravel()) - np.dot(s,s.T)

    def backward(self, grad_output : np.array, learning_rate : float) -> np.array:
        return np.dot(self.dsoftmax(self.input),grad_output) 