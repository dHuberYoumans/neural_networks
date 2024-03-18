# activation functions
import layer_logic as ll
import numpy as np

# strategy pattern for activation functions 
class Activation(ll.Layer): # abstract activation layer
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
    
class ReLu(Activation):
    def __init__(self):
        epsilon = 0.01
        relu = lambda x : np.maximum(x,epsilon*np.ones(x.shape))
        drelu = lambda x : np.piecewise(x, [x < 0, x >= 0], [epsilon, 1])
        super().__init__(relu, drelu)
