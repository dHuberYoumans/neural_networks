# activation functions
import layer_logic as ll
import numpy as np

eps = 10**-5 # cut-off / regularization

# strategy pattern for activation functions 
class Activation(ll.Layer): # abstract activation layer for LOCAL (f(x)_i = f(x_i)) activation functions
    """ 
    abstract strategy for activation layer of *local* activation functions.

    A *local* activation functions is defined  as $f : \mathbb{R}^n \to \mathbb{R}^n$ with $f(x)_i = f(x_i)$.

    Examples are tanh(x), sigmoid(x) etc. A counter example is softmax(x) since the denominator considers information of all $x_i$ 
    """
    def __init__(self, activation, dactivation):
        self.activation = activation # need to pass functions, e.g. lambda functions
        self.dactivation = dactivation 

    def forward(self, input : np.array) -> np.array:
        self.input = input
        return self.activation(self.input)

    def backward(self, grad_output : np.array, learning_rate : float) -> np.array:
        self.grad_E = grad_output
        return np.multiply(self.dactivation(self.input),self.grad_E)
    
class Tanh(Activation): # concrete activation layer
     """ 
     implements tanh activation
     """
     def __init__(self):
        tanh = lambda x : np.tanh(x)
        dtanh = lambda x : (1 - np.tanh(x)**2)
        super().__init__(tanh,dtanh)

class Sigmoid(Activation): # concrete activation layer
    """ 
    implements sigmoid activation s(x) = 1 / (1 + exp(-x)) 
    """
    def __init__(self):
        sigmoid = lambda x : 1 / (1 + np.exp(-x))
        dsigmoid = lambda x : sigmoid(x)*(1 - sigmoid(x))
        super().__init__(sigmoid, dsigmoid)


    
class ReLu(Activation):
    """
    implements rectified linear unit (relu) activation relu(x) = max(epsilon,x)

    the regularization epsilon is set to 0.01
    """
    def __init__(self):
        epsilon = 0.01
        relu = lambda x : np.maximum(x,epsilon*np.ones(x.shape))
        drelu = lambda x : np.piecewise(x, [x < 0, x >= 0], [epsilon, 1])
        super().__init__(relu, drelu)


