# define neural network (for simplicity use 2 dense layers 2->3 and 3->1; learn (call activation) after each desne layer )
import layer_logic as ll
import numpy as np
import time 

class network:
    def __init__(self, layers : list[ll.Layer]):
        self.layers = layers

    def predict(self, input : np.array) -> np.array:
        self.output = input
        for layer in self.layers:
            self.output = layer.forward(self.output)
        return self.output
    
    def learn(self, grad_output : np.array, learning_rate : float):
        self.grad_output = grad_output
        for layer in reversed(self.layers):
            self.grad_output = layer.backward(self.grad_output,learning_rate)
        return self.grad_output

    def train(self, loss, dloss, x_train : list[np.array], y_train : list[np.array], epochs : int, learning_rate : float, verbose : bool = True):
        for e in range(epochs):
            self.error = 0
            for x,y in zip(x_train,y_train):
                
                # forward propagation
                self.y_hat = self.predict(x)

                # update error
                self.error += loss(y,self.y_hat)

                # backward propagation
                self.grad = self.learn(dloss(y,self.y_hat),learning_rate)

                # normalize error 
                self.error /= len(x_train)

                if verbose:
                    print(f"{e + 1}/{epochs}, error = {self.error:.5E}",end='\r')
        print()
