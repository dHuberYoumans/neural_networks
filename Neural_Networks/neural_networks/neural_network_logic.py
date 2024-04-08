import layer_logic as ll
import numpy as np

class network:
    def __init__(self, layers : list[ll.Layer]):
        self.layers = layers

    def predict(self, input : np.array) -> np.array:
        self.output = input

        for layer in self.layers:
            self.output = layer.forward(self.output)

        return self.output

    def train(self, loss, dloss, x_train : list[np.array], y_train : list[np.array], epochs : int, learning_rate : float, verbose : bool = True):
        print('\n')
        for e in range(epochs):
            error = 0.0
            for i, (x,y) in enumerate(zip(x_train,y_train)):
                
                # forward propagation
                self.y_hat = self.predict(x)

                # update error
                error += loss(y,self.y_hat)

                # backward propagation 
                self.grad_output = dloss(y,self.y_hat)

                for layer in reversed(self.layers):
                    self.grad_output = layer.backward(self.grad_output,learning_rate)
                # self.learn(dloss(y,self.y_hat),learning_rate)

                # normalize error 
                error /= len(x_train)

                if verbose:
                    print(f"epoch {e + 1}/{epochs} {'':<10} sample {i+1}/{x_train.shape[0]} {'':<10} error = {self.error:.5e}",end='\r')
        print()

