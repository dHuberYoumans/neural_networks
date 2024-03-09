# xor test
import layer_logic as ll
import neural_network_logic as nnl
from loss_functions import mse, dmse
import numpy as np
import time

# define data: x,y = list of input / expectet states
x = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1)) # shape: 4 states | each 2d (column) vector | list
y = np.reshape([[0], [1], [1], [0]], (4, 1, 1)) # shape: 4 states | each a number (0 or 1) | list

# define network
tanh_activation = [
    ll.Dense(2,3), # dense(input_size,output_size)
    ll.Tanh(),
    ll.Dense(3,1),
    ll.Tanh()]

# sigmoid_activation = [
#     ll.Dense(2,3), # dense(input_size,output_size)
#     ll.Sigmoid(),
#     ll.Dense(3,1),
#     ll.Sigmoid()]

xor_network = nnl.network(tanh_activation)
    

# train the network
epochs = 10_000
alpha = 0.1# learning rate
verbose = True

tstart = time.time()
xor_network.train(mse,dmse,x,y,epochs,alpha,verbose)
tend = time.time()
print(f'run time [s]: {tend - tstart:.2f}')
input()

