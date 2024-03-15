# (1d) convolution nn to learn binary numbers

import layer_logic as ll
import neural_network_logic as nnl
from loss_functions import binary_cross_entropy, dbinary_cross_entropy
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import binary_decimal_representation as bdr

if __name__=='__main__':
    # define data 
    # input x: array representing binary number; output y
    #x[0] = sign-bit (0 for positive numbers, 1 for negative numbers); len(x) = padding + 1
    x = [] 
    y = [] 

    samples = 1000
    bits = 10
    x = np.random.randint(0, 2, size=(samples, bits))
    y = np.array([[[int(c) for c in bin(abs(i)).split('b')[1].zfill(bits)]] for i in range(samples)])

    # defining the network 
    kernel_length = 3
    depth = 16
    y_hidden_length = bits - kernel_length + 1

    sigmoid= [
        ll.one_dim_convolution(bits,kernel_length,depth), 
        ll.Sigmoid(),
        # ll.BinaryToDecimal(x_length,kernel_length), # returns list of integers of length depth 
        # ll.Reshape((depth,y_hidden_length),(y_hidden_length,depth)),
        # ll.Dense(y_hidden_length,y_length),
        # ll.Sigmoid(),
        # ll.Reshape((y_length,depth),(depth*y_length,1)),
        # ll.Dense(depth,y_length),
        # ll.Sigmoid(),
        ll.Reshape((depth,y_hidden_length),(y_hidden_length*depth,-1)),
        ll.Dense(y_hidden_length*depth,bits),
        ll.Reshape((bits,1),(1,bits)),
        ll.Sigmoid()
        ]

    bin_network = nnl.network(sigmoid)

    # train the network
    epochs = 200
    alpha = 0.1 # learning rate
    verbose = True

    tstart = time.time()
    bin_network.train(binary_cross_entropy,dbinary_cross_entropy,x,y,epochs,alpha,verbose)
    tend = time.time()
    print(f'run time [s]: {tend - tstart:.2f}')
    for i in range(10):
        y_test = np.random.randint(0, 2, size=(1, bits))
        y_hat_prob = bin_network.predict(y_test[0])[0]
        y_hat = np.array([ int(np.round(a)) for a in bin_network.predict(y_test[0])[0] ])
        print(f'y_test = {y_test[0]}')
        print(f'y_hat  = {y_hat}\n')