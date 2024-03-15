# (1d) convolution nn to learn binary numbers

import layer_logic as ll
import neural_network_logic as nnl
import activation_layer_logic as all
from loss_functions import binary_cross_entropy, dbinary_cross_entropy
import numpy as np
import time

if __name__=='__main__':
    # generate data (as column vectors)
    samples = 1000
    bits = len(bin(samples).split('b')[1])
    x = np.random.randint(0, 2, size=(samples, bits,1))
    y = np.array([ [ [int(c) for c in bin(i).split('b')[1].zfill(bits)] ] for i in range(samples)]).reshape(samples,bits,1)

    # defining the network 
    kernel_length = 2
    depth = 16
    y_hidden_length = bits - kernel_length + 1

    network= [
        ll.one_dim_convolution(bits,kernel_length,depth), 
        all.ReLu(),
        ll.Reshape((depth,y_hidden_length,1),(depth*y_hidden_length,-1)),
        ll.Dense(y_hidden_length*depth,bits),
        all.Sigmoid()
        ]

    bin_network = nnl.network(network)

    # train the network
    epochs = 100
    alpha = 0.1 # learning rate
    verbose = True

    tstart = time.time()
    bin_network.train(binary_cross_entropy,dbinary_cross_entropy,x,y,epochs,alpha,verbose)
    tend = time.time()
    print(f'run time [s]: {tend - tstart:.2f}')
    for i in range(10):
        y_test = np.random.randint(0, 2, size=(bits,1))
        # y_hat_prob = bin_network.predict(y_test)
        y_hat = np.array([ int(np.round(a)) for a in bin_network.predict(y_test)])
        print(f'y_test = {y_test.T[0]}')
        print(f'y_hat  = {y_hat}\n')