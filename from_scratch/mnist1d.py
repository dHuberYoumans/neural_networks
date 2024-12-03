import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

import sys
sys.path.insert(0,'../logic/')
import layer_logic as ll
import neural_network_logic as nnl
import activation_layer_logic as all
from loss_functions import binary_cross_entropy, dbinary_cross_entropy
import time

def prepare_data(x,y,samples):
    # x = x[:samples]
    # y = y[:samples]
    zero_index = np.where(y == 0)[0]
    one_index = np.where(y == 1)[0]
    all_indices = np.hstack((zero_index, one_index))
    all_indices = np.random.permutation(all_indices)
    x, y = x[all_indices], y[all_indices]
    X = x[:samples].astype('float32') / 255.  
    X = X.reshape(-1,28*28,1)
    Y = to_categorical(y[:samples],10).reshape(-1,10,1)
    
    return X, Y

if __name__=='__main__':
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Preparing the data
    samples_train = 100
    samples_test = 10

    X_train, Y_train = prepare_data(x_train,y_train,samples_train)
    X_test, Y_test = prepare_data(x_test,y_test,samples_test)

    # defining the network
    ksize = 7 # 1d kernel size
    depth = 32
    isize = X_train.shape[1] # input length
    output_length = isize - ksize + 1

    network = [
        ll.one_dim_convolution(isize,ksize,depth), 
        all.Tanh(),
        ll.Reshape((depth,output_length,1),(depth*output_length,-1)),
        ll.Dense(output_length*depth,100),
        all.Tanh(),
        ll.Dense(100,10),
        all.Sigmoid()
    ]

 
    mnist_net = nnl.network(network)

    # train the network
    epochs = 20
    alpha = 0.1
    verbose = True

    tstart = time.time()
    mnist_net.train(binary_cross_entropy,dbinary_cross_entropy,X_train,Y_train,epochs,alpha,verbose)
    tend = time.time()

    # test
    print('\n')
    print('testing accuracy on training set...')
    acc_train = 0.
    for i, (X,Y) in enumerate(zip(X_train, Y_train)):
        Y_hat = mnist_net.predict(X)
        if np.argmax(Y_hat) == np.argmax(Y):
            acc_train += 1.
        
    acc_train /= samples_train
    print(f'accuracy on training: {acc_train}')
    print('\n')

    print('testing accuracy on testing set...')
    acc_test = 0.
    for i, (X,Y) in enumerate(zip(X_test, Y_test)):
        Y_hat = mnist_net.predict(X)
        if np.argmax(Y_hat) == np.argmax(Y):
            acc_test += 1.
        
    acc_test /= samples_test
    print(f'accuracy on testing: {acc_test}')
    print('\n')


