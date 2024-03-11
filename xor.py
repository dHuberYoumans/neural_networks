# xor test
import layer_logic as ll
import neural_network_logic as nnl
from loss_functions import mse, dmse
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# define data: x,y = list of input / expectet states
X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1)) # shape: 4 states | each 2d (column) vector | list
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1)) # shape: 4 states | each a number (0 or 1) | list

# define network
tanh_activation = [
    ll.Dense(2,3), 
    ll.Tanh(),
    ll.Dense(3,1),
    ll.Tanh()]

xor_network = nnl.network(tanh_activation)

# sigmoid_activation = [
#     ll.Dense(2,3), 
#     ll.Sigmoid(),
#     ll.Dense(3,1),
#     ll.Sigmoid()]

# xor_network = nnl.network(sigmoid_activation)


# train the network
epochs = 10_000
alpha = 0.1# learning rate
verbose = True

tstart = time.time()
xor_network.train(mse,dmse,X,Y,epochs,alpha,verbose)
tend = time.time()
print(f'run time [s]: {tend - tstart:.2f}')
input('Plot dicision bounndary: ')


# dicision bounddary
steps= 21
x = np.linspace(0,1,steps) # coordinate x-values
y = np.linspace(0,1,steps) # coordinate y-values
grid = np.meshgrid(x,y) #create mesh
xx,yy = grid

 # get coordinates; correct form: (# pts, dim of column vector, 1) for input into network.predict()
pts = np.array(grid).T.reshape(-1,2,1) # w/o transpose, grid gets shapes into tuples of values next to each other resulting into (x,x) and (y,y) points not (x,y) 
z = np.array(list(map(xor_network.predict,pts))) # run the network on all pts p = (x,y)

# plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(xx,yy,z)
plt.show()



