# several loss functions
import numpy as np

eps = 10**-6

def mse(y_true : np.array ,y_hat : np.array) -> float:
    return np.linalg.norm(y_true - y_hat)**2 / np.size(y_true) 

def dmse(y_true : np.array , y_hat : np.array) -> np.array:
    return  2*(y_hat - y_true) / np.size(y_true)

def binary_cross_entropy(y_true, y_hat):
    return np.mean(-y_true * np.log(y_hat+eps) - (1 - y_true) * np.log(1 - y_hat+eps))

def dbinary_cross_entropy(y_true, y_hat):
    return ((1 - y_true) / (1 - y_hat + eps) - y_true / (y_hat+eps)) / np.size(y_true)

def categorical_cross_entropy(y_true,y_hat):
    return -np.sum(y_true*np.log(y_hat + eps)) # add infinitesimal off-set pes to avoid y_hat = 0

def dcategorical_cross_entropy(y_true,y_hat):
    return - y_true / (y_hat + eps)
