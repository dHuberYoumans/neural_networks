# several loss functions
import numpy as np

def mse(y_true : np.array ,y_hat : np.array) -> float:
    return np.linalg.norm(y_true - y_hat)**2 / np.size(y_true) 

def dmse(y_true : np.array , y_hat : np.array) -> np.array:
    return  2*(y_hat - y_true) / np.size(y_true)

def binary_cross_entropy(y_true, y_hat):
    return np.mean(-y_true * np.log(y_hat) - (1 - y_true) * np.log(1 - y_hat))

def dbinary_cross_entropy(y_true, y_hat):
    return ((1 - y_true) / (1 - y_hat) - y_true / y_hat) / np.size(y_true)