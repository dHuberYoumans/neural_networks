# several loss functions
import numpy as np

def mse(y_true : np.array ,y_hat : np.array) -> float:
    return np.linalg.norm(y_true - y_hat)**2 / len(y_true) 

def dmse(y_true : np.array , y_hat : np.array) -> np.array:
    return  2*(y_hat - y_true) / len(y_true)