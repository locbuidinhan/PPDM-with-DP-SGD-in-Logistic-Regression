import numpy as np

def sigmoid(z):
    z = np.clip(z, -500, 500)   
    return 1.0 / (1.0 + np.exp(-z))

def compute_loss(y_true, y_pred, eps=1e-10):
    p = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(
        y_true * np.log(p) + (1 - y_true) * np.log(1 - p)
    )
