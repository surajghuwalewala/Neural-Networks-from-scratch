import numpy as np

def accuracy(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    return np.mean([1 if a==b else 0 for a,b in zip(y_true, y_pred)])