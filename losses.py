import numpy as np


class sigmoid():
    
    def activation(self, x):
        return (1.0 / ( 1.0 + np.exp(-x)))

    def backpropagation(self, x):
        return self.activation(x) * (1.0 - self.activation(x))

class relu():

    def activation(self, x):
        # return np.max(0, x)
        return((x > 0)*x)


    def backpropagation(self, x):
        # return 0 if x < 0 else 1
        return((x > 0)*1.0)


class softmax():

    def activation(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x, axis=0)

    def backpropagation(self, x):
        ##REF:https://medium.com/@aerinykim/how-to-implement-the-softmax-derivative-independently-from-any-loss-function-ae6d44363a9d
        # Reshape the 1-d softmax to 2-d so that np.dot will do the matrix multiplication
        s = x.reshape(-1,1)
        return np.diagflat(s) - np.dot(s, s.T)



