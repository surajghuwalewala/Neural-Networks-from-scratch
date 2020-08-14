import numpy as np
from losses import *
from metrics import *
from tqdm import tqdm
from sklearn.utils import shuffle

class DNN():


    def __init__(self, input_dim, hidden_dims, num_classes, h_act_fn = 'relu', use_bias=True):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = num_classes
        self.layer_dims = [self.input_dim] + self.hidden_dims + [self.output_dim] 
        
        ## Init weights
        self._init_weights()
        
        ## Init bias
        self.use_bias = use_bias
        self._init_bias()

        ## Init layers
        self._init_layers()

        ## hidden activations
        if h_act_fn == 'relu':
            self.loss_fn = relu()
        elif h_act_fn == 'sigmoid':
            self.loss_fn = sigmoid()

        ##output loss
        if num_classes == 2:
            self.out_loss = sigmoid()
        elif num_classes > 2:
            self.out_loss = softmax()
        

    def _init_weights(self):
        """
        Initializes random weights.
        """
        self.weights = [np.random.rand(i,j) for i,j in zip(self.layer_dims[1:], self.layer_dims[:-1])]


    def _init_bias(self):
        """
        Initializes random bias if True else zeros.
        """
        if self.use_bias == True:
            self.bias = [np.random.rand(i,1) for i in self.hidden_dims]
        else:
            self.bias = [np.zeros((i,1)) for i in self.hidden_dims]

    def _init_layers(self):
        self.layers = [np.zeros((i,1)) for i in self.layer_dims]

    def make_batches(self, X, y, batch_size):
        num_batches = len(X)//batch_size
        batched_X, batched_y = [], []
        X, y = shuffle(X, y)
        for batch in range(1, num_batches):
            batched_X.append(X[ (batch-1)*batch_size: batch*batch_size])
            batched_y.append(y[ (batch-1)*batch_size: batch*batch_size])
        
        ## Add remaining data in other batch
        batched_X.append(X[batch*batch_size:])
        batched_y.append(y[batch*batch_size:])
        
        return batched_X, batched_y


    def train_on_batch(self, batch_X, batch_y, learning_rate):
        
        for X,y in zip(batch_X, batch_y):
            self.layers[0] = X.copy()

            for i in range(len(self.weights[:-1])):
                self.layers[i+1] = self.loss_fn.activation(np.dot( self.weights[i], self.layers[i]))
            
            out = self.out_loss.activation(np.dot( self.weights[-1], self.layers[-2]))

            error = out - y

            delta = np.dot(self.out_loss.backpropagation(out), error)

            self.weights[-1] +=  learning_rate * np.dot(delta, self.layers[-2].T)

            for i in reversed(range(len(self.weights) - 1 )):
                delta = np.dot(self.weights[i+1].T, delta) * self.loss_fn.backpropagation(self.layers[i+1])
                self.weights[i] += learning_rate * np.dot(delta, self.layers[i].T)


    def train(self, X, y, val_X=None, val_y=None, batch_size = 20, n_epochs=10, learning_rate = 0.001):
        """
        X = input array (num_data, data_dim)
        y = labels (num_data)

        """

        assert X.shape[1] == self.input_dim
        X = np.expand_dims(X, axis=-1)

        batched_X, batched_y = self.make_batches(X,y, batch_size = batch_size)

        for epoch in range(n_epochs):
            print("\nEpoch : ", epoch+1)

            for batch_X, batch_y in tqdm(zip(batched_X, batched_y)):
                self.train_on_batch(batch_X, batch_y, learning_rate)
            if val_X is not None and val_y is not None:
                print("Validation accuracy: ",self.evaluate(val_X, val_y))


    def predict(self, test_X):
        preds = []
        for X in test_X:
            self.layers[0] = X.copy()

            for i in range(len(self.weights[:-1])):
                self.layers[i+1] = self.loss_fn.activation(np.dot( self.weights[i], self.layers[i]))
            
            preds.append(self.out_loss.activation(np.dot( self.weights[-1], self.layers[-2])))

        return np.array(preds)

    def evaluate(self, test_X, test_y):
        preds = self.predict(test_X)
        preds = np.argmax(preds, axis=1)
        return accuracy(test_y, preds)
        

