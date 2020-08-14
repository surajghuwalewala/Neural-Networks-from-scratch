import nn
import numpy as np
import pandas as pd
from metrics import *

## Loading train data
train_df = pd.read_csv("data/mnist_train.csv")
train_y = train_df['label'].to_numpy()
train_X = train_df.drop(columns=['label']).to_numpy()
print(train_X.shape, train_y.shape)

## Loading test data
test_df = pd.read_csv("data/mnist_test.csv")
test_y = test_df['label'].to_numpy()
test_X = test_df.drop(columns=['label']).to_numpy()
print(test_X.shape, test_y.shape)

model = nn.DNN(input_dim = train_X.shape[1], 
               hidden_dims = [512,256,128], 
               num_classes = 10, 
               h_act_fn='relu')

model.train(train_X[:5000], train_y[:5000], val_X = train_X[-100:], val_y = train_y[-100:], n_epochs=50, batch_size=100)

acc = model.evaluate(test_X, test_y)
print(acc)