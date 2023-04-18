import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import scipy as sp

import torch
import torch.nn as nn
import torch.nn.functional as nn_functional

train=pd.read_csv(r'E:\Python Projects\Churn Modelling using Deep Learning\data\processed\trainTrans.csv',header=None)
test=pd.read_csv(r'E:\Python Projects\Churn Modelling using Deep Learning\data\processed\testTrans.csv',header=None)
y_train=pd.read_csv(r'E:\Python Projects\Churn Modelling using Deep Learning\data\processed\trainLabels.csv')
y_test=pd.read_csv(r'E:\Python Projects\Churn Modelling using Deep Learning\data\processed\testLabels.csv')

display(train.shape,y_train.shape,test.shape,y_test.shape)

#converting train,test,trainLabels,testLabels to tensors

x_train=torch.from_numpy(np.array(train).astype('float32'))
x_test=torch.from_numpy(np.array(test).astype('float32'))
y_train=torch.LongTensor(np.array(y_train).astype('int32'))
y_test=torch.LongTensor(np.array(y_test).astype('int32'))






