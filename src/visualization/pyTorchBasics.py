import torch
x=torch.empty(3)
print(x)

x=torch.ones(2,2,dtype=torch.float32)
print(x)

x=torch.rand(2,2)
y=torch.rand(2,2)
print(x,y)
z=x+y
print(z)

print(x[:,1])
print(x[1,1].item()) #works only if you want to extract one value

print(y)
print(y.add_(x))#add_ is inplace operation, so x gets added to y and value in y gets replaced by sum
print(y)

y.view(4) #converts y to 4-dimensional tensor
y.view(4,1) #converts y 1-dimmensional tensor with 4 elements

x=torch.ones(4,dtype=torch.float32,requires_grad=True)
for i in range(3):
    y=(x*3).sum()
    y.backward()
    print(x.grad)
    # as one can see each time the loop is run, the gadient
    #gets accumulated and it is added to the previous gradient
    # to overcome this we needs to re-set the grad to zero
for i in range(3):
    y=(x*3).sum()
    y.backward()
    print(x.grad)
    x.grad.zero_()
    #now the gradient is zeroed and the value of x is updated
    
#### x=1, y=2, w=1
x=torch.tensor([1],dtype=torch.float32)
y=torch.tensor([2],dtype=torch.float32)
w=torch.tensor([1],dtype=torch.float32,requires_grad=True)

##forward pass and compute loss
y_hat=x*w
loss=(y_hat-y)**2
print(loss)

#backward loss
loss.backward()
print(w.grad)

################################### linear regression-manual 
import numpy as np
# f=2*x
x=np.array([1,2,3],dtype=np.float32)
y=np.array([2,4,6],dtype=np.float32)
w=0

#model prediction
def forward(x):
    return(w*x)

#loss function
def loss(y,y_predicted):
    return(((y_predicted-y)**2).mean())

#gradient
#mse=1/n*(w*x-y)**2
#dj/dw=1/n 2x(w*x-y)
def gradient(x,y,y_predicted):
    return(np.dot(2*x,y_predicted-y).mean())

print('prediction before training:',forward(5))

learning_rate = 0.01
n_iter=30
for epoch in range(n_iter):
    #prediction=forwardpass
    y_pred=forward(x)
    
    #loss
    l=loss(y,y_pred)
    
    #gradients
    dw=gradient(x,y,y_pred)
    
    #update weights
    w-=learning_rate*dw
    
    if epoch%2==0:
        print('epoch:',epoch,'weight:',w,'loss:',l.item())
print('prediction after training:',forward(5))

########################################################################################

################################### linear regression-using pytorch

# f=2*x
x=torch.tensor([1,2,3],dtype=torch.float32)
y=torch.tensor([2,4,6],dtype=torch.float32)
w=torch.tensor([0],dtype=torch.float32,requires_grad=True)

#model prediction
def forward(x):
    return(w*x)

#loss function:MSE
def loss(y,y_predicted):
    return(((y_predicted-y)**2).mean())

# #gradient
# #mse=1/n*(w*x-y)**2
# #dj/dw=1/n 2x(w*x-y)
# def gradient(x,y,y_predicted):
#     return(np.dot(2*x,y_predicted-y).mean())

print('prediction before training:',forward(5))

learning_rate = 0.01
n_iter=100
for epoch in range(n_iter):
    #prediction=forwardpass
    y_pred=forward(x)
    
    #loss
    l=loss(y,y_pred)
    
    #gradients
    # dw=gradient(x,y,y_pred)
    l.backward()
    
    #update weights
    with torch.no_grad():
        w-=learning_rate*w.grad
    w.grad.zero_()
    if epoch%2==0:
        print('epoch:',epoch,'weight:',w,'loss:',l.item())
print('prediction after training:',forward(5))

###################linear regression suing pytorch-back propogation, optimizer (in this case its gradient descent) and loss function

# design our model (input, output and forward pass)
# construct loss and optimizer
# training loop
  #-forward pass
  #-backward pass: gradients
  #-update weights

import torch .nn as nn 

# f=2*x
x=torch.tensor([1,2,3],dtype=torch.float32)
y=torch.tensor([2,4,6],dtype=torch.float32)
w=torch.tensor([0],dtype=torch.float32,requires_grad=True)
# model=nn.linear

#model prediction
def forward(x):
    return(w*x)

# #loss function:MSE
# def loss(y,y_predicted):
#     return(((y_predicted-y)**2).mean())

loss=nn.MSELoss()
optimizer=torch.optim.SGD([w],lr=learning_rate)

# #gradient
# #mse=1/n*(w*x-y)**2
# #dj/dw=1/n 2x(w*x-y)
# def gradient(x,y,y_predicted):
#     return(np.dot(2*x,y_predicted-y).mean())

print('prediction before training:',forward(5))

learning_rate = 0.01
n_iter=100
for epoch in range(n_iter):
    #prediction=forwardpass
    y_pred=forward(x)
    
    #loss
    l=loss(y,y_pred)
    
    #gradients
    # dw=gradient(x,y,y_pred)
    l.backward()
    
    #update weights
    optimizer.step()
    # with torch.no_grad():
    #     w-=learning_rate*w.grad
    optimizer.zero_grad()
    if epoch%10==0:
        print('epoch:',epoch,'weight:',w,'loss:',l.item())
print('prediction after training:',forward(5))

#########################################################################
import torch .nn as nn 

# f=2*x
x=torch.tensor([[1],[2],[3]],dtype=torch.float32)
y=torch.tensor([[2],[4],[6]],dtype=torch.float32)
x_test=torch.tensor([5],dtype=torch.float32)
# w=torch.tensor([0],dtype=torch.float32,requires_grad=True)
n_samples,n_features=x.shape
input_size=n_features
output_size=n_features
model=nn.Linear(input_size,output_size)

# #model prediction
# def forward(x):
#     return(w*x)

# #loss function:MSE
# def loss(y,y_predicted):
#     return(((y_predicted-y)**2).mean())

loss=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)

# #gradient
# #mse=1/n*(w*x-y)**2
# #dj/dw=1/n 2x(w*x-y)
# def gradient(x,y,y_predicted):
#     return(np.dot(2*x,y_predicted-y).mean())

print('prediction before training:',model(x_test).item())

learning_rate = 0.001
n_iter=150
for epoch in range(n_iter):
    #prediction=forwardpass
    y_pred=forward(x)
    
    #loss
    l=loss(y,y_pred)
    
    #gradients
    # dw=gradient(x,y,y_pred)
    l.backward()
    
    #update weights
    optimizer.step()
    # with torch.no_grad():
    #     w-=learning_rate*w.grad
    optimizer.zero_grad()
    if epoch%10==0:
        [w,b]=model.parameters()
        print('epoch:',epoch,'weight:',w[0][0].item(),'loss:',l.item())
print('prediction after training:',model(x_test).item())

########################### linear regression using pytorch and large dataset

import torch.nn as nn 
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

#0-prepare dataset
x_numpy,y_numpy=datasets.make_regression(n_samples=1000,noise=0.05,
                                             n_features=1,
                                             random_state=100)
x=torch.from_numpy(x_numpy.astype(np.float32))
y=torch.from_numpy(y_numpy.astype(np.float32))#shape:1,1000
y=y.view(y.shape[0],1)#shape:1000,1

n_samples,n_features=x.shape

#1-build model-forward pass
input_size=n_features
output_size=n_features
model=nn.Linear(input_size,output_size)
#2-build loss function and optimizer
l=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
#3-training loop
num_epochs=100
for epoch in range(num_epochs):
    #forward pass for training and loss calculation
    y_predicted=model(x)
    loss=l(y,y_predicted)
    #backward pass
    loss.backward()
    #update weights using optmizer
    optimizer.step()
    
    #need to empty the gradient. whenever we call loss.backward(), it accumulates
    #the gradients and sums them up
    optimizer.zero_grad()
    
    if (epoch+1)%10==0:
        [w,b]=model.parameters()
        print('epoch:',epoch,'loss:',loss.item())
    
    predicted=model(x).detach().numpy() #detaching it so that its not tracked in our 
    #computational graph
    plt.plot(x_numpy,y_numpy,'ro')
    plt.plot(x_numpy,predicted,'b')
    plt.show()
    
################################## logistic regression using pytorch
# 1. design model-input, output, forward pass
# 2. construct loss and optimizer
# 3. training loop
#    -forward pass:compute prediction and loss
#    -backward pass:compute gradients
#    -update weights

import torch.nn as nn 
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df=datasets.load_breast_cancer()
x_numpy,y_numpy=df.data,df.target
display(x_numpy.shape, y_numpy.shape)

x_train,x_test,y_train,y_test=train_test_split(x_numpy,y_numpy,
                                               test_size=0.2,
                                               random_state=100)
# display(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

x_train=torch.from_numpy(x_train.astype(np.float32))
y_train=torch.from_numpy(y_train.astype(np.float32))#shape:1,569
x_test=torch.from_numpy(x_test.astype(np.float32))
y_test=torch.from_numpy(y_test.astype(np.float32))#shape:1,569

y_train=y_train.view(y_train.shape[0],1)#shape:569,1
y_test=y_test.view(y_test.shape[0],1)

n_samples,n_features=x_train.shape

class LogisticRegression(nn.Module):
    def __init__(self,n_input_features):
        super(LogisticRegression,self).__init__()
        self.linear=nn.Linear(n_input_features,1)
    def forward(self,x):
        y_predicted=torch.sigmoid(self.linear(x))
        return y_predicted

model = LogisticRegression(n_features)
loss=nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)
num_epochs=100
for epoch in range(num_epochs):
    #forward pass
    y_predicted=model(x_train)
    l=loss(y_predicted,y_train)
    #backward pass
    l.backward()
    
    #update weights
    optimizer.step()
    
    optimizer.zero_grad()

    if epoch%10==0:
        print("epoch:",epoch,"loss:",l.item())

with torch.no_grad():
    
    y_predicted=model(x_test)    
    y_predicted_cls=y_predicted.round()
    acc=accuracy_score(y_test,y_predicted_cls)
    print("Accuracy score:",acc)
        
        








