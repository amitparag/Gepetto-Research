import numpy as np
import crocoddyl as c
import numdifftools as nd
from time import perf_counter
c.switchToNumpyArray()

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')


def data():
    # get the cost
    positions = []
    cost = []
    model = c.ActionModelUnicycle()
    for _ in range(1000):

        x0 = np.array([np.random.uniform(-2.1, 2.1), np.random.uniform(-2.1, 2.1), np.random.uniform(0,1)])
        T = 30
        problem = c.ShootingProblem(x0.T, [ model ] * T, model)
        ddp = c.SolverDDP(problem)
        ddp.solve()


        positions.append(x0)
        cost.append(np.array([ddp.cost]))
    end = perf_counter()
    positions = np.asarray(positions)
    cost = np.asarray(cost)
    del model
    return positions, cost


class Net(nn.Module):
    def __init__(self, input_features, output_features):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(input_features, 16)
        torch.nn.init.normal_(self.fc1.weight)
        self.fc1.bias.data.fill_(0.0)
        
        self.fc2 = nn.Linear(16, 16)
        torch.nn.init.normal_(self.fc2.weight)
        self.fc2.bias.data.fill_(0.0)
        
        self.fc3 = nn.Linear(16, output_features)
        torch.nn.init.normal_(self.fc3.weight)
        self.fc3.bias.data.fill_(0.0)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x
    
def pyNet():
    positions, cost = data()
    x_train = torch.as_tensor(positions, device = device, dtype = torch.float32)
    y_train = torch.as_tensor(cost, device = device, dtype = torch.float32)

      
    net = Net(x_train.shape[1], y_train.shape[1])
    net = net.float()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    
    n_epochs = 150 
    batch_size = 2 
    for epoch in range(n_epochs):

        # X is a torch Variable
        permutation = torch.randperm(x_train.size()[0])

        for i in range(0,x_train.size()[0], batch_size):
            optimizer.zero_grad()

            indices = permutation[i:i+batch_size]
            batch_x, batch_y = x_train[indices], y_train[indices]

            # in case you wanted a semi-full example
            outputs = net(batch_x)
            loss = criterion(outputs,batch_y)

            loss.backward()
            optimizer.step()
        
    return net

def pytorch_output(net, x):
    return net(x)

def jacobian(y, x, create_graph=False):                                                               
    jac = []                                                                                          
    flat_y = y.reshape(-1)                                                                            
    grad_y = torch.zeros_like(flat_y)                                                                 
    for i in range(len(flat_y)):                                                                      
        grad_y[i] = 1.                                                                                
        grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
        jac.append(grad_x.reshape(x.shape))                                                           
        grad_y[i] = 0.                                                                                
    return torch.stack(jac).reshape(y.shape + x.shape)                                                
                                                                                                      
def hessian(y, x):                                                                                    
    return jacobian(jacobian(y, x, create_graph=True), x)
