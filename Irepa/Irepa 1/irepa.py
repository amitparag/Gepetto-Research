import os
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

def irepa_0():
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
        self.input_dims = input_features
        self.output_dims = output_features

        self.fc1 = nn.Linear(self.input_dims, 8)
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        torch.nn.init.normal_(self.fc1.weight.data, -f1, f1)
        torch.nn.init.uniform_(self.fc1.bias, -f1, f1)        
        self.bn1 = nn.LayerNorm(8)

        self.fc2 = nn.Linear(8, 8)
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        torch.nn.init.normal_(self.fc2.weight.data, -f2, f2)
        torch.nn.init.uniform_(self.fc2.bias, -f2, f2)        
        self.bn2 = nn.LayerNorm(8)

        
        
        self.fc3 = nn.Linear(8, output_features)
        f3 = 1 / np.sqrt(self.fc3.weight.data.size()[0])
        torch.nn.init.normal_(self.fc3.weight.data, -f3, f3)
        torch.nn.init.uniform_(self.fc3.bias, -f3, f3)    

        
        self.device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x = torch.tanh(self.bn1(self.fc1(x)))
        x = torch.tanh(self.bn2(self.fc2(x)))
        return self.fc3(x)
    
def pyNet():
    positions, cost = irepa_0()
    
    
    x_train = torch.as_tensor(positions, device = device, dtype = torch.float32)
    y_train = torch.as_tensor(cost, device = device, dtype = torch.float32)

      
    net = Net(x_train.shape[1], y_train.shape[1])
    net = net.float()
    net.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    
    n_epochs = 150 
    batch_size = 4 
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
            
    del x_train, y_train
        
    return net

   
def pyNet_train(neuralnet, positions, cost):
   
    x_train = torch.as_tensor(positions, device = device, dtype = torch.float32)
    y_train = torch.as_tensor(cost, device = device, dtype = torch.float32)

      

    neuralnet.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(neuralnet.parameters(), lr=0.01)
    
    n_epochs = 100 
    batch_size = 4 
    for epoch in range(n_epochs):

        # X is a torch Variable
        permutation = torch.randperm(x_train.size()[0])

        for i in range(0,x_train.size()[0], batch_size):
            optimizer.zero_grad()

            indices = permutation[i:i+batch_size]
            batch_x, batch_y = x_train[indices], y_train[indices]

            # in case you wanted a semi-full example
            outputs = neuralnet(batch_x)
            loss = criterion(outputs,batch_y)

            loss.backward()
            optimizer.step()
            
    del x_train, y_train
        
    return neuralnet



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


import crocoddyl
import pinocchio

crocoddyl.switchToNumpyMatrix()


def a2m(a):
    return np.matrix(a).T


def m2a(m):
    return np.array(m).squeeze()

class UnicycleTerminal(crocoddyl.ActionModelAbstract):
    def __init__(self, net):
        crocoddyl.ActionModelAbstract.__init__(self, crocoddyl.StateVector(3), 2, 5)
        self.net = net
        self.dt = .1
        self.costWeights = [10., 1.]
        self.net.eval()

        
    def calc(self, data, x, u=None):
        if u is None:
            u = self.unone
        x0 = torch.as_tensor(m2a(x.T), device = device, dtype = torch.float32).resize_(1, 3)
        data.cost = self.net(x0).item()

    def calcDiff(self, data, x, u=None, recalc=True):               
        if u is None:
            u = self.unone
        if recalc:
            self.calc(data, x, u)
            
        x0 = torch.as_tensor(m2a(x.T), device = device, dtype = torch.float32).resize_(1, 3)
        x0.requires_grad_(True)
        j = jacobian(self.net(x0), x0)
        h = hessian(self.net(x0), x0)
        data.Lx = a2m(j.cpu().detach().numpy())
        data.Lxx = a2m(h.cpu().detach().numpy())
        

        

def irepa_data(net):

    positions = []
    cost = []
    model = c.ActionModelUnicycle()
    terminal_model = UnicycleTerminal(net)
    for _ in range(1000):

        x0 = np.array([np.random.uniform(-2.1, 2.1), np.random.uniform(-2.1, 2.1), np.random.uniform(0,1)])
        T = 30
        problem = c.ShootingProblem(x0.T, [ model ] * T, terminal_model)
        ddp = c.SolverDDP(problem)
        ddp.solve()

        
        positions.append(x0)
        cost.append(np.array([ddp.cost]))
    end = perf_counter()
    positions = np.asarray(positions)
    cost = np.asarray(cost)
    del model
    return positions, cost

    

def irepa():
    # run irepa zero and get the net
    
    net = pyNet()
    
    for _ in range(21):
        positions, cost = irepa_data(net)
        net = pyNet_train(net, positions, cost)
        if _ in [1, 4, 10, 15, 20]:
            name = f"Net{_}.pth"
            save_path = os.path.join(os.getcwd(), name)
            torch.save(net, save_path)
        

if __name__ == "__main__":
    from time import perf_counter
    start = perf_counter()
    irepa()
    end = perf_counter()
    print("Time for Irepa ", end - start )
