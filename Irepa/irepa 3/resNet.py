import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ResNet(nn.Module):
    def __init__(self, input_features, output_features):
        super(TwoLayerNet, self).__init__()
        self.input_dims = input_features
        self.output_dims = output_features

        self.fc1 = nn.Linear(self.input_dims, 16)
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        torch.nn.init.normal_(self.fc1.weight.data, -f1, f1)
        torch.nn.init.uniform_(self.fc1.bias, -f1, f1)        
        self.bn1 = nn.LayerNorm(8)

        self.fc2 = nn.Linear(16, 16)
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        torch.nn.init.normal_(self.fc2.weight.data, -f2, f2)
        torch.nn.init.uniform_(self.fc2.bias, -f2, f2)        
        self.bn2 = nn.LayerNorm(16)
                
        self.fc3 = nn.Linear(16, 16)
        f3 = 1 / np.sqrt(self.fc3.weight.data.size()[0])
        torch.nn.init.normal_(self.fc3.weight.data, -f3, f3)
        torch.nn.init.uniform_(self.fc3.bias, -f3, f3)        
        self.bn3 = nn.LayerNorm(16)        
        
        self.fc4 = nn.Linear(16, 5)
        f4 = 1 / np.sqrt(self.fc4.weight.data.size()[0])
        torch.nn.init.normal_(self.fc4.weight.data, -f4, f4)
        torch.nn.init.uniform_(self.fc4.bias, -f4, f4)
        
        self.fc5 = nn.Linear(5, self.output_dims)
        f5 = 1 / np.sqrt(self.fc5.weight.data.size()[0])
        torch.nn.init.normal_(self.fc5.weight.data, -f5, f5)
        torch.nn.init.uniform_(self.fc5.bias, -f5, f5)

        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        
        x = torch.tanh(self.bn1(self.fc1(x)))
        x = torch.tanh(self.bn2(self.fc2(x)))
        x = torch.tanh(self.bn3(self.fc3(x)))
        residual = self.bn4(self.fc4(x))
        value_function = self.bn5(self.fc5(x))
        return value_function, residual
    
    
def squared_loss(output, target):
    loss = torch.abs(output - target) ** 2
    return loss.sum() 


def train_valueNet(net, positions, cost, residual):
    from tqdm import tqdm

    assert net.input_dims == 3
    assert net.output_dims == 1
    x = torch.as_tensor(positions, device = device, dtype = torch.float32)
    y = torch.as_tensor(cost, device = device, dtype = torch.float32)
    r = torch.as_tensor(residual, device = device, dtype = torch.float32)
    net = net.float()
    net.train()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    n_epochs = 150 
    for epoch in tqdm(range(n_epochs)):

        for position, cost, residual in zip(x, y, r):
            optimizer.zero_grad()
            cost_predicted, residual_predicted = net(position)
            loss_cost = squared_loss(cost_predicted,cost)
            loss_residual = squared_loss(residual_predicted, residual)
            loss = loss_cost + loss_residual
            loss.backward()
            optimizer.step()
    del x_train, y_train
    return net
    