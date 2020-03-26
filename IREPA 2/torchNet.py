import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TwoLayerNet(nn.Module):
    def __init__(self, input_features, output_features):
        super(TwoLayerNet, self).__init__()
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

        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x = torch.tanh(self.bn1(self.fc1(x)))
        x = torch.tanh(self.bn2(self.fc2(x)))
        return self.fc3(x)
    
def squared_loss(output, target):
    loss = torch.abs(output - target) ** 2
    return loss.sum()

def train_net(net, positions, cost):
    assert net.input_dims == 3
    assert net.output_dims == 1
    x_train = torch.as_tensor(positions, device = device, dtype = torch.float32)
    y_train = torch.as_tensor(cost, device = device, dtype = torch.float32)
    net = net.float()
    net.train()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    x_ref = torch.zeros(1, 3).to(device)
    x_ref.requires_grad=True
    n_epochs = 150 
    for epoch in range(n_epochs):

        for x, y in zip(x_train, y_train):
            optimizer.zero_grad()       
            loss = squared_loss(x_ref,x_train)
            loss.backward()
            optimizer.step()
    del x_train, y_train
    return net