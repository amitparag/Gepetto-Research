import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class xRefNet(nn.Module):
    def __init__(self, input_features, output_features):
        super(xRefNet, self).__init__()
        self.input_dims = input_features
        self.output_dims = output_features

        self.fc1 = nn.Linear(self.input_dims, 16)
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        torch.nn.init.normal_(self.fc1.weight.data, -f1, f1)
        torch.nn.init.uniform_(self.fc1.bias, -f1, f1)        
        self.bn1 = nn.LayerNorm(16)

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
        
        self.fc4 = nn.Linear(16, 3)
        f4 = 1 / np.sqrt(self.fc4.weight.data.size()[0])
        torch.nn.init.normal_(self.fc4.weight.data, -f4, f4)
        torch.nn.init.uniform_(self.fc4.bias, -f4, f4)
        


        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        
        x = torch.tanh(self.bn1(self.fc1(x)))
        x = torch.tanh(self.bn2(self.fc2(x)))
        x = torch.tanh(self.bn3(self.fc3(x)))
        x_ref = self.fc4(x)
        return x_ref
    
    
def squared_loss(output, target):
    loss = torch.abs(output - target) ** 2
    return loss.sum() 

def train_xRefNet(net, start_x, terminal_x):
    from tqdm import tqdm

    x = torch.as_tensor(start_x, device = device, dtype = torch.float32)
    y = torch.as_tensor(terminal_x, device = device, dtype = torch.float32)
    
    net = net.float()
    net.train()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    
    n_epochs = 100 
    for epoch in tqdm(range(n_epochs)):

        for position, x_ref in zip(x, y):
            optimizer.zero_grad()
            x_ref_predicted = net(position)
            loss = squared_loss(x_ref_predicted,x_ref)
            loss.backward()
            optimizer.step()
    del x, y
    return net