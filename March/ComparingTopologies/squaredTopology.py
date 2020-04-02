import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import torch.utils.data as data_utils
torch.set_default_dtype(torch.float64)


class SquaredNet(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()
        
        self.input_features = input_features
        self.output_features = output_features

        self.fc1 = nn.Linear(self.input_features, 8)
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        torch.nn.init.normal_(self.fc1.weight.data, -f1, f1)
        torch.nn.init.uniform_(self.fc1.bias, -f1, f1)        
        

        self.fc2 = nn.Linear(8, 8)
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        torch.nn.init.normal_(self.fc2.weight.data, -f2, f2)
        torch.nn.init.uniform_(self.fc2.bias, -f2, f2)        
        
                
        self.fc3 = nn.Linear(8, 3)
        f3 = 1 / np.sqrt(self.fc3.weight.data.size()[0])
        torch.nn.init.normal_(self.fc3.weight.data, -f3, f3)
        torch.nn.init.uniform_(self.fc3.bias, -f3, f3)   
        
        self.fc4 = nn.Linear(3, self.output_features)        
        self.device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x)) 
        #x = torch.tensor([torch.sum(self.fc3(x) **2)])
        #x = torch.tensor([torch.sum(x **2)])
        x = self.fc3(x)
        #x.requires_grad_(True)
        return self.fc4(x)
    
    
def mse_loss(x, y):
    return torch.sum((torch.abs(x - y))**2)
    
def trainNetwork(net, start, cost, epochs = 50):
    from tqdm import tqdm
    
    net = net.float()
    net.train()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    n_epochs = epochs
    criterion = nn.CrossEntropyLoss()
    for epoch in tqdm(range(n_epochs)):
        for x, y in zip(start, cost):
            x.to('cpu')
            y.to('cpu')
            y.requires_grad=True
            y_ = net(x)          
            loss = F.mse_loss(y_, y)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()  
    del start, cost        
    return net
    

