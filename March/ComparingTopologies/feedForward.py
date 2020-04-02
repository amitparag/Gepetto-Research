import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import torch.utils.data as data_utils


class FeedForwardNet(nn.Module):
    def __init__(self, input_features, output_features):
        super(FeedForwardNet, self).__init__()
        

        self.fc1 = nn.Linear(input_features, 8)
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        torch.nn.init.normal_(self.fc1.weight.data, -f1, f1)
        torch.nn.init.uniform_(self.fc1.bias, -f1, f1)        
        

        self.fc2 = nn.Linear(8, 8)
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        torch.nn.init.normal_(self.fc2.weight.data, -f2, f2)
        torch.nn.init.uniform_(self.fc2.bias, -f2, f2)        
        
                
        self.fc3 = nn.Linear(8, 1)
        f3 = 1 / np.sqrt(self.fc3.weight.data.size()[0])
        torch.nn.init.normal_(self.fc3.weight.data, -f3, f3)
        torch.nn.init.uniform_(self.fc3.bias, -f3, f3)    
      
        self.device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        
        x1 = torch.tanh(self.fc1(x)) 
        x2 = torch.tanh(self.fc2(x1)) 
        x3 = self.fc3(x2) 
        
        return x3
    
    
def trainFfn(net, start, cost, epochs = 50):
    from tqdm import tqdm
    
    net = net.float()
    net.train()
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    n_epochs = epochs
    
    for epoch in tqdm(range(n_epochs)):
        for x, y in zip(start, cost):
                       
            optimizer.zero_grad()
            y_ = net(x)          
            loss = torch.sum((torch.abs(y_ - y))**2)
            loss.backward()
            optimizer.step()  
    del start, cost        
    return net
    


    