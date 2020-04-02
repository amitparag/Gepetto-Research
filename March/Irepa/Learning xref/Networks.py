
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np

class xRefNet(nn.Module):
    def __init__(self, input_features, output_features):
        super(xRefNet, self).__init__()
        

        self.fc1 = nn.Linear(input_features, 16)
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        torch.nn.init.normal_(self.fc1.weight.data, -f1, f1)
        torch.nn.init.uniform_(self.fc1.bias, -f1, f1)        
        

        self.fc2 = nn.Linear(16, 16)
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        torch.nn.init.normal_(self.fc2.weight.data, -f2, f2)
        torch.nn.init.uniform_(self.fc2.bias, -f2, f2)        
        
                
        self.fc3 = nn.Linear(16, 16)
        f3 = 1 / np.sqrt(self.fc3.weight.data.size()[0])
        torch.nn.init.normal_(self.fc3.weight.data, -f3, f3)
        torch.nn.init.uniform_(self.fc3.bias, -f3, f3)        
               
        
        self.fc4 = nn.Linear(16, output_features)
        f4 = 1 / np.sqrt(self.fc4.weight.data.size()[0])
        torch.nn.init.normal_(self.fc4.weight.data, -f4, f4)
        torch.nn.init.uniform_(self.fc4.bias, -f4, f4)


        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x_ref = self.fc4(x)
        
        return x_ref
    
    
def squared_loss(output, target):
    loss = torch.abs(output - target) ** 2
    return loss.mean() 


def train_xrefNet(net, initial_positions, x_refs):
    from tqdm import tqdm
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    initial_positions.to(device)
    x_refs.to(device)
    net = net.float()
    net.train()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    n_epochs = 100 
    for epoch in tqdm(range(n_epochs)):

        for data, target in zip(initial_positions, x_refs):
            optimizer.zero_grad()
            x_ref_predicted = net(data.to(device))
            
            loss = squared_loss(x_ref_predicted, target.to(device))
            
            loss.backward()
            optimizer.step()
    del initial_positions, x_refs
    return net
    


