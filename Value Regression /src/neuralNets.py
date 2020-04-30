
import torch 
import numpy as np
import torch.nn as nn


class FeedForwardNet(nn.Module):
    def __init__(self, 
                 in_features:int  = 3,
                 out_features:int = 1,
                 nhiddenunits:int = 256):
        
        super(FeedForwardNet, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.nhiddenunits = nhiddenunits
        
        # Structure
        self.fc1 = nn.Linear(self.in_features, self.nhiddenunits)
        self.fc2 = nn.Linear(self.nhiddenunits, 3)
        self.fc3 = nn.Linear(3, self.out_features)

        # Weight Initialization protocol
        nn.init.kaiming_uniform_(self.fc1.weight)
        nn.init.kaiming_uniform_(self.fc2.weight)
        nn.init.kaiming_uniform_(self.fc3.weight)
        
        # Bias Initialization protocol
        self.fc1.bias.data.fill_(0)
        self.fc2.bias.data.fill_(0)
        self.fc3.bias.data.fill_(0)
        
        # Activation
        self.activation = nn.Tanh()
      
        self.device = torch.device('cpu')
        self.to(self.device)
        print(self)
        

    def forward(self, x):
        
        x1 = self.activation(self.fc1(x)) 
        x2 = self.activation(self.fc2(x1)) 
        x3 = self.fc3(x2) 
        
        return x3
    

    def jacobian(self, x):
        j = torch.autograd.functional.jacobian(self.forward, x).squeeze()
        return j

    def hessian(self, x):
        h = torch.autograd.functional.hessian(self.forward, x).squeeze()
        return h

    def batch_hessian(self, x):
        h = [torch.autograd.functional.hessian(self.forward, x) for x in x]
        return torch.stack(h).squeeze()
    
    def batch_jacobian(self, x):
        j = [torch.autograd.functional.jacobian(self.forward, x) for x in x]
        return torch.stack(j).squeeze()
            


class SquaredNet(nn.Module):
    def __init__(self, 
                 in_features:int  = 3,
                 out_features:int = 3,
                 nhiddenunits:int = 256):
        
        super(SquaredNet, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.nhiddenunits = nhiddenunits
        
        # Structure
        self.fc1 = nn.Linear(self.in_features, self.nhiddenunits)
        self.fc2 = nn.Linear(self.nhiddenunits, 3)
        self.fc3 = nn.Linear(3, self.out_features)

        # Weight Initialization protocol
        nn.init.kaiming_uniform_(self.fc1.weight)
        nn.init.kaiming_uniform_(self.fc2.weight)
        nn.init.kaiming_uniform_(self.fc3.weight)
        
        # Bias Initialization protocol
        self.fc1.bias.data.fill_(0)
        self.fc2.bias.data.fill_(0)
        self.fc3.bias.data.fill_(0)
        
        # Activation
        self.activation = nn.Tanh()
      
        self.device = torch.device('cpu')
        self.to(self.device)
        print(self)
        

    def forward(self, x):
        
        x = self.activation(self.fc1(x)) 
        x = self.activation(self.fc2(x)) 
        x = self.fc3(x) 
        return x.pow(2).sum(dim=1, keepdim=True)


    def jacobian(self, x):
        x = x.reshape(1, 3)
        j = torch.autograd.functional.jacobian(self.forward, x).squeeze()
        return j
    
    def hessian(self, x):
        x = x.reshape(1, 3)
        h = torch.autograd.functional.hessian(self.forward, x).squeeze()
        return h
    
    def batch_jacobian(self, x):
        jj = []
        for x in x:
            x = x.reshape(1, 3)
            j = torch.autograd.functional.jacobian(self.forward, x).squeeze()
            jj.append(j)
        return torch.stack(jj).squeeze()
    
    def batch_hessian(self, x):
        jj = []
        for x in x:
            x = x.reshape(1, 3)
            j = torch.autograd.functional.hessian(self.forward, x).squeeze()
            jj.append(j)
        return torch.stack(jj).squeeze()