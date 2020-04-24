import numpy as np
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SQNet(nn.Module):
    def __init__(self, 
                 input_features,
                 output_features,
                n_hidden_units = 16):
        
        super(SQNet, self).__init__()
        
        # Structure
        self.fc1 = nn.Linear(input_features, n_hidden_units)
        self.fc2 = nn.Linear(n_hidden_units, 3)
        self.fc3 = nn.Linear(3, 3)
        self.fc4 = nn.Linear(3, output_features)
        
        # Initialization protocol
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)

      
        self.device = torch.device('cpu')
        self.to(self.device)

    def forward(self, x):
        
        x1 = torch.tanh(self.fc1(x)) 
        x2 = torch.tanh(self.fc2(x1)) 
        x3 = self.fc3(x2) 
        x4 = self.fc4(x3) ** 2
        
        return x4, x3
    

def jacobian(y, x, create_graph=False):
    
    """
    @params:
        1: y      = Neural net to be differentiated
        2: x        = Input tensor to the neural net.
        
    @returns: d(N(x)) / d(x)
    
    usage: jacobian(net(x), x)
    """
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
    """
    Returns the hessian of the function y with respect to x
    Usage: hessian(net(x), x)
    """
    
    return jacobian(jacobian(y, x, create_graph=True), x)    