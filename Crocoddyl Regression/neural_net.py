"""
This contains the class of a generic feedforward neural network in pytorch. 

It also contains methods to calculate jacobian and hessian of the output of
the neural network with respect to input. 


"""

import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from jacobian import JacobianReg
from tqdm import tqdm

   



class FeedForwardNet(nn.Module):
    def __init__(self, 
                 input_features,
                 output_features,
                 n_hidden_units = 512,
                 activation = 'relu'):
        
        
        """
        A two hidden layered neural network.
        
        @params:
            1: input_features = number of input features of the dataset
            2: output_features = number of output features of the dataset
            3: n_hidden_units = number of hidden units in hidden layer
            4: activation = either relu or tanh
            
            
        @returns:
            A fully connected feedforward network 
        
        """       
        
        super(FeedForwardNet, self).__init__()
        
        if activation == 'relu':
            self.activation = F.relu
            
        else:
            self.activation = torch.tanh
        # Structure
        self.fc1 = nn.Linear(input_features, n_hidden_units)
        self.fc2 = nn.Linear(n_hidden_units, n_hidden_units)
        self.fc3 = nn.Linear(n_hidden_units, output_features)
        
        # Initialization protocol
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        
      
        self.device = torch.device('cpu')
        self.to(self.device)

    def forward(self, x):     
        
        x1 = self.activation(self.fc1(x)) 
        x2 = self.activation(self.fc2(x1)) 
        x3 = self.fc3(x2) 
        return x3

    
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