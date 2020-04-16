
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
import numpy as np
from jacobian import JacobianReg
from tqdm import tqdm    
from data import get_data
from neural_net import FeedForwardNet

def train(nhiddenUnits = 512, epochs= 10000, batchsize:int = 1000, lr = 1e-3):
    """
    
    Generates and returns a trained neural network
    
    
    
    """
    
    
    # Make data and create a data generator to be used in training
    xtrain, ytrain = get_data(1000)
    dataset = torch.utils.data.TensorDataset(xtrain,ytrain)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batchsize) # DataGenerator

    
    # Generate a Neural Net
    net = FeedForwardNet(input_features = xtrain.shape[1], 
                         output_features = ytrain.shape[1],
                         n_hidden_units = nhiddenUnits)
    
    # set the net to training mode
    net = net.float()
    net.train()
    
    
    # Define the loss function for optimizer
    criteria = torch.nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay = 0.1)   
    
    # Jacobian regularization
    reg = JacobianReg() 
    lambda_JR = 0.01 
    
    t0 = time.time()    
    # Training    
    for epoch in tqdm(range(epochs)):        
        for data, target in dataloader:   
            data.requires_grad=True
            optimizer.zero_grad()
            output = net.forward(data)          
            loss_super = criteria(output, target)
            R = reg(data, output)                            # Jacobian regularization
            loss = loss_super + lambda_JR*R                  # full loss
            loss.backward()
            optimizer.step()                                      
                     
    print('Training lasted = %.0f seconds' % (time.time()-t0))        
        
    
    del xtrain, ytrain

    return net