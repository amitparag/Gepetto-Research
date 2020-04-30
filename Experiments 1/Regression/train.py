import torch
import numpy as np
import torch.optim as optim
from torch.utils import data
from neuralnet import FNet, SQNet
from data import dataGen
from tqdm import tqdm
import time
import crocoddyl



def _sq_net():
    
    # Tensor data for training
    positions, costs, _, _ = dataGen(size = 3000)

    # Torch dataloader
    dataset = torch.utils.data.TensorDataset(positions,costs)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = 500) 

    
    # Generate a Neural Net
    net = SQNet(input_features = positions.shape[1], 
                 output_features = costs.shape[1],
                 n_hiddenunits = 256)
    # Initialize loss and optimizer
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-2)


    for epoch in tqdm(range(10000)):
        for i, (input, target) in enumerate(dataloader):  

            net.eval()
            output = net(input)               

            net.train()
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    torch.save(net, "sqnet.pth")



def _ffn_net():
    
    # Tensor data for training
    positions, costs, _, _ = dataGen(size = 3000)

    # Torch dataloader
    dataset = torch.utils.data.TensorDataset(positions,costs)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = 500) 

    
    # Generate a Neural Net
    net = FNet(input_features = positions.shape[1], 
                 output_features = costs.shape[1],
                 n_hiddenunits = 16)
    # Initialize loss and optimizer
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-2)


    for epoch in tqdm(range(10000)):
        for i, (input, target) in enumerate(dataloader):  

            net.eval()
            output = net(input)               

            net.train()
            loss = criterion(output, target)
           

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


      
    torch.save(net, "fnet.pth")
    
    
if __name__=='__main__':
    _ffn_net()
    _sq_net()