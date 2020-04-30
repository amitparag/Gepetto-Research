
from data import Datagen
from neuralNets import FeedForwardNet, SquaredNet

import torch
from tqdm import tqdm
import numpy as np
import crocoddyl
import torch.optim as optim
import torch.nn.functional as F

def train(num_epochs:int = 10000,
          datasize:int = 3000,
          train_feedforward:bool = True,
          train_squared:bool=True,
          save_fn = True,
          save_sqn = True):
    
    if train_feedforward:
        net = FeedForwardNet()
        data = Datagen()
        x, y = data.value(datasize)
        dataset = torch.utils.data.TensorDataset(x, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = 1000)  
        net = net.float()
        net.train()
        criterion = torch.nn.MSELoss(reduction='sum')
        optimizer = optim.Adam(net.parameters(), lr= 1e-3, weight_decay = 0.1) 
        for epoch in tqdm(range(num_epochs)):        
            for data, target in dataloader:  
                outputs = net(data)
                loss = criterion(outputs, target)


                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        if save_fn:
            torch.save(net, 'fnet.pth')
        else: return net
        
    if train_squared:
        sqn = SquaredNet()
        data = Datagen()
        x, y = data.value(datasize)
        dataset = torch.utils.data.TensorDataset(x, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = 1000)  
        sqn = sqn.float()
        sqn.train()
        criterion = torch.nn.MSELoss(reduction='sum')
        optimizer = optim.Adam(sqn.parameters(), lr= 1e-3, weight_decay = 0.1) 
        for epoch in tqdm(range(num_epochs)):        
            for data, target in dataloader:  
                outputs = sqn(data).reshape(target.shape)
                loss = criterion(outputs, target)


                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        if save_sqn:
            torch.save(net, 'sqnet.pth')
        else: return sqn


if __name__=='__main__':
	train(num_epochs=10000,
          datasize = 5000,
          train_feedforward = True,
          train_squared = True,
          save_fn = True,
          save_sqn = True)
        
        
        
