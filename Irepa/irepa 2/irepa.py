import os
import torch
import numpy as np
import crocoddyl

from base_crocoddyl_data import *
from terminal_model import *
from torchNet import *
from terminal_data import *

crocoddyl.switchToNumpyArray()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def irepa():
    # run irepa zero and get the net
    net_path = "./torchNetworks/"
    position, cost = base_data()
    net = TwoLayerNet(input_features=3, output_features=1)
    net = train_net(net, position, cost[:,0].reshape(position.shape[0], 1))
    # this is the first net trained on data from pure crocoddyl
    name = "Net0.pth"
    save_path = os.path.join(net_path, name)
    torch.save(net, save_path)
    del position, cost
    
    
    
    # Use this net to generate data from terminal model
    position1, cost1 = terminal_data(net)
    net1 = train_net(net, position1, cost1[:,0].reshape(position1.shape[0], 1))
    name1 = "Net1.pth"
    save_path1 = os.path.join(net_path, name1)
    torch.save(net1, save_path1)
    del net1, position1, cost1

    """
    positions, cost = base_data()
    # Create the first net
    net = TwoLayerNet(input_features=3, output_features=1)
    # train the net
    net = train_net(net, positions, cost[:,0].reshape(positions.shape[0], 1))
    
    net_path = "./torchNetworks/"
    name_ = "Net1.pth"
    save_path_ = os.path.join(net_path, name_)
    torch.save(net, save_path_)
    data_path = "./data/"
    for _ in range(21):
        positions, cost = terminal_data(net)
        net = train_net(net, positions, cost[:,0].reshape(positions.shape[0], 1))
        if _ in [0, 4, 9, 14, 19]:
            name = f"Net{_ + 1}.pth"
            save_path = os.path.join(net_path, name)
            torch.save(net, save_path)
            
            p = f"p{_ + 1}.out"
            c = f"c{_ + 1}.out"
            pname = os.path.join(data_path, p)
            cname = os.path.join(data_path, c)

            np.savetxt(pname, positions, delimiter = ',')
            np.savetxt(cname, cost, delimiter = ',')
        del positions, cost    
      """

if __name__=='__main__':
    irepa()