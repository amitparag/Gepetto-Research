

import numpy as np
import crocoddyl
import torch    
from utils import solve_crocoddyl, random_array
"""
Base crocoddyl data

"""


def dataGen(size:int = 100, theta:float = 0.):
    """
    Returns position and cost
    """


    x = random_array(size)
    y = []

    
    for state in x:        
        ddp = solve_crocoddyl(state)
        y.append([ddp.cost])
        
    positions = torch.tensor(x, dtype = torch.float32)
    cost = torch.tensor(y, dtype = torch.float32)
    del ddp,x, y    
    return positions, cost




def solver_grads(xtest):
    """
    Returns position, cost, vx[0], vxx[1]
    """


    
    y   = []
    vx  = []
    vxx = []
    
    for state in xtest:        
        ddp = solve_crocoddyl(state)
        y.append([ddp.cost])
        
        vx_ = np.array(ddp.Vx)
        vx.append(vx_[0])
        
        vxx_ = np.array(ddp.Vxx)
        vxx.append(vxx_[0])
        
    
    cost  = torch.tensor(y,   dtype = torch.float32)
    grad1 = torch.tensor(vx,  dtype = torch.float32)
    grad2 = torch.tensor(vxx, dtype = torch.float32)
    del ddp, y, vx, vxx    
    return cost, grad1, grad2






