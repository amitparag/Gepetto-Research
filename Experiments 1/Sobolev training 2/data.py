

import numpy as np
import crocoddyl
import torch    
from utils import solve_crocoddyl, random_array
"""
Base crocoddyl data

"""


def dataGen(size:int = 100, theta:float = 0.):
    """
    Returns position, cost, vx[0], vxx[1]
    """


    x   = random_array(size)
    y   = []
    vx  = []
    vxx = []
    
    for state in x:        
        ddp = solve_crocoddyl(state)
        y.append([ddp.cost])
        
        vx_ = np.array(ddp.Vx)
        vx.append(vx_[0])
        
        vxx_ = np.array(ddp.Vxx)
        vxx.append(vxx_[0])
        
    start = torch.tensor(x,   dtype = torch.float32)
    cost  = torch.tensor(y,   dtype = torch.float32)
    grad1 = torch.tensor(vx,  dtype = torch.float32)
    grad2 = torch.tensor(vxx, dtype = torch.float32)
    del ddp,x, y, vx, vxx    
    return start, cost, grad1, grad2




def solver_norms(xtest):
    solutions = []
    for xyz in xtest:
        ddp = solve_crocoddyl(xyz)
        solutions.append(ddp)


    vx    = []
    vxx   = []
    cost  = []


    for ddp in solutions:
        nodes  = np.array(ddp.xs)

        diff1  = np.array(ddp.Vx)
        diff2  = np.array(ddp.Vxx)

        vx.append([np.linalg.norm(diff1[0])])
        vxx.append([np.linalg.norm(diff2[0])])

        cost.append([ddp.cost])



    vx    = np.array(vx).reshape(-1, 1)
    vxx   = np.array(vxx).reshape(-1, 1)
    cost  = np.array(cost).reshape(-1,1)
    return cost, vx, vxx

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



