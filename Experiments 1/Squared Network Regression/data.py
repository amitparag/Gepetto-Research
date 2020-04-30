

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









