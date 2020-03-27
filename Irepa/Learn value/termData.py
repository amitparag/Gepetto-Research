
import numpy as np
import torch
import crocoddyl 
crocoddyl.switchToNumpyArray()
from tqdm import tqdm
from termModel import *
def terminalData(net):
    # get the cost
    positions = []
    value = []
    model = crocoddyl.ActionModelUnicycle()
    model.costWeights = np.matrix([1,1]).T
    terminal = UnicycleTerminal(net)
    for _ in tqdm(range(1000)):

        x0 = np.array([np.random.uniform(-2.1, 2.1), np.random.uniform(-2.1, 2.1), np.random.uniform(0,1)])
        T = 30
        problem = crocoddyl.ShootingProblem(x0.T, [ model ] * T, terminal)
        ddp = crocoddyl.SolverDDP(problem)
        ddp.solve()
        positions.append(x0)
        value.append(np.array([ddp.cost]))
    
    positions = torch.tensor(positions, dtype = torch.float32)
    value = torch.tensor(value, dtype = torch.float32)
    del model
    return positions, value