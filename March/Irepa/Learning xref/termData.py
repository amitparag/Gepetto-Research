import torch
import numpy as np
import crocoddyl 
from UniTerminal import *
crocoddyl.switchToNumpyArray()
from tqdm import tqdm
def terminalData(net):
    # get the cost
    positions = []
    x_ref = []
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
        x_ref.append(ddp.xs[-1].tolist())
    
    positions = torch.tensor(positions, dtype = torch.float32)
    x_ref = torch.tensor(x_ref, dtype = torch.float32)
    del model
    return positions, x_ref