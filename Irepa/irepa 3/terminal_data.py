import numpy as np
import crocoddyl as c
from terminal_model import *
c.switchToNumpyArray()
def terminalData(net):
    # get the cost
    positions = []
    x_ref = []
    model = c.ActionModelUnicycle()
    terminal = UnicycleTerminal(net)
    for _ in range(1000):

        x0 = np.array([np.random.uniform(-2.1, 2.1), np.random.uniform(-2.1, 2.1), np.random.uniform(0,1)])
        T = 30
        problem = c.ShootingProblem(x0.T, [ model ] * T, terminal)
        ddp = c.SolverDDP(problem)
        ddp.solve()
        positions.append(x0)
        x_ref.append(ddp.xs[-1].tolist())
    
    positions = np.asarray(positions)
    x_ref = np.squeeze(np.asarray(x_ref))
    del model
    return positions, x_ref