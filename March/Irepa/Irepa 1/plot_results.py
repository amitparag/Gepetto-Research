import os
import numpy as np
import crocoddyl as c
import numdifftools as nd
from time import perf_counter
c.switchToNumpyArray()

import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.stats import kde
import torch
from irepa import *


def irepa_run0():
    # X, Y, cost, iters 
    data = []
    model = c.ActionModelUnicycle()
    for _ in range(2000):

        x0 = np.array([np.random.uniform(-2.1, 2.1), np.random.uniform(-2.1, 2.1), np.random.uniform(0,1)])
        T = 30
        problem = c.ShootingProblem(x0.T, [ model ] * T, model)
        ddp = c.SolverDDP(problem)
        ddp.solve([], [], 1000)
        data.append([float(x0[0]), float(x0[1]), ddp.cost, ddp.iter])
    return np.asarray(data)

def irepa_runs(net):
    # X, Y, cost, iters 
    data = []
    model = c.ActionModelUnicycle()
    terminal_model = UnicycleTerminal(net)
    for _ in range(2000):
        x0 = np.array([np.random.uniform(-2.1, 2.1), np.random.uniform(-2.1, 2.1), np.random.uniform(0,1)])
        T = 30
        problem = c.ShootingProblem(x0.T, [ model ] * T, terminal_model)
        ddp = c.SolverDDP(problem)
        ddp.solve([], [], 1000)

        data.append([float(x0[0]), float(x0[1]), ddp.cost, ddp.iter])
    return np.asarray(data)

def plot_cost(data, n = 0, cmap = 'plasma'):
    pass
    """
    grid_x, grid_y = np.mgrid[-2.1:2.1:100j, -2.1:2.1:200j]
    points = data[:,0:2]
    values = data[:,2]

    iters = data[:,3]

    grid_z0 = griddata(points, values, (grid_x, grid_y))

    plt.imshow(grid_z0.T, extent=(-2.1,2.1,-2.1,2.1), cmap = cmap)
    plt.xticks(np.arange(-2.1, 2.1, 1))
    plt.yticks(np.arange(-2.1, 2.1, 1))
    clb = plt.colorbar()
    clb.set_label('Cost', labelpad=-40, y=1.10, rotation=0)
    plt.title(f"Irepa {n}th iteration")
    plt.savefig(f"Irepa {n}th run cost.png")
    """
    
def plot_iters(data, n= 0, cmap = "plasma"):
    pass
    """
    grid_x, grid_y = np.mgrid[-2.1:2.1:100j, -2.1:2.1:200j]
    points = data[:,0:2]
    iters = data[:,3]
    grid_z0 = griddata(points, iters, (grid_x, grid_y))
    plt.imshow(grid_z0.T, extent=(-2.1,2.1,-2.1,2.1), cmap = cmap)
    plt.xticks(np.arange(-2.1, 2.1, 1))
    plt.yticks(np.arange(-2.1, 2.1, 1))
    clb = plt.colorbar()
    clb.set_label('Iterations', labelpad=-40, y=1.10, rotation=0)
    plt.title(f"Irepa {n}th iteration")
    plt.savefig(f"Irepa {n}th run iterations.png")
    """
    
    
if __name__ == "__main__":

    import torch
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    from irepa import *
    net1 = torch.load('Net1.pth', map_location= device)
    net4 = torch.load('Net4.pth', map_location= device)
    net10 = torch.load('Net10.pth', map_location= device)
    net15 = torch.load('Net15.pth', map_location= device)
    net20 = torch.load('Net20.pth', map_location= device)


    n = 2000
    # Irepa Run 1
    data1 = irepa_runs(net1)
    plot_cost(data1)
    plot_iters(data1)
    np.savetxt(f"irepa_run1_{n}.out", data1, delimiter=',')
    del data1

    # Irepa Run 0
    data2 = irepa_runs(net4)
    plot_cost(data2)
    plot_iters(data2)
    np.savetxt(f"irepa_run4_{n}.out", data2, delimiter=',')
    del data2

    # Irepa Run 0
    data3 = irepa_runs(net10)
    plot_cost(data3)
    plot_iters(data3)
    np.savetxt(f"irepa_run10_{n}.out", data3, delimiter=',')
    del data3

    # Irepa Run 0
    data4 = irepa_runs(net15)
    plot_cost(data4)
    plot_iters(data4)
    np.savetxt(f"irepa_run15_{n}.out", data4, delimiter=',')
    del data4

    # Irepa Run 0
    data5 = irepa_runs(net20)
    plot_cost(data5)
    plot_iters(data5)
    np.savetxt(f"irepa_run20_{n}.out", data5, delimiter=',')
    del data5
