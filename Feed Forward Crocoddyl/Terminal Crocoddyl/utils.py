
import math
import numpy as np
import random
from neuralNet import FeedForwardNet
import torch
import crocoddyl
from terminalNet import UnicycleTerminal


def a2m(a):
    return np.matrix(a).T


def m2a(m):
    return np.array(m).squeeze()



def circular(r=[2], n=[100]):
    """
    @params:
        r = list of radius
        n = list of points required from each radius
        
    @returns:
        array of points from the circumference of circle of radius r centered on origin
        
    Usage: circle_points([2, 1, 3], [100, 20, 40])
    
    """
    circles = []
    for r, n in zip(r, n):
        t = np.linspace(0, 2* np.pi, n)
        x = r * np.cos(t)
        y = r * np.sin(t)
        z = np.zeros(x.size,)
        circles.append(np.c_[x, y, z])
    return np.array(circles).squeeze()




def trajectories_crocoddyl():
    xtest = circular()
    cost = []
    trajectory = []
    iterations = []
    for xyz in xtest:
        model = crocoddyl.ActionModelUnicycle()
        T = 30
        model.costWeights = np.matrix([1,1]).T
        problem = crocoddyl.ShootingProblem(m2a(xyz).T, [ model ] * T, model)
        ddp = crocoddyl.SolverDDP(problem)
        ddp.solve([], [], 1000)
        if ddp.iter < 100:
            xs_ = np.array(ddp.xs)
            xs = xs_[:,0:2]
            cost.append(ddp.cost)
            trajectory.append(xs)
            iterations.append(ddp.iter)

    return cost, trajectory, iterations




def trajectories_terminal_net(net):
    xtest = circular()
    cost = []
    trajectory = []
    iterations = []
    
    i = 0

    model = crocoddyl.ActionModelUnicycle()
    terminal_model = UnicycleTerminal(net)
    T = 30
    model.costWeights = np.matrix([1,1]).T
    
    
    for xyz in xtest:

        problem = crocoddyl.ShootingProblem(m2a(xyz).T, [ model ] * T, terminal_model)
        ddp = crocoddyl.SolverDDP(problem)
        ddp.solve([], [], 1000)
        #if ddp.iter < 100:

        xs_ = np.array(ddp.xs)
        xs = xs_[:,0:2]
        cost.append(ddp.cost)
        trajectory.append(xs)
        iterations.append(ddp.iter)

    return cost, trajectory, iterations

def plot_trajectories(cost, trajectories, name = "Cost", save= True):
    """
    
    @params:
        cost           = list of keys for cmap
        trajectories   = list of corresponding trajectories
        name           = str, to distinguish between cost and iterations
        
    @ returns plot of trajectories colored according to keys.    
    
    """

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams['figure.dpi'] = 80
    fig = plt.figure(figsize=(8, 6))

    norm = mpl.colors.Normalize(vmin=float(min(cost)), vmax=float(max(cost)))
    cmap = mpl.cm.ScalarMappable(norm = norm, cmap=mpl.cm.plasma)
    cmap.set_array([])


    for key, trajectory in zip(cost, trajectories):
        plt.scatter(trajectory[:, 0], trajectory[:, 1], 
                    marker = '',
                    zorder=2, 
                    s=50,
                    linewidths=0.2,
                    alpha=.8, 
                    cmap = cmap )
        plt.plot(trajectory[:, 0], trajectory[:, 1], c=cmap.to_rgba(key))

    plt.xlabel("X Coordinates", fontsize = 20)
    plt.ylabel("Y Coordinates", fontsize = 20)
    plt.colorbar(cmap).set_label(name, labelpad=2, size=15)
    if save:
        plt.savefig(name+".png")
    plt.show()