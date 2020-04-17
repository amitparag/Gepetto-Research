

import crocoddyl
import numpy as np
import torch
import matplotlib.pyplot as plt
from utils import *
def terminal_data(net, size = 40):
    """
    Returns trajectories starting from the circumference of a circle
    
    """
    xrange = np.linspace(-1.,1.,size)
    xtest = torch.tensor([ [x1,x2, 0.] for x1 in xrange for x2 in xrange ], dtype = torch.float32)
    
    cost_net = []
    
    iterations_net = []
    
    
    
    for xyz in xtest:
        model = crocoddyl.ActionModelUnicycle()
        terminal_model = UnicycleTerminal(net)
        T = 30
        model.costWeights = np.matrix([1,1]).T
    

        problem = crocoddyl.ShootingProblem(m2a(xyz).T, [ model ] * T, terminal_model)
        ddp = crocoddyl.SolverDDP(problem)
        ddp.solve([], [], 1000)

        cost_net.append(ddp.cost)
        iterations_net.append(ddp.iter)
    
    
    
    #..........Now solve for crocoddyl
    cost = []    
    iterations = []
    for xyz in xtest:
        model = crocoddyl.ActionModelUnicycle()
        T = 30
        model.costWeights = np.matrix([1,1]).T

        problem = crocoddyl.ShootingProblem(m2a(xyz).T, [ model ] * T, model)
        ddp = crocoddyl.SolverDDP(problem)
        ddp.solve()

        cost.append(ddp.cost)
        iterations.append(ddp.iter)
  

    return cost_net, iterations_net, cost, iterations


def plot_utils(y_true, y_pred, size =35, name  = 'cost', savefig=True):
    xrange = np.linspace(-1.,1.,size)
    xtest = torch.tensor([ [x1,x2, 0.] for x1 in xrange for x2 in xrange ], dtype = torch.float32)

    plt.set_cmap('plasma')
    plt.figure(figsize=[8,10])

    trange = [ min(y_true),max(y_true) ]
    prange = [ min(y_pred),max(y_pred) ]
    vrange = [ min(trange+prange),max(trange+prange)]
    vd  = vrange[1]-vrange[0]
    vrange = [ vrange[0]-vd*.1, vrange[1]+vd*.1 ]

    plt.subplot(3,1,1)
    plt.scatter(xtest[:,0],xtest[:,1],c=y_true,vmin=vrange[0],vmax=vrange[1])
    plt.colorbar().set_label(name,labelpad=2, size=15)
    plt.title('Crocoddyl')   


    plt.subplot(3,1,2)
    plt.scatter(xtest[:,0],xtest[:,1],c=y_pred,vmin=vrange[0],vmax=vrange[1])
    plt.colorbar().set_label(name,labelpad=2, size=15)
    plt.title('Terminal Crocoddyl')


    plt.subplot(3, 1, 3)
    z = np.array(y_true) - np.array(y_pred)
    plt.scatter(xtest[:,0],xtest[:,1],c=z)
    plt.title("Crocoddyl(x) - Terminal_Crocoddyl(x)")
    plt.colorbar().set_label('Error', labelpad=2, size = 15)
    plt.subplots_adjust(hspace=0.25)

    #if savefig:
    #    plt.savefig(name + "TerminalCrocoddyl.png")

    plt.show()
    plt.close()