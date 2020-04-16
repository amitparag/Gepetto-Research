
from neuralNet import FeedForwardNet
from terminalNet import *
import numpy as np
import matplotlib.pyplot as plt

def plot_comparisions(net, size = 50, savename=None):
    """
    @param:
         1: neural network --> This will go inside the terminal model
         2: Size of xrange   
         
    @returns 
         1: Scatter plot of position w.r.t ddp.cost of plain crocoddyl
         2: Scatter plot of position w.r.t terminal(Net) cost
         3: Scatter plot of position w.r.t crocoddyl iterations
         4: Scatter plot of position w.r.t crocoddyl with terminal net iterations
    
    """
    
    xrange = np.linspace(-2.,2.,size)
    xtest = np.array([ [x1,x2, 0.] for x1 in xrange for x2 in xrange ])
    
    
    # To store results of crocoddyl with Neural Network inside it.
    terminal_net_iterations = []
    terminal_net_cost = []
    
    # To store results of plain crocoddyl
    crocoddyl_iterations = []
    crocoddyl_cost = []
    
    # Solve problems with both terminal crocoddyl model and crocoddyl model
    model = crocoddyl.ActionModelUnicycle()
    T = 30
    model.costWeights = np.matrix([1,1]).T
    modelValueTerminal = UnicycleTerminal(net)
    for xyz in xtest:
        
        # Solve the problem for crocoddyl with torch network

        problem = crocoddyl.ShootingProblem(m2a(xyz).T, [ model ] * T, modelValueTerminal)
        ddp = crocoddyl.SolverDDP(problem)
        ddp.solve([], [], 1000)
        if ddp.iter < 100:

            terminal_net_iterations.append([ddp.iter])
            terminal_net_cost.append([ddp.cost])
        
     
        # Solve the problem with just crocoddyl

        problem2 = crocoddyl.ShootingProblem(m2a(xyz).T, [ model ] * T, model)
        ddp2 = crocoddyl.SolverDDP(problem2)
        ddp2.solve([], [], 1000)
        crocoddyl_iterations.append([ddp2.iter])
        crocoddyl_cost.append([ddp2.cost])
        
        
        
    terminal_net_iterations = np.array(terminal_net_iterations)
    terminal_net_cost = np.array(terminal_net_cost)
    crocoddyl_iterations = np.array(crocoddyl_iterations)
    crocoddyl_cost = np.array(crocoddyl_cost)
    
    
    
    
    # .... Plotting Cost, Iterations    
    fig, axs = plt.subplots(2, 2, figsize=(10, 5),sharex='col', sharey='row',
                        gridspec_kw={'hspace': 0.5, 'wspace': 0.5}, dpi=80) 
    plt.set_cmap('plasma')
    
    #........Setting Vrange for cost
    
    trange = [ min(crocoddyl_cost),max(crocoddyl_cost) ]
    prange = [ min(terminal_net_cost),max(terminal_net_cost) ]
    vrange = [ min(trange+prange),max(trange+prange)]
    vd  = vrange[1]-vrange[0]
    vrange = [ vrange[0]-vd*.1, vrange[1]+vd*.1 ]
    
    #....... Setting Vrange for iterations
    
    itrange = [ min(crocoddyl_iterations),max(crocoddyl_iterations) ]
    iprange = [ min(terminal_net_iterations),max(terminal_net_iterations) ]
    ivrange = [ min(itrange+iprange),max(itrange+iprange)]
    ivd  = ivrange[1]-ivrange[0]
    ivrange = [ ivrange[0]-ivd*.1, ivrange[1]+ivd*.1 ]
    

        
    im1 = axs[0,0].scatter(xtest[:,0],xtest[:,1],c= crocoddyl_cost.flat,vmin=vrange[0],vmax=vrange[1])
    axs[0,0].set_title("Crocoddyl cost, ie ddp.cost")
    fig.colorbar(im1, ax = axs[0,0]).set_label("cost",labelpad=2, size=15)
    

    im2 = axs[0,1].scatter(xtest[:,0],xtest[:,1],c= terminal_net_cost.flat, vmin=vrange[0],vmax=vrange[1])
    axs[0,1].set_title("Crocoddyl with Neural Net cost")
    fig.colorbar(im2, ax = axs[0,1]).set_label("cost",labelpad=2, size=15)
    
    
    

        
    im3 = axs[1,0].scatter(xtest[:,0],xtest[:,1],c=crocoddyl_iterations.flat,vmin=ivrange[0],vmax=ivrange[1])
    axs[1,0].set_title("Crocoddyl iterations")
    fig.colorbar(im3, ax = axs[1, 0]).set_label("iterations",labelpad=2, size=15)

    
    im4 = axs[1,1].scatter(xtest[:,0],xtest[:,1],c=terminal_net_iterations.flat,vmin=ivrange[0],vmax=ivrange[1])
    axs[1,1].set_title("Crocoddyl with Neural Net iterations")
    fig.colorbar(im4, ax = axs[1, 1]).set_label("iterations",labelpad=2, size=15)
    if savename is not None:
        plt.savefig(savename)
    plt.show()