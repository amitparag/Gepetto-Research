
import numpy as np
import matplotlib.pyplot as plt
import crocoddyl
import torch
from utils import *
from neural_net import *


# Solves problems with terminal crocoddyl
def solve_terminal_crocoddyl(starting_positions, net):
    """
    Returns the iterations , cost, trajectories by solving the terminal_net_crocoddyl.
    
    """
    
    iterations, cost, trajectories = [], [], []
    
    
    # Running model
    model = crocoddyl.ActionModelUnicycle()
    # Terminal model
    terminal_model = UnicycleTerminal(net)
    # Time horizon
    T = 30
    # model costweights
    model.costWeights = np.matrix([1,1]).T
    
    for starting_position in starting_positions:


        problem = crocoddyl.ShootingProblem(m2a(starting_positions).T, [ model ] * T, terminal_model)
        ddp = crocoddyl.SolverDDP(problem)
        ddp.solve([], [], 1000)
        iterations.append(ddp.iter)
        cost.append(ddp.cost)
        trajectories.append(m2a(ddp.xs))
        
    return iterations, cost, trajectories  


# Return gradients from a neural network
def grad(inputs, net):
    """
    Gradient of neural net with respect to each input in xtest
    
    """
    gradient = []
    double_gradient = []
    for x in inputs:
        x.requires_grad=True
        grad = jacobian(net(x), x).detach().numpy().tolist()
        double_grad = hessian(net(x), x).detach().numpy().squeeze()
        
        gradient.append(grad)
        double_gradient.append(double_grad)
    
    return np.array(gradient).squeeze(), np.array(double_gradient)


# Plot predictions of the neural net
def plot_neural_net(net, name = None):
    xrange = np.linspace(-2,2, 100)
    xtest = torch.tensor([ [x1,x2, 0.] for x1 in xrange for x2 in xrange ], dtype = torch.float32)
    
    
    # Get neural net related predictions, gradients, hessians
    preds = net(xtest).detach().numpy()
    g, h = grad(xtest, net)
    g_norm = []
    h_norm = []
    for i in g:
        g_norm.append(np.linalg.norm(i))
    for i in h:
        h_norm.append(np.linalg.norm(i))
        
        
        
    # Get crocoddyl related cost, gradients, hessians
    xtest = xtest.detach().numpy()

    model = crocoddyl.ActionModelUnicycle()
    g_c = []
    h_c = []
    T = 30
    cost = []
    model.costWeights = np.matrix([1,1]).T
    for xyz in xtest:
        problem = crocoddyl.ShootingProblem(m2a(xyz).T, [ model ] * T, model)
        ddp = crocoddyl.SolverDDP(problem)
        ddp.solve([], [], 1000)
        vx = np.array(ddp.Vx)
        vxx = np.array(ddp.Vxx)
        g_c.append(vx[0])
        h_c.append(vxx[0])
        cost.append(ddp.cost)
    

    gc = np.array(g_c)        
    hc = np.array(h_c)
    
    
    gc_norm = []
    hc_norm = []
    for i in gc:
        gc_norm.append(np.linalg.norm(i))
    for i in hc:
        hc_norm.append(np.linalg.norm(i))
    
    
    
    
    plt.clf()
    
    # Make the figure:

    fig, axs = plt.subplots(5, 2, figsize=(18, 20), sharex=True, sharey ='row')
    fig.subplots_adjust(left=0.02, bottom=0.2, right=0.95, top=0.94, wspace=0.25)

    # Plot prediction, cost
    im1 = axs[0, 0].scatter(x = xtest[:,0], y = xtest[:,1], c= preds)
    fig.colorbar(im1, ax=axs[0, 0])
    
    
    im2 = axs[0, 1].scatter(x = xtest[:,0], y = xtest[:,1], c= cost)
    fig.colorbar(im2, ax=axs[0, 1])
    
    
    # Plot g_p/ x0, g_c/x0
    im3 = axs[1, 0].scatter(x = xtest[:,0], y = xtest[:,1], c=  g[:,0])
    fig.colorbar(im3, ax=axs[1, 0])
    
    im4 = axs[1, 1].scatter(x = xtest[:,0], y = xtest[:,1], c=  gc[:,0])
    fig.colorbar(im4, ax=axs[1, 1])
    
    # Plot g_p/ x1, g_c/x1
    im5 = axs[2, 0].scatter(x = xtest[:,0], y = xtest[:,1], c=  g[:,1])
    fig.colorbar(im5, ax=axs[2, 0])
    
    im6 = axs[2, 1].scatter(x = xtest[:,0], y = xtest[:,1], c=  gc[:,1])
    fig.colorbar(im6, ax=axs[2, 1])
    
    # Plot norm of the gradient
    im7 = axs[3, 0].scatter(x = xtest[:,0], y = xtest[:,1], c=  g_norm,)
    fig.colorbar(im7, ax=axs[3, 0])
    
    im8 = axs[3, 1].scatter(x = xtest[:,0], y = xtest[:,1], c=  gc_norm)
    fig.colorbar(im8, ax=axs[3, 1])
    
    
    # Plot Hessian norm
    im9 = axs[4, 0].scatter(x = xtest[:,0], y = xtest[:,1], c=  h_norm)
    fig.colorbar(im9, ax=axs[4, 0])
    
    im10 = axs[4, 1].scatter(x = xtest[:,0], y = xtest[:,1], c=  hc_norm)
    fig.colorbar(im10, ax=axs[4, 1])
    
    
    # Set titles
    axs[0, 0].set_title("Predictions, p", fontsize  = 15)
    axs[0, 1].set_title("Data, v", fontsize  = 15)
    axs[1, 0].set_title("dp/dx0", fontsize  = 15)
    axs[1, 1].set_title("dv/dx0", fontsize  = 15)
    axs[2, 0].set_title("dp/dx1", fontsize  = 15)
    axs[2, 1].set_title("dv/dx1", fontsize  = 15)
    axs[3, 0].set_title("norm(grad(p))", fontsize  = 15)
    axs[3, 1].set_title("norm(Vx)", fontsize  = 15)
    axs[4, 0].set_title("norm hessian(p)", fontsize  = 15)
    axs[4, 1].set_title("norm Vxx", fontsize  = 15)


    if name is not None:
        plt.savefig("./images/"+name+".png")

    
def plot_2(net, name = None):
    x = 0.99
    y = np.linspace(-1, 1, 50)
    xtest = torch.tensor([ [x,x2, 0.] for x2 in y ], dtype = torch.float32)
    
    # Get neural net related predictions, gradients, hessians
    preds = net(xtest).detach().numpy()
    g, h = grad(xtest, net)
    g_norm = []
    h_norm = []
    for i in g:
        g_norm.append(np.linalg.norm(i))
    for i in h:
        h_norm.append(np.linalg.norm(i))
        
        
        
    # Get crocoddyl related cost, gradients, hessians
    xtest = xtest.detach().numpy()

    model = crocoddyl.ActionModelUnicycle()
    g_c = []
    h_c = []
    T = 30
    cost = []
    model.costWeights = np.matrix([1,1]).T
    for xyz in xtest:
        problem = crocoddyl.ShootingProblem(m2a(xyz).T, [ model ] * T, model)
        ddp = crocoddyl.SolverDDP(problem)
        ddp.solve([], [], 1000)
        vx = np.array(ddp.Vx)
        vxx = np.array(ddp.Vxx)
        g_c.append(vx[0])
        h_c.append(vxx[0])
        cost.append(ddp.cost)
    

    gc = np.array(g_c)        
    hc = np.array(h_c)
    
    
    gc_norm = []
    hc_norm = []
    for i in gc:
        gc_norm.append(np.linalg.norm(i))
    for i in hc:
        hc_norm.append(np.linalg.norm(i))
        
        
    plt.clf()
    
    # Make the figure:

    fig, axs = plt.subplots(5, 2, figsize=(18, 20), sharex=True, sharey ='row')
    fig.suptitle("X = 0.99, Theta = 0", fontsize = 20)
    fig.subplots_adjust(left=0.02, bottom=0.2, right=0.95, top=0.94, wspace=0.25)
    
    # Plot prediction, cost
    im1 = axs[0, 0].scatter(xtest[:,1], y = preds, marker = "*")
    im2 = axs[0, 1].scatter(xtest[:,1], y = cost, marker = "*")
    
    im3 = axs[1, 0].scatter(xtest[:,1], y = g[:,0], marker = "*")
    im4 = axs[1, 1].scatter(xtest[:,1], y = gc[:,0], marker = "*")
    
    im5 = axs[2, 0].scatter(xtest[:,1], y = g[:,1], marker = "*")
    im6 = axs[2, 1].scatter(xtest[:,1], y = gc[:,1], marker = "*")
    
    im7 = axs[3, 0].scatter(xtest[:,1], y = g_norm, marker = "*")
    im8 = axs[3, 1].scatter(xtest[:,1], y = gc_norm, marker = "*")
    
    im9 = axs[4, 0].scatter(xtest[:,1], y = h_norm, marker = "*")
    im10 = axs[4, 1].scatter(xtest[:,1], y = hc_norm, marker = "*")
    


    
    
    
    # Set titles
    axs[0, 0].set_title("Predictions, p", fontsize  = 15)
    axs[0, 1].set_title("Data, v", fontsize  = 15)
    axs[1, 0].set_title("dp/dx0", fontsize  = 15)
    axs[1, 1].set_title("dv/dx0", fontsize  = 15)
    axs[2, 0].set_title("dp/dx1", fontsize  = 15)
    axs[2, 1].set_title("dv/dx1", fontsize  = 15)
    axs[3, 0].set_title("norm(grad(p))", fontsize  = 15)
    axs[3, 1].set_title("norm(Vx)", fontsize  = 15)
    axs[4, 0].set_title("norm hessian(p)", fontsize  = 15)
    axs[4, 1].set_title("norm Vxx", fontsize  = 15)

    
    if name is not None:
        plt.savefig("./images/"+name+".png")
    
    
    