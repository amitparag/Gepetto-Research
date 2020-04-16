import numpy as np
import crocoddyl
import torch    

"""
Base crocoddyl data

"""


def get_data(size:int = 1000, theta:float = 0.):
    """
    @params:
        1: size  = size of the dataset
        2: theta = float, between 0, 1
    
    Returns xtrain, ytrain.
    Returns data in the form of x --> V(x)
    
    """

    _xtrain = []
    _ytrain = []

    
    for _ in range(size):
        # Generate random starting configuration
        xyz = [np.random.uniform(-2.1, 2.1), 
               np.random.uniform(-2.1, 2.1),
               theta]
        
        
        model = crocoddyl.ActionModelUnicycle()
        T = 30
        model.costWeights = np.matrix([1,1]).T
        
        problem = crocoddyl.ShootingProblem(m2a(xyz).T, [ model ] * T, model)
        ddp = crocoddyl.SolverDDP(problem)
        ddp.solve([], [], 1000)
        
        cost = [ddp.cost]
        
        _xtrain.append(xyz)
        _ytrain.append(cost)
        
    xtrain = torch.tensor(_xtrain, dtype = torch.float32)
    ytrain = torch.tensor(_ytrain, dtype = torch.float32)
        
    return xtrain, ytrain


    
def a2m(a):
    return np.matrix(a).T


def m2a(m):
    return np.array(m).squeeze()



def crocoddyl_cost(xtest):
    """
    Returns the crocddyl cost array for a given array of starting positions

    """
    cost = []
    for xyz in xtest:
        model = crocoddyl.ActionModelUnicycle()
        T = 30
        model.costWeights = np.matrix([1,1]).T

        problem = crocoddyl.ShootingProblem(m2a(xyz).T, [ model ] * T, model)
        ddp = crocoddyl.SolverDDP(problem)
        ddp.solve()

        cost.append([ddp.cost])

    return torch.tensor(cost, dtype = torch.float32)
