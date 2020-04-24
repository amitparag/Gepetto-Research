import numpy as np
import crocoddyl
import torch


def a2m(a):
    return np.matrix(a).T


def m2a(m):
    return np.array(m).squeeze()


def solve_crocoddyl(xyz):
    """
    xyz should either be a list of an array
    """
   

    model = crocoddyl.ActionModelUnicycle()
    T = 30
    model.costWeights = np.matrix([1,1]).T

    problem = crocoddyl.ShootingProblem(m2a(xyz).T, [ model ] * T, model)
    ddp = crocoddyl.SolverDDP(problem)
    ddp.solve([], [], 1000)
    return ddp


def random_array(size, theta = 0.):
    """
    @ returns numpy random array of size = size, 3
    """
    x = []
    for _ in range(size):
        # Generate random starting configuration
        xyz = [np.random.uniform(-2.1, 2.1), 
               np.random.uniform(-2.1, 2.1),
               theta]
        
        x.append(xyz)
    return np.array(x)
    
def griddata(size):
    """
    Will return size^2 points
    """
    xrange = np.linspace(-1.,1.,size)
    xtest = np.array([ [x1,x2, 0.] for x1 in xrange for x2 in xrange ])
    return xtest


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