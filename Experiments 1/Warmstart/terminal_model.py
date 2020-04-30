import numpy as np
import crocoddyl
import torch
from neural_net import FeedForwardNet, jacobian, hessian

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



class UnicycleTerminal(crocoddyl.ActionModelAbstract):
    """
    This class is the terminal model for unicycle crocoddyl with neural net inside it.
    
    Given a state x, the neural net predicts the value function in the calc.
            # net(x) = data.cost
            
    Given the state x, the jacobian and hessian of the net(x) with respect to x are calculated
            # jacobian(net(x), x) = data.Lx
            # hessian(net(x), x)  = data.Lxx
    
    """
    def __init__(self, net):
        """
        @params
            1: network
        
        usage: terminal_model = UnicycleTerminal(net)   
        
        """
        crocoddyl.ActionModelAbstract.__init__(self, crocoddyl.StateVector(3), 2, 5)
        self.net = net
        
        
    def calc(self, data, x, u=None):
        
        x = torch.as_tensor(m2a(x), dtype = torch.float32).resize_(1, 3)
        
        with torch.no_grad():
            data.cost = self.net(x).item()

    def calcDiff(self, data, x, u=None):
        
        if u is None:
            u = self.unone
        
        x0 = torch.as_tensor(m2a(x), dtype = torch.float32).resize_(1, 3)
        
        x0.requires_grad_(True)
        
        j = jacobian(self.net(x0), x0)        
        h = hessian(self.net(x0), x0)
        
        data.Lx = a2m(j.detach().numpy())
        data.Lxx = a2m(h.detach().numpy())
        
        