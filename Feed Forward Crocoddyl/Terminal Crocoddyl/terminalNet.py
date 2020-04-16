
import numpy as np
import crocoddyl
import torch
from neuralNet import jacobian, hessian


   
    
def a2m(a):
    return np.matrix(a).T


def m2a(m):
    return np.array(m).squeeze()




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
        
        