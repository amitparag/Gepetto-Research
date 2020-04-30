import torch
import crocoddyl
import numpy as np
import numdifftools as nd
from neuralnet import SQNet


def a2m(a):
    return np.matrix(a).T


def m2a(m):
    return np.array(m).squeeze()





        
class SQTerminal(crocoddyl.ActionModelAbstract):
    
    def __init__(self, net):
       
        crocoddyl.ActionModelAbstract.__init__(self, crocoddyl.StateVector(3), 2, 5)
        self.net = net        
    def calc(self, data, x, u=None):        
        x = torch.as_tensor(m2a(x), dtype = torch.float32).resize_(1, 3)
        data.cost = self.net(x).item()

    def calcDiff(self, data, x, u=None):        
        if u is None:
            u = self.unone        
        x = m2a(x)
        
        def function(x):
            x = torch.tensor(x, dtype = torch.float32)
            x = x.resize_(1, 3)
            jacobian = self.net.jacobian(x).squeeze().numpy()
            return np.array(jacobian).reshape(3,)
        
        j = function(x)
        hessian = nd.Jacobian(function)
        h = hessian(x)
        data.Lx = a2m(j)
        data.Lxx = a2m(h)
        
        
class HessianTerminal(crocoddyl.ActionModelAbstract):
    
    def __init__(self, net):
       
        crocoddyl.ActionModelAbstract.__init__(self, crocoddyl.StateVector(3), 2, 5)
        self.net = net        
    def calc(self, data, x, u=None):        
        x = torch.as_tensor(m2a(x), dtype = torch.float32).resize_(1, 3)
        data.cost = self.net(x).item()

    def calcDiff(self, data, x, u=None):        
        if u is None:
            u = self.unone        
        x = m2a(x)
        

        x = torch.tensor(x, dtype = torch.float32)
        x = x.resize_(1, 3)
        j = self.net.jacobian(x).squeeze().numpy()
        h = self.net.hessian(x).squeeze().numpy()
        data.Lx = a2m(j)
        data.Lxx = a2m(h)
        
        
        