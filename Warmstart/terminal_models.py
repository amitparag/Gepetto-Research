
import torch
import numpy
from sqnet import *
import crocoddyl
import numdifftools as nd

def a2m(a):
    return np.matrix(a).T


def m2a(m):
    return np.array(m).squeeze()

class TrueHessian(crocoddyl.ActionModelAbstract):
    
    def __init__(self, net):
       
        crocoddyl.ActionModelAbstract.__init__(self, crocoddyl.StateVector(3), 2, 5)
        self.net = net
        
        
    def calc(self, data, x, u=None):
        
        x = torch.as_tensor(m2a(x), dtype = torch.float32).resize_(1, 3)
        
        with torch.no_grad():
            data.cost = self.net(x)[0].item()

    def calcDiff(self, data, x, u=None):
        
        if u is None:
            u = self.unone
        
        x0 = torch.as_tensor(m2a(x), dtype = torch.float32).resize_(1, 3)
        
        x0.requires_grad_(True)
        
        j = jacobian(self.net(x0)[0], x0)        
        h = hessian(self.net(x0)[0], x0)
        
        data.Lx = a2m(j.detach().numpy())
        data.Lxx = a2m(h.detach().numpy())
        
        
        
class GaussNewtonHessian(crocoddyl.ActionModelAbstract):
    
    def __init__(self, net):
       
        crocoddyl.ActionModelAbstract.__init__(self, crocoddyl.StateVector(3), 2, 5)
        self.net = net
        
        
    def calc(self, data, x, u=None):
        
        x = torch.as_tensor(m2a(x), dtype = torch.float32).resize_(1, 3)
        
        with torch.no_grad():
            data.cost = self.net(x)[0].item()

    def calcDiff(self, data, x, u=None):
        
        if u is None:
            u = self.unone
        
        x0 = torch.as_tensor(m2a(x), dtype = torch.float32).resize_(1, 3)
        
        x0.requires_grad_(True)
        j = jacobian(self.net(x0)[0], x0) 
        
        dr = self.net(x0)[1].detach().numpy().reshape(3, 1)
        
        drT_dx = jacobian(self.net(x0)[1].T, x0).detach().numpy().reshape(3, 3)
        
        dr_dx = jacobian(self.net(x0)[1], x0).detach().numpy().reshape(3, 3)
        
        drt_dx_dr = drT_dx @ dr
        #j =  drt_dx_dr.reshape(3,)
        data.Lx = a2m(j.detach().numpy())
        
        diff2 = drT_dx @ dr_dx
        h = diff2.reshape(3, 3)
        data.Lxx = a2m(h)