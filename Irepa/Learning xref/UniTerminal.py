import torch
import numpy as np
import crocoddyl

crocoddyl.switchToNumpyMatrix()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def jacobian(y, x, create_graph=False):                                                               
    jac = []                                                                                          
    flat_y = y.reshape(-1)                                                                            
    grad_y = torch.zeros_like(flat_y)                                                                 
    for i in range(len(flat_y)):                                                                      
        grad_y[i] = 1.                                                                                
        grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
        jac.append(grad_x.reshape(x.shape))                                                           
        grad_y[i] = 0.                                                                                
    return torch.stack(jac).reshape(y.shape + x.shape)                                                
                                                                                                      
def hessian(y, x):                                                                                    
    return jacobian(jacobian(y, x, create_graph=True), x)


def a2m(a):
    return np.matrix(a).T


def m2a(m):
    return np.array(m).squeeze()

class UnicycleTerminal(crocoddyl.ActionModelAbstract):
    def __init__(self, net):
        crocoddyl.ActionModelAbstract.__init__(self, crocoddyl.StateVector(3), 2, 5)
        self.net = net
        self.net.eval()

        
    def calc(self, data, x, u=None):
        if u is None:
            u = self.unone
        x0 = torch.as_tensor(m2a(x.T), device = device, dtype = torch.float32).resize_(1, 3)
        value = torch.sum(torch.abs(x0 - self.net(x0)) ** 2)
        data.cost = value.item()
        

    def calcDiff(self, data, x, u=None, recalc=True):               
        if u is None:
            u = self.unone
        if recalc:
            self.calc(data, x, u)
            
        x0 = torch.as_tensor(m2a(x.T), device = device, dtype = torch.float32).resize_(1, 3)
        x0.requires_grad_(True)
        
        def value_function(net, x):
            return torch.sum(torch.abs(x - net(x)) ** 2)
        
        j = jacobian(value_function(self.net, x0), x0)
        h = hessian(value_function(self.net, x0), x0)
        data.Lx = a2m(j.cpu().detach().numpy())
        data.Lxx = a2m(h.cpu().detach().numpy())
        