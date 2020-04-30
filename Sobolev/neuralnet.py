import torch
import torch.nn as nn



import torch
import torch.nn as nn

class SQNet(nn.Module):
    def __init__(self,
                 hiddenUnits = 16):
        
        super(SQNet, self).__init__()
        self.nUnits = hiddenUnits
        
        # Structure
        self.fc1 = nn.Linear(3, self.nUnits)
        self.fc2 = nn.Linear(self.nUnits, 3)
        self.fc3 = nn.Linear(3, 3)
        self.fc4 = nn.Linear(3, 1)



        
        # Initialization protocol
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.xavier_normal_(self.fc4.weight)


      
        self.device = torch.device('cpu')
        self.to(self.device)

      
            
    
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        x = self.fc4(x)
        return x
               
        



    

    def jacobian(self, x):
        """
        The output of net.eval() can be more than one element
        """
        if x.dim() == 1:
            return torch.autograd.functional.jacobian(self.forward, x).squeeze()
        else:
            j = [torch.autograd.functional.jacobian(self.forward, x) for x in x]
            return torch.stack(j).squeeze()

        
        
        
    def hessian(self, x):
        """
        The output of net.eval() can be more than one element
        """
        if x.dim() == 1:
            return torch.autograd.functional.hessian(self.forward, x).squeeze()
        else:
            j = [torch.autograd.functional.hessian(self.forward, x) for x in x]
            return torch.stack(j).squeeze()

