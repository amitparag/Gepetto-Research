import torch
import torch.nn as nn


class SQNet(nn.Module):
    def __init__(self, 
                 input_features,
                 output_features,
                 n_hiddenunits = 16):
        
        super(SQNet, self).__init__()
        
        # Structure
        self.fc1 = nn.Linear(input_features, n_hiddenunits)
        self.fc2 = nn.Linear(n_hiddenunits, 3)
        self.fc3 = nn.Linear(3, 3)
        self.fc4 = nn.Linear(3, output_features)

        
        # Initialization protocol
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)


      
        self.device = torch.device('cpu')
        self.to(self.device)

    def forward(self, x):
        
        x1 = torch.tanh(self.fc1(x)) 
        x2 = torch.tanh(self.fc2(x1)) 
        x3 = self.fc3(x2) 
        x4 = self.fc4(x3) ** 2

        
        return x4
    

    def jacobian(self, x):
        """
        The output of net.eval() can be more than one element
        """
        if x.shape[0] > 1:
            j = [torch.autograd.functional.jacobian(self.forward, x) for x in x]
            return torch.stack(j).squeeze()
        else:
            return torch.autograd.functional.jacobian(self.forward, x)
                
    def hessian(self, x):
        """
        The output of the net should always be a single element. Will throw and error if output of 
        self.forward(x) is more than 1 element.
        
        """
        if x.shape[0] > 1:
            h = [torch.autograd.functional.hessian(self.forward, x) for x in x]
            return torch.stack(h).squeeze()
        else:
            return torch.autograd.functional.hessian(self.forward, x)
            

