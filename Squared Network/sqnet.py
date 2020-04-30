import torch
import torch.nn as nn

class SQNet(nn.Module):
    def __init__(self, 
                 in_features,
                 out_features,
                 units = 16):        
        super(SQNet, self).__init__()
        
        self.in_features  = in_features
        self.out_features = out_features
        self.units =  units
        
        # Structure
        self.fc1 = nn.Linear(self.in_features, self.units)
        self.fc2 = nn.Linear(units, 3)
        self.fc3 = nn.Linear(3, 3)
        self.ln4 = nn.Linear(1, self.out_features)

        
        # Weight Initialization protocol
        nn.init.kaiming_uniform_(self.fc1.weight)
        nn.init.kaiming_uniform_(self.fc2.weight)
        nn.init.kaiming_uniform_(self.fc3.weight)
        nn.init.kaiming_uniform_(self.ln4.weight)

        # Bias Initialization protocol
        self.fc1.bias.data.fill_(0)
        self.fc2.bias.data.fill_(0)
        self.fc3.bias.data.fill_(0)
        self.ln4.bias.data.fill_(0)
        
    def predict(self, x):
        
        x = torch.tanh(self.fc1(x)) 
        x = torch.tanh(self.fc2(x)) 
        x = self.fc3(x) 
        return self.ln4.weight.item() * sum(x ** 2) + self.ln4.bias.item()
        
        #return out
    
    def predict_batch(self, x):
        prediction = [self.predict(x) for x in x]
        return torch.stack(prediction).squeeze().reshape(x.shape[0], 1)
    
    
    def jacobian(self, x):
        j = torch.autograd.functional.jacobian(self.predict, x).squeeze()
        return j
    
    def hessian(self, x):
        h = torch.autograd.functional.hessian(self.predict, x).squeeze()
        return h
    def forward(self, x):
        x = torch.tanh(self.fc1(x)) 
        x = torch.tanh(self.fc2(x)) 
        x = self.fc3(x) 
        if x.dim() > 1:
            out =  self.ln4(torch.stack([torch.sum(x ** 2) for x in x]).reshape(x.shape[0], 1))
        else:
            out = self.ln4(torch.tensor([torch.sum(x ** 2)]).reshape(1, 1))
        return out