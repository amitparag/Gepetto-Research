import torch 
import torch.nn as nn

class FNet(nn.Module):
    def __init__(self, 
                 in_features,
                 out_features,
                 units = 16):        
        super(FNet, self).__init__()
        
        self.in_features  = in_features
        self.out_features = out_features
        self.units =  units
        
        # Structure
        self.fc1 = nn.Linear(self.in_features, self.units)
        self.fc2 = nn.Linear(units, 3)
        self.fc3 = nn.Linear(3, self.out_features)

        
        # Weight Initialization protocol
        nn.init.kaiming_uniform_(self.fc1.weight)
        nn.init.kaiming_uniform_(self.fc2.weight)
        nn.init.kaiming_uniform_(self.fc3.weight)
        
        # Bias Initialization protocol
        self.fc1.bias.data.fill_(0)
        self.fc2.bias.data.fill_(0)
        self.fc3.bias.data.fill_(0)


    def forward(self, x):
        
        x = torch.tanh(self.fc1(x)) 
        x = torch.tanh(self.fc2(x)) 
        x = self.fc3(x) 
       
        return x

    
    