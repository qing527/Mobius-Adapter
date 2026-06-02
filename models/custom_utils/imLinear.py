import torch
import torch.nn as nn

class imLinear(nn.Module):
    
    def __init__(self, in_channels, out_channels, bias=True,zero_init=True):
        super(imLinear, self).__init__()
              
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=bias)
        if zero_init:
            nn.init.zeros_(self.lin.weight)
            if bias:
                nn.init.zeros_(self.lin.bias)

    def forward(self, x):

        return torch.permute(self.lin(torch.permute(x, (0, 2, 3, 1))), (0, 3, 1, 2))