import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as tcp
from .mobius_conv import MobiusConv
from .custom_utils.FRDirichlet import FRDirichlet
from .custom_utils.imLinear import imLinear
################################################
###     Mobius Adapter Block      ###
################################################

# adapter
class MCResNetBlockAdapter(torch.nn.Module):
    
    def __init__(self, in_channels, out_channels, B, D1=1, D2=1, M=2, Q=30, mid_channels=None, checkpoint=False):
        super(MCResNetBlockAdapter, self).__init__()

        iC1 = in_channels
        oC2 = out_channels
        
        if (mid_channels is None):
            oC1 = out_channels
            iC2 = out_channels
        else:
            oC1 = mid_channels
            iC2 = mid_channels
        
        # Convolution blocks
        self.conv1 = MobiusConv(iC1, oC1, B, D1, D2, M, Q)
        self.conv2 = MobiusConv(iC2, oC2, B, D1, D2, M, Q)

        # Normalization blocks
        self.FR1 = FRDirichlet(B, oC1)
        self.FR2 = FRDirichlet(B, oC2)
        
        # Residual connection
        if (in_channels == out_channels):
            self.res = torch.nn.Identity()
        else:
            self.res = imLinear(in_channels, out_channels, bias=False)
        self.res1 = imLinear(in_channels, oC1, bias=False)
        self.res2 = imLinear(oC1,oC2)
        if checkpoint:
            self.wrapper = tcp.checkpoint
        else:
            self.wrapper = lambda f, x: f(x)
    
    def _forward(self, x):
        # Mobius Convolution        
        x_conv = self.conv1(x)

        # Normalization + Nonlinearity
        x_conv = self.FR1(x_conv,self.res1(x))
        
        # Mobius Convolution 
        x_conv1 = self.conv2(x_conv)
        
        if x_conv1.shape != x_conv.shape:
            x2 = self.res2(x_conv)
        else:
            x2 = x_conv
        x_res2 = x2 + x_conv1

        # Normalization + Nonlinearity w/ residiual connection
        xOut = self.FR2(x_res2, self.res(x))
        
        return xOut
    
    def forward(self, x):

        return self.wrapper(self._forward, x)

class MCDownAdapter(nn.Module):

    def __init__(self, in_channels, out_channels, B, D1=1, D2=1, M=2, Q=30, checkpoint=True):
        super(MCDownAdapter, self).__init__()

        assert B % 2 == 0
        self.pool = torch.nn.AdaptiveMaxPool2d( (B, B) )
        self.conv = MCResNetBlockAdapter(in_channels, out_channels, B//2, D1=D1, D2=D2, M=M, Q=Q, checkpoint=checkpoint)


    def forward(self, x):
        x = self.pool(x)

        return self.conv(x)