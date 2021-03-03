import torch
import torch.nn as nn
import torch.nn.init as init


class L2Norm(nn.Module):
    """
    Layer learns to scale the l2 normalized features from conv4_3
    source:
    https://github.com/amdegroot/ssd.pytorch/blob/5b0b77faa955c1917b0c710d770739ba8fbff9b7/layers/modules/l2norm.py#L7 
    """ 
    def __init__(self,
                 in_channels,
                 scale):
        super(L2Norm, self).__init__()
        self.in_channels = in_channels
        self.gamma = scale
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.in_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight,self.gamma)

    def forward(self, x):
        # consider the l2norm along the axis 1
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        # normalize data on l2norm
        x = torch.div(x,norm)
        # multiply by weights
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out