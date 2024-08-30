import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    
    def forward(self, input):
        mean = input.mean(-1, keep_dim=True)
        var = input.var(-1, keep_dim=True)
        output = (input - mean)/torch.sqrt(var + self.eps)
        output = self.gamma * output + self.beta 
        return output
