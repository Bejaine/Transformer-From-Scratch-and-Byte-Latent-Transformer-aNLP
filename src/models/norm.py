import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    """
    Standard Layer Normalization
    Shapes:
        x: [batch_size, seq_len, d_model]
    """

    def __init__(self, d_model: int, eps: float = 1e-5):

        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization
    Shapes:
        x: [batch_size, seq_len, d_model]
    """
    
    def __init__(self, d_model: int, eps: float = 1e-5):

        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
       
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        
        x_norm = x / rms
        return self.gamma * x_norm