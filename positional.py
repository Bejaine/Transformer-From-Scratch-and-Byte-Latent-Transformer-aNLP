import torch
import torch.nn as nn
import math

class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal Absolute Positional Encoding
    Shapes:
        pe: [max_seq_len, d_model] or [1, max_seq_len, d_model]
        x: [batch_size, seq_len, d_model]
    """

    def __init__(self, d_model: int, max_seq_len: int = 5000):

        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE)
    """

    def __init__(self, head_dim: int, max_seq_len: int = 5000):

        super().__init__()
        self.head_dim = head_dim
        
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len, dtype=torch.float)
        
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        
        # Concatenate to match head_dim: [max_seq_len, head_dim]
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos().unsqueeze(0).unsqueeze(0)) # [1, 1, max_seq_len, head_dim]
        self.register_buffer('sin_cached', emb.sin().unsqueeze(0).unsqueeze(0))

    def forward(self, q: torch.Tensor, k: torch.Tensor):

        # q, k shape: [batch_size, num_heads, seq_len, head_dim]
        seq_len = q.size(2)
        
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]
        
        q_rotated = (q * cos) + (self._rotate_half(q) * sin)
        k_rotated = (k * cos) + (self._rotate_half(k) * sin)
        
        return q_rotated, k_rotated

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:

        x1, x2 = x[..., :self.head_dim // 2], x[..., self.head_dim // 2:]
        return torch.cat((-x2, x1), dim=-1)
