import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalByteEncoder(nn.Module):
    """
    Local Encoder for the Byte Latent Transformer (BLT) approach[cite: 5].
    Groups raw bytes into latent patches to reduce sequence length[cite: 5].
    """
    def __init__(self, d_model: int, patch_size: int = 4, vocab_size: int = 256, is_tgt: bool = False):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.is_tgt = is_tgt
        
        self.byte_embedding = nn.Embedding(vocab_size, d_model)
        
        # Core Patcher: Linear projection followed by GELU and LayerNorm 
        # to prevent variance explosion and catastrophic forgetting during high LR.
        self.patcher = nn.Linear(d_model * patch_size, d_model)
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(d_model)
        
        if self.is_tgt:
            self.sos_patch = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.size()
        
        remainder = seq_len % self.patch_size
        if remainder != 0:
            pad_len = self.patch_size - remainder
            x = F.pad(x, (0, pad_len), value=0)
            seq_len += pad_len

        byte_embeds = self.byte_embedding(x)
        
        num_patches = seq_len // self.patch_size
        grouped_bytes = byte_embeds.view(batch_size, num_patches, -1) 
        
        # Apply projection, non-linearity, and normalization
        latent_patches = self.patcher(grouped_bytes)
        latent_patches = self.norm(self.activation(latent_patches))
        
        if self.is_tgt:
            sos_expanded = self.sos_patch.expand(batch_size, -1, -1)
            latent_patches = torch.cat([sos_expanded, latent_patches[:, :-1, :]], dim=1)
            
        return latent_patches


class LocalByteDecoder(nn.Module):
    """
    Local Decoder for the Byte Latent Transformer (BLT) approach[cite: 5].
    Unrolls global transformer latent patches back into raw byte logits[cite: 5].
    """
    def __init__(self, d_model: int, patch_size: int = 4, vocab_size: int = 256):
        super().__init__()
        self.patch_size = patch_size

        # Unpatcher with activation to stabilize the unrolled representations
        self.unpatcher = nn.Linear(d_model, d_model * patch_size)
        self.activation = nn.GELU()
        
        self.byte_classifier = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)

        batch_size, num_patches, _ = x.size()
        
        # Expand latent patches and apply non-linearity
        unrolled = self.unpatcher(x)
        unrolled = self.activation(unrolled)
        
        byte_sequence = unrolled.view(batch_size, num_patches * self.patch_size, -1)
        logits = self.byte_classifier(byte_sequence)
        
        return logits