import torch
import torch.nn as nn

class LocalByteEncoder(nn.Module):

    def __init__(self, d_model: int, patch_size: int = 4, vocab_size: int = 256):

        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        
        self.byte_embedding = nn.Embedding(vocab_size, d_model)
        
        self.patcher = nn.Linear(d_model * patch_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # x shape: [batch_size, seq_len]
        batch_size, seq_len = x.size()
        
        remainder = seq_len % self.patch_size
        if remainder != 0:
            pad_len = self.patch_size - remainder
            x = nn.functional.pad(x, (0, pad_len), value=0)
            seq_len += pad_len

        # Embed bytes: [batch_size, seq_len, d_model]
        byte_embeds = self.byte_embedding(x)
        
        # Reshape to group bytes: [batch_size, num_patches, patch_size, d_model]
        num_patches = seq_len // self.patch_size
        grouped_bytes = byte_embeds.view(batch_size, num_patches, -1) 
        
        # Project to latent space: [batch_size, num_patches, d_model]
        latent_patches = self.patcher(grouped_bytes)
        
        return latent_patches


class LocalByteDecoder(nn.Module):

    def __init__(self, d_model: int, patch_size: int = 4, vocab_size: int = 256):

        super().__init__()
        self.patch_size = patch_size

        self.unpatcher = nn.Linear(d_model, d_model * patch_size)
        
        self.byte_classifier = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # x shape: [batch_size, num_patches, d_model]
        batch_size, num_patches, _ = x.size()
        
        # Unroll: [batch_size, num_patches, patch_size * d_model]
        unrolled = self.unpatcher(x)
        
        # Reshape to sequence of bytes: [batch_size, num_patches * patch_size, d_model]
        byte_sequence = unrolled.view(batch_size, num_patches * self.patch_size, -1)
        
        # Predict byte logits: [batch_size, seq_len, 256]
        logits = self.byte_classifier(byte_sequence)
        
        return logits