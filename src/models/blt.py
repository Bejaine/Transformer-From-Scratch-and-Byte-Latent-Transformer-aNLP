import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalByteEncoder(nn.Module):
    """
    Local Encoder for the Byte Latent Transformer (BLT) approach[cite: 4].
    Instead of using subword vocabularies, we feed raw bytes into a local encoder 
    to create patch representations[cite: 4]. This reduces sequence length, lowering 
    computational overhead and peak GPU memory during training[cite: 4].
    """
    def __init__(self, d_model: int, patch_size: int = 4, vocab_size: int = 256, is_tgt: bool = False):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.is_tgt = is_tgt
        
        # Maps raw bytes (0-255) + special tokens to vectors
        self.byte_embedding = nn.Embedding(vocab_size, d_model)
        
        # Projects a group of embedded bytes into a single latent patch
        self.patcher = nn.Linear(d_model * patch_size, d_model)
        
        if self.is_tgt:
            # A learnable generic "Start of Sequence" patch for the target sequence
            self.sos_patch = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.size()
        
        # Ensure sequence length is divisible by patch_size by padding
        remainder = seq_len % self.patch_size
        if remainder != 0:
            pad_len = self.patch_size - remainder
            x = F.pad(x, (0, pad_len), value=0)
            seq_len += pad_len

        byte_embeds = self.byte_embedding(x)
        
        # Group bytes into patches
        num_patches = seq_len // self.patch_size
        grouped_bytes = byte_embeds.view(batch_size, num_patches, -1) 
        
        latent_patches = self.patcher(grouped_bytes)
        
        if self.is_tgt:
            # Shift patches right by 1 to enforce patch-level causality.
            # This ensures Patch N is forced to predict Patch N+1 using the global transformer,
            # preventing the model from trivial 1-to-1 byte copying.
            sos_expanded = self.sos_patch.expand(batch_size, -1, -1)
            latent_patches = torch.cat([sos_expanded, latent_patches[:, :-1, :]], dim=1)
            
        return latent_patches


class LocalByteDecoder(nn.Module):
    """
    Local Decoder for the Byte Latent Transformer (BLT) approach[cite: 4].
    After the global transformer processes the latent patches, we decode them 
    using a local byte-level decoder[cite: 4] to predict the raw output sequence.
    """
    def __init__(self, d_model: int, patch_size: int = 4, vocab_size: int = 256):
        super().__init__()
        self.patch_size = patch_size

        # Expands a single latent patch back into `patch_size` byte vectors
        self.unpatcher = nn.Linear(d_model, d_model * patch_size)
        
        # Maps the unrolled byte vectors to raw vocabulary logits
        self.byte_classifier = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fallback to handle greedy decoding which may pass 2D tensors
        if x.dim() == 2:
            x = x.unsqueeze(1)

        batch_size, num_patches, _ = x.size()
        
        # Expand latent patches
        unrolled = self.unpatcher(x)
        
        # Reshape into a flat sequence of bytes
        byte_sequence = unrolled.view(batch_size, num_patches * self.patch_size, -1)
        
        # Predict logits for each byte
        logits = self.byte_classifier(byte_sequence)
        
        return logits