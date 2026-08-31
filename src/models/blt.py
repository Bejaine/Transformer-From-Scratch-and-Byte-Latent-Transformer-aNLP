import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalByteEncoder(nn.Module):
    """
    Local Encoder for the Byte Latent Transformer (BLT) architecture.
    Instead of using subword vocabularies (like BPE), this module takes raw bytes,
    embeds them, and groups them into 'patches' (latent representations).
    This significantly reduces the sequence length the global transformer has to process,
    saving computational overhead and peak GPU memory[cite: 4].
    """
    def __init__(self, d_model: int, patch_size: int = 4, vocab_size: int = 256):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        
        # 1. Byte-level Embedding: Maps raw bytes (0-255) + special tokens to vectors
        self.byte_embedding = nn.Embedding(vocab_size, d_model)
        
        # 2. Patcher: Projects a group of embedded bytes into a single latent patch.
        # This is the core of BLT: reducing length L to L / patch_size.
        self.patcher = nn.Linear(d_model * patch_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len]
        batch_size, seq_len = x.size()
        
        # We must ensure the sequence length is perfectly divisible by the patch_size.
        # If not, we pad the sequence with 0s (PAD_IDX) at the end.
        remainder = seq_len % self.patch_size
        if remainder != 0:
            pad_len = self.patch_size - remainder
            x = F.pad(x, (0, pad_len), value=0)
            seq_len += pad_len

        # Embed the raw bytes: [batch_size, seq_len, d_model]
        byte_embeds = self.byte_embedding(x)
        
        # Group the bytes into patches: [batch_size, num_patches, patch_size * d_model]
        num_patches = seq_len // self.patch_size
        grouped_bytes = byte_embeds.view(batch_size, num_patches, -1) 
        
        # Compress into latent space: [batch_size, num_patches, d_model]
        latent_patches = self.patcher(grouped_bytes)
        
        return latent_patches


class LocalByteDecoder(nn.Module):
    """
    Local Decoder for the Byte Latent Transformer (BLT) architecture.
    Takes the global transformer's latent patch predictions and unrolls them
    back into a sequence of raw byte logits.
    """
    def __init__(self, d_model: int, patch_size: int = 4, vocab_size: int = 256):
        super().__init__()
        self.patch_size = patch_size

        # 1. Unpatcher: Expands a single latent patch back into `patch_size` byte vectors.
        self.unpatcher = nn.Linear(d_model, d_model * patch_size)
        
        # 2. Byte Classifier: Maps the unrolled byte vectors to raw vocabulary logits.
        self.byte_classifier = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fallback to handle greedy decoding which may pass 2D tensors
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # x shape: [batch_size, num_patches, d_model]
        batch_size, num_patches, _ = x.size()
        
        # Expand latent patches: [batch_size, num_patches, patch_size * d_model]
        unrolled = self.unpatcher(x)
        
        # Reshape into a flat sequence of bytes: [batch_size, num_patches * patch_size, d_model]
        byte_sequence = unrolled.view(batch_size, num_patches * self.patch_size, -1)
        
        # Predict logits for each byte: [batch_size, unrolled_seq_len, vocab_size]
        logits = self.byte_classifier(byte_sequence)
        
        return logits