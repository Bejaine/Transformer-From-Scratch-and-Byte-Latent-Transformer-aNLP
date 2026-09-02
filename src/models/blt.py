import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention import MultiHeadAttention

class LocalByteEncoder(nn.Module):
    """
    Local Encoder for the Byte Latent Transformer (BLT) approach[cite: 5].
    Matches the architectural diagram: Byte-Level Transformer -> Cross-Attention Pooling.
    """
    def __init__(self, d_model: int, patch_size: int = 4, vocab_size: int = 256, is_tgt: bool = False):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.is_tgt = is_tgt
        
        self.byte_embedding = nn.Embedding(vocab_size, d_model)
        
        # FIX: Intra-Patch Positional Encoding. 
        # This prevents the MultiHeadAttention from scrambling the byte order!
        self.intra_patch_pos = nn.Parameter(torch.randn(1, patch_size, d_model))
        
        # 1. Byte-Level Small Transformer
        self.local_self_attn = MultiHeadAttention(d_model, num_heads=4, use_rope=False)
        self.norm1 = nn.LayerNorm(d_model)
        
        # 2. Encoder Patch Cross Attention
        self.patch_query = nn.Parameter(torch.randn(1, 1, d_model))
        self.patch_cross_attn = MultiHeadAttention(d_model, num_heads=4, use_rope=False)
        self.norm2 = nn.LayerNorm(d_model)
        
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
        grouped_bytes = byte_embeds.view(batch_size * num_patches, self.patch_size, -1) 
        
        # FIX: Inject positional information before the attention layer
        grouped_bytes = grouped_bytes + self.intra_patch_pos
        
        # 1. Local Byte-Level Context
        local_out = self.local_self_attn(q=self.norm1(grouped_bytes), k=self.norm1(grouped_bytes), v=self.norm1(grouped_bytes))
        grouped_bytes = grouped_bytes + local_out
        
        # 2. Cross-Attention Pooling
        queries = self.patch_query.expand(batch_size * num_patches, -1, -1)
        patch_repr = self.patch_cross_attn(q=self.norm2(queries), k=self.norm2(grouped_bytes), v=self.norm2(grouped_bytes))
        
        latent_patches = patch_repr.view(batch_size, num_patches, self.d_model)
        
        if self.is_tgt:
            sos_expanded = self.sos_patch.expand(batch_size, -1, -1)
            latent_patches = torch.cat([sos_expanded, latent_patches[:, :-1, :]], dim=1)
            
        return latent_patches


class LocalByteDecoder(nn.Module):
    """
    Local Decoder for the Byte Latent Transformer (BLT) approach[cite: 5].
    Matches the architectural diagram: Cross-Attention Unpatching -> Byte-Level Transformer.
    """
    def __init__(self, d_model: int, patch_size: int = 4, vocab_size: int = 256):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model

        # 1. Decoder Patch Cross Attention
        # These learned queries inherently act as positional encodings for the unrolling!
        self.unpatch_queries = nn.Parameter(torch.randn(1, patch_size, d_model))
        self.unpatch_cross_attn = MultiHeadAttention(d_model, num_heads=4, use_rope=False)
        self.norm1 = nn.LayerNorm(d_model)
        
        # 2. Small Byte-Level Transformer
        self.local_self_attn = MultiHeadAttention(d_model, num_heads=4, use_rope=False)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.byte_classifier = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)

        batch_size, num_patches, _ = x.size()
        
        global_patches = x.view(batch_size * num_patches, 1, self.d_model)
        queries = self.unpatch_queries.expand(batch_size * num_patches, -1, -1)
        
        # 1. Cross-Attention Expansion
        unrolled = self.unpatch_cross_attn(q=self.norm1(queries), k=self.norm1(global_patches), v=self.norm1(global_patches))
        unrolled = queries + unrolled
        
        # 2. Local Byte-Level Smoothing 
        causal_mask = torch.tril(torch.ones((self.patch_size, self.patch_size), device=x.device)).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        
        local_out = self.local_self_attn(q=self.norm2(unrolled), k=self.norm2(unrolled), v=self.norm2(unrolled), mask=causal_mask)
        unrolled = unrolled + local_out
        
        byte_sequence = unrolled.view(batch_size, num_patches * self.patch_size, self.d_model)
        logits = self.byte_classifier(byte_sequence)
        
        return logits