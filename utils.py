import torch

def create_padding_mask(seq: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """
    Shape:
        [batch_size, 1, 1, seq_len]
    """
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    return mask

def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Shape:
        [1, 1, seq_len, seq_len]
    """
    mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).type(torch.bool)
    return mask.unsqueeze(0).unsqueeze(0)

def create_masks(src: torch.Tensor, tgt: torch.Tensor, pad_idx: int) -> tuple[torch.Tensor, torch.Tensor]:

    src_mask = create_padding_mask(src, pad_idx)
    
    tgt_pad_mask = create_padding_mask(tgt, pad_idx)
    tgt_causal_mask = create_causal_mask(tgt.size(1), tgt.device)
    tgt_mask = tgt_pad_mask & tgt_causal_mask
    
    return src_mask, tgt_mask
