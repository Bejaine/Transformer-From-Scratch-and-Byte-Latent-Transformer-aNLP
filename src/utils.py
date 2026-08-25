import torch
import Levenshtein
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

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

def greedy_decode(model, src, src_mask, max_len, start_symbol, device, pad_idx=0):
    """
    Autoregressively generates a sequence using the trained model.
    """
    batch_size = src.size(0)
    # Initialize the target tensor with just the <SOS> token
    tgt = torch.full((batch_size, 1), start_symbol, dtype=torch.long, device=device)
    
    # Pre-compute encoder output so we don't recalculate it every step
    enc_out = model.encode(src, src_mask)
    
    for _ in range(max_len - 1):
        tgt_pad_mask = create_padding_mask(tgt, pad_idx)
        tgt_causal_mask = create_causal_mask(tgt.size(1), device)
        tgt_mask = tgt_pad_mask & tgt_causal_mask
        
        dec_out = model.decode(tgt, enc_out, src_mask, tgt_mask)
        logits = model.generator(dec_out[:, -1, :]) # Get predictions for the last token
        
        _, next_word = torch.max(logits, dim=-1)
        tgt = torch.cat([tgt, next_word.unsqueeze(1)], dim=1)
        
    return tgt

def calculate_metrics(pred_strings: list[str], tgt_strings: list[str]):
    """
    Calculates Sequence Accuracy, Bit/Character-Level Accuracy, Levenshtein, and BLEU.
    """
    exact_matches = 0
    total_chars = 0
    correct_chars = 0
    total_levenshtein = 0
    total_bleu = 0
    
    smoothie = SmoothingFunction().method4
    
    for pred, tgt in zip(pred_strings, tgt_strings):
        # 1. Sequence Accuracy
        if pred == tgt:
            exact_matches += 1
            
        # 2. Levenshtein Distance
        total_levenshtein += Levenshtein.distance(pred, tgt)
        
        # 3. Character/Bit-Level Accuracy
        min_len = min(len(pred), len(tgt))
        match_count = sum(1 for i in range(min_len) if pred[i] == tgt[i])
        correct_chars += match_count
        total_chars += max(len(pred), len(tgt))
        
        # 4. BLEU Score
        ref_tokens = [tgt.split()]
        pred_tokens = pred.split()
        total_bleu += sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothie)

    num_samples = len(pred_strings)
    return {
        "seq_accuracy": (exact_matches / num_samples) * 100,
        "char_accuracy": (correct_chars / total_chars) * 100 if total_chars > 0 else 0,
        "avg_levenshtein": total_levenshtein / num_samples,
        "avg_bleu": total_bleu / num_samples
    }
