import torch
import Levenshtein
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

def create_padding_mask(seq: torch.Tensor, pad_idx: int) -> torch.Tensor:
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).type(torch.bool)
    return mask.unsqueeze(0).unsqueeze(0)

def create_masks(src: torch.Tensor, tgt: torch.Tensor, pad_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    src_mask = create_padding_mask(src, pad_idx)
    tgt_pad_mask = create_padding_mask(tgt, pad_idx)
    tgt_causal_mask = create_causal_mask(tgt.size(1), tgt.device)
    tgt_mask = tgt_pad_mask & tgt_causal_mask
    return src_mask, tgt_mask

def greedy_decode(model, src, src_mask, max_len, start_symbol, device, pad_idx=0):
    batch_size = src.size(0)
    tgt = torch.full((batch_size, 1), start_symbol, dtype=torch.long, device=device)
    enc_out = model.encode(src, src_mask)
    
    for _ in range(max_len - 1):
        tgt_pad_mask = create_padding_mask(tgt, pad_idx)
        tgt_causal_mask = create_causal_mask(tgt.size(1), device)
        tgt_mask = tgt_pad_mask & tgt_causal_mask
        
        dec_out = model.decode(tgt, enc_out, src_mask, tgt_mask)
        logits = model.generator(dec_out[:, -1, :])
        
        _, next_word = torch.max(logits, dim=-1)
        tgt = torch.cat([tgt, next_word.unsqueeze(1)], dim=1)
        
        # Stop early if all sequences in batch have generated EOS
        if (tgt == 2).any(dim=1).all(): # Assuming EOS_IDX = 2
            break
            
    return tgt

def calculate_metrics(pred_strings: list[str], tgt_strings: list[str], is_tokenized: bool = True):
    exact_matches = 0
    total_bits = 0
    correct_bits = 0
    total_levenshtein = 0
    total_bleu = 0
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1_total, rouge2_total, rougeL_total = 0, 0, 0
    smoothie = SmoothingFunction().method4
    
    for pred, tgt in zip(pred_strings, tgt_strings):
        # 1. Sequence Accuracy
        if pred == tgt:
            exact_matches += 1
            
        # 2. Levenshtein Distance
        total_levenshtein += Levenshtein.distance(pred, tgt)
        
        # 3. Bit-Level Accuracy (Convert chars to 8-bit binary representation)
        pred_bits = ''.join(format(ord(c), '08b') for c in pred)
        tgt_bits = ''.join(format(ord(c), '08b') for c in tgt)
        
        max_len = max(len(pred_bits), len(tgt_bits))
        min_len = min(len(pred_bits), len(tgt_bits))
        
        if max_len > 0:
            match_count = sum(1 for i in range(min_len) if pred_bits[i] == tgt_bits[i])
            correct_bits += match_count
            total_bits += max_len
            
        # 4. BLEU and ROUGE (Only if tokenized model C1-C4)
        if is_tokenized:
            ref_tokens = [tgt.split()]
            pred_tokens = pred.split()
            total_bleu += sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothie)
            
            scores = scorer.score(tgt, pred)
            rouge1_total += scores['rouge1'].fmeasure
            rouge2_total += scores['rouge2'].fmeasure
            rougeL_total += scores['rougeL'].fmeasure

    num_samples = len(pred_strings)
    metrics = {
        "seq_accuracy": (exact_matches / num_samples) * 100 if num_samples > 0 else 0,
        "bit_accuracy": (correct_bits / total_bits) * 100 if total_bits > 0 else 0,
        "avg_levenshtein": total_levenshtein / num_samples if num_samples > 0 else 0,
    }
    
    if is_tokenized:
        metrics.update({
            "bleu": total_bleu / num_samples if num_samples > 0 else 0,
            "rouge1": rouge1_total / num_samples if num_samples > 0 else 0,
            "rouge2": rouge2_total / num_samples if num_samples > 0 else 0,
            "rougeL": rougeL_total / num_samples if num_samples > 0 else 0
        })
        
    return metrics