import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import wandb
import argparse
import os
import time

from models.attention import MultiHeadAttention, GroupedQueryAttention
from models.norm import LayerNorm, RMSNorm
from models.positional import SinusoidalPositionalEncoding
from models.blt import LocalByteEncoder, LocalByteDecoder
from dataset import CipherDataset, CustomBPE, collate_fn, PAD_IDX, SOS_IDX, EOS_IDX, bits_to_byte_str
from utils import create_masks, greedy_decode, calculate_metrics

CONFIGS = {
    "C1": {
        "run_name": "C1_Base_Model",
        "d_model": 256, "num_heads": 8, "num_layers": 4, "d_ff": 1024, "dropout": 0.1,
        "max_seq_len": 64, "batch_size": 64, "learning_rate": 0.0005, "epochs": 100,
        "tokenization": "subword", "pos_type": "sinusoidal", "attn_type": "mha", "norm_type": "layernorm"
    },
    "C2": {
        "run_name": "C2_RoPE_Model",
        "d_model": 256, "num_heads": 8, "num_layers": 4, "d_ff": 1024, "dropout": 0.1,
        "max_seq_len": 64, "batch_size": 64, "learning_rate": 0.0005, "epochs": 100,
        "tokenization": "subword", "pos_type": "rope", "attn_type": "mha", "norm_type": "layernorm"
    },
    "C3": {
        "run_name": "C3_GQA_Model",
        "d_model": 256, "num_heads": 8, "num_q_heads": 8, "num_kv_heads": 2, "num_layers": 4, "d_ff": 1024, "dropout": 0.1,
        "max_seq_len": 64, "batch_size": 64, "learning_rate": 0.0005, "epochs": 100,
        "tokenization": "subword", "pos_type": "sinusoidal", "attn_type": "gqa", "norm_type": "layernorm"
    },
    "C4": {
        "run_name": "C4_RMSNorm_Model",
        "d_model": 256, "num_heads": 8, "num_layers": 4, "d_ff": 1024, "dropout": 0.1,
        "max_seq_len": 64, "batch_size": 64, "learning_rate": 0.0005, "epochs": 100,
        "tokenization": "subword", "pos_type": "sinusoidal", "attn_type": "mha", "norm_type": "rmsnorm"
    }
}

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        d_model = config['d_model']
        use_rope = config.get('pos_type') == 'rope'

        if config.get('attn_type') == 'gqa':
            self.self_attn = GroupedQueryAttention(d_model, config['num_q_heads'], config['num_kv_heads'], use_rope=use_rope)
        else:
            self.self_attn = MultiHeadAttention(d_model, config['num_heads'], use_rope=use_rope)

        NormClass = RMSNorm if config.get('norm_type') == 'rmsnorm' else LayerNorm
        self.norm1, self.norm2 = NormClass(d_model), NormClass(d_model)
        self.ffn = PositionwiseFeedForward(d_model, config['d_ff'], config['dropout'])
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.dropout(self.self_attn(q=self.norm1(x), k=self.norm1(x), v=self.norm1(x), mask=mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        d_model = config['d_model']
        use_rope = config.get('pos_type') == 'rope'

        if config.get('attn_type') == 'gqa':
            self.self_attn = GroupedQueryAttention(d_model, config['num_q_heads'], config['num_kv_heads'], use_rope=use_rope)
            self.cross_attn = GroupedQueryAttention(d_model, config['num_q_heads'], config['num_kv_heads'], use_rope=False)
        else:
            self.self_attn = MultiHeadAttention(d_model, config['num_heads'], use_rope=use_rope)
            self.cross_attn = MultiHeadAttention(d_model, config['num_heads'], use_rope=False)

        NormClass = RMSNorm if config.get('norm_type') == 'rmsnorm' else LayerNorm
        self.norm1, self.norm2, self.norm3 = NormClass(d_model), NormClass(d_model), NormClass(d_model)
        self.ffn = PositionwiseFeedForward(d_model, config['d_ff'], config['dropout'])
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, x: torch.Tensor, enc_out: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.self_attn(q=self.norm1(x), k=self.norm1(x), v=self.norm1(x), mask=tgt_mask))
        x = x + self.dropout(self.cross_attn(q=self.norm2(x), k=enc_out, v=enc_out, mask=src_mask))
        x = x + self.dropout(self.ffn(self.norm3(x)))
        return x

class Seq2SeqTransformer(nn.Module):
    def __init__(self, config: dict, vocab_size: int):
        super().__init__()
        self.is_blt = config.get('tokenization') == 'blt'
        d_model = config['d_model']

        if self.is_blt:
            self.src_embed = LocalByteEncoder(d_model, patch_size=config['patch_size'], vocab_size=vocab_size)
            self.tgt_embed = LocalByteEncoder(d_model, patch_size=config['patch_size'], vocab_size=vocab_size)
            self.generator = LocalByteDecoder(d_model, patch_size=config['patch_size'], vocab_size=vocab_size)
        else:
            self.src_embed = nn.Embedding(vocab_size, d_model)
            self.tgt_embed = nn.Embedding(vocab_size, d_model)
            self.generator = nn.Linear(d_model, vocab_size)

        self.pos_enc = None if config.get('pos_type') == 'rope' else SinusoidalPositionalEncoding(d_model)
        self.encoder_layers = nn.ModuleList([EncoderLayer(config) for _ in range(config['num_layers'])])
        self.decoder_layers = nn.ModuleList([DecoderLayer(config) for _ in range(config['num_layers'])])

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.pos_enc(self.src_embed(src)) if self.pos_enc else self.src_embed(src)
        for layer in self.encoder_layers: x = layer(x, src_mask)
        return x

    def decode(self, tgt: torch.Tensor, enc_out: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        x = self.pos_enc(self.tgt_embed(tgt)) if self.pos_enc else self.tgt_embed(tgt)
        for layer in self.decoder_layers: x = layer(x, enc_out, src_mask, tgt_mask)
        return x

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        return self.generator(self.decode(tgt, self.encode(src, src_mask), src_mask, tgt_mask))


def train_epoch(model, dataloader, optimizer, criterion, device, scheduler):
    model.train()
    total_loss = 0
    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        tgt_input, tgt_expected = tgt[:, :-1], tgt[:, 1:]

        src_mask, tgt_mask = create_masks(src, tgt_input, PAD_IDX)
        optimizer.zero_grad()

        logits = model(src, tgt_input, src_mask, tgt_mask)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_expected.reshape(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate_epoch(model, dataloader, tokenizer, device, config, epoch):
    model.eval()
    all_preds = []
    all_targets = []
    is_tokenized = config['tokenization'] == 'subword'

    print(f"\n--- Validation Decoded Sequences (Epoch {epoch+1}) ---")

    with torch.no_grad():
        for i, (src, tgt) in enumerate(dataloader):
            if i >= 1:  
                break

            src, tgt = src.to(device), tgt.to(device)
            src_mask, _ = create_masks(src, tgt, PAD_IDX)

            pred_ids = greedy_decode(model, src, src_mask, config['max_seq_len'] + 2, SOS_IDX, device)

            for batch_idx, (p, t) in enumerate(zip(pred_ids.cpu().tolist(), tgt.cpu().tolist())):
                if EOS_IDX in p:
                    p = p[:p.index(EOS_IDX)]
                if EOS_IDX in t:
                    t = t[:t.index(EOS_IDX)]
                    
                if is_tokenized:
                    pred_str = tokenizer.decode(p, skip_special_tokens=True)
                    tgt_str = tokenizer.decode(t, skip_special_tokens=True)
                else:
                    pred_str = "".join([chr(max(0, b - 4)) for b in p if b not in (PAD_IDX, SOS_IDX, EOS_IDX)])
                    tgt_str = "".join([chr(max(0, b - 4)) for b in t if b not in (PAD_IDX, SOS_IDX, EOS_IDX)])

                all_preds.append(pred_str)
                all_targets.append(tgt_str)

                if i == 0 and batch_idx < 3: 
                    print(f"[TARGET]: {tgt_str}")
                    print(f"[PRED]  : {pred_str}\n")

    return calculate_metrics(all_preds, all_targets, is_tokenized=is_tokenized)

def corpus_iterator(cipher_path, plain_path, chunk_size):
    """Chunks the dataset strictly within line boundaries to maintain XOR key phases."""
    texts = []
    is_cipher_flags = []
    
    with open(cipher_path, 'r', encoding='utf-8') as fc, open(plain_path, 'r', encoding='utf-8') as fp:
        for c_line, p_line in zip(fc, fp):
            c_line = c_line.strip()
            p_line = p_line.strip()
            c_bytes = bits_to_byte_str(c_line)
            
            for i in range(0, len(c_bytes), chunk_size):
                c_chunk = c_bytes[i:i + chunk_size]
                p_chunk = p_line[i:i + chunk_size]
                if len(c_chunk) == chunk_size:
                    texts.extend([c_chunk, p_chunk])
                    is_cipher_flags.extend([True, False])
                    
    return texts, is_cipher_flags

def get_or_build_tokenizer(cipher_path, plain_path, vocab_size, chunk_size, save_path):
    """Dynamically builds and saves a Custom BPE tokenizer using specified hyperparameters."""
    if os.path.exists(save_path):
        print(f"Loading existing Custom BPE tokenizer from {save_path}...")
        return CustomBPE.from_file(save_path)

    print(f"Building Custom BPE tokenizer (Vocab Size: {vocab_size}, Window: {chunk_size})...")
    tokenizer = CustomBPE(vocab_size=vocab_size)

    texts, is_cipher_flags = corpus_iterator(cipher_path, plain_path, chunk_size)
    tokenizer.train(texts, is_cipher_flags)
    tokenizer.save(save_path)
    return tokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, choices=['C1', 'C2', 'C3', 'C4'], default='C1')
    args = parser.parse_args()

    config = CONFIGS[args.config]

    wandb.init(project="aNLP-Assignment-1", name=config['run_name'], config=config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    print(f"Training Config {args.config} on {device} with {config['epochs']} epochs...")

    if config['tokenization'] == 'subword':
        # Safely name the tokenizer using the config to prevent parallel GPU runs from crashing
        tok_path = f"tokenizer_{config['run_name']}.json"
        tokenizer = get_or_build_tokenizer(
            'dataset/brown_cipher.txt', 
            'dataset/brown_plain.txt', 
            vocab_size=512, 
            chunk_size=config['max_seq_len'],
            save_path=tok_path
        )
        vocab_size = tokenizer.get_vocab_size()
    else:
        tokenizer = None
        vocab_size = 260

    full_dataset = CipherDataset('dataset/brown_cipher.txt', 'dataset/brown_plain.txt', config, tokenizer=tokenizer) 

    total_size = len(full_dataset)
    val_size = int(0.1 * total_size)
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)

    model = Seq2SeqTransformer(config, vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=config['learning_rate'], 
        steps_per_epoch=len(train_loader), 
        epochs=config['epochs']
    )

    for epoch in range(config['epochs']):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
            
        start_time = time.time()
        
        avg_train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scheduler)
        
        epoch_time = time.time() - start_time
        peak_memory_mb = 0
        if torch.cuda.is_available():
            peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

        metrics = evaluate_epoch(model, val_loader, tokenizer, device, config, epoch)

        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{config['epochs']} | Train Loss: {avg_train_loss:.4f} | LR: {current_lr:.6f}")
        print(f"Speed & Memory     -> Time: {epoch_time:.2f}s | Peak VRAM: {peak_memory_mb:.2f} MB")
        print(f"Validation Metrics -> Seq Acc: {metrics['seq_accuracy']:.2f}% | Bit Acc: {metrics['bit_accuracy']:.2f}% | Levenshtein: {metrics['avg_levenshtein']:.2f}")
        
        if config['tokenization'] == 'subword':
            print(f"Validation Scores  -> BLEU: {metrics['bleu']:.4f} | ROUGE-L: {metrics['rougeL']:.4f}")

        wandb.log({
            "epoch": epoch, 
            "train_loss": avg_train_loss,
            "learning_rate": current_lr,
            "epoch_time_seconds": epoch_time,
            "peak_gpu_memory_mb": peak_memory_mb,
            **metrics
        })

if __name__ == "__main__":
    main()