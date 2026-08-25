import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import yaml
import argparse
from tokenizers import Tokenizer

from models.attention import MultiHeadAttention, GroupedQueryAttention
from models.norm import LayerNorm, RMSNorm
from models.positional import SinusoidalPositionalEncoding
from models.blt import LocalByteEncoder, LocalByteDecoder
from dataset import CipherDataset, collate_fn, PAD_IDX
from utils import create_masks

# --- MODEL ASSEMBLY ---
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
        if config.get('attn_type') == 'gqa':
            self.self_attn = GroupedQueryAttention(d_model, config['num_q_heads'], config['num_kv_heads'])
        else:
            self.self_attn = MultiHeadAttention(d_model, config['num_heads'])
            
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
        if config.get('attn_type') == 'gqa':
            self.self_attn = GroupedQueryAttention(d_model, config['num_q_heads'], config['num_kv_heads'])
            self.cross_attn = GroupedQueryAttention(d_model, config['num_q_heads'], config['num_kv_heads'])
        else:
            self.self_attn = MultiHeadAttention(d_model, config['num_heads'])
            self.cross_attn = MultiHeadAttention(d_model, config['num_heads'])
            
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
            self.src_embed = LocalByteEncoder(d_model, patch_size=config['patch_size'])
            self.tgt_embed = LocalByteEncoder(d_model, patch_size=config['patch_size'])
            self.generator = LocalByteDecoder(d_model, patch_size=config['patch_size'])
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


# --- TRAINING LOOP ---
def train_epoch(model, dataloader, optimizer, criterion, device):
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
        
        total_loss += loss.item()
    return total_loss / len(dataloader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    wandb.init(project="aNLP-Assignment-1", name=config['run_name'], config=config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}...")
    
    # Load the Tokenizer for C1
    if config['tokenization'] == 'subword':
        tokenizer = Tokenizer.from_file("tokenizer.json")
        vocab_size = tokenizer.get_vocab_size()
    else:
        tokenizer = None
        vocab_size = 260 # For BLT
        
    dataset = CipherDataset('dataset/brown_cipher.txt', 'dataset/brown_plain.txt', config, tokenizer=tokenizer) 
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
    
    model = Seq2SeqTransformer(config, vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    
    for epoch in range(config['epochs']):
        loss = train_epoch(model, dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{config['epochs']} | Loss: {loss:.4f}")
        wandb.log({"epoch": epoch, "train_loss": loss})

if __name__ == "__main__":
    main()
