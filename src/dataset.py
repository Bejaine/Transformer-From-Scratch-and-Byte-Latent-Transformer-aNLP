import torch
from torch.utils.data import Dataset
import os
import json
from collections import defaultdict

# Special byte/token constants
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

def bits_to_byte_str(bit_string: str) -> str:
    n_bytes = len(bit_string) // 8
    raw = bytes(int(bit_string[i * 8:(i + 1) * 8], 2) for i in range(n_bytes))
    return raw.decode('latin-1')

class CustomBPE:
    """
    BPE Tokenizer implemented from scratch to comply with Assignment 1 rules.
    Maps Base Bytes (0-255) to vocab IDs (4-259), reserving 0-3 for special tokens.
    """
    def __init__(self, vocab_size=512):
        self.vocab_size = vocab_size
        self.merges = {} # Maps "id1,id2" string to new_id

    def get_vocab_size(self):
        return 4 + 256 + len(self.merges)

    def train(self, texts: list[str], is_cipher_flags: list[bool]):
        sequences = []
        for text, is_cipher in zip(texts, is_cipher_flags):
            raw_bytes = text.encode('latin-1') if is_cipher else text.encode('utf-8')
            sequences.append([b + 4 for b in raw_bytes])

        current_vocab_size = 260
        target_merges = self.vocab_size - current_vocab_size
        
        print(f"Training BPE from scratch: learning {target_merges} merges...")
        
        for i in range(target_merges):
            counts = defaultdict(int)
            for seq in sequences:
                for j in range(len(seq) - 1):
                    counts[(seq[j], seq[j+1])] += 1
                    
            if not counts:
                break
                
            best_pair = max(counts, key=counts.get)
            new_id = current_vocab_size
            self.merges[f"{best_pair[0]},{best_pair[1]}"] = new_id
            
            new_sequences = []
            for seq in sequences:
                new_seq = []
                j = 0
                while j < len(seq):
                    if j < len(seq) - 1 and seq[j] == best_pair[0] and seq[j+1] == best_pair[1]:
                        new_seq.append(new_id)
                        j += 2
                    else:
                        new_seq.append(seq[j])
                        j += 1
                new_sequences.append(new_seq)
                
            sequences = new_sequences
            current_vocab_size += 1
            
            if (i + 1) % 50 == 0 or (i + 1) == target_merges:
                print(f"Learned {i+1}/{target_merges} merges...")

    def encode(self, text: str, is_cipher: bool = False) -> list[int]:
        raw_bytes = text.encode('latin-1') if is_cipher else text.encode('utf-8')
        seq = [b + 4 for b in raw_bytes]
        
        for pair_str, new_id in self.merges.items():
            p0, p1 = map(int, pair_str.split(','))
            new_seq = []
            j = 0
            while j < len(seq):
                if j < len(seq) - 1 and seq[j] == p0 and seq[j+1] == p1:
                    new_seq.append(new_id)
                    j += 2
                else:
                    new_seq.append(seq[j])
                    j += 1
            seq = new_seq
        return seq

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        expand_map = {v: tuple(map(int, k.split(','))) for k, v in self.merges.items()}
        
        def expand(token_id):
            if token_id < 4:
                return [] if skip_special_tokens else [token_id] 
            elif token_id < 260:
                return [token_id - 4]
            else:
                left, right = expand_map[token_id]
                return expand(left) + expand(right)
                
        raw_bytes = []
        for tid in ids:
            if skip_special_tokens and tid < 4:
                continue
            raw_bytes.extend(expand(tid))
            
        b = bytes(raw_bytes)
        try:
            return b.decode('utf-8')
        except UnicodeDecodeError:
            return b.decode('latin-1', errors='replace')

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump({"merges": self.merges, "vocab_size": self.vocab_size}, f)

    @classmethod
    def from_file(cls, path: str):
        with open(path, 'r') as f:
            data = json.load(f)
        obj = cls(vocab_size=data.get("vocab_size", 512))
        obj.merges = data.get("merges", {})
        return obj

class CipherDataset(Dataset):
    def __init__(self, cipher_path: str, plain_path: str, config: dict, tokenizer=None):
        self.config = config
        self.is_blt = config.get('tokenization') == 'blt'
        self.tokenizer = tokenizer
        self.chunk_size = config['max_seq_len'] 

        with open(cipher_path, 'r', encoding='utf-8') as fc, open(plain_path, 'r', encoding='utf-8') as fp:
            cipher_lines = [line.strip() for line in fc.readlines()]
            plain_lines = [line.strip() for line in fp.readlines()]

        assert len(cipher_lines) == len(plain_lines)
        self.data = []
        
        # Bug 2 Fix: Chunk WITHIN lines to guarantee the XOR key phase (i % 8 == 0) resets perfectly
        for c_line, p_line in zip(cipher_lines, plain_lines):
            c_bytes = bits_to_byte_str(c_line)
            
            for i in range(0, len(c_bytes), self.chunk_size):
                c_chunk = c_bytes[i : i + self.chunk_size]
                p_chunk = p_line[i : i + self.chunk_size]

                # if len(c_chunk) < self.chunk_size:
                #    continue

                if self.is_blt:
                    c_encoded_bytes = [b + 4 for b in c_chunk.encode('latin-1')]
                    p_encoded_bytes = [b + 4 for b in p_chunk.encode('utf-8')]
                    src_encoded = [SOS_IDX] + c_encoded_bytes + [EOS_IDX]
                    tgt_encoded = [SOS_IDX] + p_encoded_bytes + [EOS_IDX]
                else:
                    src_ids = self.tokenizer.encode(c_chunk, is_cipher=True)
                    tgt_ids = self.tokenizer.encode(p_chunk, is_cipher=False)

                    src_encoded = [SOS_IDX] + src_ids + [EOS_IDX]
                    tgt_encoded = [SOS_IDX] + tgt_ids + [EOS_IDX]

                self.data.append((
                    torch.tensor(src_encoded, dtype=torch.long),
                    torch.tensor(tgt_encoded, dtype=torch.long)
                ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]

def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]):
    src_batch, tgt_batch = zip(*batch)
    src_padded = torch.nn.utils.rnn.pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=True)
    return src_padded, tgt_padded