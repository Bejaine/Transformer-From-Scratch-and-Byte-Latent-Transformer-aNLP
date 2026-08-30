import torch
from torch.utils.data import Dataset, DataLoader
import os

# Special byte/token constants
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

def bits_to_byte_str(bit_string: str) -> str:
    n_bytes = len(bit_string) // 8
    raw = bytes(int(bit_string[i * 8:(i + 1) * 8], 2) for i in range(n_bytes))
    return raw.decode('latin-1')

class CipherDataset(Dataset):
    def __init__(self, cipher_path: str, plain_path: str, config: dict, tokenizer=None):
        self.config = config
        self.is_blt = config.get('tokenization') == 'blt'
        self.tokenizer = tokenizer
        
        # 1. Sequence length is now strictly our 64-byte window
        self.chunk_size = config['max_seq_len'] 

        with open(cipher_path, 'r', encoding='utf-8') as fc, open(plain_path, 'r', encoding='utf-8') as fp:
            # 2. Concatenate the entire corpus to prevent phase-shifting the XOR key
            full_cipher_bits = "".join([line.strip() for line in fc.readlines()])
            full_plain = "".join([line.strip() for line in fp.readlines()])

        # 3. Combine 8 bits to 1 byte for the entire corpus at once
        full_cipher_bytes = bits_to_byte_str(full_cipher_bits)

        self.data = []
        
        # 4. The "Hashmap" Pre-encoding Loop
        for i in range(0, len(full_cipher_bytes), self.chunk_size):
            c_chunk = full_cipher_bytes[i : i + self.chunk_size]
            p_chunk = full_plain[i : i + self.chunk_size]

            # Prune remaining bytes that don't form a complete 64-byte window
            if len(c_chunk) < self.chunk_size:
                continue

            if self.is_blt:
                c_bytes = [b + 4 for b in c_chunk.encode('latin-1')]
                p_bytes = [b + 4 for b in p_chunk.encode('utf-8')]
                src_encoded = [SOS_IDX] + c_bytes + [EOS_IDX]
                tgt_encoded = [SOS_IDX] + p_bytes + [EOS_IDX]
            else:
                # Tokenize the 64-byte pruned chunk
                src_ids = self.tokenizer.encode(c_chunk).ids
                tgt_ids = self.tokenizer.encode(p_chunk).ids

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