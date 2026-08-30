import torch
from torch.utils.data import Dataset, DataLoader
import os

# Special byte/token constants
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

class CipherDataset(Dataset):

    def __init__(self, cipher_path: str, plain_path: str, config: dict, tokenizer=None):
        """
        config: 'tokenization' ('subword' or 'blt'), 'max_seq_len'
        """
        
        self.config = config
        self.is_blt = config.get('tokenization') == 'blt'
        self.tokenizer = tokenizer
        self.max_len = config['max_seq_len']
        
        with open(cipher_path, 'r', encoding='utf-8') as fc, open(plain_path, 'r', encoding='utf-8') as fp:
            self.cipher_lines = fc.readlines()
            self.plain_lines = fp.readlines()
            
        assert len(self.cipher_lines) == len(self.plain_lines)

    def __len__(self):

        return len(self.cipher_lines)

    def _encode_blt(self, text: str) -> list[int]:

        bytes_list = [b + 4 for b in text.strip().encode('utf-8')]
        return [SOS_IDX] + bytes_list[:self.max_len - 2] + [EOS_IDX]

    def __getitem__(self, idx: int):

        cipher_text = self.cipher_lines[idx].strip()
        plain_text = self.plain_lines[idx].strip()

        if self.is_blt:
            src_encoded = self._encode_blt(cipher_text)
            tgt_encoded = self._encode_blt(plain_text)
        else:
            src_encoded = self.tokenizer.encode(cipher_text).ids[:self.max_len]
            tgt_encoded = self.tokenizer.encode(plain_text).ids[:self.max_len]

        return torch.tensor(src_encoded, dtype=torch.long), torch.tensor(tgt_encoded, dtype=torch.long)

def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]):
    
    src_batch, tgt_batch = zip(*batch)
    
    src_padded = torch.nn.utils.rnn.pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=True)
    
    return src_padded, tgt_padded