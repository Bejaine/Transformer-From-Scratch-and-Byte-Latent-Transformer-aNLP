import torch
from torch.utils.data import Dataset, DataLoader
import os

# Special byte/token constants
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

def bits_to_byte_str(bit_string: str) -> str:
    """
    Pack a raw '0'/'1' bit string into a string with exactly ONE character per
    original byte (8 bits), using the Latin-1 mapping (codepoint == byte value,
    0-255) so the conversion is lossless and reversible.

    Without this, an 8-bit chunk like "01000001" is fed to BPE/BLT as an
    8-CHARACTER string over a 2-symbol alphabet ('0'/'1'). That is 8x longer
    than necessary and forces BPE to spend most of its merge budget just
    re-discovering "these 8 bits are one byte" instead of learning cipher-level
    patterns. This converts it to the single character chr(65) = 'A' instead,
    which is the actual byte the cipher produced.
    """
    n_bytes = len(bit_string) // 8
    raw = bytes(int(bit_string[i * 8:(i + 1) * 8], 2) for i in range(n_bytes))
    return raw.decode('latin-1')

class CipherDataset(Dataset):

    def __init__(self, cipher_path: str, plain_path: str, config: dict, tokenizer=None):
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

    def _encode_blt(self, text: str, is_cipher: bool) -> list[int]:
        # The cipher side is a raw '0'/'1' bit string, NOT text -- it must be
        # packed into actual bytes first. Calling .encode('utf-8') directly on
        # it (the old behavior) just gives you the ASCII codes of the
        # characters '0' (48) and '1' (49), i.e. only 2 distinct "byte" values
        # ever appear, and the sequence is 8x too long. The plaintext side is
        # already real text, so utf-8 encoding it directly is correct.
        if is_cipher:
            raw_bytes = bits_to_byte_str(text).encode('latin-1')
        else:
            raw_bytes = text.strip().encode('utf-8')
        bytes_list = [b + 4 for b in raw_bytes]
        return [SOS_IDX] + bytes_list[:self.max_len - 2] + [EOS_IDX]

    def __getitem__(self, idx: int):
        cipher_text = self.cipher_lines[idx].strip()
        plain_text = self.plain_lines[idx].strip()

        if self.is_blt:
            src_encoded = self._encode_blt(cipher_text, is_cipher=True)
            tgt_encoded = self._encode_blt(plain_text, is_cipher=False)
        else:
            # Pack the raw bits into actual bytes (1 char per 8 bits) BEFORE
            # tokenizing, so BPE operates on the real 256-symbol byte alphabet
            # instead of an 8x-longer sequence over a 2-symbol '0'/'1' alphabet.
            byte_cipher = bits_to_byte_str(cipher_text)
            
            src_ids = self.tokenizer.encode(byte_cipher).ids
            tgt_ids = self.tokenizer.encode(plain_text).ids
            
            src_encoded = [SOS_IDX] + src_ids[:self.max_len - 2] + [EOS_IDX]
            tgt_encoded = [SOS_IDX] + tgt_ids[:self.max_len - 2] + [EOS_IDX]

        return torch.tensor(src_encoded, dtype=torch.long), torch.tensor(tgt_encoded, dtype=torch.long)

def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]):
    src_batch, tgt_batch = zip(*batch)
    
    src_padded = torch.nn.utils.rnn.pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=True)
    
    return src_padded, tgt_padded