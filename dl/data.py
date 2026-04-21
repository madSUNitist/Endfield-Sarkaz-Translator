import torch
from torch.utils.data import Dataset, DataLoader

from dl.hyperparameters import SOS_ID, EOS_ID, PAD_ID

import sentencepiece as spm

src_sp = spm.SentencePieceProcessor()
src_sp.Load(model_file="zh2fic.model")
tgt_sp = spm.SentencePieceProcessor()
tgt_sp.Load(model_file="zh2fic_tgt.model")

class TranslationDataset(Dataset):
    def __init__(self, src_file, tgt_file, src_sp, tgt_sp, max_len=128):
        self.src_lines = open(src_file, encoding="utf-8").readlines()
        self.tgt_lines = open(tgt_file, encoding="utf-8").readlines()
        self.src_sp, self.tgt_sp = src_sp, tgt_sp
        self.max_len = max_len

    def __len__(self): return len(self.src_lines)

    def __getitem__(self, idx):
        src_ids = [SOS_ID] + self.src_sp.encode(self.src_lines[idx].strip())[:self.max_len] + [EOS_ID]
        tgt_ids = [SOS_ID] + self.tgt_sp.encode(self.tgt_lines[idx].strip())[:self.max_len] + [EOS_ID]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = torch.tensor(src_batch)
    tgt_batch = torch.tensor(tgt_batch)
    src_lens = [len(s) for s in src_batch]
    tgt_lens = [len(t) for t in tgt_batch]
    src_pad = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=PAD_ID)
    tgt_pad = torch.nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_ID)
    return src_pad, tgt_pad, torch.tensor(src_lens), torch.tensor(tgt_lens)