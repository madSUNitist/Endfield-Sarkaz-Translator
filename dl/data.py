import torch
from torch.utils.data import Dataset
import sentencepiece as spm
from hyperparameters import *

src_sp = spm.SentencePieceProcessor()
src_sp.Load(model_file=SP_SRC_MODEL)
tgt_sp = spm.SentencePieceProcessor()
tgt_sp.Load(model_file=SP_TGT_MODEL)

class TranslationDataset(Dataset):
    def __init__(self, max_len=MAX_SEQ_LEN):
        self.src_lines = open(SRC_FILE, encoding="utf-8").readlines()
        self.tgt_lines = open(TGT_FILE, encoding="utf-8").readlines()
        self.max_len = max_len

    def __len__(self): return len(self.src_lines)

    def __getitem__(self, idx):
        # 密文走 src_sp，中文走 tgt_sp
        s = [SOS_ID] + src_sp.encode(self.src_lines[idx].strip())[:self.max_len-2] + [EOS_ID]
        t = [SOS_ID] + tgt_sp.encode(self.tgt_lines[idx].strip())[:self.max_len-2] + [EOS_ID]
        return torch.tensor(s, dtype=torch.long), torch.tensor(t, dtype=torch.long)

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_pad = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=PAD_ID)
    tgt_pad = torch.nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_ID)
    return src_pad, tgt_pad  # 仅返回2个值，匹配训练循环