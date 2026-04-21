from data import TranslationDataset, collate_fn, src_sp, tgt_sp
from model import Seq2SeqTransformer
from train import train_model
from predict import translate
from hyperparameters import *

from torch.utils.data import DataLoader

dataset = TranslationDataset()
loader = DataLoader(
    dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    collate_fn=collate_fn, 
    num_workers=NUM_WORKERS, 
    pin_memory=True
)

model = Seq2SeqTransformer(
    src_vocab=src_sp.vocab_size(),
    tgt_vocab=tgt_sp.vocab_size(),
    d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS, dropout=DROPOUT
)
print(f"词表对齐: src={src_sp.vocab_size()}, tgt={tgt_sp.vocab_size()}")

model = train_model(model, loader)

# 测试推理
print(translate(model, "bsbtjyrernx", src_sp, tgt_sp))