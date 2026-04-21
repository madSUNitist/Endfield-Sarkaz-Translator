import torch.nn as nn
import torch
import math
from hyperparameters import *

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=SP_MAX_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])

class Seq2SeqTransformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead, num_encoder_layers=num_layers,
            num_decoder_layers=num_layers, dim_feedforward=DIM_FEEDFORWARD, dropout=dropout, batch_first=True
        )
        self.out = nn.Linear(d_model, tgt_vocab)
        self.d_model = d_model

    def forward(self, src, tgt, src_pad_mask=None, tgt_pad_mask=None, tgt_mask=None):
        src = self.pos_enc(self.src_emb(src) * math.sqrt(self.d_model))
        tgt = self.pos_enc(self.tgt_emb(tgt) * math.sqrt(self.d_model))
        memory = self.transformer.encoder(src, src_key_padding_mask=src_pad_mask)
        out = self.transformer.decoder(tgt, memory, tgt_mask=tgt_mask,
                                       tgt_key_padding_mask=tgt_pad_mask, memory_key_padding_mask=src_pad_mask)
        return self.out(out)