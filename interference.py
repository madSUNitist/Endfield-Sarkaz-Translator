import torch.nn as nn
import torch

import sentencepiece as spm

import math


PAD_ID, SOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3

# self & dataset paths
SP_SRC_MODEL = "models/sp_src.model"
SP_TGT_MODEL = "models/sp_zh.model"
MODEL_PATH = "models/checkpoint.pt"

# self
D_MODEL = 256
NHEAD = 4
NUM_LAYERS = 4
DIM_FEEDFORWARD = 1024
DROPOUT = 0.1
SP_MAX_LEN = 5000


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
        out = self.transformer.decoder(
            tgt, memory, 
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask, 
            memory_key_padding_mask=src_pad_mask
        )
        return self.out(out)

    def translate(self, cipher_text, src_sp, tgt_sp, max_len=64):
        self.eval()
        device = next(self.parameters()).device
        src_ids = torch.tensor([SOS_ID] + src_sp.encode(cipher_text) + [EOS_ID], dtype=torch.long).unsqueeze(0).to(device)
        tgt_ids = torch.tensor([SOS_ID], dtype=torch.long).unsqueeze(0).to(device)
        src_pad = src_ids == PAD_ID

        for _ in range(max_len - 1):
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_ids.size(1)).to(device)
            tgt_pad = tgt_ids == PAD_ID
            out = self(src_ids, tgt_ids, src_pad_mask=src_pad, tgt_pad_mask=tgt_pad, tgt_mask=tgt_mask)
            next_token = out[:, -1, :].argmax(dim=-1)
            tgt_ids = torch.cat([tgt_ids, next_token.unsqueeze(0)], dim=1)
            if next_token.item() == EOS_ID: break
        return tgt_sp.decode(tgt_ids.squeeze().tolist()[1:-1])

    def beam_search(self, src_text, src_sp, tgt_sp, beam_size=5, max_len=64, length_penalty=0.6):
        self.eval()
        device = next(self.parameters()).device
        
        # 1. 编码源端
        src_ids = torch.tensor([SOS_ID] + src_sp.encode(src_text) + [EOS_ID], dtype=torch.long).unsqueeze(0).to(device)
        src_pad_mask = (src_ids == PAD_ID)
        
        # 2. 初始 Beam
        beams = [{"seq": [SOS_ID], "score": 0.0, "finished": False}]
        
        for step in range(1, max_len):
            active = [b for b in beams if not b["finished"]]
            if not active:
                break
                
            # 批量构建 tgt 序列
            batch_seqs = [torch.tensor(b["seq"], dtype=torch.long, device=device).unsqueeze(0) for b in active]
            seqs_tensor = torch.cat(batch_seqs, dim=0)
            tgt_pad_mask = (seqs_tensor == PAD_ID)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seqs_tensor.size(1)).to(device).bool()
            
            # 扩展 src 匹配 Batch 大小
            src_exp = src_ids.expand(seqs_tensor.size(0), -1)
            src_pad_exp = src_pad_mask.expand(seqs_tensor.size(0), -1)
            
            with torch.no_grad():
                out = self(src_exp, seqs_tensor, tgt_mask=tgt_mask, 
                            src_pad_mask=src_pad_exp, tgt_pad_mask=tgt_pad_mask)
                
            # 获取最后一步 log_prob
            log_probs = torch.log_softmax(out[:, -1, :], dim=-1)
            
            # 收集所有候选
            candidates = []
            for i, b in enumerate(active):
                for token_id in range(log_probs.size(1)):
                    prob = log_probs[i, token_id].item()
                    new_score = b["score"] + prob
                    
                    # 长度惩罚
                    if length_penalty != 0:
                        new_score /= ((5 + len(b["seq"])) / 6) ** length_penalty
                        
                    new_seq = b["seq"] + [token_id]
                    finished = token_id == EOS_ID
                    
                    candidates.append({"seq": new_seq, "score": new_score, "finished": finished})
                    
            # 保留 top K
            candidates.sort(key=lambda x: x["score"], reverse=True)
            beams = candidates[:beam_size]
            
        # 返回最优序列 (去掉 SOS)
        best_seq = beams[0]["seq"][1:]
        if best_seq and best_seq[-1] == EOS_ID:
            best_seq = best_seq[:-1]
            
        return tgt_sp.decode(best_seq)

if __name__ == '__main__':
    model = Seq2SeqTransformer(
        src_vocab=256,
        tgt_vocab=10000,
        d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS, dropout=DROPOUT
    )
    
    state_dict = torch.load(MODEL_PATH)
    model.load_state_dict(state_dict)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    src_sp = spm.SentencePieceProcessor()
    src_sp.Load(model_file=SP_SRC_MODEL)
    tgt_sp = spm.SentencePieceProcessor()
    tgt_sp.Load(model_file=SP_TGT_MODEL)
    
    translate_result = model.translate("bsbtjyrernx", src_sp, tgt_sp)
    beam_search_result = model.beam_search("bsbtjyrernx", src_sp, tgt_sp, beam_size=5, max_len=64, length_penalty=0.6)

    print(translate_result)
    print(beam_search_result)