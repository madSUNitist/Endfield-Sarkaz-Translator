import torch.nn as nn

from dl.hyperparameters import SOS_ID, EOS_ID, PAD_ID

import torch


def translate(model, sentence, src_sp, tgt_sp, max_len=64):
    model.eval()
    device = next(model.parameters()).device
    src_ids = torch.tensor([SOS_ID] + src_sp.encode(sentence) + [EOS_ID], dtype=torch.long).unsqueeze(0).to(device)
    
    tgt_ids = torch.tensor([SOS_ID], dtype=torch.long).unsqueeze(0).to(device)
    src_pad_mask = src_ids == PAD_ID
    
    for _ in range(max_len - 1):
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_ids.size(1)).to(device)
        tgt_pad_mask = tgt_ids == PAD_ID
        out = model(src_ids, tgt_ids, tgt_mask=tgt_mask, src_pad_mask=src_pad_mask, tgt_pad_mask=tgt_pad_mask)
        next_token = out[:, -1, :].argmax(dim=-1)
        tgt_ids = torch.cat([tgt_ids, next_token.unsqueeze(0)], dim=1)
        if next_token.item() == EOS_ID: break
            
    return tgt_sp.decode(tgt_ids.squeeze().tolist()[1:-1])  # 去除 SOS/EOS