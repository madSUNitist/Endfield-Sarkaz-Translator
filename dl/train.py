import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import torch

import tqdm

from hyperparameters import *

def train_model(model, train_loader, epochs=EPOCHS, lr=LR):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: min((step+1)**-0.5, (step+1)*WARMUP_STEPS**-1.5))

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for src, tgt in tqdm.tqdm(train_loader):
            src, tgt = src.to(device), tgt.to(device)
            tgt_in = tgt[:, :-1]
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_in.size(1)).to(device)
            src_pad = src == PAD_ID
            tgt_pad = tgt_in == PAD_ID

            optimizer.zero_grad()
            out = model(src, tgt_in, src_pad_mask=src_pad, tgt_pad_mask=tgt_pad, tgt_mask=tgt_mask)
            loss = criterion(out.transpose(1, 2), tgt[:, 1:])
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")
    
    torch.save(model.state_dict(), SAVE_PATH)
    return model