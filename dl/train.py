import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from hyperparameters import PAD_ID

import torch

def train_model(model, train_loader, epochs=10, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: min(step**(-0.5), step * 4000**(-1.5)))

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for src, tgt, _, _ in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            src_mask = None  # 编码器不需要因果掩码
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(device)
            src_pad_mask = src == PAD_ID
            tgt_pad_mask = tgt == PAD_ID

            optimizer.zero_grad()
            output = model(src, tgt[:, :-1], tgt_mask=tgt_mask, 
                           src_pad_mask=src_pad_mask, tgt_pad_mask=tgt_pad_mask[:, :-1])
            loss = criterion(output.transpose(1, 2), tgt[:, 1:])
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f}")
    return model