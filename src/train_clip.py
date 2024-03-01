import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from src.utils import get_rank


def train_clip(dataloader, condition_encoder, image_encoder, optimizer_condition, optimizer_image, args):
    device = args.device
    epoch = args.current_epoch
    max_epoch = args.clip_epoch
    t = torch.tensor(args.clip_t, dtype=torch.float32)
    loss_fn = torch.nn.CrossEntropyLoss()

    if (args.multi_gpus==True) and (get_rank() != 0):
        None
    else:
        loop = tqdm(total=len(dataloader))
    
    total_loss = 0.0
    for step, data in enumerate(dataloader, 0):
        optimizer_image.zero_grad()
        optimizer_condition.zero_grad()
        
        imgs, spec_emb, gauge_emb, _, _ = data
        imgs = imgs.to(device)
        spec_emb = spec_emb.to(device)
        gauge_emb = gauge_emb.to(device)
        
        # Condition_encoder
        C_e = condition_encoder(spec_emb, gauge_emb)

        # Image_encoder(ex, CNN, ViT)
        I_e = image_encoder(imgs)

        logits = torch.matmul(C_e, I_e.T) * torch.exp(t)
        labels = torch.arange(logits.shape[0], device=device)
        loss = (loss_fn(logits, labels) + loss_fn(logits.T, labels)) / 2
        total_loss += loss.item()
        
        loss.backward()
        
        optimizer_image.step()
        optimizer_condition.step()
        # update loop information
        if (args.multi_gpus==True) and (get_rank() != 0):
            None
        else:
            loop.update(1)
            loop.set_description(f'Training Epoch [{epoch}/{max_epoch}]')
            loop.set_postfix()
    if (args.multi_gpus==True) and (get_rank() != 0):
        None
    else:
        loop.close()
    print(f"Loss at epoch {epoch}: {total_loss / len(dataloader)}") 