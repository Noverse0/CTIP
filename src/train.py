import torch
import os
import os.path as osp
import numpy as np 
import torchvision.utils as vutils
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from src.utils import get_rank, truncated_noise, mkdir_p
from fastprogress import progress_bar
import random


def noise_images(x, t, alpha_hat):
    "Add noise to images at instant t"
    sqrt_alpha_hat = torch.sqrt(alpha_hat[t])[:, None, None, None]
    sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat[t])[:, None, None, None]
    Ɛ = torch.randn_like(x)
    return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ
    

def sample_timesteps(n, noise_steps):
    return torch.randint(low=1, high=noise_steps, size=(n,))


def train_step(model, condition_encoder, train_dataloader, optimizer, scheduler, loss_fn, ema, ema_model, scaler, alpha_hat, args):
    for epoch in progress_bar(range(args.max_epoch), total=args.max_epoch, leave=True):
        print(f"Starting epoch {args.current_epoch}:")
        avg_loss = 0.
        if args.train: model.train()
        else: model.eval()
        pbar = progress_bar(train_dataloader, leave=False)
        for i, (imgs, spec_emb, gauge_emb, filename, imgs_latent) in enumerate(pbar):
        # for i, (imgs, spec_emb, test_emb, gauge_emb, filename, imgs_latent) in enumerate(pbar):
            with torch.autocast("cuda") and (torch.inference_mode() if not args.train else torch.enable_grad()):
                # imgs = imgs.to(args.device)
                spec_emb = spec_emb.to(args.device)
                # test_emb = test_emb.to(args.device)
                gauge_emb = gauge_emb.to(args.device)
                condi_emb = condition_encoder(spec_emb, gauge_emb)
                # condi_emb = torch.cat([condi_emb, test_emb], dim=1)
                imgs_latent = imgs_latent.to(args.device)
                t = sample_timesteps(args.batch_size, args.noise_steps).to(args.device)
                x_t, noise = noise_images(imgs_latent, t, alpha_hat)

                predicted_noise = model(x_t, t, condi_emb)
                loss = loss_fn(noise, predicted_noise)
                avg_loss += loss

            if args.train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                ema.step_ema(ema_model, model)
                scheduler.step()
            pbar.comment = f"MSE={loss.item():2.3f}"
        return avg_loss.mean().item()



def prepare_noise_schedule(beta_start, beta_end, noise_steps):
    return torch.linspace(beta_start, beta_end, noise_steps)


def set_alpha_hat(args):
    beta = prepare_noise_schedule(args.beta_start, args.beta_end, args.noise_steps).to(args.device)
    alpha = 1. - beta
    alpha_hat = torch.cumprod(alpha, dim=0)
    return alpha_hat, alpha, beta
