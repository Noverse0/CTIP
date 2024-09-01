import torch
import os
import torchvision.utils as vutils
from src.utils import mkdir_p
from fastprogress import progress_bar
from tqdm import tqdm

def sample(model, ema_model, use_ema, condi_emb, batch_size, alpha_hat_sc, alpha_sc, beta_sc, args):
    model = ema_model if use_ema else model
    model.eval()
    with torch.inference_mode():
        x = torch.randn(torch.Size([batch_size, 4, 32, 32])).to(args.device)
        for i in progress_bar(reversed(range(1, args.noise_steps)), total=args.noise_steps-1, leave=False):
            t = (torch.ones(batch_size, device=args.device) * i).long()
            predicted_noise = model(x, t, condi_emb)

            alpha = alpha_sc[t.int()][:, None, None, None]
            alpha_hat = alpha_hat_sc[t.int()][:, None, None, None]
            beta = beta_sc[t.int()][:, None, None, None]

            if i > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

    return x

def test(dataloader, model, condition_encoder, ema_model, use_ema, ae_model,
         alpha_hat, alpha, beta, save_dir, epoch, args):
    for step, data in enumerate(dataloader, 0):
        ######################################################
        # (1) Prepare_data
        ######################################################
        imgs, spec_emb, gauge_emb, filename, _ = data
        # imgs, spec_emb, test_emb, gauge_emb, filename, _ = data

        spec_emb = spec_emb.to(args.device)
        # test_emb = test_emb.to(args.device)
        gauge_emb = gauge_emb.to(args.device)
        condi_emb = condition_encoder(spec_emb, gauge_emb)
        # condi_emb = torch.cat([condi_emb, test_emb], dim=1)
        
        samples_z = sample(model, ema_model, use_ema, condi_emb, condi_emb.shape[0], alpha_hat, alpha, beta, args)
        ######################################################
        # (2) Generate fake images
        ######################################################
        batch_size = condi_emb.size(0)
        ae_model.eval()
        with torch.inference_mode():
            fake_imgs = ae_model.decode(samples_z).sample.mean(dim=1, keepdim=True)
            
        if use_ema:
            folder = '%s/single/%s_ema' % (save_dir, epoch)
        else:
            folder = '%s/single/%s' % (save_dir, epoch)
        if not os.path.isdir(folder):
            print('Make a new folder: ', folder)
            mkdir_p(folder)

        fullpath = '%s/test_%s_Generate.png' % (folder, str(step))
        vutils.save_image(fake_imgs.data, fullpath, nrow=8, normalize=True)
        fullpath = '%s/test_%s_Real.png' % (folder, str(step))
        vutils.save_image(imgs.data, fullpath, nrow=8, normalize=True)
        for j in range(batch_size):
            ######################################################
            # (3) Save fake images
            ######################################################
            fullpath = '%s/%s_Generate.png' % (folder, filename[j])
            vutils.save_image(fake_imgs[j].data, fullpath, normalize=True)
            
            fullpath = '%s/%s_Real.png' % (folder, filename[j])
            vutils.save_image(imgs[j].data, fullpath, normalize=True)
            