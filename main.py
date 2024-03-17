import os
import os.path as osp
import time
import random
import argparse
import numpy as np
import pprint
import pandas as pd
import torch
import torchvision.utils as vutils
import torch.nn as nn
import copy

from fastprogress import progress_bar
from diffusers import AutoencoderKL
from lpips import LPIPS
from model.encoder import Condition_encoder, VisionTransformer
from src.utils import *
from src.train import train_step, prepare_noise_schedule
from data.data_loader import get_images_dataloader
from src.test import sample, test
from src.train_clip import train_clip
from model.modules import EMA, UNet_conditional
from torchvision.utils import make_grid, save_image

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  "."))

def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='Text2Img')
    parser.add_argument('--cfg', dest='cfg_file', type=str, default='./config.yml',
                        help='optional config file')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='number of workers(default: 0)')
    parser.add_argument('--stamp', type=str, default='normal',
                        help='the stamp of model')
    parser.add_argument('--train', type=bool, default=True,
                        help='if train model')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--noise_steps', type=int, default=1000,
                        help='noise steps')
    parser.add_argument('--resume_epoch', type=int, default=1,
                        help='resume epoch')
    parser.add_argument('--resume_model_path', type=str, default='model',
                        help='the model for resume training')
    parser.add_argument('--multi_gpus', type=bool, default=False,
                        help='if multi-gpu training under ddp')
    parser.add_argument('--world_size', type=int, default=4,
                        help='world_size id')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='local_rank id')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='gpu id')
    parser.add_argument('--clip_train', type=bool, default=True,
                        help='clip train')
    parser.add_argument('--clip_model', type=str, default='ViT',
                        help='clip model')
    parser.add_argument('--clip_epoch', type=int, default=30,
                        help='clip training max epoch')
    parser.add_argument('--clip_path', type=str, default='./clip_saved_models/',
                        help='clip model path')
    parser.add_argument('--clip_t', type=float, default=0.01,
                        help='clip t')
    parser.add_argument('--vae_train', type=bool, default=False,
                        help='vae train')
    parser.add_argument('--vae_epoch', type=int, default=5,
                        help='vae training max epoch')
    parser.add_argument('--vae_path', type=str, default='./ae_saved_models/',
                        help='vae model path')
    parser.add_argument('--random_sample', action='store_true',default=True, 
                        help='whether to sample the dataset with random sampler')

    args = parser.parse_args()
    return args


def main(args):
    time_stamp = get_time_stamp()
    stamp = '_'.join([str(args.model),str(args.stamp),str(args.CONFIG_NAME),str(args.imsize),time_stamp])
    args.model_save_file = osp.join(ROOT_PATH, 'saved_models', stamp)
    log_dir = osp.join(ROOT_PATH, 'logs/{0}'.format(osp.join(stamp)))
    args.img_save_dir = osp.join(ROOT_PATH, 'imgs/{0}'.format(osp.join(stamp)))
    if (args.multi_gpus==True) and (get_rank() != 0):
        None
    else:
        mkdir_p(osp.join(ROOT_PATH, 'logs'))
        mkdir_p(args.model_save_file)
        mkdir_p(args.img_save_dir)
        mkdir_p(args.clip_path)
        
    # prepare dataloader, models, data
    train_dl, train_sampler, test_dl, test_sampler = get_images_dataloader('./data', batch_size = args.batch_size, image_size = args.imsize, max_gauge_dim=args.max_gauge_dim,
                                                    TEST_SPEC = args.TEST_SPEC,
                                                    num_workers= args.num_workers, world_size = args.world_size, multi_gpus=args.multi_gpus)
    fixed_img, fixed_spec, fiexed_gague, fixed_z, fixed_latent = get_fix_data(train_dl, args)
    
    condition_encoder = Condition_encoder(
        input_features = fixed_spec.shape[1], 
        condi_dim = args.condi_dim,
        max_gauge_dim = args.max_gauge_dim, 
        num_gauge=fiexed_gague.shape[1]
    ).to(args.device)

    image_encoder = VisionTransformer(
        input_resolution= args.imsize,
        patch_size= int(args.imsize/64),
        width= args.imsize,
        layers= 4,
        heads= args.imsize // 64,
        output_dim= args.condi_dim
    ).to(args.device)

    
    # train clip
    if (args.clip_train==True):
        start_epoch = 1
        optimizer_image = torch.optim.Adam(image_encoder.parameters(), lr=0.001, betas=(0.0, 0.9))
        optimizer_condition = torch.optim.Adam(condition_encoder.parameters(), lr=0.001, betas=(0.0, 0.9))
        for epoch in range(start_epoch, args.clip_epoch + 1, 1):
            args.current_epoch = epoch
            train_clip(train_dl, condition_encoder, image_encoder, optimizer_condition, optimizer_image, args)
            if epoch%10==0:
                save_clip_models(condition_encoder, epoch, args.multi_gpus, args.clip_path)
        torch.cuda.empty_cache()
    else:
        args.clip_path = osp.join(args.clip_path, 'state_epoch_030_ctip_new.pth')
        condition_encoder = load_condition_encoder(condition_encoder, args.clip_path)
    
    # load vae
    print('vae model loaded')
    ae_model = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(args.device)

    # train dataset에 대한 ground truth image 생성
    if (args.multi_gpus==True) and (get_rank() != 0):
        None
    else:
        img_name = 'gt.png'
        img_save_path = osp.join(args.img_save_dir, img_name)
        vutils.save_image(fixed_img.data, img_save_path, nrow=8, normalize=True)

        # Test AutoEncoder
        ae_model.eval()
        with torch.no_grad(): 
            print('fixed_latent:', fixed_latent.shape)
            reconstructed_img = ae_model.decode(fixed_latent).sample
            img_save_path = osp.join(args.img_save_dir, 'reconstructed_img.png')
            vutils.save_image(reconstructed_img.data, img_save_path, nrow=8, normalize=True)
            args.latent_size = fixed_latent.shape

    unet = UNet_conditional(fixed_latent.shape[1:], n_head=8, time_dim=256, condi_dim=args.condi_dim).to(args.device)
    
    
    # prepare optimizer
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.lr, eps=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, 
                                                steps_per_epoch=len(train_dl), epochs=args.max_epoch)
    loss_fn = nn.MSELoss()
    ema = EMA(0.995)
    ema_model = copy.deepcopy(unet).eval().requires_grad_(False)
    scaler = torch.cuda.amp.GradScaler()
    beta = prepare_noise_schedule(args.beta_start, args.beta_end, args.noise_steps).to(args.device)
    alpha = 1. - beta
    alpha_hat = torch.cumprod(alpha, dim=0)

    # print args
    if (args.multi_gpus==True) and (get_rank() != 0):
        None
    else:
        # pprint.pprint(args)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        arg_save_path = osp.join(log_dir, 'args.yaml')
        save_args(arg_save_path, args)
        print("Start Training")

    print('**************u-net_paras: ',params_count(unet))
    print('**************total: ',params_count(unet)+params_count(ae_model))

    # Start training
    start_epoch = 1
    avg_losses = []
    test_interval, gen_interval, save_interval = args.test_interval, args.gen_interval, args.save_interval
    for epoch in range(start_epoch, args.max_epoch + 1, 1):
        if (args.multi_gpus==True):
            train_sampler.set_epoch(epoch)
        start_t = time.time()
        # training
        args.current_epoch = epoch
        
        avg_loss = train_step(unet, condition_encoder, train_dl, optimizer, scheduler, loss_fn, ema, ema_model, scaler, alpha_hat, args)
        avg_losses.append(avg_loss)
        print('The epoch %d avg_loss: %.2f'%(epoch, avg_loss))
        # save
        if epoch%save_interval==0:
            save_models(unet, optimizer, alpha_hat, alpha, beta, epoch, args.multi_gpus, args.model_save_file)
            
        # sample
        if epoch%gen_interval==0:
            condi_emb = condition_encoder(fixed_spec, fiexed_gague)
            samples_z = sample(unet, ema_model, False, condi_emb, args.batch_size, alpha_hat, alpha, beta, args)
            with torch.inference_mode():
                decoder_samples = ae_model.decode(samples_z).sample
            sample_grid = make_grid(decoder_samples.mean(dim=1, keepdim=True), nrow=8)
            img_name = 'samples_epoch_%03d.png'%(epoch)
            img_save_path = osp.join(args.img_save_dir, img_name)
            save_image(sample_grid, img_save_path)

            torch.cuda.empty_cache()
        # end epoch
            
        # test
        if epoch%test_interval==0:
            unet.eval()
            test(test_dl, unet, condition_encoder, ema_model, False, ae_model,
                alpha_hat, alpha, beta, args.img_save_dir, epoch, args)
            torch.cuda.empty_cache()
            end_t = time.time()
            print('The epoch %d costs %.2fs'%(epoch, end_t-start_t))
            print('*'*40)
    avg_losses = pd.DataFrame(avg_losses, columns=['loss'])
    avg_losses.to_csv(osp.join(log_dir, 'avg_losses.csv'), index=False)
    print('Training finished')
        
        

if __name__ == "__main__":
    args = merge_args_yaml(parse_args())
    # set seed
    if args.manual_seed is None:
        args.manual_seed = 100
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    if args.cuda:
        if args.multi_gpus:
            torch.cuda.manual_seed_all(args.manual_seed)
            torch.distributed.init_process_group(backend="nccl", world_size=4)
            local_rank = torch.distributed.get_rank()
            torch.cuda.set_device(local_rank)
            args.device = torch.device("cuda", local_rank)
            args.local_rank = local_rank
        else:
            torch.cuda.manual_seed_all(args.manual_seed)
            torch.cuda.set_device(args.gpu_id)
            args.device = torch.device("cuda")
    else:
        args.device = torch.device('cpu')

    main(args)
