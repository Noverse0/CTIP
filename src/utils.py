import numpy as np
import torch
import os
import errno
import dateutil.tz
import datetime
import yaml
from torch import distributed as dist
from easydict import EasyDict as edict

# utils

def load_yaml(filename):
    with open(filename, 'r') as f:
        cfg = edict(yaml.load(f, Loader=yaml.FullLoader))
    return cfg

def merge_args_yaml(args):
    if args.cfg_file is not None:
        opt = vars(args)
        args = load_yaml(args.cfg_file)
        args.update(opt)
        args = edict(args)
    return args

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def params_count(model):
    model_size = np.sum([p.numel() for p in model.parameters()]).item()
    return model_size

def get_time_stamp():
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')  
    return timestamp

def save_args(save_path, args):
    fp = open(save_path, 'w')
    fp.write(yaml.dump(args))
    fp.close()

# DDP utils

def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


# save and load models
def load_opt_weights(optimizer, weights):
    optimizer.load_state_dict(weights)
    return optimizer


def load_model_opt(netG, netD, netC, optim_G, optim_D, condition_encoder, path, clip_encoder_path, multi_gpus):
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    netG = load_model_weights(netG, checkpoint['model']['netG'], multi_gpus)
    netD = load_model_weights(netD, checkpoint['model']['netD'], multi_gpus)
    netC = load_model_weights(netC, checkpoint['model']['netC'], multi_gpus)
    condition_encoder = load_condition_encoder(condition_encoder, clip_encoder_path)
    optim_G = load_opt_weights(optim_G, checkpoint['optimizers']['optimizer_G'])
    optim_D = load_opt_weights(optim_D, checkpoint['optimizers']['optimizer_D'])
    return netG, netD, netC, optim_G, optim_D, condition_encoder


def load_models(unet, condition_encoder, path):
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    unet.load_state_dict(checkpoint['model']['unet'])
    condition_encoder.load_state_dict(checkpoint['encoder']['condition_encoder'])
    alpha_hat= checkpoint['alpha_hat']
    alpha= checkpoint['alpha']
    beta= checkpoint['beta']
    return unet, condition_encoder, alpha_hat, alpha, beta


def load_unet(unet, path, multi_gpus, train):
    checkpoint = torch.load(path, map_location="cpu")
    unet = load_model_weights(unet, checkpoint['model']['unet'], multi_gpus, train)
    return unet


def load_condition_encoder(condition_encoder, path):
    print('load clip encoder')
    checkpoint = torch.load(path, map_location="cpu")
    condition_encoder.load_state_dict(checkpoint['encoder']['condition_encoder'])
    return condition_encoder


def load_ae(ae_model, path):
    print('load autoencoder')
    checkpoint = torch.load(path, map_location="cpu")
    ae_model.load_state_dict(checkpoint['autoencoder']['autoencoder'])
    return load_ae


def load_model_weights(model, weights, multi_gpus, train=True):
    if list(weights.keys())[0].find('module')==-1:
        pretrained_with_multi_gpu = False
    else:
        pretrained_with_multi_gpu = True
    if (multi_gpus==False) or (train==False):
        if pretrained_with_multi_gpu:
            state_dict = {
                key[7:]: value
                for key, value in weights.items()
            }
        else:
            state_dict = weights
    else:
        state_dict = weights
    model.load_state_dict(state_dict)
    return model


def save_models(unet, optimizer, alpha_hat, alpha, beta, epoch, multi_gpus, save_path):
    if (multi_gpus==True) and (get_rank() != 0):
        None
    else:
        state = {'model': {'unet': unet.state_dict()}, \
                'optimizers': {'optimizer': optimizer.state_dict()},\
                'alpha_hat': alpha_hat, 'alpha': alpha, 'beta': beta, 'epoch': epoch}
        torch.save(state, '%s/state_epoch_%03d.pth' % (save_path, epoch))


def save_clip_models(condition_encoder, epoch, multi_gpus, save_path):
    if (multi_gpus==True) and (get_rank() != 0):
        None
    else:
        state = {'encoder': {'condition_encoder': condition_encoder.state_dict()}, 'epoch': epoch}
        torch.save(state, '%s/state_epoch_%03d_ctip_new.pth' % (save_path, epoch))


def save_vae_models(vae_model, epoch, multi_gpus, ae_path):
    if (multi_gpus==True) and (get_rank() != 0):
        None
    else:
        state = {'vae': {'vae': vae_model.state_dict()}, 'epoch': epoch}
        torch.save(state, '%s/vae_state_epoch_%03d.pth' % (ae_path, epoch))

# fix data
        
def truncated_noise(batch_size=1, dim_z=100, truncation=1., seed=None):
    from scipy.stats import truncnorm
    state = None if seed is None else np.random.RandomState(seed)
    values = truncnorm.rvs(-2, 2, size=(batch_size, dim_z), random_state=state).astype(np.float32)
    return truncation * values

def get_one_batch_data(dataloader):
    imgs, spec_feature, gague_feature, _, img_latents = next(iter(dataloader))
    return imgs, spec_feature, gague_feature, img_latents

def get_fix_data(train_dl, args):
    fixed_images, fixed_spec, fixed_gague, fixed_latents = get_one_batch_data(train_dl)
    
    if args.truncation==True:
        noise = truncated_noise(fixed_images.size(0), args.z_dim, args.trunc_rate)
        fixed_noise = torch.tensor(noise, dtype=torch.float)
    else:
        fixed_noise = torch.randn(fixed_images.size(0), args.z_dim)
    return fixed_images.to(args.device), fixed_spec.to(args.device), fixed_gague.to(args.device), fixed_noise.to(args.device), fixed_latents.to(args.device)

def get_test_data(dataloader):
    imgs, spec_feature, gague_feature = next(iter(dataloader))
    return imgs, spec_feature, gague_feature

def get_test_data(test_dl, args):
    test_imgs, test_spec_feature, test_gague_feature = get_test_data(test_dl)
    
    if args.truncation==True:
        noise = truncated_noise(test_imgs.size(0), args.z_dim, args.trunc_rate)
        test_fixed_noise = torch.tensor(noise, dtype=torch.float).to(args.device)
    else:
        test_fixed_noise = torch.randn(test_imgs.size(0), args.z_dim).to(args.device)
        
    return test_imgs, test_spec_feature, test_gague_feature, test_fixed_noise


