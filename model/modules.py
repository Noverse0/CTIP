
import torch
import torch.nn as nn
import torch.nn.functional as F

def one_param(m):
    "get model first parameter"
    return next(iter(m.parameters()))

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels        
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        size = x.shape[-1]
        x = x.view(-1, self.channels, size * size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, size, size)
    

class CrossAttention(nn.Module):
    def __init__(self, channels, num_heads, condition_dim):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.channels = channels
        self.condition_dim = condition_dim
        self.mha = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_cross = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        # condition_dim을 channels로 매핑하는 레이어 추가
        self.condition_mapping = nn.Linear(condition_dim, channels)

    def forward(self, x, c):
        size = x.shape[-1]  # H 차원
        batch_size = x.shape[0]

        # c의 차원을 [batch_size, channel]로 매핑 이후 차원 조정
        c = self.condition_mapping(c)
        c = c.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, size, size)
        c = c.view(-1, self.channels, size * size).swapaxes(1, 2)

        # x를 적절한 차원으로 조정
        x = x.view(batch_size, self.channels, size * size).transpose(1, 2)

        x_ln = self.ln(x)
        # Query: x, Key: c_expanded, Value: c_expanded로 Cross Attention 수행
        attention_value, _ = self.mha(x_ln, c, c)  
        attention_value = attention_value + x
        attention_value = self.ff_cross(attention_value) + attention_value

        return attention_value.transpose(1, 2).view(batch_size, self.channels, size, size)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = torch.cat([skip_x, x], dim=1)
        x = self.up(x)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet(nn.Module):
    def __init__(self, latent_size, n_head=8, time_dim=256, condi_dim=256, remove_deep_conv=False):
        super().__init__()
        self.time_dim = time_dim
        self.condi_dim = condi_dim
        self.remove_deep_conv = remove_deep_conv
        self.c_in = latent_size[0]
        self.c_out = latent_size[0]
        self.head = n_head
        self.inc = DoubleConv(self.c_in, 64)
        self.down1 = Down(64, 128)
        self.ca1 = CrossAttention(128, n_head, self.condi_dim)
        self.down2 = Down(128, 256)
        self.ca2 = CrossAttention(256, n_head, self.condi_dim)
        self.down3 = Down(256, 512)
        self.ca3 = CrossAttention(512, n_head, self.condi_dim)
        self.down4 = Down(512, 512)
        self.ca4 = CrossAttention(512, n_head, self.condi_dim)

        if remove_deep_conv:
            self.bot1 = DoubleConv(512, 512)
            self.bot3 = DoubleConv(512, 512)
        else:
            self.bot1 = DoubleConv(512, 1024)
            self.bot2 = DoubleConv(1024, 1024)
            self.bot3 = DoubleConv(1024, 512)

        self.ca5 = CrossAttention(512, n_head, self.condi_dim)
        self.up1 = Up(1024, 512)
        self.ca6 = CrossAttention(512, n_head, self.condi_dim)
        self.up2 = Up(1024, 256)
        self.ca7 = CrossAttention(256, n_head, self.condi_dim)
        self.up3 = Up(512, 128)
        self.ca8 = CrossAttention(128, n_head, self.condi_dim)
        self.up4 = Up(256, 128)
        
        self.outc = nn.Conv2d(128, self.c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=one_param(self).device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def unet_forwad(self, x, t, condi):
        x1 = self.inc(x) # 64, 64, 32, 32
        x2 = self.down1(x1, t) # 64, 128, 16, 16
        x2 = self.ca1(x2, condi)
        x3 = self.down2(x2, t) # 64, 256, 8, 8
        x3 = self.ca2(x3, condi)
        x4 = self.down3(x3, t) # 64, 512, 4, 4
        x4 = self.ca3(x4, condi)
        x5 = self.down4(x4, t) # 64, 512, 2, 2
        x5 = self.ca4(x5, condi)

        x6 = self.bot1(x5) # 64, 1024, 2, 2
        if not self.remove_deep_conv:
            x6 = self.bot2(x6) # 64, 1024, 2, 2
        x6 = self.bot3(x6) # 64, 512, 2, 2
        
        x = self.ca5(x6, condi)
        x = self.up1(x, x5, t)
        x = self.ca6(x, condi)
        x = self.up2(x, x4, t)
        x = self.ca7(x, condi)
        x = self.up3(x, x3, t)
        x = self.ca8(x, condi)
        x = self.up4(x, x2, t)
        
        output = self.outc(x)
        return output
    
    def forward(self, x, t):
        t = t.unsqueeze(-1)
        t = self.pos_encoding(t, self.time_dim)
        return self.unet_forwad(x, t)


class UNet_conditional(UNet):
    def __init__(self, latent_size, n_head=4, time_dim=256, condi_dim=256):
        super().__init__(latent_size, n_head, time_dim)

    def forward(self, x, t, condi):
        t = t.unsqueeze(-1)
        t = self.pos_encoding(t, self.time_dim)

        return self.unet_forwad(x, t, condi)