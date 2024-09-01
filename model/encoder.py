from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F



class Spec_encoder(nn.Module):
    def __init__(self, input_features, image_size):
        super(Spec_encoder, self).__init__()
        self.fc1_n = nn.Linear(input_features, int(image_size / 4))
        self.fc2_n = nn.Linear(int(image_size / 4), int(image_size / 2))

    def forward(self, x):
        x = torch.relu(self.fc1_n(x))
        x = self.fc2_n(x)
        return x


class Gauge_encoder(nn.Module):
    def __init__(self, num_gauge, input_dim=400, image_size=100):
        super(Gauge_encoder, self).__init__()
        filter_size = 10
        stride = 1
        out_channels = 1

        self.conv_layer = nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=(num_gauge, filter_size), stride=stride, padding=0)
        self.flatten = nn.Flatten()

        flattened_size = out_channels * ((input_dim - filter_size + stride) // stride)
        self.fc_layer = nn.Linear(int(flattened_size), int(image_size / 2))

    def forward(self, x):
        x = x.unsqueeze(1) 
        x = self.conv_layer(x)
        x = self.flatten(x)
        x = self.fc_layer(x)
        return x


class Condition_encoder(nn.Module):
    def __init__(self, input_features, condi_dim, max_gauge_dim, num_gauge):
        super(Condition_encoder, self).__init__()
        self.spec_encoder = Spec_encoder(input_features = input_features, image_size = condi_dim)
        self.gauge_encoder = Gauge_encoder(num_gauge = num_gauge, input_dim = max_gauge_dim, image_size = condi_dim)

    def forward(self, spec_emb, gauge_emb):
        spec_emb = self.spec_encoder(spec_emb)
        gauge_emb = self.gauge_encoder(gauge_emb)
        condi_emb = torch.cat((spec_emb, gauge_emb), dim=1)
        return condi_emb


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x
