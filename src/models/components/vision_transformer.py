# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# import timm
from timm.models.vision_transformer import Block, DropPath, Mlp

from src.utils.masks import unstructured_mask
from src.utils.pos_embed import get_2d_sincos_pos_embed


def expand_size(sz):
    if isinstance(sz, int):
        return [sz, sz]
    return sz


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding -- borrowed from https://pypi.org/project/timm/0.4.12/
    """

    def __init__(self, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        patch_size = expand_size(patch_size)
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.permute(0, 2, 3, 1)  # channels-last
        x = self.norm(x)
        return x


class FlashAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 use_rotary: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.attn_drop = attn_drop

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.use_rotary = use_rotary
        if self.use_rotary:
            powers = -2 * torch.arange(head_dim // 2, dtype=torch.float) / dim
            self.thetas = nn.Parameter((10000 ** powers).repeat_interleave(2).unsqueeze(0),
                                       requires_grad=False)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv  # each one has shape batch_size, num_heads, seq_length, embed_dim

        if self.use_rotary:
            q, k = self._add_rotary_encoding(q, k)

        x = F.scaled_dot_product_attention(q, k, v,
                                           dropout_p=self.attn_drop if self.training else 0.,
                                           scale=self.scale).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def _add_rotary_encoding(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        print("rotary", self.thetas.shape)
        positions = torch.arange(q.size(-2), dtype=torch.float, device=q.device).unsqueeze(1)
        angles = positions * self.thetas  # angles[i, j] = i*theta_j, shape (seq_len, embed_dim)

        perm_q = torch.stack((-q[..., 1::2], q[..., 0::2]), dim=-1).view_as(q)
        q = torch.cos(angles) * q + torch.sin(angles) * perm_q

        perm_k = torch.stack((-k[..., 1::2], q[..., 0::2]), dim=-1).view_as(k)
        k = torch.cos(angles) * k + torch.sin(angles) * perm_k

        return q, k


class FlashAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_rotary: bool = False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = FlashAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            use_rotary=use_rotary
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# Masked Modeling Duo (M2D)
class ViTEncoder(nn.Module):
    """ Vision Transformer encoder (M2D) implementation based on the MAE.
    """

    def __init__(self, img_size=224, in_chans=3, patch_size=16,
                 embed_dim=1024, depth=24, num_heads=16,
                 masking_ratio: float = 0.6, masking_method: str = "unstructured",
                 mask_targets: bool = True, shift_pos_encoding: bool = False,
                 time_pos_encoding: str = "absolute",
                 mlp_ratio=4., norm_layer=nn.LayerNorm, flash_attn: bool = False):
        super().__init__()
        self.in_chans = in_chans
        self.masking_ratio = masking_ratio
        self.img_size, self.patch_size = expand_size(img_size), expand_size(patch_size)
        self.grid_size = [s // p for s, p in zip(self.img_size, self.patch_size)]

        self.time_pos_encoding = time_pos_encoding
        if self.time_pos_encoding != "absolute":
            self.grid_size[1] = 1  # put 1 in time dimension since embeddings are relative in this dimension anyway

        # masking method
        self.masking_fn = unstructured_mask

        self.patch_embed = PatchEmbed(self.patch_size, in_chans, embed_dim)

        self.pos_embed = nn.Parameter(torch.zeros(1, *self.grid_size, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        attn_layer = FlashAttentionBlock if flash_attn else Block
        self.blocks = nn.ModuleList([
            attn_layer(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                       use_rotary=self.time_pos_encoding == "rotary")
            for _ in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.mask_targets = mask_targets
        self.shift_pos_encoding = shift_pos_encoding

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        embed_dim = self.pos_embed.size(-1)
        if self.time_pos_encoding == "absolute":
            pos_embed = get_2d_sincos_pos_embed(embed_dim, self.grid_size)
        else:  # in that case we add frequency positional embeddings on all dimensions
            pos_embed = get_2d_sincos_pos_embed(2 * embed_dim, self.grid_size)[..., embed_dim:]
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def generate_masks(self, inputs: torch.Tensor) -> Tuple[torch.BoolTensor, torch.BoolTensor]:
        batch_size, *_, freq_bins, time_steps = inputs.size()
        ph, pw = self.patch_size

        context_mask, target_mask = self.masking_fn(freq_bins // ph, time_steps // pw, masking_ratio=self.masking_ratio,
                                    num_masks=batch_size, device=inputs.device)

        if self.mask_targets:
            return context_mask, target_mask

        return context_mask, torch.ones_like(context_mask)

    def forward(self, x: torch.Tensor, mask: torch.BoolTensor | None = None, return_layers: bool = False):
        r"""

        Args:
        """
        # embed patches
        x = self.patch_embed(x)
        batch_size, freq_patches, time_patches, embed_dim = x.size()

        # add pos embed w/o cls token
        if self.time_pos_encoding == "absolute":
            time_shift = torch.randint(self.pos_embed.size(2) - time_patches, ()) if self.shift_pos_encoding else 0
            x = x + self.pos_embed[:, :, time_shift: time_shift + time_patches, :]
        else:
            x = x + self.pos_embed

        # mask inputs
        if mask is not None:  # WARNING: this only works since the number of masked patches is constant within a batch!
            x = x[mask]
        x = x.view(batch_size, -1, embed_dim)

        # apply Transformer blocks
        layers = []
        for blk in self.blocks:
            x = blk(x)
            if return_layers:
                layers.append(x)

        x = self.norm(x)
        if return_layers:
            layers.pop()  # replace the last feature with the normalized one.
            layers.append(x)

        if return_layers:
            return torch.stack(layers)

        return x


class ViTPredictor(nn.Module):
    """ Masked Modeling Duo (M2D) implementation based on the MAE.
    """

    def __init__(self, img_size=224, patch_size=16, encoder_embed_dim=1024,
                 embed_dim=512, depth=8, num_heads=16,
                 time_pos_encoding: str = "absolute",
                 mlp_ratio=4., norm_layer=nn.LayerNorm, flash_attn: bool = False):
        super().__init__()
        img_size, patch_size = expand_size(img_size), expand_size(patch_size)
        self.grid_size = [s // p for s, p in zip(img_size, patch_size)]

        self.time_pos_encoding = time_pos_encoding
        if self.time_pos_encoding != "absolute":
            self.grid_size[1] = 1  # put 1 in time dimension since embeddings are relative in this dimension anyway

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(encoder_embed_dim, embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, *self.grid_size, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        attn_layer = FlashAttentionBlock if flash_attn else Block
        self.blocks = nn.ModuleList([
            attn_layer(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                       use_rotary=self.time_pos_encoding == "rotary")
            for _ in range(depth)])

        self.decoder_norm = norm_layer(embed_dim)
        self.decoder_pred = nn.Linear(embed_dim, encoder_embed_dim, bias=True)  # predict target embeddings

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        embed_dim = self.pos_embed.size(-1)
        if self.time_pos_encoding == "absolute":
            pos_embed = get_2d_sincos_pos_embed(embed_dim, self.grid_size)
        else:  # in that case we add frequency positional embeddings on all dimensions
            pos_embed = get_2d_sincos_pos_embed(2 * embed_dim, self.grid_size)[..., embed_dim:]
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self,
                x: torch.Tensor,
                context_mask: torch.BoolTensor,
                target_mask: torch.BoolTensor) -> torch.Tensor:
        batch_size, num_patches, embed_dim = x.size()

        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence (eventually cast to x.dtype to handle mixed precision)
        context_tokens = self.mask_token.to(x.dtype).repeat(*context_mask.size(), 1)
        context_tokens.masked_scatter_(context_mask.unsqueeze(-1), x)

        # add pos embed and flatten for making it a sequence
        x = torch.flatten(context_tokens + self.pos_embed, 1, 2)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        x = x.view(*target_mask.size(), -1)

        preds = x[target_mask].view(batch_size, -1, embed_dim)

        return preds


if __name__ == '__main__':
    # dummy test for FlashAttention
    f = FlashAttention(512)
    a = torch.randn(2048, 65, 512)

    y = f(a)
    print(y.size())
