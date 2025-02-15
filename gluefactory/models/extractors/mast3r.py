"""PyTorch implementation of the mast3r model,
   https://github.com/naver/mast3r
"""

from copy import deepcopy
from itertools import repeat
import collections.abc
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Union, Tuple, Iterable, List, Optional, Dict

from ..base_model import BaseModel
from ..utils.misc import pad_and_stack

try:
    from ..backbones.curope import cuRoPE2D
    RoPE2D = cuRoPE2D
except ImportError:
    print('Warning, cannot find cuda-compiled version of RoPE2D, using a slow pytorch version instead')

    class RoPE2D(torch.nn.Module):
        
        def __init__(self, freq=100.0, F0=1.0):
            super().__init__()
            self.base = freq 
            self.F0 = F0
            self.cache = {}

        def get_cos_sin(self, D, seq_len, device, dtype):
            if (D,seq_len,device,dtype) not in self.cache:
                inv_freq = 1.0 / (self.base ** (torch.arange(0, D, 2).float().to(device) / D))
                t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
                freqs = torch.einsum("i,j->ij", t, inv_freq).to(dtype)
                freqs = torch.cat((freqs, freqs), dim=-1)
                cos = freqs.cos() # (Seq, Dim)
                sin = freqs.sin()
                self.cache[D,seq_len,device,dtype] = (cos,sin)
            return self.cache[D,seq_len,device,dtype]
            
        @staticmethod
        def rotate_half(x):
            x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)
            
        def apply_rope1d(self, tokens, pos1d, cos, sin):
            assert pos1d.ndim==2
            cos = torch.nn.functional.embedding(pos1d, cos)[:, None, :, :]
            sin = torch.nn.functional.embedding(pos1d, sin)[:, None, :, :]
            return (tokens * cos) + (self.rotate_half(tokens) * sin)
            
        def forward(self, tokens, positions):
            """
            input:
                * tokens: batch_size x nheads x ntokens x dim
                * positions: batch_size x ntokens x 2 (y and x position of each token)
            output:
                * tokens after appplying RoPE2D (batch_size x nheads x ntokens x dim)
            """
            assert tokens.size(3)%2==0, "number of dimensions should be a multiple of two"
            D = tokens.size(3) // 2
            assert positions.ndim==3 and positions.shape[-1] == 2 # Batch, Seq, 2
            cos, sin = self.get_cos_sin(D, int(positions.max())+1, tokens.device, tokens.dtype)
            # split features into two along the feature dimension, and apply rope1d on each half
            y, x = tokens.chunk(2, dim=-1)
            y = self.apply_rope1d(y, positions[:,:,0], cos, sin)
            x = self.apply_rope1d(x, positions[:,:,1], cos, sin)
            tokens = torch.cat((y, x), dim=-1)
            return tokens


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class Attention(nn.Module):

    def __init__(self, dim, rope=None, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope 

    def forward(self, x, xpos):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(1,3)
        q, k, v = [qkv[:,:,i] for i in range(3)]
        # q,k,v = qkv.unbind(2)  # make torchscript happy (cannot use tensor as tuple)
               
        if self.rope is not None:
            q = self.rope(q, xpos)
            k = self.rope(k, xpos)
               
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, rope=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, rope=rope, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, xpos):
        x = x + self.drop_path(self.attn(self.norm1(x), xpos))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class CrossAttention(nn.Module):
    
    def __init__(self, dim, rope=None, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.projq = nn.Linear(dim, dim, bias=qkv_bias)
        self.projk = nn.Linear(dim, dim, bias=qkv_bias)
        self.projv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.rope = rope
        
    def forward(self, query, key, value, qpos, kpos):
        B, Nq, C = query.shape
        Nk = key.shape[1]
        Nv = value.shape[1]
        
        q = self.projq(query).reshape(B,Nq,self.num_heads, C// self.num_heads).permute(0, 2, 1, 3)
        k = self.projk(key).reshape(B,Nk,self.num_heads, C// self.num_heads).permute(0, 2, 1, 3)
        v = self.projv(value).reshape(B,Nv,self.num_heads, C// self.num_heads).permute(0, 2, 1, 3)
        
        if self.rope is not None:
            q = self.rope(q, qpos)
            k = self.rope(k, kpos)
            
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class DecoderBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, norm_mem=True, rope=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, rope=rope, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.cross_attn = CrossAttention(dim, rope=rope, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm_y = norm_layer(dim) if norm_mem else nn.Identity()

    def forward(self, x, y, xpos, ypos):
        x = x + self.drop_path(self.attn(self.norm1(x), xpos))
        y_ = self.norm_y(y)
        x = x + self.drop_path(self.cross_attn(self.norm2(x), y_, y_, xpos, ypos))
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        return x, y
        
        
# patch embedding
class PositionGetter(object):
    """ return positions of patches """

    def __init__(self):
        self.cache_positions = {}
        
    def __call__(self, b, h, w, device):
        if not (h,w) in self.cache_positions:
            x = torch.arange(w, device=device)
            y = torch.arange(h, device=device)
            self.cache_positions[h,w] = torch.cartesian_prod(y, x) # (h, w, 2)
        pos = self.cache_positions[h,w].view(1, h*w, 2).expand(b, -1, 2).clone()
        return pos

class PatchEmbed(nn.Module):
    """ just adding _init_weights + position getter compared to timm.models.layers.patch_embed.PatchEmbed"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        
        self.position_getter = PositionGetter()
        
    def forward(self, x):
        B, C, H, W = x.shape
        torch._assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        torch._assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        pos = self.position_getter(B, x.size(2), x.size(3), x.device)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x, pos
        
    def _init_weights(self):
        w = self.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1])) 
        

class PatchEmbedDust3R(PatchEmbed):
    def forward(self, x, **kw):
        B, C, H, W = x.shape
        assert H % self.patch_size[0] == 0, f"Input image height ({H}) is not a multiple of patch size ({self.patch_size[0]})."
        assert W % self.patch_size[1] == 0, f"Input image width ({W}) is not a multiple of patch size ({self.patch_size[1]})."
        x = self.proj(x)
        pos = self.position_getter(B, x.size(2), x.size(3), x.device)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x, pos


class RandomMask(nn.Module):
    """
    random masking
    """
    def __init__(self, num_patches, mask_ratio):
        super().__init__()
        self.num_patches = num_patches
        self.num_mask = int(mask_ratio * self.num_patches)
    
    def __call__(self, x):
        noise = torch.rand(x.size(0), self.num_patches, device=x.device) 
        argsort = torch.argsort(noise, dim=1) 
        return argsort < self.num_mask


def get_2d_sincos_pos_embed(embed_dim, grid_size, n_cls_token=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [n_cls_token+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if n_cls_token>0:
        pos_embed = np.concatenate([np.zeros([n_cls_token, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_pos_embed(pos_embed, enc_embed_dim=None, dec_embed_dim=None, num_patches=None):
    if pos_embed=='cosine':
        # positional embedding of the encoder 
        enc_pos_embed = get_2d_sincos_pos_embed(enc_embed_dim, int(num_patches**.5), n_cls_token=0)
        # positional embedding of the decoder  
        dec_pos_embed = get_2d_sincos_pos_embed(dec_embed_dim, int(num_patches**.5), n_cls_token=0)
        # pos embedding in each block
        rope = None # nothing for cosine 
    elif pos_embed.startswith('RoPE'): # eg RoPE100 
        enc_pos_embed = None # nothing to add in the encoder with RoPE
        dec_pos_embed = None # nothing to add in the decoder with RoPE
        if RoPE2D is None: raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
        freq = float(pos_embed[len('RoPE'):])
        rope = RoPE2D(freq=freq)
    else:
        raise NotImplementedError('Unknown pos_embed '+pos_embed)
    return enc_pos_embed, dec_pos_embed, rope


@torch.no_grad()
def init_weights(m):
    if isinstance(m, nn.Linear):
        # we use xavier_uniform following official JAX ViT:
        torch.nn.init.xavier_uniform_(m.weight)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    out_shape4 = out_shape
    if expand == True:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0],
        out_shape1,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1],
        out_shape2,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2],
        out_shape3,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer4_rn = nn.Conv2d(
        in_shape[3],
        out_shape4,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )

    scratch.layer_rn = nn.ModuleList([
        scratch.layer1_rn,
        scratch.layer2_rn,
        scratch.layer3_rn,
        scratch.layer4_rn,
    ])

    return scratch

class ResidualConvUnit_custom(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation, bn):
        """Init.
        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups = 1

        self.conv1 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.bn,
            groups=self.groups,
        )

        self.conv2 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.bn,
            groups=self.groups,
        )

        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)

class FeatureFusionBlock_custom(nn.Module):
    """Feature fusion block."""

    def __init__(
        self,
        features,
        activation,
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        width_ratio=1,
    ):
        """Init.
        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock_custom, self).__init__()
        self.width_ratio = width_ratio

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups = 1

        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2

        self.out_conv = nn.Conv2d(
            features,
            out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            groups=1,
        )

        self.resConfUnit1 = ResidualConvUnit_custom(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit_custom(features, activation, bn)

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs):
        """Forward pass.
        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            if self.width_ratio != 1:
                res = F.interpolate(res, size=(output.shape[2], output.shape[3]), mode='bilinear')

            output = self.skip_add.add(output, res)
            # output += res

        output = self.resConfUnit2(output)

        if self.width_ratio != 1:
            # and output.shape[3] < self.width_ratio * output.shape[2]
            #size=(image.shape[])
            if (output.shape[3] / output.shape[2]) < (2 / 3) * self.width_ratio:
                shape = 3 * output.shape[3]
            else:
                shape = int(self.width_ratio * 2 * output.shape[2])
            output  = F.interpolate(output, size=(2* output.shape[2], shape), mode='bilinear')
        else:
            output = nn.functional.interpolate(output, scale_factor=2,
                    mode="bilinear", align_corners=self.align_corners)
        output = self.out_conv(output)
        return output

def make_fusion_block(features, use_bn, width_ratio=1):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        width_ratio=width_ratio,
    )

class Interpolate(nn.Module):
    """Interpolation module."""

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.
        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )

        return x

class DPTOutputAdapter(nn.Module):
    """DPT output adapter.

    :param num_cahnnels: Number of output channels
    :param stride_level: tride level compared to the full-sized image.
        E.g. 4 for 1/4th the size of the image.
    :param patch_size_full: Int or tuple of the patch size over the full image size.
        Patch size for smaller inputs will be computed accordingly.
    :param hooks: Index of intermediate layers
    :param layer_dims: Dimension of intermediate layers
    :param feature_dim: Feature dimension
    :param last_dim: out_channels/in_channels for the last two Conv2d when head_type == regression
    :param use_bn: If set to True, activates batch norm
    :param dim_tokens_enc:  Dimension of tokens coming from encoder
    """

    def __init__(self,
                 num_channels: int = 1,
                 stride_level: int = 1,
                 patch_size: Union[int, Tuple[int, int]] = 16,
                 main_tasks: Iterable[str] = ('rgb',),
                 hooks: List[int] = [2, 5, 8, 11],
                 layer_dims: List[int] = [96, 192, 384, 768],
                 feature_dim: int = 256,
                 last_dim: int = 32,
                 use_bn: bool = False,
                 dim_tokens_enc: Optional[int] = None,
                 head_type: str = 'regression',
                 output_width_ratio=1,
                 **kwargs):
        super().__init__()
        self.num_channels = num_channels
        self.stride_level = stride_level
        self.patch_size = pair(patch_size)
        self.main_tasks = main_tasks
        self.hooks = hooks
        self.layer_dims = layer_dims
        self.feature_dim = feature_dim
        self.dim_tokens_enc = dim_tokens_enc * len(self.main_tasks) if dim_tokens_enc is not None else None
        self.head_type = head_type

        # Actual patch height and width, taking into account stride of input
        self.P_H = max(1, self.patch_size[0] // stride_level)
        self.P_W = max(1, self.patch_size[1] // stride_level)

        self.scratch = make_scratch(layer_dims, feature_dim, groups=1, expand=False)

        self.scratch.refinenet1 = make_fusion_block(feature_dim, use_bn, output_width_ratio)
        self.scratch.refinenet2 = make_fusion_block(feature_dim, use_bn, output_width_ratio)
        self.scratch.refinenet3 = make_fusion_block(feature_dim, use_bn, output_width_ratio)
        self.scratch.refinenet4 = make_fusion_block(feature_dim, use_bn, output_width_ratio)

        if self.head_type == 'regression':
            # The "DPTDepthModel" head
            self.head = nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim // 2, kernel_size=3, stride=1, padding=1),
                Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(feature_dim // 2, last_dim, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(last_dim, self.num_channels, kernel_size=1, stride=1, padding=0)
            )
        elif self.head_type == 'semseg':
            # The "DPTSegmentationModel" head
            self.head = nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(feature_dim) if use_bn else nn.Identity(),
                nn.ReLU(True),
                nn.Dropout(0.1, False),
                nn.Conv2d(feature_dim, self.num_channels, kernel_size=1),
                Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            )
        else:
            raise ValueError('DPT head_type must be "regression" or "semseg".')

        if self.dim_tokens_enc is not None:
            self.init(dim_tokens_enc=dim_tokens_enc)

    def init(self, dim_tokens_enc=768):
        """
        Initialize parts of decoder that are dependent on dimension of encoder tokens.
        Should be called when setting up MultiMAE.

        :param dim_tokens_enc: Dimension of tokens coming from encoder
        """
        #print(dim_tokens_enc)

        # Set up activation postprocessing layers
        if isinstance(dim_tokens_enc, int):
            dim_tokens_enc = 4 * [dim_tokens_enc]

        self.dim_tokens_enc = [dt * len(self.main_tasks) for dt in dim_tokens_enc]

        self.act_1_postprocess = nn.Sequential(
            nn.Conv2d(
                in_channels=self.dim_tokens_enc[0],
                out_channels=self.layer_dims[0],
                kernel_size=1, stride=1, padding=0,
            ),
            nn.ConvTranspose2d(
                in_channels=self.layer_dims[0],
                out_channels=self.layer_dims[0],
                kernel_size=4, stride=4, padding=0,
                bias=True, dilation=1, groups=1,
            )
        )

        self.act_2_postprocess = nn.Sequential(
            nn.Conv2d(
                in_channels=self.dim_tokens_enc[1],
                out_channels=self.layer_dims[1],
                kernel_size=1, stride=1, padding=0,
            ),
            nn.ConvTranspose2d(
                in_channels=self.layer_dims[1],
                out_channels=self.layer_dims[1],
                kernel_size=2, stride=2, padding=0,
                bias=True, dilation=1, groups=1,
            )
        )

        self.act_3_postprocess = nn.Sequential(
            nn.Conv2d(
                in_channels=self.dim_tokens_enc[2],
                out_channels=self.layer_dims[2],
                kernel_size=1, stride=1, padding=0,
            )
        )

        self.act_4_postprocess = nn.Sequential(
            nn.Conv2d(
                in_channels=self.dim_tokens_enc[3],
                out_channels=self.layer_dims[3],
                kernel_size=1, stride=1, padding=0,
            ),
            nn.Conv2d(
                in_channels=self.layer_dims[3],
                out_channels=self.layer_dims[3],
                kernel_size=3, stride=2, padding=1,
            )
        )

        self.act_postprocess = nn.ModuleList([
            self.act_1_postprocess,
            self.act_2_postprocess,
            self.act_3_postprocess,
            self.act_4_postprocess
        ])

    def adapt_tokens(self, encoder_tokens):
        # Adapt tokens
        x = []
        x.append(encoder_tokens[:, :])
        x = torch.cat(x, dim=-1)
        return x

    def forward(self, encoder_tokens: List[torch.Tensor], image_size):
            #input_info: Dict):
        assert self.dim_tokens_enc is not None, 'Need to call init(dim_tokens_enc) function first'
        H, W = image_size
        
        # Number of patches in height and width
        N_H = H // (self.stride_level * self.P_H)
        N_W = W // (self.stride_level * self.P_W)

        # Hook decoder onto 4 layers from specified ViT layers
        layers = [encoder_tokens[hook] for hook in self.hooks]

        # Extract only task-relevant tokens and ignore global tokens.
        layers = [self.adapt_tokens(l) for l in layers]

        # Reshape tokens to spatial representation
        layers = [rearrange(l, 'b (nh nw) c -> b c nh nw', nh=N_H, nw=N_W) for l in layers]

        layers = [self.act_postprocess[idx](l) for idx, l in enumerate(layers)]
        # Project layers to chosen feature dim
        layers = [self.scratch.layer_rn[idx](l) for idx, l in enumerate(layers)]

        # Fuse layers using refinement stages
        path_4 = self.scratch.refinenet4(layers[3])
        path_3 = self.scratch.refinenet3(path_4, layers[2])
        path_2 = self.scratch.refinenet2(path_3, layers[1])
        path_1 = self.scratch.refinenet1(path_2, layers[0])

        # Output head
        out = self.head(path_1)

        return out


class DPTOutputAdapter_fix(DPTOutputAdapter):
    """
    Adapt croco's DPTOutputAdapter implementation for dust3r:
    remove duplicated weigths, and fix forward for dust3r
    """

    def init(self, dim_tokens_enc=768):
        super().init(dim_tokens_enc)
        # these are duplicated weights
        del self.act_1_postprocess
        del self.act_2_postprocess
        del self.act_3_postprocess
        del self.act_4_postprocess

    def forward(self, encoder_tokens: List[torch.Tensor], image_size=None):
        assert self.dim_tokens_enc is not None, 'Need to call init(dim_tokens_enc) function first'
        # H, W = input_info['image_size']
        image_size = self.image_size if image_size is None else image_size
        H, W = image_size
        # Number of patches in height and width
        N_H = H // (self.stride_level * self.P_H)
        N_W = W // (self.stride_level * self.P_W)

        # Hook decoder onto 4 layers from specified ViT layers
        layers = [encoder_tokens[hook] for hook in self.hooks]

        # Extract only task-relevant tokens and ignore global tokens.
        layers = [self.adapt_tokens(l) for l in layers]

        # Reshape tokens to spatial representation
        layers = [rearrange(l, 'b (nh nw) c -> b c nh nw', nh=N_H, nw=N_W) for l in layers]

        layers = [self.act_postprocess[idx](l) for idx, l in enumerate(layers)]
        # Project layers to chosen feature dim
        layers = [self.scratch.layer_rn[idx](l) for idx, l in enumerate(layers)]

        # Fuse layers using refinement stages
        path_4 = self.scratch.refinenet4(layers[3])[:, :, :layers[2].shape[2], :layers[2].shape[3]]
        path_3 = self.scratch.refinenet3(path_4, layers[2])
        path_2 = self.scratch.refinenet2(path_3, layers[1])
        path_1 = self.scratch.refinenet1(path_2, layers[0])

        # Output head
        out = self.head(path_1)

        return out


class PixelwiseTaskWithDPT(nn.Module):
    """ DPT module for dust3r, can return 3D points + confidence for all pixels"""

    def __init__(self, *, n_cls_token=0, hooks_idx=None, dim_tokens=None,
                 output_width_ratio=1, num_channels=1, postprocess=None, depth_mode=None, conf_mode=None, **kwargs):
        super(PixelwiseTaskWithDPT, self).__init__()
        self.return_all_layers = True  # backbone needs to return all layers
        self.postprocess = postprocess
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode

        assert n_cls_token == 0, "Not implemented"
        dpt_args = dict(output_width_ratio=output_width_ratio,
                        num_channels=num_channels,
                        **kwargs)
        if hooks_idx is not None:
            dpt_args.update(hooks=hooks_idx)
        self.dpt = DPTOutputAdapter_fix(**dpt_args)
        dpt_init_args = {} if dim_tokens is None else {'dim_tokens_enc': dim_tokens}
        self.dpt.init(**dpt_init_args)

    def forward(self, x, img_info):
        out = self.dpt(x, image_size=(img_info[0], img_info[1]))
        if self.postprocess:
            out = self.postprocess(out, self.depth_mode, self.conf_mode)
        return out


class Cat_MLP_LocalFeatures_DPT_Pts3d(PixelwiseTaskWithDPT):
    """ Mixture between MLP and DPT head that outputs 3d points and local features (with MLP).
    The input for both heads is a concatenation of Encoder and Decoder outputs
    """

    def __init__(self, net, has_conf=False, local_feat_dim=16, hidden_dim_factor=4., hooks_idx=None, dim_tokens=None,
                 num_channels=1, postprocess=None, feature_dim=256, last_dim=32, depth_mode=None, conf_mode=None, head_type="regression", **kwargs):
        super().__init__(num_channels=num_channels, feature_dim=feature_dim, last_dim=last_dim, hooks_idx=hooks_idx,
                         dim_tokens=dim_tokens, depth_mode=depth_mode, postprocess=postprocess, conf_mode=conf_mode, head_type=head_type)
        self.local_feat_dim = local_feat_dim

        patch_size = net.patch_embed.patch_size
        if isinstance(patch_size, tuple):
            assert len(patch_size) == 2 and isinstance(patch_size[0], int) and isinstance(
                patch_size[1], int), "What is your patchsize format? Expected a single int or a tuple of two ints."
            assert patch_size[0] == patch_size[1], "Error, non square patches not managed"
            patch_size = patch_size[0]
        self.patch_size = patch_size

        self.desc_mode = net.conf.desc_mode
        self.has_conf = has_conf
        self.two_confs = net.conf.two_confs  # independent confs for 3D regr and descs
        self.desc_conf_mode = net.conf.desc_conf_mode
        idim = net.conf.enc_embed_dim + net.conf.dec_embed_dim

        self.head_local_features = Mlp(in_features=idim,
                                       hidden_features=int(hidden_dim_factor * idim),
                                       out_features=(self.local_feat_dim + self.two_confs) * self.patch_size**2)

    def forward(self, decout, img_shape):
        # pass through the heads
        pts3d = self.dpt(decout, image_size=(img_shape[0], img_shape[1]))

        # recover encoder and decoder outputs
        enc_output, dec_output = decout[0], decout[-1]
        cat_output = torch.cat([enc_output, dec_output], dim=-1)  # concatenate
        H, W = img_shape
        B, S, D = cat_output.shape

        # extract local_features
        local_features = self.head_local_features(cat_output)  # B,S,D
        local_features = local_features.transpose(-1, -2).view(B, -1, H // self.patch_size, W // self.patch_size)
        local_features = F.pixel_shuffle(local_features, self.patch_size)  # B,d,H,W

        # post process 3D pts, descriptors and confidences
        out = torch.cat([pts3d, local_features], dim=1)
        if self.postprocess:
            out = self.postprocess(out,
                                   depth_mode=self.depth_mode,
                                   conf_mode=self.conf_mode,
                                   desc_dim=self.local_feat_dim,
                                   desc_mode=self.desc_mode,
                                   two_confs=self.two_confs,
                                   desc_conf_mode=self.desc_conf_mode)
        return out


def reg_dense_depth(xyz, mode):
    """
    extract 3D points from prediction head output
    """
    mode, vmin, vmax = mode

    no_bounds = (vmin == -float('inf')) and (vmax == float('inf'))
    assert no_bounds

    if mode == 'linear':
        if no_bounds:
            return xyz  # [-inf, +inf]
        return xyz.clip(min=vmin, max=vmax)

    # distance to origin
    d = xyz.norm(dim=-1, keepdim=True)
    xyz = xyz / d.clip(min=1e-8)

    if mode == 'square':
        return xyz * d.square()

    if mode == 'exp':
        return xyz * torch.expm1(d)

    raise ValueError(f'bad {mode=}')


def reg_dense_conf(x, mode):
    """
    extract confidence from prediction head output
    """
    mode, vmin, vmax = mode
    if mode == 'exp':
        return vmin + x.exp().clip(max=vmax-vmin)
    if mode == 'sigmoid':
        return (vmax - vmin) * torch.sigmoid(x) + vmin
    raise ValueError(f'bad {mode=}')


def reg_desc(desc, mode):
    if 'norm' in mode:
        desc = desc / desc.norm(dim=-1, keepdim=True)
    else:
        raise ValueError(f"Unknown desc mode {mode}")
    return desc


def postprocess(out, depth_mode, conf_mode, desc_dim=None, desc_mode='norm', two_confs=False, desc_conf_mode=None):
    if desc_conf_mode is None:
        desc_conf_mode = conf_mode
    fmap = out.permute(0, 2, 3, 1)  # B,H,W,D
    res = dict(pts3d=reg_dense_depth(fmap[..., 0:3], mode=depth_mode))
    if conf_mode is not None:
        res['conf'] = reg_dense_conf(fmap[..., 3], mode=conf_mode)
    if desc_dim is not None:
        start = 3 + int(conf_mode is not None)
        res['desc'] = reg_desc(fmap[..., start:start + desc_dim], mode=desc_mode)
        if two_confs:
            res['desc_conf'] = reg_dense_conf(fmap[..., start + desc_dim], mode=desc_conf_mode)
        else:
            res['desc_conf'] = res['conf'].clone()
    return res


def mast3r_head_factory(head_type, output_mode, net, has_conf=False):
    """" build a prediction head for the decoder 
    """
    if head_type == 'catmlp+dpt' and output_mode.startswith('pts3d+desc'):
        local_feat_dim = int(output_mode[10:])
        assert net.conf.dec_depth > 9
        l2 = net.conf.dec_depth
        feature_dim = 256
        last_dim = feature_dim // 2
        out_nchan = 3
        ed = net.conf.enc_embed_dim
        dd = net.conf.dec_embed_dim
        return Cat_MLP_LocalFeatures_DPT_Pts3d(net, local_feat_dim=local_feat_dim, has_conf=has_conf,
                                               num_channels=out_nchan + has_conf,
                                               feature_dim=feature_dim,
                                               last_dim=last_dim,
                                               hooks_idx=[0, l2 * 2 // 4, l2 * 3 // 4, l2],
                                               dim_tokens=[ed, dd, dd, dd],
                                               postprocess=postprocess,
                                               depth_mode=net.conf.depth_mode,
                                               conf_mode=net.conf.conf_mode,
                                               head_type='regression')
    else:
        raise NotImplementedError(
            f"unexpected {head_type=} and {output_mode=}")


def transpose_to_landscape(head, activate=True):
    """ Predict in the correct aspect-ratio,
        then transpose the result in landscape 
        and stack everything back together.
    """
    def wrapper_no(decout, true_shape):
        B = len(true_shape)
        assert true_shape[0:1].allclose(true_shape), 'true_shape must be all identical'
        H, W = true_shape[0].cpu().tolist()
        res = head(decout, (H, W))
        return res

    def wrapper_yes(decout, true_shape):
        B = len(true_shape)
        # by definition, the batch is in landscape mode so W >= H
        H, W = int(true_shape.min()), int(true_shape.max())

        height, width = true_shape.T
        is_landscape = (width >= height)
        is_portrait = ~is_landscape

        # true_shape = true_shape.cpu()
        def transposed(dic): return {k: v.swapaxes(1, 2) for k, v in dic.items()}
        if is_landscape.all():
            return head(decout, (H, W))
        if is_portrait.all():
            return transposed(head(decout, (W, H)))

        # batch is a mix of both portraint & landscape
        def selout(ar): return [d[ar] for d in decout]
        l_result = head(selout(is_landscape), (H, W))
        p_result = transposed(head(selout(is_portrait), (W, H)))

        # allocate full result
        result = {}
        for k in l_result | p_result:
            x = l_result[k].new(B, *l_result[k].shape[1:])
            x[is_landscape] = l_result[k]
            x[is_portrait] = p_result[k]
            result[k] = x

        return result

    return wrapper_yes if activate else wrapper_no


def freeze_all_params(modules):
    for module in modules:
        try:
            for n, param in module.named_parameters():
                param.requires_grad = False
        except AttributeError:
            # module is directly a parameter
            module.requires_grad = False


def top_k_mask(scores, k):
    b, h, w = scores.shape
    if k >= h * w:
        return torch.ones_like(scores, dtype=torch.bool)
    topk_scores, topk_indices = torch.topk(scores.view(b, -1), k, dim=1)
    mask = torch.zeros_like(scores, dtype=torch.bool)
    b_indices = torch.arange(b, device=scores.device).unsqueeze(1)
    mask.view(b, -1)[b_indices, topk_indices] = True
    return mask


def sample_k_mask(scores, k):
    b, h, w = scores.shape
    if k >= h * w:
        return torch.ones_like(scores, dtype=torch.bool)
    indices = torch.multinomial(scores.view(b, -1), k, replacement=False)
    mask = torch.zeros_like(scores, dtype=torch.bool)
    b_indices = torch.arange(b, device=scores.device).unsqueeze(1)
    mask.view(-1)[b_indices, indices] = True
    return mask


def get_image_coords(img):
    b, _, h, w = img.shape
    y_coords = torch.arange(h, device=img.device)  # Shape: (H,)
    x_coords = torch.arange(w, device=img.device)  # Shape: (W,)

    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")  # Shape: (H, W)
    coords = torch.stack((yy, xx), dim=-1)  # Shape: (H, W, 2)

    return coords.unsqueeze(0).expand(b, -1, -1, -1)


def sample_descriptors(keypoints, descriptors):
    """Interpolate descriptors at keypoint locations"""
    h, w, d = descriptors.shape
    keypoints = keypoints / (keypoints.new_tensor([w, h]) - 1)
    keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
    keypoints = torch.flip(keypoints, [1])  # (y, x) order
    descriptors = torch.nn.functional.grid_sample(
        descriptors.unsqueeze(0).permute(0, 3, 1, 2), keypoints.view(1, -1, 1, 2), mode="bilinear", align_corners=True
    )
    return descriptors.squeeze(0).squeeze(-1).permute(1, 0)


class MASt3R(BaseModel):
    default_conf = {
        "siamese_input": True,

        # MASt3R
        "desc_mode": ('norm'),
        "two_confs": True,
        "desc_conf_mode": ('exp', 0, float('inf')),
        "max_num_keypoints": -1,
        "max_num_keypoints_val": None,
        "force_num_keypoints": False,
        "confidence_threshold": 1.001,
        "remove_borders": None,
        "sparse_outputs": False,
        "dense_outputs": True,
        "weights": None,  # local path of pretrained weights
        "randomize_keypoints_training": False,
        # DUSt3R
        "output_mode": 'pts3d+desc24',
        "head_type": 'catmlp+dpt',
        "depth_mode": ('exp', -float('inf'), float('inf')),
        "conf_mode": ('exp', 1, float('inf')),
        "freeze": 'none',
        "landscape_only": False,
        "patch_embed_cls": 'PatchEmbedDust3R',  # PatchEmbedDust3R or ManyAR_PatchEmbed
        # CROCO
        "img_size": (512, 512),           # input image size
        "patch_size": 16,          # patch_size 
        "mask_ratio": 0.9,         # ratios of masked tokens 
        "enc_embed_dim": 1024,      # encoder feature dimension
        "enc_depth": 24,           # encoder depth 
        "enc_num_heads": 16,       # encoder number of heads in the transformer block 
        "dec_embed_dim": 768,      # decoder feature dimension 
        "dec_depth": 12,            # decoder depth 
        "dec_num_heads": 12,       # decoder number of heads in the transformer block 
        "mlp_ratio": 4,
        "norm_layer": 'LayerNorm',
        "norm_im2_in_dec": True,   # whether to apply normalization of the 'memory' = (second image) in the decoder 
        "pos_embed": 'RoPE100',       # positional embedding (either cosine or RoPE100)
    }
    required_data_keys = ["image"]

    checkpoint_url = "https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
    pe_classes = {'PatchEmbedDust3R': PatchEmbedDust3R}
    norm_layers = {
        "BatchNorm2d": partial(nn.BatchNorm2d, eps=1e-6),
        "LayerNorm": partial(nn.LayerNorm, eps=1e-6),
        "InstanceNorm2d": partial(nn.InstanceNorm2d, eps=1e-6),
        "GroupNorm": partial(nn.GroupNorm, eps=1e-6),
    }

    def _init(self, conf):
        self.conf = SimpleNamespace(**conf)
        if self.conf.desc_conf_mode is None:
            self.conf.desc_conf_mode = self.conf.conf_mode

        if conf.force_num_keypoints:
            assert conf.confidence_threshold <= 0 and conf.max_num_keypoints > 0

        ### croco specific initialization
        # Encoder
        self.patch_embed = self.pe_classes[conf.patch_embed_cls](conf.img_size, conf.patch_size, 3, conf.enc_embed_dim)

        self.mask_generator = RandomMask(self.patch_embed.num_patches, conf.mask_ratio)
        enc_pos_embed, dec_pos_embed, self.rope = get_pos_embed(conf.pos_embed)
        if self.rope is None:
            self.register_buffer('enc_pos_embed', torch.from_numpy(enc_pos_embed).float())
            self.register_buffer('dec_pos_embed', torch.from_numpy(dec_pos_embed).float())

        norm_layer = self.norm_layers[conf.norm_layer]

        enc_blocks = []
        for i in range(conf.enc_depth):
            block = Block(conf.enc_embed_dim, 
                        conf.enc_num_heads, 
                        conf.mlp_ratio, 
                        qkv_bias=True, 
                        norm_layer=norm_layer, 
                        rope=self.rope)
            enc_blocks.append(block)
        self.enc_blocks = nn.ModuleList(enc_blocks)
        self.enc_norm = norm_layer(conf.enc_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, conf.dec_embed_dim))

        # Decoder
        self.decoder_embed = nn.Linear(conf.enc_embed_dim, conf.dec_embed_dim, bias=True)
        dec_blocks = []
        for i in range(conf.dec_depth):
            block = DecoderBlock(conf.dec_embed_dim, 
                        conf.dec_num_heads, 
                        mlp_ratio=conf.mlp_ratio, 
                        qkv_bias=True, 
                        norm_layer=norm_layer, 
                        norm_mem=conf.norm_im2_in_dec, 
                        rope=self.rope)
            dec_blocks.append(block)
        self.dec_blocks = nn.ModuleList(dec_blocks)
        self.dec_norm = norm_layer(conf.dec_embed_dim)
        
        # weight initialization
        self.patch_embed._init_weights()
        if self.mask_token is not None: 
            torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(init_weights)

        # dust3r specific initialization
        self.dec_blocks2 = deepcopy(self.dec_blocks)

        # mast3r specific initialization
        # downstream head
        assert conf.img_size[0] % conf.patch_size == 0 and conf.img_size[1] % conf.patch_size == 0, f'{conf.img_size=} must be multiple of {conf.patch_size=}'
        self.downstream_head1 = mast3r_head_factory(conf.head_type, conf.output_mode, self, has_conf=bool(conf.conf_mode))
        self.downstream_head2 = mast3r_head_factory(conf.head_type, conf.output_mode, self, has_conf=bool(conf.conf_mode))
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=conf.landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=conf.landscape_only)

        self.max_pool = torch.nn.MaxPool2d(kernel_size=conf.patch_size, return_indices=True)
        self.max_unpool = torch.nn.MaxUnpool2d(conf.patch_size)

        to_be_frozen = {
            'none': [],
            'mask': [self.mask_token],
            'encoder': [self.mask_token, self.patch_embed, self.enc_blocks],
        }
        freeze_all_params(to_be_frozen[conf.freeze])

        if conf.weights is not None and Path(conf.weights).exists():
            state_dict = torch.load(conf.weights, map_location="cpu")
        else:
            state_dict = torch.hub.load_state_dict_from_url(self.checkpoint_url, map_location="cpu")['model']
        self.load_state_dict(state_dict)
        # self.load_state_dict(state_dict, strict=False) # mast3r

    def encode_image(self, image, true_shape):
        x, pos = self.patch_embed(image, true_shape=true_shape)

        for blk in self.enc_blocks:
            x = blk(x, pos)
        x = self.enc_norm(x)
        
        return x, pos

    def decode_image(self, f0, f1, pos0, pos1):
        final_output = [(f0, f1)]  # before projection

        # project to decoder dim
        f0 = self.decoder_embed(f0)
        f1 = self.decoder_embed(f1)
        
        final_output.append((f0, f1))
        for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):
            f0, _ = blk1(*final_output[-1][::+1], pos0, pos1) # img1 side
            f1, _ = blk2(*final_output[-1][::-1], pos1, pos0) # img2 side
            final_output.append((f0, f1))
        
        # normalize last output
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))
        
        return zip(*final_output)

    def _forward(self, data):
        data0, data1 = data
        image0, image1 = data0['image'], data1['image']

        # encode the two images --> B,S,D
        b, _, _, _ = image0.shape
        shape0 = data0['image_size'][:, [1, 0]] if 'image_size' in data0 else torch.tensor(image0.shape[-2:])[None].repeat(b, 1)
        shape1 = data1['image_size'][:, [1, 0]] if 'image_size' in data1 else torch.tensor(image1.shape[-2:])[None].repeat(b, 1)

        if image0.shape[-2:] == image1.shape[-2:]:
            feat, pos = self.encode_image(torch.cat((image0, image1), dim=0),
                                         torch.cat((shape0, shape1), dim=0))
            feat0, feat1 = feat.chunk(2, dim=0)
            pos0, pos1 = pos.chunk(2, dim=0)
        else:
            feat0, pos0 = self.encode_image(image0, shape0)
            feat1, pos1 = self.encode_image(image1, shape1)

        # combine all ref images into object-centric representation
        feat0, feat1 = self.decode_image(feat0, feat1, pos0, pos1)

        # dense feature description
        with torch.amp.autocast(device_type=feat0[0].device.type, enabled=False):
            res0 = self.head1([tok.float() for tok in feat0], shape0)
            res1 = self.head2([tok.float() for tok in feat1], shape1)

        pred0 = {
            # "keypoints": get_image_coords(image0),
            "keypoint_scores": res0['desc_conf'],
            "descriptors": res0['desc'],
        }
        pred1 = {
            # "keypoints": get_image_coords(image1),
            "keypoint_scores": res1['desc_conf'],
            "descriptors": res1['desc'],
        }

        if self.conf.sparse_outputs:
            max_kps = self.conf.max_num_keypoints

            # for val we allow different max_kps
            if not self.training and self.conf.max_num_keypoints_val is not None:
                max_kps = self.conf.max_num_keypoints_val

            for pred, image, data in zip([pred0, pred1], [image0, image1], [data0, data1]):
                if max_kps <= 0:
                    continue

                valid_mask = torch.ones_like(pred['keypoint_scores'], dtype=torch.bool)

                # Discard keypoints near the image borders
                if self.conf.remove_borders is not None and border >= 1:
                    valid_mask[:, : self.conf.remove_borders] = False
                    valid_mask[:, :, : self.conf.remove_borders] = False
                    valid_mask[:, -self.conf.remove_borders :] = False
                    valid_mask[:, :, -self.conf.remove_borders :] = False

                # Remove uncertain keypoints
                valid_mask &= (pred['keypoint_scores'] >= self.conf.confidence_threshold)
                
                # First, find best pixel in each 16x16 tile by non-maximum surpression,
                # input should be (n, c, h, w)
                mp_scores, mp_indices = self.max_pool(pred['keypoint_scores'] * valid_mask)
                mp_mask = self.max_unpool(mp_scores, mp_indices).bool()
                valid_mask &= mp_mask
                # Then, select k keypoints from the best pixels.
                if self.conf.randomize_keypoints_training and self.training:
                    # instead of selecting top-k, sample k by score weights
                    valid_mask &= sample_k_mask(pred['keypoint_scores'] * valid_mask, max_kps)
                else:
                    valid_mask &= top_k_mask(pred['keypoint_scores'] * valid_mask, max_kps)

                # Filtering out and listify
                keypoints = get_image_coords(image)
                scores = pred["keypoint_scores"]
                desc = pred["descriptors"]
                keypoints, scores, desc = list(
                    zip(
                        *[
                            (k[m], s[m], d[m])
                            for k, s, d, m in zip(keypoints, scores, desc, valid_mask)
                        ]
                    )
                )
                keypoints, scores, desc = list(keypoints), list(scores), list(desc)

                # Convert (h, w) to (x, y)
                keypoints = [torch.flip(k, [1]).float() for k in keypoints]

                if self.conf.force_num_keypoints:
                    pads = [max_kps - k.shape[-2] for k in keypoints]

                    keypoints = pad_and_stack(
                        keypoints,
                        max_kps,
                        -2,
                        mode="random_c",
                        bounds=(
                            0,
                            data.get("image_size", torch.tensor(image.shape[-2:]))
                            .min()
                            .item(),
                        ),
                    )
                    scores = pad_and_stack(scores, max_kps, -1, mode="zeros")

                    dense_desc = pred["descriptors"]  # b, h, w, d
                    desc = [
                        torch.cat([x, sample_descriptors(k[-p:], xn)], dim=-2) if p > 0 else x 
                        for k, x, xn, p in zip(keypoints, desc, dense_desc, pads)
                    ]
                    desc = torch.stack(desc, 0)
                    
                else:
                    keypoints = torch.stack(keypoints, 0)
                    scores = torch.stack(scores, 0)
                    desc = torch.stack(desc, 0)

                if self.conf.dense_outputs:
                    pred["dense_keypoint_scores"] = pred["keypoint_scores"]
                    pred["dense_descriptors"] = pred["descriptors"]

                pred["keypoints"] = keypoints
                pred["keypoint_scores"] = scores
                pred["descriptors"] = desc

        return pred0, pred1
        
    def loss(self, pred, data):
        raise NotImplementedError