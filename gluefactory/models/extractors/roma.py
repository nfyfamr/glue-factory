"""PyTorch implementation of the roma model,
   https://github.com/Parskatt/RoMa
"""

from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from functools import partial
import warnings
import math
from typing import Sequence, Tuple, Union, Callable, Optional, List, Any, Dict

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from torchvision import transforms
from torchvision import models as tvm
from torchvision.transforms.functional import InterpolationMode
from einops import rearrange

from ..base_model import BaseModel
from ..utils.misc import pad_and_stack


def get_autocast_params(device=None, enabled=False, dtype=None):
    if device is None:
        autocast_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        #strip :X from device
        autocast_device = str(device).split(":")[0]
    if 'cuda' in str(device):
        out_dtype = dtype
        enabled = True
    else:
        out_dtype = torch.bfloat16
        enabled = False
        # mps is not supported
        autocast_device = "cpu"
    return autocast_device, enabled, out_dtype

@torch.no_grad()
def cls_to_flow_refine(cls):
    B,C,H,W = cls.shape
    device = cls.device
    res = round(math.sqrt(C))
    G = torch.meshgrid(
        *[torch.linspace(-1+1/res, 1-1/res, steps = res, device = device) for _ in range(2)],
        indexing = 'ij'
        )
    G = torch.stack([G[1],G[0]],dim=-1).reshape(C,2)
    # FIXME: below softmax line causes mps to bug, don't know why.
    if device.type == 'mps':
        cls = cls.log_softmax(dim=1).exp()
    else:
        cls = cls.softmax(dim=1)
    mode = cls.max(dim=1).indices
    
    index = torch.stack((mode-1, mode, mode+1, mode - res, mode + res), dim = 1).clamp(0,C - 1).long()
    neighbours = torch.gather(cls, dim = 1, index = index)[...,None]
    flow = neighbours[:,0] * G[index[:,0]] + neighbours[:,1] * G[index[:,1]] + neighbours[:,2] * G[index[:,2]] + neighbours[:,3] * G[index[:,3]] + neighbours[:,4] * G[index[:,4]]
    tot_prob = neighbours.sum(dim=1)  
    flow = flow / tot_prob
    return flow

def kde(x, std = 0.1, half = True, down = None):
    # use a gaussian kernel to estimate density
    if half:
        x = x.half() # Do it in half precision TODO: remove hardcoding
    if down is not None:
        scores = (-torch.cdist(x,x[::down])**2/(2*std**2)).exp()
    else:
        scores = (-torch.cdist(x,x)**2/(2*std**2)).exp()
    density = scores.sum(dim=-1)
    return density


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x

    assert isinstance(x, int)
    return (x, x)

class PatchEmbed(nn.Module):
    """
    2D image to patch embedding: (B,C,H,W) -> (B,N,D)

    Args:
        img_size: Image size.
        patch_size: Patch token size.
        in_chans: Number of input image channels.
        embed_dim: Number of linear projection output channels.
        norm_layer: Normalization layer.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = (
            image_HW[0] // patch_HW[0],
            image_HW[1] // patch_HW[1],
        )

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        _, _, H, W = x.shape
        patch_H, patch_W = self.patch_size

        assert H % patch_H == 0, f"Input image height {H} is not a multiple of patch height {patch_H}"
        assert W % patch_W == 0, f"Input image width {W} is not a multiple of patch width: {patch_W}"

        x = self.proj(x)  # B C H W
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # B HW C
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)  # B H W C
        return x

    def flops(self) -> float:
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module

class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)

try:
    from xformers.ops import SwiGLU

    XFORMERS_AVAILABLE = True
except ImportError:
    SwiGLU = SwiGLUFFN
    XFORMERS_AVAILABLE = False

class SwiGLUFFNFused(SwiGLU):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            bias=bias,
        )

try:
    from xformers.ops import memory_efficient_attention, unbind, fmha

    XFORMERS_AVAILABLE = True
except ImportError:
    # logger.warning("xFormers not available")
    XFORMERS_AVAILABLE = False

class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            assert attn_bias is None, "xFormers is required for nested tensors usage"
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    output = x * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, Tensor] = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

try:
    from xformers.ops import fmha
    from xformers.ops import scaled_index_add, index_select_cat

    XFORMERS_AVAILABLE = True
except ImportError:
    # logger.warning("xFormers not available")
    XFORMERS_AVAILABLE = False

class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
    ) -> None:
        super().__init__()
        # print(f"biases: qkv: {qkv_bias}, proj: {proj_bias}, ffn: {ffn_bias}")
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    def forward(self, x: Tensor) -> Tensor:
        def attn_residual_func(x: Tensor) -> Tensor:
            return self.ls1(self.attn(self.norm1(x)))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x))
            x = x + self.drop_path1(ffn_residual_func(x))  # FIXME: drop_path2
        else:
            x = x + attn_residual_func(x)
            x = x + ffn_residual_func(x)
        return x

def drop_add_residual_stochastic_depth(
    x: Tensor,
    residual_func: Callable[[Tensor], Tensor],
    sample_drop_ratio: float = 0.0,
) -> Tensor:
    # 1) extract subset using permutation
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]

    # 2) apply residual_func to get residual
    residual = residual_func(x_subset)

    x_flat = x.flatten(1)
    residual = residual.flatten(1)

    residual_scale_factor = b / sample_subset_size

    # 3) add the residual
    x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    return x_plus_residual.view_as(x)

def get_branges_scales(x, sample_drop_ratio=0.0):
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    residual_scale_factor = b / sample_subset_size
    return brange, residual_scale_factor

def add_residual(x, brange, residual, residual_scale_factor, scaling_vector=None):
    if scaling_vector is None:
        x_flat = x.flatten(1)
        residual = residual.flatten(1)
        x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    else:
        x_plus_residual = scaled_index_add(
            x, brange, residual.to(dtype=x.dtype), scaling=scaling_vector, alpha=residual_scale_factor
        )
    return x_plus_residual

attn_bias_cache: Dict[Tuple, Any] = {}

def get_attn_bias_and_cat(x_list, branges=None):
    """
    this will perform the index select, cat the tensors, and provide the attn_bias from cache
    """
    batch_sizes = [b.shape[0] for b in branges] if branges is not None else [x.shape[0] for x in x_list]
    all_shapes = tuple((b, x.shape[1]) for b, x in zip(batch_sizes, x_list))
    if all_shapes not in attn_bias_cache.keys():
        seqlens = []
        for b, x in zip(batch_sizes, x_list):
            for _ in range(b):
                seqlens.append(x.shape[1])
        attn_bias = fmha.BlockDiagonalMask.from_seqlens(seqlens)
        attn_bias._batch_sizes = batch_sizes
        attn_bias_cache[all_shapes] = attn_bias

    if branges is not None:
        cat_tensors = index_select_cat([x.flatten(1) for x in x_list], branges).view(1, -1, x_list[0].shape[-1])
    else:
        tensors_bs1 = tuple(x.reshape([1, -1, *x.shape[2:]]) for x in x_list)
        cat_tensors = torch.cat(tensors_bs1, dim=1)

    return attn_bias_cache[all_shapes], cat_tensors

def drop_add_residual_stochastic_depth_list(
    x_list: List[Tensor],
    residual_func: Callable[[Tensor, Any], Tensor],
    sample_drop_ratio: float = 0.0,
    scaling_vector=None,
) -> Tensor:
    # 1) generate random set of indices for dropping samples in the batch
    branges_scales = [get_branges_scales(x, sample_drop_ratio=sample_drop_ratio) for x in x_list]
    branges = [s[0] for s in branges_scales]
    residual_scale_factors = [s[1] for s in branges_scales]

    # 2) get attention bias and index+concat the tensors
    attn_bias, x_cat = get_attn_bias_and_cat(x_list, branges)

    # 3) apply residual_func to get residual, and split the result
    residual_list = attn_bias.split(residual_func(x_cat, attn_bias=attn_bias))  # type: ignore

    outputs = []
    for x, brange, residual, residual_scale_factor in zip(x_list, branges, residual_list, residual_scale_factors):
        outputs.append(add_residual(x, brange, residual, residual_scale_factor, scaling_vector).view_as(x))
    return outputs

class NestedTensorBlock(Block):
    def forward_nested(self, x_list: List[Tensor]) -> List[Tensor]:
        """
        x_list contains a list of tensors to nest together and run
        """
        assert isinstance(self.attn, MemEffAttention)

        if self.training and self.sample_drop_ratio > 0.0:

            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.attn(self.norm1(x), attn_bias=attn_bias)

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.mlp(self.norm2(x))

            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=self.ls1.gamma if isinstance(self.ls1, LayerScale) else None,
            )
            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=self.ls2.gamma if isinstance(self.ls1, LayerScale) else None,
            )
            return x_list
        else:

            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls1(self.attn(self.norm1(x), attn_bias=attn_bias))

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls2(self.mlp(self.norm2(x)))

            attn_bias, x = get_attn_bias_and_cat(x_list)
            x = x + attn_residual_func(x, attn_bias=attn_bias)
            x = x + ffn_residual_func(x)
            return attn_bias.split(x)

    def forward(self, x_or_x_list):
        if isinstance(x_or_x_list, Tensor):
            return super().forward(x_or_x_list)
        elif isinstance(x_or_x_list, list):
            assert XFORMERS_AVAILABLE, "Please install xFormers for nested tensors usage"
            return self.forward_nested(x_or_x_list)
        else:
            raise AssertionError

class BlockChunk(nn.ModuleList):
    def forward(self, x):
        for b in self:
            x = b(x)
        return x

class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=1,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            drop_path_rate (float): stochastic depth rate
            drop_path_uniform (bool): apply uniform drop rate across blocks
            weight_init (str): weight init scheme
            init_values (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            act_layer (nn.Module): MLP activation layer
            block_fn (nn.Module): transformer block class
            ffn_layer (str): "mlp", "swiglu", "swiglufused" or "identity"
            block_chunks: (int) split block sequence into block_chunks units for FSDP wrap
        """
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        if ffn_layer == "mlp":
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            )
            for i in range(depth)
        ]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                # this is to keep the block index consistent if we chunk the block list
                chunked_blocks.append([nn.Identity()] * i + blocks_list[i : i + chunksize])
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        self.init_weights()
        for param in self.parameters():
            param.requires_grad = False
    
    @property
    def device(self):
        return self.cls_token.device

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )

        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(self, x, masks=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)

        return x

    def forward_features_list(self, x_list, masks_list):
        x = [self.prepare_tokens_with_masks(x, masks) for x, masks in zip(x_list, masks_list)]
        for blk in self.blocks:
            x = blk(x)

        all_x = x
        output = []
        for x, masks in zip(all_x, masks_list):
            x_norm = self.norm(x)
            output.append(
                {
                    "x_norm_clstoken": x_norm[:, 0],
                    "x_norm_patchtokens": x_norm[:, 1:],
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward_features(self, x, masks=None):
        if isinstance(x, list):
            return self.forward_features_list(x, masks)

        x = self.prepare_tokens_with_masks(x, masks)

        for blk in self.blocks:
            x = blk(x)

        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_patchtokens": x_norm[:, 1:],
            "x_prenorm": x,
            "masks": masks,
        }

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def _get_intermediate_layers_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        output, i, total_block_len = [], 0, len(self.blocks[-1])
        # If n is an int, take the n last blocks. If it's a list, take them
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for block_chunk in self.blocks:
            for blk in block_chunk[i:]:  # Passing the nn.Identity()
                x = blk(x)
                if i in blocks_to_take:
                    output.append(x)
                i += 1
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        if self.chunked_blocks:
            outputs = self._get_intermediate_layers_chunked(x, n)
        else:
            outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1:] for out in outputs]
        if reshape:
            B, _, w, h = x.shape
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def forward(self, *args, is_training=False, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])

def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

def vit_large(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        **kwargs,
    )
    return model


class ResNet50(nn.Module):
    def __init__(self, pretrained=False, high_res = False, weights = None, 
                 dilation = None, freeze_bn = True, anti_aliased = False, early_exit = False, amp = False, amp_dtype = torch.float16) -> None:
        super().__init__()
        if dilation is None:
            dilation = [False,False,False]
        if anti_aliased:
            pass
        else:
            if weights is not None:
                self.net = tvm.resnet50(weights = weights,replace_stride_with_dilation=dilation)
            else:
                self.net = tvm.resnet50(pretrained=pretrained,replace_stride_with_dilation=dilation)
            
        self.high_res = high_res
        self.freeze_bn = freeze_bn
        self.early_exit = early_exit
        self.amp = amp
        self.amp_dtype = amp_dtype

    def forward(self, x, **kwargs):
        autocast_device, autocast_enabled, autocast_dtype = get_autocast_params(x.device, self.amp, self.amp_dtype)
        with torch.autocast(autocast_device, enabled=autocast_enabled, dtype = autocast_dtype):
            net = self.net
            feats = {1:x}
            x = net.conv1(x)
            x = net.bn1(x)
            x = net.relu(x)
            feats[2] = x 
            x = net.maxpool(x)
            x = net.layer1(x)
            feats[4] = x 
            x = net.layer2(x)
            feats[8] = x
            if self.early_exit:
                return feats
            x = net.layer3(x)
            feats[16] = x
            x = net.layer4(x)
            feats[32] = x
            return feats

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                pass

class VGG19(nn.Module):
    def __init__(self, pretrained=False, amp = False, amp_dtype = torch.float16) -> None:
        super().__init__()
        self.layers = nn.ModuleList(tvm.vgg19_bn(pretrained=pretrained).features[:40])
        self.amp = amp
        self.amp_dtype = amp_dtype

    def forward(self, x, **kwargs):
        autocast_device, autocast_enabled, autocast_dtype = get_autocast_params(x.device, self.amp, self.amp_dtype)
        with torch.autocast(device_type=autocast_device, enabled=autocast_enabled, dtype = autocast_dtype):
            feats = {}
            scale = 1
            for layer in self.layers:
                if isinstance(layer, nn.MaxPool2d):
                    feats[scale] = x
                    scale = scale*2
                x = layer(x)
            return feats

class CNNandDinov2(nn.Module):
    def __init__(self, cnn_kwargs = None, amp = False, use_vgg = False, dinov2_weights = None, amp_dtype = torch.float16):
        super().__init__()
        if dinov2_weights is None:
            dinov2_weights = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth", map_location="cpu")
        vit_kwargs = dict(img_size= 518,
            patch_size= 14,
            init_values = 1.0,
            ffn_layer = "mlp",
            block_chunks = 0,
        )

        dinov2_vitl14 = vit_large(**vit_kwargs).eval()
        dinov2_vitl14.load_state_dict(dinov2_weights)
        cnn_kwargs = cnn_kwargs if cnn_kwargs is not None else {}
        if not use_vgg:
            self.cnn = ResNet50(**cnn_kwargs)
        else:
            self.cnn = VGG19(**cnn_kwargs)
        self.amp = amp
        self.amp_dtype = amp_dtype
        if self.amp:
            dinov2_vitl14 = dinov2_vitl14.to(self.amp_dtype)
        self.dinov2_vitl14 = [dinov2_vitl14] # ugly hack to not show parameters to DDP
    
    
    def train(self, mode: bool = True):
        return self.cnn.train(mode)
    
    def forward(self, x, upsample = False):
        B,C,H,W = x.shape
        feature_pyramid = self.cnn(x)
        
        if not upsample:
            with torch.no_grad():
                if self.dinov2_vitl14[0].device != x.device:
                    self.dinov2_vitl14[0] = self.dinov2_vitl14[0].to(x.device).to(self.amp_dtype)
                dinov2_features_16 = self.dinov2_vitl14[0].forward_features(x.to(self.amp_dtype))
                features_16 = dinov2_features_16['x_norm_patchtokens'].permute(0,2,1).reshape(B,1024,H//14, W//14)
                del dinov2_features_16
                feature_pyramid[16] = features_16
        return feature_pyramid


def get_grid(b, h, w, device):
    grid = torch.meshgrid(
        *[
            torch.linspace(-1 + 1 / n, 1 - 1 / n, n, device=device)
            for n in (b, h, w)
        ],
        indexing = 'ij'
    )
    grid = torch.stack((grid[2], grid[1]), dim=-1).reshape(b, h, w, 2)
    return grid

class TransformerDecoder(nn.Module):
    def __init__(self, blocks, hidden_dim, out_dim, is_classifier = False, *args, 
                 amp = False, pos_enc = True, learned_embeddings = False, embedding_dim = None, amp_dtype = torch.float16, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.blocks = blocks
        self.to_out = nn.Linear(hidden_dim, out_dim)
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self._scales = [16]
        self.is_classifier = is_classifier
        self.amp = amp
        self.amp_dtype = amp_dtype
        self.pos_enc = pos_enc
        self.learned_embeddings = learned_embeddings
        if self.learned_embeddings:
            self.learned_pos_embeddings = nn.Parameter(nn.init.kaiming_normal_(torch.empty((1, hidden_dim, embedding_dim, embedding_dim))))

    def scales(self):
        return self._scales.copy()

    def forward(self, gp_posterior, features, old_stuff, new_scale):
        autocast_device, autocast_enabled, autocast_dtype = get_autocast_params(gp_posterior.device, enabled=self.amp, dtype=self.amp_dtype)
        with torch.autocast(autocast_device, enabled=autocast_enabled, dtype = autocast_dtype):
            B,C,H,W = gp_posterior.shape
            x = torch.cat((gp_posterior, features), dim = 1)
            B,C,H,W = x.shape
            grid = get_grid(B, H, W, x.device).reshape(B,H*W,2)
            if self.learned_embeddings:
                pos_enc = F.interpolate(self.learned_pos_embeddings, size = (H,W), mode = 'bilinear', align_corners = False).permute(0,2,3,1).reshape(1,H*W,C)
            else:
                pos_enc = 0
            tokens = x.reshape(B,C,H*W).permute(0,2,1) + pos_enc
            z = self.blocks(tokens)
            out = self.to_out(z)
            out = out.permute(0,2,1).reshape(B, self.out_dim, H, W)
            warp, certainty = out[:, :-1], out[:, -1:]
            return warp, certainty, None


class ConvRefiner(nn.Module):
    def __init__(
        self,
        in_dim=6,
        hidden_dim=16,
        out_dim=2,
        dw=False,
        kernel_size=5,
        hidden_blocks=3,
        displacement_emb = None,
        displacement_emb_dim = None,
        local_corr_radius = None,
        corr_in_other = None,
        no_im_B_fm = False,
        amp = False,
        concat_logits = False,
        use_bias_block_1 = True,
        use_cosine_corr = False,
        disable_local_corr_grad = False,
        is_classifier = False,
        sample_mode = "bilinear",
        norm_type = nn.BatchNorm2d,
        bn_momentum = 0.1,
        amp_dtype = torch.float16,
    ):
        super().__init__()
        self.bn_momentum = bn_momentum
        self.block1 = self.create_block(
            in_dim, hidden_dim, dw=dw, kernel_size=kernel_size, bias = use_bias_block_1,
        )
        self.hidden_blocks = nn.Sequential(
            *[
                self.create_block(
                    hidden_dim,
                    hidden_dim,
                    dw=dw,
                    kernel_size=kernel_size,
                    norm_type=norm_type,
                )
                for hb in range(hidden_blocks)
            ]
        )
        self.hidden_blocks = self.hidden_blocks
        self.out_conv = nn.Conv2d(hidden_dim, out_dim, 1, 1, 0)
        if displacement_emb:
            self.has_displacement_emb = True
            self.disp_emb = nn.Conv2d(2,displacement_emb_dim,1,1,0)
        else:
            self.has_displacement_emb = False
        self.local_corr_radius = local_corr_radius
        self.corr_in_other = corr_in_other
        self.no_im_B_fm = no_im_B_fm
        self.amp = amp
        self.concat_logits = concat_logits
        self.use_cosine_corr = use_cosine_corr
        self.disable_local_corr_grad = disable_local_corr_grad
        self.is_classifier = is_classifier
        self.sample_mode = sample_mode
        self.amp_dtype = amp_dtype
        
    def create_block(
        self,
        in_dim,
        out_dim,
        dw=False,
        kernel_size=5,
        bias = True,
        norm_type = nn.BatchNorm2d,
    ):
        num_groups = 1 if not dw else in_dim
        if dw:
            assert (
                out_dim % in_dim == 0
            ), "outdim must be divisible by indim for depthwise"
        conv1 = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=num_groups,
            bias=bias,
        )
        norm = norm_type(out_dim, momentum = self.bn_momentum) if norm_type is nn.BatchNorm2d else norm_type(num_channels = out_dim)
        relu = nn.ReLU(inplace=True)
        conv2 = nn.Conv2d(out_dim, out_dim, 1, 1, 0)
        return nn.Sequential(conv1, norm, relu, conv2)
        
    def forward(self, x, y, flow, scale_factor = 1, logits = None):
        b,c,hs,ws = x.shape
        autocast_device, autocast_enabled, autocast_dtype = get_autocast_params(x.device, enabled=self.amp, dtype=self.amp_dtype)
        with torch.autocast(autocast_device, enabled=autocast_enabled, dtype = autocast_dtype):            
            x_hat = F.grid_sample(y, flow.permute(0, 2, 3, 1), align_corners=False, mode = self.sample_mode)
            if self.has_displacement_emb:
                im_A_coords = torch.meshgrid(
                (
                    torch.linspace(-1 + 1 / hs, 1 - 1 / hs, hs, device=x.device),
                    torch.linspace(-1 + 1 / ws, 1 - 1 / ws, ws, device=x.device),
                ), indexing='ij'
                )
                im_A_coords = torch.stack((im_A_coords[1], im_A_coords[0]))
                im_A_coords = im_A_coords[None].expand(b, 2, hs, ws)
                in_displacement = flow-im_A_coords
                emb_in_displacement = self.disp_emb(40/32 * scale_factor * in_displacement)
                if self.local_corr_radius:
                    if self.corr_in_other:
                        # Corr in other means take a kxk grid around the predicted coordinate in other image
                        local_corr = local_correlation(x,y,local_radius=self.local_corr_radius,flow = flow, 
                                                       sample_mode = self.sample_mode)
                    else:
                        raise NotImplementedError("Local corr in own frame should not be used.")
                    if self.no_im_B_fm:
                        x_hat = torch.zeros_like(x)
                    d = torch.cat((x, x_hat, emb_in_displacement, local_corr), dim=1)
                else:    
                    d = torch.cat((x, x_hat, emb_in_displacement), dim=1)
            else:
                if self.no_im_B_fm:
                    x_hat = torch.zeros_like(x)
                d = torch.cat((x, x_hat), dim=1)
            if self.concat_logits:
                d = torch.cat((d, logits), dim=1)
            d = self.block1(d)
            d = self.hidden_blocks(d)
        d = self.out_conv(d.float())
        displacement, certainty = d[:, :-1], d[:, -1:]
        return displacement, certainty

class CosKernel(nn.Module):  # similar to softmax kernel
    def __init__(self, T, learn_temperature=False):
        super().__init__()
        self.learn_temperature = learn_temperature
        if self.learn_temperature:
            self.T = nn.Parameter(torch.tensor(T))
        else:
            self.T = T

    def __call__(self, x, y, eps=1e-6):
        c = torch.einsum("bnd,bmd->bnm", x, y) / (
            x.norm(dim=-1)[..., None] * y.norm(dim=-1)[:, None] + eps
        )
        if self.learn_temperature:
            T = self.T.abs() + 0.01
        else:
            T = torch.tensor(self.T, device=c.device)
        K = ((c - 1.0) / T).exp()
        return K

class GP(nn.Module):
    def __init__(
        self,
        kernel,
        T=1,
        learn_temperature=False,
        only_attention=False,
        gp_dim=64,
        basis="fourier",
        covar_size=5,
        only_nearest_neighbour=False,
        sigma_noise=0.1,
        no_cov=False,
        predict_features = False,
    ):
        super().__init__()
        self.K = kernel(T=T, learn_temperature=learn_temperature)
        self.sigma_noise = sigma_noise
        self.covar_size = covar_size
        self.pos_conv = torch.nn.Conv2d(2, gp_dim, 1, 1)
        self.only_attention = only_attention
        self.only_nearest_neighbour = only_nearest_neighbour
        self.basis = basis
        self.no_cov = no_cov
        self.dim = gp_dim
        self.predict_features = predict_features

    def get_local_cov(self, cov):
        K = self.covar_size
        b, h, w, h, w = cov.shape
        hw = h * w
        cov = F.pad(cov, 4 * (K // 2,))  # pad v_q
        delta = torch.stack(
            torch.meshgrid(
                torch.arange(-(K // 2), K // 2 + 1), torch.arange(-(K // 2), K // 2 + 1),
                indexing = 'ij'),
            dim=-1,
        )
        positions = torch.stack(
            torch.meshgrid(
                torch.arange(K // 2, h + K // 2), torch.arange(K // 2, w + K // 2),
                indexing = 'ij'),
            dim=-1,
        )
        neighbours = positions[:, :, None, None, :] + delta[None, :, :]
        points = torch.arange(hw)[:, None].expand(hw, K**2)
        local_cov = cov.reshape(b, hw, h + K - 1, w + K - 1)[
            :,
            points.flatten(),
            neighbours[..., 0].flatten(),
            neighbours[..., 1].flatten(),
        ].reshape(b, h, w, K**2)
        return local_cov

    def reshape(self, x):
        return rearrange(x, "b d h w -> b (h w) d")

    def project_to_basis(self, x):
        if self.basis == "fourier":
            return torch.cos(8 * math.pi * self.pos_conv(x))
        elif self.basis == "linear":
            return self.pos_conv(x)
        else:
            raise ValueError(
                "No other bases other than fourier and linear currently im_Bed in public release"
            )

    def get_pos_enc(self, y):
        b, c, h, w = y.shape
        coarse_coords = torch.meshgrid(
            (
                torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=y.device),
                torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=y.device),
            ),
            indexing = 'ij'
        )

        coarse_coords = torch.stack((coarse_coords[1], coarse_coords[0]), dim=-1)[
            None
        ].expand(b, h, w, 2)
        coarse_coords = rearrange(coarse_coords, "b h w d -> b d h w")
        coarse_embedded_coords = self.project_to_basis(coarse_coords)
        return coarse_embedded_coords

    def forward(self, x, y, **kwargs):
        b, c, h1, w1 = x.shape
        b, c, h2, w2 = y.shape
        f = self.get_pos_enc(y)
        b, d, h2, w2 = f.shape
        x, y, f = self.reshape(x.float()), self.reshape(y.float()), self.reshape(f)
        K_xx = self.K(x, x)
        K_yy = self.K(y, y)
        K_xy = self.K(x, y)
        K_yx = K_xy.permute(0, 2, 1)
        sigma_noise = self.sigma_noise * torch.eye(h2 * w2, device=x.device)[None, :, :]
        with warnings.catch_warnings():
            K_yy_inv = torch.linalg.inv(K_yy + sigma_noise)

        mu_x = K_xy.matmul(K_yy_inv.matmul(f))
        mu_x = rearrange(mu_x, "b (h w) d -> b d h w", h=h1, w=w1)
        if not self.no_cov:
            cov_x = K_xx - K_xy.matmul(K_yy_inv.matmul(K_yx))
            cov_x = rearrange(cov_x, "b (h w) (r c) -> b h w r c", h=h1, w=w1, r=h1, c=w1)
            local_cov_x = self.get_local_cov(cov_x)
            local_cov_x = rearrange(local_cov_x, "b h w K -> b K h w")
            gp_feats = torch.cat((mu_x, local_cov_x), dim=1)
        else:
            gp_feats = mu_x
        return gp_feats

class Decoder(nn.Module):
    def __init__(
        self, embedding_decoder, gps, proj, conv_refiner, detach=False, scales="all", pos_embeddings = None,
        num_refinement_steps_per_scale = 1, warp_noise_std = 0.0, displacement_dropout_p = 0.0, gm_warp_dropout_p = 0.0,
        flow_upsample_mode = "bilinear", amp_dtype = torch.float16,
    ):
        super().__init__()
        self.embedding_decoder = embedding_decoder
        self.num_refinement_steps_per_scale = num_refinement_steps_per_scale
        self.gps = gps
        self.proj = proj
        self.conv_refiner = conv_refiner
        self.detach = detach
        if pos_embeddings is None:
            self.pos_embeddings = {}
        else:
            self.pos_embeddings = pos_embeddings
        if scales == "all":
            self.scales = ["32", "16", "8", "4", "2", "1"]
        else:
            self.scales = scales
        self.warp_noise_std = warp_noise_std
        self.refine_init = 4
        self.displacement_dropout_p = displacement_dropout_p
        self.gm_warp_dropout_p = gm_warp_dropout_p
        self.flow_upsample_mode = flow_upsample_mode
        self.amp_dtype = amp_dtype
        
    def get_placeholder_flow(self, b, h, w, device):
        coarse_coords = torch.meshgrid(
            (
                torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=device),
                torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=device),
            ),
            indexing = 'ij'
        )
        coarse_coords = torch.stack((coarse_coords[1], coarse_coords[0]), dim=-1)[
            None
        ].expand(b, h, w, 2)
        coarse_coords = rearrange(coarse_coords, "b h w d -> b d h w")
        return coarse_coords
    
    def get_positional_embedding(self, b, h ,w, device):
        coarse_coords = torch.meshgrid(
            (
                torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=device),
                torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=device),
            ),
            indexing = 'ij'
        )

        coarse_coords = torch.stack((coarse_coords[1], coarse_coords[0]), dim=-1)[
            None
        ].expand(b, h, w, 2)
        coarse_coords = rearrange(coarse_coords, "b h w d -> b d h w")
        coarse_embedded_coords = self.pos_embedding(coarse_coords)
        return coarse_embedded_coords

    def forward(self, f1, f2, gt_warp = None, gt_prob = None, upsample = False, flow = None, certainty = None, scale_factor = 1):
        coarse_scales = self.embedding_decoder.scales()
        all_scales = self.scales if not upsample else ["8", "4", "2", "1"] 
        sizes = {scale: f1[scale].shape[-2:] for scale in f1}
        h, w = sizes[1]
        b = f1[1].shape[0]
        device = f1[1].device
        coarsest_scale = int(all_scales[0])
        old_stuff = torch.zeros(
            b, self.embedding_decoder.hidden_dim, *sizes[coarsest_scale], device=f1[coarsest_scale].device
        )
        corresps = {}
        if not upsample:
            flow = self.get_placeholder_flow(b, *sizes[coarsest_scale], device)
            certainty = 0.0
        else:
            flow = F.interpolate(
                    flow,
                    size=sizes[coarsest_scale],
                    align_corners=False,
                    mode="bilinear",
                )
            certainty = F.interpolate(
                    certainty,
                    size=sizes[coarsest_scale],
                    align_corners=False,
                    mode="bilinear",
                )
        displacement = 0.0
        for new_scale in all_scales:
            ins = int(new_scale)
            corresps[ins] = {}
            f1_s, f2_s = f1[ins], f2[ins]
            if new_scale in self.proj:
                autocast_device, autocast_enabled, autocast_dtype = get_autocast_params(f1_s.device, str(f1_s)=='cuda', self.amp_dtype)
                with torch.autocast(autocast_device, enabled=autocast_enabled, dtype = autocast_dtype):
                    if not autocast_enabled:
                        f1_s, f2_s = f1_s.to(torch.float32), f2_s.to(torch.float32)
                    f1_s, f2_s = self.proj[new_scale](f1_s), self.proj[new_scale](f2_s)

            if ins in coarse_scales:
                old_stuff = F.interpolate(
                    old_stuff, size=sizes[ins], mode="bilinear", align_corners=False
                )
                gp_posterior = self.gps[new_scale](f1_s, f2_s)
                gm_warp_or_cls, certainty, old_stuff = self.embedding_decoder(
                    gp_posterior, f1_s, old_stuff, new_scale
                )
                
                if self.embedding_decoder.is_classifier:
                    flow = cls_to_flow_refine(
                        gm_warp_or_cls,
                    ).permute(0,3,1,2)
                    corresps[ins].update({"gm_cls": gm_warp_or_cls,"gm_certainty": certainty,}) if self.training else None
                else:
                    corresps[ins].update({"gm_flow": gm_warp_or_cls,"gm_certainty": certainty,}) if self.training else None
                    flow = gm_warp_or_cls.detach()
                    
            if new_scale in self.conv_refiner:
                corresps[ins].update({"flow_pre_delta": flow}) if self.training else None
                delta_flow, delta_certainty = self.conv_refiner[new_scale](
                    f1_s, f2_s, flow, scale_factor = scale_factor, logits = certainty,
                )                    
                corresps[ins].update({"delta_flow": delta_flow,}) if self.training else None
                displacement = ins*torch.stack((delta_flow[:, 0].float() / (self.refine_init * w),
                                                delta_flow[:, 1].float() / (self.refine_init * h),),dim=1,)
                flow = flow + displacement
                certainty = (
                    certainty + delta_certainty
                )  # predict both certainty and displacement
            corresps[ins].update({
                "certainty": certainty,
                "flow": flow,             
            })
            if new_scale != "1":
                flow = F.interpolate(
                    flow,
                    size=sizes[ins // 2],
                    mode=self.flow_upsample_mode,
                )
                certainty = F.interpolate(
                    certainty,
                    size=sizes[ins // 2],
                    mode=self.flow_upsample_mode,
                )
                if self.detach:
                    flow = flow.detach()
                    certainty = certainty.detach()
            #torch.cuda.empty_cache()                
        return corresps


class TupleResize(object):
    def __init__(self, size, mode=InterpolationMode.BICUBIC):
        self.size = size
        self.resize = transforms.Resize(size, mode)
    def __call__(self, im_tuple):
        return [self.resize(im) for im in im_tuple]

    def __repr__(self):
        return "TupleResize(size={})".format(self.size)

class TupleToTensorScaled(object):
    def __init__(self):
        self.to_tensor = ToTensorScaled()

    def __call__(self, im_tuple):
        return [self.to_tensor(im) for im in im_tuple]

    def __repr__(self):
        return "TupleToTensorScaled(./255)"

class ToTensorScaled(object):
    """Convert a RGB PIL Image to a CHW ordered Tensor, scale the range to [0, 1]"""

    def __call__(self, im):
        if not isinstance(im, torch.Tensor):
            im = np.array(im, dtype=np.float32).transpose((2, 0, 1))
            im /= 255.0
            return torch.from_numpy(im)
        else:
            return im

    def __repr__(self):
        return "ToTensorScaled(./255)"

class TupleNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, im_tuple):
        c,h,w = im_tuple[0].shape
        if c > 3:
            warnings.warn(f"Number of channels c={c} > 3, assuming first 3 are rgb")
        return [self.normalize(im[:3]) for im in im_tuple]

    def __repr__(self):
        return "TupleNormalize(mean={}, std={})".format(self.mean, self.std)

class TupleCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, im_tuple):
        for t in self.transforms:
            im_tuple = t(im_tuple)
        return im_tuple

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

def get_tuple_transform_ops(resize=None, normalize=True, unscale=False, clahe = False, colorjiggle_params = None):
    ops = []
    if resize:
        ops.append(TupleResize(resize))
    ops.append(TupleToTensorScaled())
    if normalize:
        ops.append(
            TupleNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )  # Imagenet mean/std
    return TupleCompose(ops)

def local_correlation(
    feature0,
    feature1,
    local_radius,
    padding_mode="zeros",
    flow = None,
    sample_mode = "bilinear",
):
    r = local_radius
    K = (2*r+1)**2
    B, c, h, w = feature0.size()
    corr = torch.empty((B,K,h,w), device = feature0.device, dtype=feature0.dtype)
    if flow is None:
        # If flow is None, assume feature0 and feature1 are aligned
        coords = torch.meshgrid(
                (
                    torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=feature0.device),
                    torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=feature0.device),
                ),
                indexing = 'ij'
                )
        coords = torch.stack((coords[1], coords[0]), dim=-1)[
            None
        ].expand(B, h, w, 2)
    else:
        coords = flow.permute(0,2,3,1) # If using flow, sample around flow target.
    local_window = torch.meshgrid(
                (
                    torch.linspace(-2*local_radius/h, 2*local_radius/h, 2*r+1, device=feature0.device),
                    torch.linspace(-2*local_radius/w, 2*local_radius/w, 2*r+1, device=feature0.device),
                ),
                indexing = 'ij'
                )
    local_window = torch.stack((local_window[1], local_window[0]), dim=-1)[
            None
        ].expand(1, 2*r+1, 2*r+1, 2).reshape(1, (2*r+1)**2, 2)
    for _ in range(B):
        with torch.no_grad():
            local_window_coords = (coords[_,:,:,None]+local_window[:,None,None]).reshape(1,h,w*(2*r+1)**2,2)
            window_feature = F.grid_sample(
                feature1[_:_+1], local_window_coords, padding_mode=padding_mode, align_corners=False, mode = sample_mode, #
            )
            window_feature = window_feature.reshape(c,h,w,(2*r+1)**2)
        corr[_] = (feature0[_,...,None]/(c**.5)*window_feature).sum(dim=0).permute(2,0,1)
    return corr

def to_pixel_coordinates(coords, H_A, W_A, H_B = None, W_B = None):
    def _to_pixel_coordinates(coords, H, W):
        kpts = torch.stack((W/2 * (coords[...,0]+1), H/2 * (coords[...,1]+1)),axis=-1)
        return kpts

    if coords.shape[-1] == 2:
        return _to_pixel_coordinates(coords, H_A, W_A) 
    
    if isinstance(coords, (list, tuple)):
        kpts_A, kpts_B = coords[0], coords[1]
    else:
        kpts_A, kpts_B = coords[...,:2], coords[...,2:]
    return _to_pixel_coordinates(kpts_A, H_A, W_A), _to_pixel_coordinates(kpts_B, H_B, W_B)

def sample(matches, certainty, num=10000, sample_mode="threshold", sample_thresh=0.05):
    if "threshold" in sample_mode:
        upper_thresh = sample_thresh
        certainty = certainty.clone()
        certainty[certainty > upper_thresh] = 1
    matches, certainty = (
        matches.reshape(-1, 4),
        certainty.reshape(-1),
    )
    expansion_factor = 4 if "balanced" in sample_mode else 1
    good_samples = torch.multinomial(certainty, 
                        num_samples = min(expansion_factor*num, len(certainty)), 
                        replacement=False)
    good_matches, good_certainty = matches[good_samples], certainty[good_samples]
    if "balanced" not in sample_mode:
        return good_matches, good_certainty
    density = kde(good_matches, std=0.1)
    p = 1 / (density+1)
    p[density < 10] = 1e-7 # Basically should have at least 10 perfect neighbours, or around 100 ok ones
    balanced_samples = torch.multinomial(p, 
                        num_samples = min(num,len(good_certainty)), 
                        replacement=False)
    return good_matches[balanced_samples], good_certainty[balanced_samples]

def match_keypoints(self, x_A, x_B, warp, certainty, sample_thresh, return_tuple = True, return_inds = False):
    x_A_to_B = F.grid_sample(warp[...,-2:].permute(2,0,1)[None], x_A[None,None], align_corners = False, mode = "bilinear")[0,:,0].mT
    cert_A_to_B = F.grid_sample(certainty[None,None,...], x_A[None,None], align_corners = False, mode = "bilinear")[0,0,0]
    D = torch.cdist(x_A_to_B, x_B)
    inds_A, inds_B = torch.nonzero((D == D.min(dim=-1, keepdim = True).values) * (D == D.min(dim=-2, keepdim = True).values) * (cert_A_to_B[:,None] > sample_thresh), as_tuple = True)
    
    if return_tuple:
        if return_inds:
            return inds_A, inds_B
        else:
            return x_A[inds_A], x_B[inds_B]
    else:
        if return_inds:
            return torch.cat((inds_A, inds_B),dim=-1)
        else:
            return torch.cat((x_A[inds_A], x_B[inds_B]),dim=-1)

def visualize_warp(self, warp, certainty, im_A = None, im_B = None, 
                    im_A_path = None, im_B_path = None, device = "cuda", symmetric = True, save_path = None, unnormalize = False):
    #assert symmetric == True, "Currently assuming bidirectional warp, might update this if someone complains ;)"
    H,W2,_ = warp.shape
    W = W2//2 if symmetric else W2
    if im_A is None:
        from PIL import Image
        im_A, im_B = Image.open(im_A_path).convert("RGB"), Image.open(im_B_path).convert("RGB")
    if not isinstance(im_A, torch.Tensor):
        im_A = im_A.resize((W,H))
        im_B = im_B.resize((W,H))    
        x_B = (torch.tensor(np.array(im_B)) / 255).to(device).permute(2, 0, 1)
        if symmetric:
            x_A = (torch.tensor(np.array(im_A)) / 255).to(device).permute(2, 0, 1)
    else:
        if symmetric:
            x_A = im_A
        x_B = im_B
    im_A_transfer_rgb = F.grid_sample(
    x_B[None], warp[:,:W, 2:][None], mode="bilinear", align_corners=False
    )[0]
    if symmetric:
        im_B_transfer_rgb = F.grid_sample(
        x_A[None], warp[:, W:, :2][None], mode="bilinear", align_corners=False
        )[0]
        warp_im = torch.cat((im_A_transfer_rgb,im_B_transfer_rgb),dim=2)
        white_im = torch.ones((H,2*W),device=device)
    else:
        warp_im = im_A_transfer_rgb
        white_im = torch.ones((H, W), device = device)
    vis_im = certainty * warp_im + (1 - certainty) * white_im
    if save_path is not None:
        from romatch.utils import tensor_to_pil
        tensor_to_pil(vis_im, unnormalize=unnormalize).save(save_path)
    return vis_im
    
            
class RoMa(BaseModel):
    default_conf = {
        "version": 'outdoor',  # indoor, outdoor, tiny_indoor, tiny_outdoor
        "siamese_input": True,
        "max_num_matches": 5000,
        # "batched": False,
        # "symmetric": True,
        "weights": None,  # local path of pretrained weights
        "dinov2_weights": None,  # local path of pretrained weights
        "coarse_res": (672, 672), 
        "upsample_preds": True,
        "upsample_res": (1344, 1344), 
        "amp_dtype": 'float16',
        "sample_mode": 'threshold_balanced',
        "attenuate_cert": True,
        "sample_thresh": 0.05,
        "decoder": {
            "displacement_dropout_p": 0.0,
            "gm_warp_dropout_p": 0.0,
        },
        "coord_decoder": {
            "feat_dim": 512,
            "cls_to_coord_res": 64,
        },        
        "gp": {
            "kernel": 'CosKernel',
            "T": 0.2,  # kernel_temperature
            "learn_temperature": False,
            "only_attention": False,
            "gp_dim": 512,
            "basis": "fourier",
            "no_cov": True,
        },
        "refiner": {
            "dw": True,
            "hidden_blocks": 8,
            "kernel_size": 5,
            "displacement_emb": "linear",
            "disable_local_corr_grad": True,
        },

    }
    required_data_keys = ["image"]

    checkpoint_url = {
        "outdoor": "https://github.com/Parskatt/storage/releases/download/roma/roma_outdoor.pth",
        "indoor": "https://github.com/Parskatt/storage/releases/download/roma/roma_indoor.pth",
        "tiny_outdoor": "https://github.com/Parskatt/storage/releases/download/roma/tiny_roma_v1_outdoor.pth",  # Unsupported
        "tiny_indoor": None,  # Unsupported
    }

    kernels = {"CosKernel": CosKernel}
    dtypes = {'float16': torch.float16}

    def _init(self, conf):
        self.conf = SimpleNamespace(**conf)

        assert conf.coarse_res[0] % 14 == 0, "Needs to be multiple of 14 for backbone"
        assert conf.coarse_res[1] % 14 == 0, "Needs to be multiple of 14 for backbone"

        # warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

        # encoder
        self.encoder = CNNandDinov2(
            cnn_kwargs = dict(
                pretrained=False,
                amp = True),
            amp = True,
            use_vgg = True,
            dinov2_weights = conf.dinov2_weights,
            amp_dtype=self.dtypes[conf.amp_dtype],
        )
        
        # decoder
        decoder_dim = conf.gp.gp_dim + conf.coord_decoder.feat_dim
        coord_decoder = TransformerDecoder(
            nn.Sequential(*[Block(decoder_dim, 8, attn_class=MemEffAttention) for _ in range(5)]), 
            decoder_dim, 
            conf.coord_decoder.cls_to_coord_res**2 + 1,
            is_classifier=True,
            amp = True,
            pos_enc = False,)

        gp_conf = {k: v for k, v in conf.gp.items() if k != 'kernel'}
        gp16 = GP(self.kernels[conf.gp.kernel], **gp_conf)
        gps = nn.ModuleDict({"16": gp16})

        proj16 = nn.Sequential(nn.Conv2d(1024, 512, 1, 1), nn.BatchNorm2d(512))
        proj8 = nn.Sequential(nn.Conv2d(512, 512, 1, 1), nn.BatchNorm2d(512))
        proj4 = nn.Sequential(nn.Conv2d(256, 256, 1, 1), nn.BatchNorm2d(256))
        proj2 = nn.Sequential(nn.Conv2d(128, 64, 1, 1), nn.BatchNorm2d(64))
        proj1 = nn.Sequential(nn.Conv2d(64, 9, 1, 1), nn.BatchNorm2d(9))
        proj = nn.ModuleDict({
            "16": proj16,
            "8": proj8,
            "4": proj4,
            "2": proj2,
            "1": proj1,
            })

        refiner_params = {
                # (in_dim,                  hidden_dim,             out_dim, dis_embed, lc_radius, cio)
            "16": (2 * 512+128+(2*7+1)**2,  2 * 512+128+(2*7+1)**2, 3, 128, 7, True),
             "8": (2 * 512+64+(2*3+1)**2,   2 * 512+64+(2*3+1)**2,  3,  64, 3, True),
             "4": (2 * 256+32+(2*2+1)**2,   2 * 256+32+(2*2+1)**2,  3,  32, 2, True),
             "2": (2 * 64+16,               128+16,                 3,  16, None, None),
             "1": (2 * 9 + 6,               24,                     3,   6, None, None),
        }
        conv_refiner = nn.ModuleDict({
            k: ConvRefiner(
                i,
                h,
                o,
                displacement_emb_dim=dis_embed,
                local_corr_radius=lc_radius,
                corr_in_other=cio,
                amp=True,
                bn_momentum=0.01,
                **conf.refiner
            )
            for k, (i, h, o, dis_embed, lc_radius, cio) in refiner_params.items()
        })

        self.decoder = Decoder(coord_decoder, 
                                gps, 
                                proj, 
                                conv_refiner, 
                                detach=True, 
                                scales=["16", "8", "4", "2", "1"],
                                **conf.decoder,)

        assert conf.version in ['outdoor', 'indoor'], f'Only "outdoor", "indoor" versions are supported.'
        if conf.weights is not None and Path(conf.weights).exists():
            state_dict = torch.load(conf.weights, map_location="cpu")
        else:
            state_dict = torch.hub.load_state_dict_from_url(self.checkpoint_url[conf.version], map_location="cpu")
        self.load_state_dict(state_dict)

    def get_output_resolution(self):
        if not self.conf.upsample_preds:
            return self.conf.coarse_res
        else:
            return self.conf.upsample_res
    
    def extract_backbone_features(self, batch, batched = True, upsample = False):
        x_q = batch["im_A"]
        x_s = batch["im_B"]
        if batched:
            X = torch.cat((x_q, x_s), dim = 0)
            feature_pyramid = self.encoder(X, upsample = upsample)
        else:
            feature_pyramid = self.encoder(x_q, upsample = upsample), self.encoder(x_s, upsample = upsample)
        return feature_pyramid

    def forward_symmetric(self, batch, batched = True, upsample = False, scale_factor = 1):
        feature_pyramid = self.extract_backbone_features(batch, batched = batched, upsample = upsample)
        f_q_pyramid = feature_pyramid
        f_s_pyramid = {
            scale: torch.cat((f_scale.chunk(2)[1], f_scale.chunk(2)[0]), dim = 0)
            for scale, f_scale in feature_pyramid.items()
        }
        corresps = self.decoder(f_q_pyramid, 
                                f_s_pyramid, 
                                upsample = upsample, 
                                **(batch["corresps"] if "corresps" in batch else {}),
                                scale_factor=scale_factor)
        return corresps

    def _forward(self, data):
        data0, data1 = data
        image0, image1 = data0['image'], data1['image']

        device = image0.device
        # batched = self.conf.batched       # batched forward is not supported
        # symmetric = self.conf.symmetric   # only use symmetric forward
        b, c, h0, w0 = image0.shape
        b, c, h1, w1 = image1.shape

        # check_rgb
        if image0.shape[1] != 3 or image1.shape[1] != 3:
            raise NotImplementedError("Can't handle non-RGB images")

        self.train(False)
        with torch.no_grad():
            b = 1

            # Get images in good format
            hs, ws = self.conf.coarse_res
            test_transform = get_tuple_transform_ops(
                resize=(hs, ws), normalize=True, clahe=False
            )
            image0, image1 = test_transform((data0['image'][0], data1['image'][0]))
            batch = {"im_A": image0[None], "im_B": image1[None]}

            finest_scale = 1

            # Run matcher
            corresps = self.forward_symmetric(batch)

            if self.conf.upsample_preds:
                hs, ws = self.conf.upsample_res

            if self.conf.attenuate_cert:
                low_res_certainty = F.interpolate(
                    corresps[16]["certainty"], size=(hs, ws), align_corners=False, mode="bilinear"
                )
                cert_clamp = 0
                factor = 0.5
                low_res_certainty = factor * low_res_certainty * (low_res_certainty < cert_clamp)

            if self.conf.upsample_preds:
                finest_corresps = corresps[finest_scale]
                torch.cuda.empty_cache()
                test_transform = get_tuple_transform_ops(
                    resize=(hs, ws), normalize=True
                )
                image0, image1 = test_transform((data0['image'][0], data1['image'][0]))
                image0, image1 = image0[None], image1[None]
                scale_factor = math.sqrt(self.conf.upsample_res[0] * self.conf.upsample_res[1] / (self.conf.coarse_res[0] * self.conf.coarse_res[1]))
                batch = {"im_A": image0, "im_B": image1, "corresps": finest_corresps}
                corresps = self.forward_symmetric(batch, upsample=True, batched=True, scale_factor=scale_factor)

            im_A_to_im_B = corresps[finest_scale]["flow"]
            certainty = corresps[finest_scale]["certainty"] - (low_res_certainty if self.conf.attenuate_cert else 0)
            if finest_scale != 1:
                im_A_to_im_B = F.interpolate(
                    im_A_to_im_B, size=(hs, ws), align_corners=False, mode="bilinear"
                )
                certainty = F.interpolate(
                    certainty, size=(hs, ws), align_corners=False, mode="bilinear"
                )
            im_A_to_im_B = im_A_to_im_B.permute(
                0, 2, 3, 1
            )
            # Create im_A meshgrid
            im_A_coords = torch.meshgrid(
                (
                    torch.linspace(-1 + 1 / hs, 1 - 1 / hs, hs, device=device),
                    torch.linspace(-1 + 1 / ws, 1 - 1 / ws, ws, device=device),
                ),
                indexing='ij'
            )
            im_A_coords = torch.stack((im_A_coords[1], im_A_coords[0]))
            im_A_coords = im_A_coords[None].expand(b, 2, hs, ws)
            certainty = certainty.sigmoid()  # logits -> probs
            im_A_coords = im_A_coords.permute(0, 2, 3, 1)
            if (im_A_to_im_B.abs() > 1).any() and True:
                wrong = (im_A_to_im_B.abs() > 1).sum(dim=-1) > 0
                certainty[wrong[:, None]] = 0
            im_A_to_im_B = torch.clamp(im_A_to_im_B, -1, 1)
            
            # symmetric
            A_to_B, B_to_A = im_A_to_im_B.chunk(2)
            q_warp = torch.cat((im_A_coords, A_to_B), dim=-1)
            im_B_coords = im_A_coords
            s_warp = torch.cat((B_to_A, im_B_coords), dim=-1)
            warp = torch.cat((q_warp, s_warp), dim=2)
            certainty = torch.cat(certainty.chunk(2), dim=3)

            dense_matches, dense_certainty = warp[0], certainty[0, 0]
            sparse_matches, sparse_certainty = sample(dense_matches, 
                                        dense_certainty, 
                                        num=self.conf.max_num_matches, 
                                        sample_mode=self.conf.sample_mode, 
                                        sample_thresh=self.conf.sample_thresh)
            keypoints0, keypoints1 = to_pixel_coordinates(sparse_matches, h0, w0, h1, w1)
            keypoints0, keypoints1 = keypoints0.unsqueeze(0), keypoints1.unsqueeze(0)

            b0, k0, c0 = keypoints0.shape
            b1, k1, c1 = keypoints1.shape

            scores0 = torch.ones(k0, device=device).unsqueeze(0)
            scores1 = scores0.clone()

            matches0 = torch.arange(k0, device=device).unsqueeze(0)
            matches1 = matches0.clone()

            mscores0 = sparse_certainty.unsqueeze(0)
            mscores1 = mscores0.clone()

            pred0 = {
                "keypoints": keypoints0,
                "keypoint_scores": scores0,
                "matches": matches0,
                "matching_scores": mscores0,
            }
            pred1 = {
                "keypoints": keypoints1,
                "keypoint_scores": scores1,
                "matches": matches1,
                "matching_scores": mscores1,
            }

            return pred0, pred1
        
    def loss(self, pred, data):
        raise NotImplementedError