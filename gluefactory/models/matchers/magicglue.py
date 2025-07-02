from math import sqrt
import warnings
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import nn
from torch.utils.checkpoint import checkpoint

from ...settings import DATA_PATH
from ..utils.losses import NLLLoss
from ..utils.metrics import matcher_metrics
from ..utils.device import dynamic_custom_fwd

FLASH_AVAILABLE = hasattr(F, "scaled_dot_product_attention")

torch.backends.cudnn.deterministic = True


@dynamic_custom_fwd(cast_inputs=torch.float32)
def normalize_keypoints(
    kpts: torch.Tensor, size: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if size is None:
        size = 1 + kpts.max(-2).values - kpts.min(-2).values
    elif not isinstance(size, torch.Tensor):
        size = torch.tensor(size, device=kpts.device, dtype=kpts.dtype)
    size = size.to(kpts)
    shift = size / 2
    scale = size.max(-1).values / 2
    kpts = (kpts - shift[..., None, :]) / scale[..., None, None]
    return kpts


@dynamic_custom_fwd(cast_inputs=torch.float32)
def denormalize_keypoints(
    kpts: torch.Tensor, size: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if size is None:
        raise ValueError("Size must be provided to denormalize keypoints.")
    elif not isinstance(size, torch.Tensor):
        size = torch.tensor(size, device=kpts.device, dtype=kpts.dtype)
    size = size.to(kpts)
    shift = size / 2
    scale = size.max(-1).values / 2
    kpts = kpts * scale[..., None, None] + shift[..., None, :]
    return kpts


@dynamic_custom_fwd(cast_inputs=torch.float32)
def get_normalization_T(size: torch.Tensor) -> torch.Tensor:
    if not isinstance(size, torch.Tensor):
        size = torch.tensor(size, device=kpts.device, dtype=kpts.dtype)
    assert size[0:1].allclose(size), "Size must be all identical"
    w, h = size[0]
    norm_T = torch.tensor([
        [2 / (w-1), 0, -1],
        [0, 2 / (h-1), -1],
        [0, 0, 1]
    ], device=size.device)
    return norm_T


@dynamic_custom_fwd(cast_inputs=torch.float32)
def normalize_keypoints2(
    kpts: torch.Tensor, size: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if size is None:
        raise ValueError("Size must be provided to normalize keypoints.")
    elif not isinstance(size, torch.Tensor):
        size = torch.tensor(size, device=kpts.device, dtype=kpts.dtype)
    assert size[0:1].allclose(size), "Size must be all identical"
    size = size.to(kpts)
    shift = ((size - 1) / 2).unsqueeze(1)
    scale = ((size - 1) / 2).unsqueeze(1)
    kpts = (kpts - shift) / scale
    return kpts


@dynamic_custom_fwd(cast_inputs=torch.float32)
def denormalize_keypoints2(
    kpts: torch.Tensor, size: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if size is None:
        raise ValueError("Size must be provided to denormalize keypoints.")
    elif not isinstance(size, torch.Tensor):
        size = torch.tensor(size, device=kpts.device, dtype=kpts.dtype)
    assert size[0:1].allclose(size), "Size must be all identical"
    size = size.to(kpts)
    shift = ((size - 1) / 2).unsqueeze(1)
    scale = ((size - 1) / 2).unsqueeze(1)
    kpts = kpts * scale + shift
    return kpts


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.unflatten(-1, (-1, 2))
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


def apply_cached_rotary_emb(freqs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return (t * freqs[0]) + (rotate_half(t) * freqs[1])


class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, M: int, dim: int, F_dim: int = None, gamma: float = 1.0) -> None:
        super().__init__()
        F_dim = F_dim if F_dim is not None else dim
        self.gamma = gamma
        self.Wr = nn.Linear(M, F_dim // 2, bias=False)
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma**-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """encode position vector"""
        projected = self.Wr(x)
        cosines, sines = torch.cos(projected), torch.sin(projected)
        emb = torch.stack([cosines, sines], 0).unsqueeze(-3)
        return emb.repeat_interleave(2, dim=-1)


class TokenConfidence(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.token = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        """get confidence tokens"""
        return (
            self.token(desc0.detach()).squeeze(-1),
            self.token(desc1.detach()).squeeze(-1),
        )

    def loss(self, desc0, desc1, la_now, la_final):
        logit0 = self.token[0](desc0.detach()).squeeze(-1)
        logit1 = self.token[0](desc1.detach()).squeeze(-1)
        la_now, la_final = la_now.detach(), la_final.detach()
        correct0 = (
            la_final[:, :-1, :].max(-1).indices == la_now[:, :-1, :].max(-1).indices
        )
        correct1 = (
            la_final[:, :, :-1].max(-2).indices == la_now[:, :, :-1].max(-2).indices
        )
        return (
            self.loss_fn(logit0, correct0.float()).mean(-1)
            + self.loss_fn(logit1, correct1.float()).mean(-1)
        ) / 2.0


class Attention(nn.Module):
    def __init__(self, allow_flash: bool) -> None:
        super().__init__()
        if allow_flash and not FLASH_AVAILABLE:
            warnings.warn(
                "FlashAttention is not available. For optimal speed, "
                "consider installing torch >= 2.0 or flash-attn.",
                stacklevel=2,
            )
        self.enable_flash = allow_flash and FLASH_AVAILABLE

        if FLASH_AVAILABLE:
            torch.backends.cuda.enable_flash_sdp(allow_flash)

    def forward(self, q, k, v, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.enable_flash and q.device.type == "cuda":
            # use torch 2.0 scaled_dot_product_attention with flash
            if FLASH_AVAILABLE:
                args = [x.half().contiguous() for x in [q, k, v]]
                v = F.scaled_dot_product_attention(*args, attn_mask=mask).to(q.dtype)
                return v if mask is None else v.nan_to_num()
        elif FLASH_AVAILABLE:
            args = [x.contiguous() for x in [q, k, v]]
            v = F.scaled_dot_product_attention(*args, attn_mask=mask)
            return v if mask is None else v.nan_to_num()
        else:
            s = q.shape[-1] ** -0.5
            sim = torch.einsum("...id,...jd->...ij", q, k) * s
            if mask is not None:
                sim.masked_fill(~mask, -float("inf"))
            attn = F.softmax(sim, -1)
            return torch.einsum("...ij,...jd->...id", attn, v)


class SelfBlock(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, flash: bool = False, bias: bool = True
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0
        self.head_dim = self.embed_dim // num_heads
        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.inner_attn = Attention(flash)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        encoding: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        qkv = self.Wqkv(x)
        qkv = qkv.unflatten(-1, (self.num_heads, -1, 3)).transpose(1, 2)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        q = apply_cached_rotary_emb(encoding, q)
        k = apply_cached_rotary_emb(encoding, k)
        context = self.inner_attn(q, k, v, mask=mask)
        message = self.out_proj(context.transpose(1, 2).flatten(start_dim=-2))
        return x + self.ffn(torch.cat([x, message], -1))


class CrossBlock(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, flash: bool = False, bias: bool = True
    ) -> None:
        super().__init__()
        self.heads = num_heads
        dim_head = embed_dim // num_heads
        self.scale = dim_head**-0.5
        inner_dim = dim_head * num_heads
        self.to_qk = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_out = nn.Linear(inner_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )
        if flash and FLASH_AVAILABLE:
            self.flash = Attention(True)
        else:
            self.flash = None

    def map_(self, func: Callable, x0: torch.Tensor, x1: torch.Tensor):
        return func(x0), func(x1)

    def forward(
        self, x0: torch.Tensor, x1: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        qk0, qk1 = self.map_(self.to_qk, x0, x1)
        v0, v1 = self.map_(self.to_v, x0, x1)
        qk0, qk1, v0, v1 = map(
            lambda t: t.unflatten(-1, (self.heads, -1)).transpose(1, 2),
            (qk0, qk1, v0, v1),
        )
        if self.flash is not None and qk0.device.type == "cuda":
            m0 = self.flash(qk0, qk1, v1, mask)
            m1 = self.flash(
                qk1, qk0, v0, mask.transpose(-1, -2) if mask is not None else None
            )
        else:
            qk0, qk1 = qk0 * self.scale**0.5, qk1 * self.scale**0.5
            sim = torch.einsum("bhid, bhjd -> bhij", qk0, qk1)
            if mask is not None:
                sim = sim.masked_fill(~mask, -float("inf"))
            attn01 = F.softmax(sim, dim=-1)
            attn10 = F.softmax(sim.transpose(-2, -1).contiguous(), dim=-1)
            m0 = torch.einsum("bhij, bhjd -> bhid", attn01, v1)
            m1 = torch.einsum("bhji, bhjd -> bhid", attn10.transpose(-2, -1), v0)
            if mask is not None:
                m0, m1 = m0.nan_to_num(), m1.nan_to_num()
        m0, m1 = self.map_(lambda t: t.transpose(1, 2).flatten(start_dim=-2), m0, m1)
        m0, m1 = self.map_(self.to_out, m0, m1)
        x0 = x0 + self.ffn(torch.cat([x0, m0], -1))
        x1 = x1 + self.ffn(torch.cat([x1, m1], -1))
        return x0, x1


class TransformerLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.self_attn = SelfBlock(*args, **kwargs)
        self.cross_attn = CrossBlock(*args, **kwargs)

    def forward(
        self,
        desc0,
        desc1,
        encoding0,
        encoding1,
        mask0: Optional[torch.Tensor] = None,
        mask1: Optional[torch.Tensor] = None,
    ):
        if mask0 is not None and mask1 is not None:
            return self.masked_forward(desc0, desc1, encoding0, encoding1, mask0, mask1)
        else:
            desc0 = self.self_attn(desc0, encoding0)
            desc1 = self.self_attn(desc1, encoding1)
            return self.cross_attn(desc0, desc1)

    # This part is compiled and allows padding inputs
    def masked_forward(self, desc0, desc1, encoding0, encoding1, mask0, mask1):
        mask = mask0 & mask1.transpose(-1, -2)
        mask0 = mask0 & mask0.transpose(-1, -2)
        mask1 = mask1 & mask1.transpose(-1, -2)
        desc0 = self.self_attn(desc0, encoding0, mask0)
        desc1 = self.self_attn(desc1, encoding1, mask1)
        return self.cross_attn(desc0, desc1, mask)


def sigmoid_log_double_softmax(
    sim: torch.Tensor, z0: torch.Tensor, z1: torch.Tensor
) -> torch.Tensor:
    """create the log assignment matrix from logits and similarity"""
    b, m, n = sim.shape
    certainties = F.logsigmoid(z0) + F.logsigmoid(z1).transpose(1, 2)
    scores0 = F.log_softmax(sim, 2)
    scores1 = F.log_softmax(sim.transpose(-1, -2).contiguous(), 2).transpose(-1, -2)
    scores = sim.new_full((b, m + 1, n + 1), 0)
    scores[:, :m, :n] = scores0 + scores1 + certainties
    scores[:, :-1, -1] = F.logsigmoid(-z0.squeeze(-1))
    scores[:, -1, :-1] = F.logsigmoid(-z1.squeeze(-1))
    return scores


class MatchAssignment(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.matchability = nn.Linear(dim, 1, bias=True)
        self.final_proj = nn.Linear(dim, dim, bias=True)

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        """build assignment matrix from descriptors"""
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        _, _, d = mdesc0.shape
        mdesc0, mdesc1 = mdesc0 / d**0.25, mdesc1 / d**0.25
        sim = torch.einsum("bmd,bnd->bmn", mdesc0, mdesc1)
        z0 = self.matchability(desc0)
        z1 = self.matchability(desc1)
        scores = sigmoid_log_double_softmax(sim, z0, z1)
        return scores, sim

    def get_matchability(self, desc: torch.Tensor):
        return torch.sigmoid(self.matchability(desc)).squeeze(-1)


def filter_matches(scores: torch.Tensor, th: float):
    """obtain matches from a log assignment matrix [Bx M+1 x N+1]"""
    max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
    m0, m1 = max0.indices, max1.indices
    indices0 = torch.arange(m0.shape[1], device=m0.device)[None]
    indices1 = torch.arange(m1.shape[1], device=m1.device)[None]
    mutual0 = indices0 == m1.gather(1, m0)
    mutual1 = indices1 == m0.gather(1, m1)
    max0_exp = max0.values.exp()
    zero = max0_exp.new_tensor(0)
    mscores0 = torch.where(mutual0, max0_exp, zero)
    mscores1 = torch.where(mutual1, mscores0.gather(1, m1), zero)
    valid0 = mutual0 & (mscores0 > th)
    valid1 = mutual1 & valid0.gather(1, m1)
    m0 = torch.where(valid0, m0, -1)
    m1 = torch.where(valid1, m1, -1)
    return m0, m1, mscores0, mscores1


def log_sigmoid_double(
    corres: torch.Tensor, z0: torch.Tensor, z1: torch.Tensor
) -> torch.Tensor:
    """create the log assignment matrix from logits and similarity"""
    b, m, n = corres.shape
    certainties = F.logsigmoid(z0) + F.logsigmoid(z1).transpose(1, 2)
    probs = corres.new_full((b, m + 1, n + 1), 0)
    probs[:, :m, :n] = F.logsigmoid(corres) + certainties
    probs[:, :-1, -1] = F.logsigmoid(-z0.squeeze(-1))
    probs[:, -1, :-1] = F.logsigmoid(-z1.squeeze(-1))
    return probs


class LooseMatchAssignment(nn.Module):
    def __init__(self, in_dim, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.matchability = nn.Linear(in_dim, 1, bias=True)
        self.final_proj0 = nn.Linear(in_dim, dim, bias=True)
        self.final_proj1 = nn.Linear(in_dim, dim, bias=True)
        self.correspondencies = nn.Linear(2 * dim, 1, bias=True)

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        """build assignment matrix from descriptors"""
        """This version generate multiple assignment for a keypoint"""
        mdesc0, mdesc1 = self.final_proj0(desc0), self.final_proj1(desc1)
        b, m, d = mdesc0.shape
        _, n, _ = mdesc1.shape
        mdesc0 = mdesc0.unsqueeze(2).expand(-1, m, n, -1)  # (b, m, n, d)
        mdesc1 = mdesc1.unsqueeze(1).expand(-1, m, n, -1)  # (b, m, n, d)
        # TODO: To retain multi-matchability, linear module instead of einsum (dot product) is applied.
        # Maybe einsum is similar accuracy, need to test it.
        feats = torch.cat([mdesc0, mdesc1], dim=-1).view(b * m * n, -1)  # (b*m*n, 2d)
        corres = self.correspondencies(feats).view(b, m, n)
        z0 = self.matchability(desc0)
        z1 = self.matchability(desc1)
        log_mutual_probs = log_sigmoid_double(corres, z0, z1)
        return log_mutual_probs, corres

    def get_matchability(self, desc: torch.Tensor):
        return torch.sigmoid(self.matchability(desc)).squeeze(-1)


def filter_matches2(probs: torch.Tensor, th: float):
    """obtain matches from a log assignment matrix [Bx M+1 x N+1]"""
    b, m, n = probs[:, :-1, :-1].shape
    mask = probs[:, :-1, :-1] > torch.tensor(th, device=probs.device).log()
    # b_idx, i_idx, j_idx = mask.nonzero(as_tuple=True)
    # m_idx = i_idx * n + j_idx
    # m0 = torch.full((b, m * n), -1, device=probs.device, dtype=torch.int32)
    # m1 = torch.full_like(m0, -1)
    # m0[b_idx, m_idx] = j_idx.to(torch.int32)
    # m1[b_idx, m_idx] = i_idx.to(torch.int32)
    # return m0, m1
    return mask


class KeyCorrection(nn.Module):
    def __init__(self, input_dim=24, hidden_dim=512, output_dim=3, bn_momentum=0.01):
        super().__init__()
        relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_dim*2, 64, kernel_size=3, padding=1),  # 48, 64
            nn.BatchNorm2d(64),
            relu,
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 64, 128
            nn.BatchNorm2d(128),
            relu,
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 128, 256
            nn.BatchNorm2d(256),
            relu,
        )
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # 4x4 -> 8x8
        self.dec1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # 256, 128
            nn.BatchNorm2d(128),
            relu,
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
        )
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # 8x8 -> 16x16
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # 128, 64
            nn.BatchNorm2d(64),
            relu,
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
        )
        self.head = nn.Conv2d(64, output_dim, kernel_size=1)

    def forward(self, feat0, feat1):
        b, d, k, p, _ = feat0.shape
        
        f = torch.cat([feat0, feat1], dim=1)  # (b, 2d, k, p, p)
        f = f.permute(0, 2, 1, 3, 4).contiguous().view(b*k, 2*d, p, p)

        e1 = self.enc1(f)  # (bk, 64, 16, 16)
        e2 = self.pool(e1)  # (bk, 64, 8, 8)

        e2 = self.enc2(e2)  # (bk, 128, 8, 8)
        e3 = self.pool(e2)  # (bk, 128, 4, 4)

        e3 = self.enc3(e3)  # (bk, 256, 4, 4)
        
        d1 = self.up1(e3)  # (bk, 128, 8, 8)
        d1 = torch.cat([d1, e2], dim=1)  # (bk, 128+128, 8, 8)
        d1 = self.dec1(d1)  # (bk, 128, 8, 8)
        
        d2 = self.up2(d1)  # (bk, 64, 16, 16)
        d2 = torch.cat([d2, e1], dim=1)  # (bk, 64+64, 16, 16)
        d2 = self.dec2(d2)  # (bk, 64, 16, 16)
        
        out = self.head(d2)  # (bk, 3, 16, 16)
        out = out.view(b, k, 3, p, p).permute(0, 1, 3, 4, 2).contiguous()  # (b, k, p, p, 3)
        offset, conf = out[..., :2], out[..., 2]
        return offset, conf


class MagicGlue(nn.Module):
    default_conf = {
        "name": "magicglue",  # just for interfacing
        "input_dim": 256,  # input descriptor dimension (autoselected from weights)
        "input_coarse_dim": 256,
        "add_scale_ori": False,
        "descriptor_dim": 256,
        "patch_size": 16,
        "n_blocks": 1,
        "n_layers": 9,
        "num_heads": 4,
        "key_sample_mode": "bilinear",
        "flash": False,  # enable FlashAttention if available.
        "mp": False,  # enable mixed precision
        "depth_confidence": -1,  # early stopping, disable with -1
        "width_confidence": -1,  # point pruning, disable with -1
        "loose_match_prob_threshold": 0.5,
        "filter_threshold": 0.0,  # match threshold
        "checkpointed": False,
        "weights": None,  # either a path or the name of pretrained weights (disk, ...)
        "weights_from_version": "v0.1_arxiv",
        "loss": {
            "gamma": 1.0,
            "fn": "nll",
            "nll_balancing": 0.5,
            "refine_conf_weight": 0.01,
        },
    }

    required_data_keys = ["keypoints0", "keypoints1", "descriptors0", "descriptors1", "dense_descriptors0", "dense_descriptors1"]

    url = "magicglue.pth"

    def __init__(self, conf) -> None:
        super().__init__()
        self.conf = conf = OmegaConf.merge(self.default_conf, conf)

        if conf.input_dim != conf.descriptor_dim:
            self.input_proj = nn.ModuleList(
                [
                    nn.Linear(conf.input_dim, conf.descriptor_dim, bias=True)
                    for _ in range(conf.n_blocks)
                ]
            )
        else:
            self.input_proj = nn.ModuleList([nn.Identity() for _ in range(conf.n_blocks)])

        head_dim = conf.descriptor_dim // conf.num_heads
        self.posenc = nn.ModuleList(
            [
                LearnableFourierPositionalEncoding(2 + 2 * conf.add_scale_ori, head_dim, head_dim)
                for _ in range(conf.n_blocks)
            ]
        )

        h, n, d = conf.num_heads, conf.n_layers, conf.descriptor_dim

        self.transformers = nn.ModuleList(
            [TransformerLayer(d, h, conf.flash) for _ in range(n)]
        )

        self.init_log_assignment = LooseMatchAssignment(conf.input_coarse_dim, conf.descriptor_dim)
        self.key_corretion = nn.ModuleList([KeyCorrection(conf.input_dim) for _ in range(conf.n_blocks)])
        self.log_assignment = nn.ModuleList([MatchAssignment(d) for _ in range(conf.n_blocks)])
        # self.token_confidence = nn.ModuleList(
        #     [TokenConfidence(d) for _ in range(n - 1)]
        # )

        self.loss_fn = NLLLoss(conf.loss)

        state_dict = None
        if conf.weights is not None:
            # weights can be either a path or an existing file from official MagicGlue
            if Path(conf.weights).exists():
                state_dict = torch.load(conf.weights, map_location="cpu")
            elif (Path(DATA_PATH) / conf.weights).exists():
                state_dict = torch.load(
                    str(DATA_PATH / conf.weights), map_location="cpu"
                )
            else:
                raise NotImplementedError("weights not found")
                # fname = (
                #     f"{conf.weights}_{conf.weights_from_version}".replace(".", "-")
                #     + ".pth"
                # )
                # state_dict = torch.hub.load_state_dict_from_url(
                #     self.url.format(conf.weights_from_version, conf.weights),
                #     file_name=fname,
                # )

        if state_dict:
            # rename old state dict entries
            for i in range(self.conf.n_layers):
                pattern = f"self_attn.{i}", f"transformers.{i}.self_attn"
                state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
                pattern = f"cross_attn.{i}", f"transformers.{i}.cross_attn"
                state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
            self.load_state_dict(state_dict, strict=False)

    def compile(self, mode="reduce-overhead"):
        if self.conf.width_confidence != -1:
            warnings.warn(
                "Point pruning is partially disabled for compiled forward.",
                stacklevel=2,
            )

        for i in range(self.conf.n_layers):
            self.transformers[i] = torch.compile(
                self.transformers[i], mode=mode, fullgraph=True
            )

    def forward(self, data: dict) -> dict:
        for key in self.required_data_keys:
            assert key in data, f"Missing key {key} in data"

        kpts0, kpts1 = data["keypoints0"], data["keypoints1"]
        b, m, _ = kpts0.shape
        b, n, _ = kpts1.shape
        device = kpts0.device
        if "view0" in data.keys() and "view1" in data.keys():
            size0 = data["view0"].get("image_size")
            size1 = data["view1"].get("image_size")
        kpts0 = normalize_keypoints2(kpts0, size0).clone()
        kpts1 = normalize_keypoints2(kpts1, size1).clone()

        if self.conf.add_scale_ori:
            sc0, o0 = data["scales0"], data["oris0"]
            sc1, o1 = data["scales1"], data["oris1"]
            kpts0 = torch.cat(
                [
                    kpts0,
                    sc0 if sc0.dim() == 3 else sc0[..., None],
                    o0 if o0.dim() == 3 else o0[..., None],
                ],
                -1,
            )
            kpts1 = torch.cat(
                [
                    kpts1,
                    sc1 if sc1.dim() == 3 else sc1[..., None],
                    o1 if o1.dim() == 3 else o1[..., None],
                ],
                -1,
            )

        desc0 = data["descriptors0"].contiguous()
        desc1 = data["descriptors1"].contiguous()

        # combine features: fine(24) + coarse(1792)
        # if "coarse_descriptors0" in data.keys() and "coarse_descriptors1" in data.keys():
        #     patch_size = self.conf.patch_size
        #     _, cd, ch0, cw0 = data["coarse_descriptors0"].shape  # d: 1024+768
        #     _, cd, ch1, cw1 = data["coarse_descriptors1"].shape

        #     coarse_desc0 = data["coarse_descriptors0"].flatten(2)
        #     coarse_desc1 = data["coarse_descriptors1"].flatten(2)

        #     kpts_coarse0 = (data["keypoints0"] // patch_size).long()
        #     kpts_coarse1 = (data["keypoints1"] // patch_size).long()
        #     kpts_coarse0_flat = kpts_coarse0[..., 1] * cw0 + kpts_coarse0[..., 0]  # (b, m)
        #     kpts_coarse1_flat = kpts_coarse1[..., 1] * cw1 + kpts_coarse1[..., 0]

        #     coarse_desc0 = torch.gather(coarse_desc0, 2, kpts_coarse0_flat.unsqueeze(1).expand(-1, cd, -1))  # (b, cd, m)
        #     coarse_desc1 = torch.gather(coarse_desc1, 2, kpts_coarse1_flat.unsqueeze(1).expand(-1, cd, -1))
        #     coarse_desc0 = coarse_desc0.transpose(1, 2)  # (b, m, cd)
        #     coarse_desc1 = coarse_desc1.transpose(1, 2)

        #     desc0 = torch.cat([desc0, coarse_desc0], dim=2)  # (b, m, xd)
        #     desc1 = torch.cat([desc1, coarse_desc1], dim=2)

        # assert desc0.shape[-1] == (self.conf.input_dim + self.conf.input_coarse_dim)
        # assert desc1.shape[-1] == (self.conf.input_dim + self.conf.input_coarse_dim)
        _, cd, ch0, cw0 = data["coarse_descriptors0"].shape
        _, cd, ch1, cw1 = data["coarse_descriptors1"].shape
        desc0 = data["coarse_descriptors0"].permute(0, 2, 3, 1).view(b, -1, cd).contiguous()  # (b, ch0*cw0, cd)
        desc1 = data["coarse_descriptors1"].permute(0, 2, 3, 1).view(b, -1, cd).contiguous()  # (b, ch0*cw0, cd)
        assert desc0.shape[-1] == self.conf.input_coarse_dim
        assert desc1.shape[-1] == self.conf.input_coarse_dim
        if torch.is_autocast_enabled():
            desc0 = desc0.half()
            desc1 = desc1.half()

        # Initial matching
        match_probs, corres = self.init_log_assignment(desc0, desc1)
        assign = filter_matches2(match_probs, self.conf.loose_match_prob_threshold)
        # m0, m1 = filter_matches2(match_probs, self.conf.loose_match_prob_threshold)
        init_match_probs = match_probs.clone()
        # init_m0 = m0.clone()
        init_assign = assign.clone()
        init_kpts0 = kpts0.clone()
        init_kpts1 = kpts1.clone()

        all_ksamples0, all_ksamples1 = [], []
        all_flow_patch, all_flow_patch_prob = [], []
        all_valid_mask0, all_valid_mask1 = [], []
        all_desc0, all_desc1 = [], []

        do_point_pruning = False
        do_early_stop = False
        # During inference, run the rest modules only with valid matches.
        # During training, run the rest modules with valid masking.
        for blk in range(self.conf.n_blocks):
            # if not self.training:
            #     if (m0 == -1).all():
            #         break
                
                # TODO: Reduce inference time workload
                # valid_indices0 = valid_mask0.nonzero(as_tuple=True)[0]
                # valid_indices1 = valid_mask1.nonzero(as_tuple=True)[0]
                # valid_mask0 = valid_mask0[valid_indices0]
                # valid_mask1 = valid_mask1[valid_indices1]
                # kpts0, kpts1 = kpts0[valid_indices0], kpts1[valid_indices1]

            # Keypoint correction
            if blk == 0:
                # with the expense of a little overhead, it is possible to use only crop_feature
                feat_crops0, kpts_smaples0 = self.crop_patch(data["dense_descriptors0"], size0)  # (b, d, m, p, p), (b, m, p, p, 2)
                feat_crops1, kpts_smaples1 = self.crop_patch(data["dense_descriptors1"], size1)

                _, d, m, p, _ = feat_crops0.shape
                k = assign.sum(dim=(-2, -1)).max().clamp(min=1)  # k: num of max matches
                k = min(k , 1024)
                f0s = torch.zeros(b, d, k, p, p, device=device, dtype=feat_crops0.dtype)
                f1s = torch.zeros(b, d, k, p, p, device=device, dtype=feat_crops1.dtype)
                k0s = torch.zeros(b, k, p, p, 2, device=device, dtype=kpts_smaples0.dtype)
                k1s = torch.zeros(b, k, p, p, 2, device=device, dtype=kpts_smaples1.dtype)
                valid_mask0 = torch.zeros(b, k, device=device, dtype=torch.bool)
                valid_mask1 = torch.zeros(b, k, device=device, dtype=torch.bool)

                for i in range(b):
                    indices = assign[i].nonzero(as_tuple=False)
                    if indices.numel() == 0:
                        m_idx = torch.tensor([0], device=device)
                        n_idx = torch.tensor([0], device=device)
                    else:
                        m_idx, n_idx = indices[:, 0], indices[:, 1]
                        valid_mask0[i, :len(m_idx)] = True
                        valid_mask1[i, :len(m_idx)] = True

                    num = min(k, len(m_idx))
                    f0s[i, :, :num] = feat_crops0[i, :, m_idx[:num], :, :]   # (d, num, p, p)
                    f1s[i, :, :num] = feat_crops1[i, :, n_idx[:num], :, :]
                    k0s[i, :num] = kpts_smaples0[i, m_idx[:num], :, :, :]  # (num, p, p, 2)
                    k1s[i, :num] = kpts_smaples1[i, n_idx[:num], :, :, :]

                    y0, x0 = m_idx // cw0, m_idx % cw0
                    y1, x1 = n_idx // cw1, n_idx % cw1
                
                # TODO: update logic more robust
                # Current trick resort to the fact that the num of keypoints will not over 768
                kpts_smaples0 = k0s[:, :k]
                kpts_smaples1 = k1s[:, :k]
                kpts0 = kpts0[:, :k]
                kpts1 = kpts1[:, :k]
                valid_mask0 = valid_mask0[:, :k]
                valid_mask1 = valid_mask1[:, :k]
                f0s = f0s[:, :, :k, :, :]
                f1s = f1s[:, :, :k, :, :]
            else:
                valid_mask0 = (m0 > -1)
                valid_mask1 = torch.zeros_like(m1, dtype=torch.bool)
                rows, cols = valid_mask0.nonzero(as_tuple=True)
                valid_mask1[rows, m0[rows, cols]] = True

                feat_crops0, kpts_smaples0 = self.crop_feature(kpts0, data["dense_descriptors0"], size0)  # (b, d, m, p, p), (b, m, p, p, 2)
                feat_crops1, kpts_smaples1 = self.crop_feature(kpts1, data["dense_descriptors1"], size1)

                f0s = feat_crops0.contiguous()
                f1s = feat_crops1.permute(0, 2, 1, 3, 4).contiguous()  # (b, d, m, p, p) -> (b, m, d, p, p)
                f1s = f1s[torch.arange(b)[:, None], m0, ...]
                f1s = f1s.permute(0, 2, 1, 3, 4).contiguous()  # (b, m, d, p, p) -> (b, d, m, p, p)

            all_valid_mask0.append(valid_mask0)
            all_valid_mask1.append(valid_mask1)

            flow_patch0to1, flow_patch_prob = self.key_corretion[blk](f0s, f1s)  # (b, k, p, p, 2), (b, k, p, p)
            # flow_patch0to1 = torch.zeros((b, k, p, p, 2), device=f0s.device, dtype=f0s.dtype)
            # flow_patch_prob = torch.zeros((b, k, p, p), device=f0s.device, dtype=f0s.dtype)
            # chunk_size = 512
            # print(k)
            # for start in range(0, k, chunk_size):
            #     end = min(start + chunk_size, k)
            #     flow, prob = self.key_corretion[blk](f0s[:, :, start:end], f1s[:, :, start:end])
            #     flow_patch0to1[:, start:end] = flow
            #     flow_patch_prob[:, start:end] = prob

            all_ksamples0.append(kpts_smaples0)
            all_ksamples1.append(kpts_smaples1)
            all_flow_patch.append(flow_patch0to1)
            all_flow_patch_prob.append(flow_patch_prob)

            # TODO: consider blk >= 1
            # shift0, shift1 = self.get_key_shifts(flow_patch0to1.detach(), flow_patch_prob.detach(), size0)
            if blk == 0:
            #     kpts0[valid_mask0] = (kpts0 + shift0)[valid_mask0]
            #     kpts1[valid_mask1] = (kpts1 + shift1)[valid_mask1]
                flow_patch_prob_flat = flow_patch_prob.view(b, k, -1)  # (b, k, p*p)
                max_idx = flow_patch_prob_flat.argmax(dim=-1)  # (b, m)
                h_idx = max_idx // p   # row index
                w_idx = max_idx % p    # col index
                batch_idx = torch.arange(b).unsqueeze(1).expand(b, k)  # (b, k)
                k_idx = torch.arange(k).unsqueeze(0).expand(b, k)      # (b, k)
                kpts0 = kpts_smaples0[batch_idx, k_idx, h_idx, w_idx]  # (b, k, 2)
                kpts1 = kpts_smaples1[batch_idx, k_idx, h_idx, w_idx] + flow_patch0to1[batch_idx, k_idx, h_idx, w_idx]  # (b, k, 2)
            else:
                shift0, shift1 = self.get_key_shifts(flow_patch0to1.detach(), flow_patch_prob.detach(), size0)
                kpts0[valid_mask0] = (kpts0 + shift0)[valid_mask0]
                # TODO: Check again kpts1 update logic.
                batch_idx = torch.arange(b, device=kpts1.device)[:, None].expand_as(m0)
                valid0_flat = valid_mask0.view(-1)
                flat_batch_idx = batch_idx.reshape(-1)[valid0_flat]
                flat_kpt_idx = m0.reshape(-1)[valid0_flat]
                flat_shift1 = shift1.reshape(-1, shift1.shape[-1])[valid0_flat]
                kpts1[flat_batch_idx, flat_kpt_idx] += flat_shift1

            desc0 = F.grid_sample(
                data["dense_descriptors0"].permute(0, 3, 1, 2).contiguous(),  # (b, d, h, w)
                kpts0.unsqueeze(2),  # (b, m, 1, 2)
                mode=self.conf.key_sample_mode,
                padding_mode="zeros",
                align_corners=True,
            ).squeeze(-1).permute(0, 2, 1).contiguous()  # (b, m, d)
            desc1 = F.grid_sample(
                data["dense_descriptors1"].permute(0, 3, 1, 2).contiguous(),  # (b, d, h, w)
                kpts1.unsqueeze(2),  # (b, n, 1, 2)
                mode=self.conf.key_sample_mode,
                padding_mode="zeros",
                align_corners=True,
            ).squeeze(-1).permute(0, 2, 1).contiguous()  # (b, n, d)

            # Fine matching
            desc0 = self.input_proj[blk](desc0)
            desc1 = self.input_proj[blk](desc1)
            # cache positional embeddings
            encoding0 = self.posenc[blk](kpts0)
            encoding1 = self.posenc[blk](kpts1)

            # GNN + final_proj + assignment
            # do_early_stop = self.conf.depth_confidence > 0 and not self.training
            # do_point_pruning = self.conf.width_confidence > 0 and not self.training
            # TODO: handle point pruning and early stopping
            # do_point_pruning = False
            # do_early_stop = False

            # do not prune
            # if do_point_pruning:
            #     ind0 = torch.arange(0, m, device=device)[None]
            #     ind1 = torch.arange(0, n, device=device)[None]
            #     # We store the index of the layer at which pruning is detected.
            #     prune0 = torch.ones_like(ind0)
            #     prune1 = torch.ones_like(ind1)
            token0, token1 = None, None
            for i in range(self.conf.n_layers):
                if self.conf.checkpointed and self.training:
                    desc0, desc1 = checkpoint(
                        self.transformers[i], desc0, desc1, encoding0, encoding1, valid_mask0[:, None, :, None], valid_mask1[:, None, :, None], use_reentrant=False
                    )
                else:
                    desc0, desc1 = self.transformers[i](desc0, desc1, encoding0, encoding1, valid_mask0[:, None, :, None], valid_mask1[:, None, :, None])
                # if self.training or i == self.conf.n_layers - 1:
            all_desc0.append(desc0)
            all_desc1.append(desc1)
                # continue  # no early stopping or adaptive width at last layer

            # only for eval
            # we conduct early stopping and point pruning at each end of blocks.
            # if do_early_stop:
            #     assert b == 1
            #     token0, token1 = self.token_confidence[i](desc0, desc1)
            #     if self.check_if_stop(token0[..., :m, :], token1[..., :n, :], i, m + n):
            #         break
            # if do_point_pruning:
            #     assert b == 1
            #     scores0 = self.log_assignment[i].get_matchability(desc0)
            #     prunemask0 = self.get_pruning_mask(token0, scores0, i)
            #     keep0 = torch.where(prunemask0)[1]
            #     ind0 = ind0.index_select(1, keep0)
            #     desc0 = desc0.index_select(1, keep0)
            #     encoding0 = encoding0.index_select(-2, keep0)
            #     prune0[:, ind0] += 1
            #     scores1 = self.log_assignment[i].get_matchability(desc1)
            #     prunemask1 = self.get_pruning_mask(token1, scores1, i)
            #     keep1 = torch.where(prunemask1)[1]
            #     ind1 = ind1.index_select(1, keep1)
            #     desc1 = desc1.index_select(1, keep1)
            #     encoding1 = encoding1.index_select(-2, keep1)
            #     prune1[:, ind1] += 1

            # desc0, desc1 = desc0[..., :m, :], desc1[..., :n, :]
            scores, _ = self.log_assignment[blk](desc0, desc1)
            m0, m1, mscores0, mscores1 = filter_matches(scores, self.conf.filter_threshold)
            m0[~valid_mask0] = -1
            m1[~valid_mask1] = -1

        if do_point_pruning:
            m0_ = torch.full((b, m), -1, device=m0.device, dtype=m0.dtype)
            m1_ = torch.full((b, n), -1, device=m1.device, dtype=m1.dtype)
            m0_[:, ind0] = torch.where(m0 == -1, -1, ind1.gather(1, m0.clamp(min=0)))
            m1_[:, ind1] = torch.where(m1 == -1, -1, ind0.gather(1, m1.clamp(min=0)))
            mscores0_ = torch.zeros((b, m), device=mscores0.device)
            mscores1_ = torch.zeros((b, n), device=mscores1.device)
            mscores0_[:, ind0] = mscores0
            mscores1_[:, ind1] = mscores1
            m0, m1, mscores0, mscores1 = m0_, m1_, mscores0_, mscores1_
        else:
            prune0 = torch.ones_like(mscores0) * self.conf.n_layers
            prune1 = torch.ones_like(mscores1) * self.conf.n_layers

        pred = {
            "keypoints0": denormalize_keypoints2(kpts0, size0),
            "keypoints1": denormalize_keypoints2(kpts1, size1),
            "matches0": m0,
            "matches1": m1,
            "matching_scores0": mscores0,
            "matching_scores1": mscores1,
            "init_keypoints0": denormalize_keypoints2(init_kpts0, size0),
            "init_keypoints1": denormalize_keypoints2(init_kpts1, size1),
            "init_assign": init_assign,
            "init_log_assignment": init_match_probs,
            "valid_mask0": torch.stack(all_valid_mask0, 1),
            "valid_mask1": torch.stack(all_valid_mask1, 1),
            # TODO: This is not intermediate keypoints nomore. Rather, it is patch grids.
            "intermediate_ksamples0": denormalize_keypoints2(torch.stack(all_ksamples0, 1).view(b, -1, 2), size0) if len(all_ksamples0) > 0 else None,
            "intermediate_ksamples1": denormalize_keypoints2(torch.stack(all_ksamples1, 1).view(b, -1, 2), size1) if len(all_ksamples1) > 0 else None,
            "intermediate_flow": torch.stack(all_flow_patch, 1).view(b, -1, 2) if len(all_flow_patch) > 0 else None,
            "intermediate_flow_conf": torch.stack(all_flow_patch_prob, 1).view(b, -1) if len(all_flow_patch_prob) > 0 else None,
            "ref_descriptors0": torch.stack(all_desc0, 1) if len(all_desc0) > 0 else None,
            "ref_descriptors1": torch.stack(all_desc1, 1) if len(all_desc1) > 0 else None,
            "log_assignment": scores,
            "prune0": prune0,
            "prune1": prune1,
        }

        return pred

    def crop_feature(self, kpts, desc, size: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Crop 16x16 features in original scale.
        """
        b, n, _ = kpts.shape
        _, h, w, d = desc.shape

        if size is None:
            size = 1 + kpts.max(-2).values - kpts.min(-2).values
        elif not isinstance(size, torch.Tensor):
            size = torch.tensor(size, device=kpts.device, dtype=kpts.dtype)
        size = size.to(kpts)

        patch_size = self.conf.patch_size
        half_patch = patch_size // 2
        patch_grid = torch.meshgrid(
            torch.linspace(-half_patch, half_patch, patch_size, device=kpts.device),
            torch.linspace(-half_patch, half_patch, patch_size, device=kpts.device),
            indexing='xy'
        )
        patch_grid = torch.stack(patch_grid, dim=-1)  # (p, p, 2)
        
        # Normalize grid displacements to feature map scale
        patch_grid = patch_grid.expand(b, n, -1, -1, -1)  # (p, p, 2) -> (b, n, p, p, 2)
        patch_grid = patch_grid * 2.0 / size[:, None, None, None, :]
        
        # Add keypoint centers to grid
        sampling_grid = patch_grid + kpts.unsqueeze(2).unsqueeze(2)  # (b, n, p, p, 2)
        sampling_grid = sampling_grid.view(b, n*patch_size*patch_size, 1, 2)  # (b, n*p*p, 1, 2)
        
        patches = torch.zeros(b, d, n, patch_size, patch_size, device=kpts.device)  # (b, d, n, p, p)
        desc = desc.permute(0, 3, 1, 2).contiguous()  # (b, d, h, w)
        for i in range(b):
            batch_patches = F.grid_sample(
                desc[i].unsqueeze(0),  # (1, d, h, w)
                sampling_grid[i].unsqueeze(0),  # (1, n*p*p, 1, 2)
                mode=self.conf.key_sample_mode,
                padding_mode="zeros",
                align_corners=True,
            )  # (1, d, n*patch_size*patch_size, 1)
            patches[i] = batch_patches.view(d, n, patch_size, patch_size)
        return patches, sampling_grid.view(b, n, patch_size, patch_size, 2)

    def crop_patch(self, desc, size: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Crop 16x16 features in original scale.
        """
        b, h, w, d = desc.shape

        p = self.conf.patch_size
        ph, pw = h // p, w // p
        n = ph * pw

        # TODO: use geometry.utils.get_image_coords(img)
        # patch_grid = get_image_coords(desc.permute(0, 3, 1, 2)) - 0.5  # Debias half-pixel Offset, (1, h, w, 2)
        patch_grid = torch.meshgrid(
            torch.linspace(-1, 1, w, device=desc.device),
            torch.linspace(-1, 1, h, device=desc.device),
            indexing='xy'
        )
        patch_grid = torch.stack(patch_grid, dim=-1)  # (h, w, 2)
        patch_grid = patch_grid.view(ph, p, pw, p, 2).permute(0, 2, 1, 3, 4).contiguous()  # (ph, pw, p, p, 2)
        patch_grid = patch_grid.view(n, p, p, 2)  # (n, p, p, 2)
        patch_grid = patch_grid.unsqueeze(0).expand(b, -1, -1, -1, -1)

        desc = desc.permute(0, 3, 1, 2).contiguous()  # (b, d, h, w)
        patches = F.unfold(desc, kernel_size=p, stride=p)  # (b, d*p*p, n)
        patches = patches.view(b, d, p, p, -1)  # (b, d, p, p, n)
        patches = patches.permute(0, 1, 4, 2, 3)  # (b, d, n, p, p)

        return patches, patch_grid

    def get_key_shifts(self, flow_patch0to1, flow_patch_prob, size) -> torch.Tensor:
        b, m, p, _ = flow_patch_prob.shape
        flow_patch_prob_flat = flow_patch_prob.view(b, m, -1)  # (b, m, p*p)
        max_idx = flow_patch_prob_flat.argmax(dim=2)  # (b, m)
        h_idx = max_idx // p   # row index
        w_idx = max_idx % p    # col index
        batch_idx = torch.arange(b).unsqueeze(1).expand(b, m)  # (b, m)
        n_idx = torch.arange(m).unsqueeze(0).expand(b, m)      # (b, m)

        size = size.to(flow_patch0to1)

        half_patch = p // 2
        patch_grid = torch.meshgrid(
            torch.linspace(-half_patch, half_patch, p, device=flow_patch0to1.device),
            torch.linspace(-half_patch, half_patch, p, device=flow_patch0to1.device),
            indexing='xy'
        )
        patch_grid = torch.stack(patch_grid, dim=-1)  # (p, p, 2)
        
        # Normalize grid displacements to feature map scale
        patch_grid = patch_grid.expand(b, m, -1, -1, -1)  # (p, p, 2) -> (b, m, p, p, 2)
        patch_grid = patch_grid * 2.0 / size[:, None, None, None, :]
        
        shift0 = patch_grid[batch_idx, n_idx, h_idx, w_idx]  # (b, m, 2)
        shift1 = shift0 + flow_patch0to1[batch_idx, n_idx, h_idx, w_idx]  # (b, m, p, p, 2)
        return shift0, shift1
        
    def confidence_threshold(self, layer_index: int) -> float:
        """scaled confidence threshold"""
        threshold = 0.8 + 0.1 * np.exp(-4.0 * layer_index / self.conf.n_layers)
        return np.clip(threshold, 0, 1)

    def get_pruning_mask(
        self, confidences: torch.Tensor, scores: torch.Tensor, layer_index: int
    ) -> torch.Tensor:
        """mask points which should be removed"""
        keep = scores > (1 - self.conf.width_confidence)
        if confidences is not None:  # Low-confidence points are never pruned.
            keep |= confidences <= self.confidence_thresholds[layer_index]
        return keep

    def check_if_stop(
        self,
        confidences0: torch.Tensor,
        confidences1: torch.Tensor,
        layer_index: int,
        num_points: int,
    ) -> torch.Tensor:
        """evaluate stopping condition"""
        confidences = torch.cat([confidences0, confidences1], -1)
        threshold = self.confidence_thresholds[layer_index]
        ratio_confident = 1.0 - (confidences < threshold).float().sum() / num_points
        return ratio_confident > self.conf.depth_confidence

    def pruning_min_kpts(self, device: torch.device):
        if self.conf.flash and FLASH_AVAILABLE and device.type == "cuda":
            return self.pruning_keypoint_thresholds["flash"]
        else:
            return self.pruning_keypoint_thresholds[device.type]

    def loss(self, pred, data):
        def loss_params(pred, i):
            la, _ = self.log_assignment[i](
                pred["ref_descriptors0"][:, i], pred["ref_descriptors1"][:, i]
            )
            return {
                "log_assignment": la,
            }

        device = pred["init_log_assignment"].device

        zero = pred["init_log_assignment"].new_tensor([0])
        nll_init, loss_refine, loss_light = zero.clone(), zero.clone(), zero.clone()
        
        # nll_init, _, loss_metrics_init = self.loss_fn({"log_assignment": pred["init_log_assignment"]}, data["gt_init"])
        # loss_metrics_init = {f"{k}_init": v for k, v in loss_metrics_init.items()}
        pos_w = torch.prod(torch.tensor(data["gt_init"]["gt_assignment"].shape[-2:])) / data["gt_init"]["gt_assignment"].float().sum((-1, -2)).view(-1, 1, 1)
        nll_pos = -(pos_w * data["gt_init"]["gt_assignment"].float() * pred["init_log_assignment"][:, :-1, :-1] + (1 - data["gt_init"]["gt_assignment"].float()) * torch.log1p(-torch.exp(pred["init_log_assignment"][:, :-1, :-1]))).mean((-1, -2))
        # Calculate negative labels
        b, cd, ch0, cw0 = data["coarse_descriptors0"].shape
        _, cd, ch1, cw1 = data["coarse_descriptors1"].shape
        desc0 = data["coarse_descriptors0"].permute(0, 2, 3, 1).view(b, -1, cd).contiguous()  # (b, ch0*cw0, cd)
        desc1 = data["coarse_descriptors1"].permute(0, 2, 3, 1).view(b, -1, cd).contiguous()  # (b, ch0*cw0, cd)
        if torch.is_autocast_enabled():
            desc0 = desc0.half()
            desc1 = desc1.half()
        true_z0 = data["gt_init"]["gt_assignment"].any(dim=2)
        true_z1 = data["gt_init"]["gt_assignment"].any(dim=1)
        z0, z1 = self.init_log_assignment.get_matchability(desc0), self.init_log_assignment.get_matchability(desc1)
        num_neg0 = (~true_z0).sum(-1).clamp(min=1.0)
        num_neg1 = (~true_z1).sum(-1).clamp(min=1.0)
        nll_neg = -(((~true_z0) * torch.log1p(-z0)).sum(-1) / num_neg0 + ((~true_z1) * torch.log1p(-z1)).sum(-1) / num_neg1) / 2
        nll_init = nll_init + nll_pos
        loss_metrics_init = {}
        
        # Regression loss
        def get_refine_loss(pred, data, i):
            b, blk_kpp, _ = pred["intermediate_flow"].shape
            _, m, n = pred["init_assign"].shape
            blk = self.conf.n_blocks
            kpp = blk_kpp // blk  # k: max num of keypoints over batch
            p = self.conf.patch_size
            pp = p**2
            k = kpp // pp
            # m0 = pred["init_assign"]  # (b, m)

            ################ patch_center1 + patch_grids1 + flow
            # valid_mask0 = pred["valid_mask0"][:, i]  # (b, k)
            # init_keypoints1 = normalize_keypoints2(pred["init_keypoints1"], data["view1"].get("image_size"))  # (b, k, 2)
            # init_keypoints1 = init_keypoints1.unsqueeze(-2).expand(-1, -1, pp, -1)  # (b, k, pp, 2)
            patch_grids1 = normalize_keypoints2(pred["intermediate_ksamples1"], data["view1"].get("image_size"))
            patch_grids1 = patch_grids1.view(b, blk, k, pp, 2)[:, i]  # (b, k, pp, 2)
            patch_warps0_1 = pred["intermediate_flow"].view(b, blk, k, pp, 2)[:, i]  # (b, k, pp, 2)

            # patch_warps0_1 = patch_warps0_1 + init_keypoints1 + patch_grids1  # (b, k, pp, 2)
            patch_warps0_1 = patch_warps0_1 + patch_grids1  # (b, k, pp, 2)
            patch_warps0_1 = patch_warps0_1.view(b, kpp, 2)
            
            ### copy from get_key_shifts...
            # half_patch = p / 2
            # patch_grid = torch.meshgrid(
            #     torch.linspace(-half_patch, half_patch, p, device=patch_warps0_1.device),
            #     torch.linspace(-half_patch, half_patch, p, device=patch_warps0_1.device),
            #     indexing='xy'
            # )
            # patch_grid = torch.stack(patch_grid, dim=-1)  # (p, p, 2)
            
            # # Normalize grid displacements to feature map scale
            # patch_grid = patch_grid.expand(b, m, -1, -1, -1)  # (p, p, 2) -> (b, m, p, p, 2)
            # patch_grid = patch_grid * 2.0 / data["view0"].get("image_size")[:, None, None, None, :]
            
            # shift1 = patch_grid.view(b, -1, 2) + pred["intermediate_flow"].view(b, blk, k, pp, 2)[:, i]
            # ###

            # patch_warps0_1 = shift1 + normalize_keypoints2(patch_warps0_1, data["view1"].get("image_size"))  # (b, k, 2)
            # TODO: in float16 mode, the generated data["gt_patch_warps0_1"] contains inf values due to range of float16.
            # Need to handle this case, possibly by generate it in normalized coordinates.
            epe = (patch_warps0_1 - normalize_keypoints2(data["gt_patch_warps0_1"].view(b, blk, kpp, 2)[:, i], data["view1"].get("image_size"))).norm(dim=-1)  # (b, kpp)
            x = epe * (data["gt_patch_warps0_1_prob"].view(b, blk, kpp)[:, i] > 0.99).float()
            reg_loss = x**2
            conf_loss = F.binary_cross_entropy_with_logits(
                pred["intermediate_flow_conf"].view(b, blk, kpp)[:, i],
                data["gt_patch_warps0_1_prob"].view(b, blk, kpp)[:, i],
                reduction="none",
            )
            valid_mask0 = pred["valid_mask0"][:, i].unsqueeze(-1).expand(-1, -1, pp)
            valid_count0 = valid_mask0.sum(dim=-1).sum(dim=-1).clamp(min=1)
            reg_loss = reg_loss.view(b, k, pp) * valid_mask0
            reg_loss = reg_loss.view(b, -1).sum(dim=-1) / valid_count0
            conf_loss = conf_loss.view(b, k, pp) * valid_mask0
            conf_loss = conf_loss.view(b, -1).sum(dim=-1) / valid_count0
            return reg_loss, conf_loss

        reg_loss, conf_loss = zero.clone(), zero.clone()
        row_norm = zero.clone()
        loss_light_last_blk = zero.clone()
        loss_metrics = {}
        if self.training or pred["intermediate_ksamples0"] is not None:
            sum_weights = 1.0
            N = self.conf.n_blocks
            for i in range(N):
                reg_loss, conf_loss = get_refine_loss(pred, data, i)
                loss_refine = loss_refine + reg_loss + self.conf.loss.refine_conf_weight * conf_loss
            loss_refine = loss_refine / N

            sum_weights = 1.0
            nll, gt_weights, loss_metrics = self.loss_fn(loss_params(pred, -1), data)
            L = pred["ref_descriptors0"].shape[1]
            loss_light_last_blk = nll.clone().detach()
            loss_light = nll

            # if self.training:
            #     losses["confidence"] = 0.0
            confidence = 0.0

            # B = pred['log_assignment'].shape[0]
            row_norm = pred["log_assignment"].exp()[:, :-1].sum(2).mean(1)
            for i in range(L - 1):
                params_i = loss_params(pred, i)
                nll, _, _ = self.loss_fn(params_i, data, weights=gt_weights)

                if self.conf.loss.gamma > 0.0:
                    weight = self.conf.loss.gamma ** (L - i - 1)
                else:
                    weight = i + 1
                sum_weights += weight
                loss_light = loss_light + nll * weight

                # confidence += self.token_confidence[i].loss(
                #     pred["ref_descriptors0"][:, i],
                #     pred["ref_descriptors1"][:, i],
                #     params_i["log_assignment"],
                #     pred["log_assignment"],
                # ) / (L - 1)

                del params_i
            loss_light /= sum_weights

            # confidences
            if self.training:
                loss_light = loss_light + confidence

        losses = {
            "total": 10 * nll_init + loss_refine + loss_light,
            "nll_init": nll_init,
            # "num_init_matches": pred["valid_mask0"][:, 0].sum(-1).float(),
            "loss_refine": loss_refine,
            "reg_loss": reg_loss,
            "conf_loss": conf_loss,
            "loss_light": loss_light,
            "row_norm": row_norm,
            "last": loss_light_last_blk,
            "init_peak_match": pred["init_assign"].sum(dim=(-2, -1)).max().float().expand(1),
            **loss_metrics,
            **loss_metrics_init,
        }

        if not self.training:
            # add metrics
            metrics = matcher_metrics(pred, data)
        else:
            metrics = {}
        return losses, metrics


__main_model__ = MagicGlue
