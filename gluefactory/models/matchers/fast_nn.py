"""
Nearest neighbor matcher for normalized descriptors.
Optionally apply the mutual check and threshold the distance or ratio.
"""

import logging

import torch
import torch.nn.functional as F

from ..base_model import BaseModel
from ..utils.metrics import matcher_metrics


###################### FAST_NN ########################
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# MASt3R Fast Nearest Neighbor
# --------------------------------------------------------
import torch
import numpy as np
import math
# from scipy.spatial import KDTree

def todevice(batch, device, callback=None, non_blocking=False):
    ''' Transfer some variables to another device (i.e. GPU, CPU:torch, CPU:numpy).

    batch: list, tuple, dict of tensors or other things
    device: pytorch device or 'numpy'
    callback: function that would be called on every sub-elements.
    '''
    if callback:
        batch = callback(batch)

    if isinstance(batch, dict):
        return {k: todevice(v, device) for k, v in batch.items()}

    if isinstance(batch, (tuple, list)):
        return type(batch)(todevice(x, device) for x in batch)

    x = batch
    if device == 'numpy':
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    elif x is not None:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if torch.is_tensor(x):
            x = x.to(device, non_blocking=non_blocking)
    return x

def to_numpy(x): return todevice(x, 'numpy')


@torch.no_grad()
def bruteforce_reciprocal_nns(A, B, block_size=None, dist='l2'):
    device = A.device
    if isinstance(A, np.ndarray):
        A = torch.from_numpy(A).to(device)
    if isinstance(B, np.ndarray):
        B = torch.from_numpy(B).to(device)

    A = A.to(device)
    B = B.to(device)

    if dist == 'l2':
        dist_func = torch.cdist
        argmin = torch.min
    elif dist == 'dot':
        def dist_func(A, B):
            return A @ B.T

        def argmin(X, dim):
            sim, nn = torch.max(X, dim=dim)
            return sim.neg_(), nn
    else:
        raise ValueError(f'Unknown {dist=}')

    if block_size is None or len(A) * len(B) <= block_size**2:
        dists = dist_func(A, B)
        _, nn_A = argmin(dists, dim=1)
        _, nn_B = argmin(dists, dim=0)
    else:
        dis_A = torch.full((A.shape[0],), float('inf'), device=device, dtype=A.dtype)
        dis_B = torch.full((B.shape[0],), float('inf'), device=device, dtype=B.dtype)
        nn_A = torch.full((A.shape[0],), -1, device=device, dtype=torch.int64)
        nn_B = torch.full((B.shape[0],), -1, device=device, dtype=torch.int64)
        number_of_iteration_A = math.ceil(A.shape[0] / block_size)
        number_of_iteration_B = math.ceil(B.shape[0] / block_size)

        for i in range(number_of_iteration_A):
            A_i = A[i * block_size:(i + 1) * block_size]
            for j in range(number_of_iteration_B):
                B_j = B[j * block_size:(j + 1) * block_size]
                dists_blk = dist_func(A_i, B_j)  # A, B, 1
                # dists_blk = dists[i * block_size:(i+1)*block_size, j * block_size:(j+1)*block_size]
                min_A_i, argmin_A_i = argmin(dists_blk, dim=1)
                min_B_j, argmin_B_j = argmin(dists_blk, dim=0)

                col_mask = min_A_i < dis_A[i * block_size:(i + 1) * block_size]
                line_mask = min_B_j < dis_B[j * block_size:(j + 1) * block_size]

                dis_A[i * block_size:(i + 1) * block_size][col_mask] = min_A_i[col_mask]
                dis_B[j * block_size:(j + 1) * block_size][line_mask] = min_B_j[line_mask]

                nn_A[i * block_size:(i + 1) * block_size][col_mask] = argmin_A_i[col_mask] + (j * block_size)
                nn_B[j * block_size:(j + 1) * block_size][line_mask] = argmin_B_j[line_mask] + (i * block_size)
    nn_A = nn_A.cpu().numpy()
    nn_B = nn_B.cpu().numpy()
    return nn_A, nn_B


class cdistMatcher:
    def __init__(self, db_pts, block_size, dist):
        self.db_pts = db_pts
        self.block_size = block_size
        self.dist = dist

    def query(self, queries, k=1):
        assert k == 1
        if queries.numel() == 0:
            return None, []
        nnA, nnB = bruteforce_reciprocal_nns(queries, self.db_pts, block_size=self.block_size, dist=self.dist)
        dis = None
        return dis, nnA


def merge_corres(idx0, idx1, shape1=None, shape2=None, ret_xy=True, ret_index=False):
    assert idx0.dtype == idx1.dtype == np.int32

    # unique and sort along idx0
    corres = np.unique(np.c_[idx1, idx0].view(np.int64), return_index=ret_index)
    if ret_index:
        corres, indices = corres
    xy1, xy0 = corres[:, None].view(np.int32).T

    if ret_xy:
        assert shape1 and shape2
        xy0 = np.unravel_index(xy0, shape1)
        xy1 = np.unravel_index(xy1, shape2)
        if ret_xy != 'y_x':
            xy0 = xy0[0].base[:, ::-1]
            xy1 = xy1[0].base[:, ::-1]

    if ret_index:
        return xy0, xy1, indices
    return xy0, xy1


class FastNN(BaseModel):
    default_conf = {
        "subsample_or_initxy0": 8, 
        "ret_xy": True, 
        "pixel_tol": 0, 
        "ret_basin": False, 
        "dist": 'dot',
        "block_size": 2**13,
        "loss": None,
        "remove_borders": None,
    }
    required_data_keys = ["descriptors0", "descriptors1"]

    def _init(self, conf):
        self.subsample_or_initxy0 = self.conf.subsample_or_initxy0
    
    def _forward(self, data):
        subsample_or_initxy0 = self.subsample_or_initxy0
        ret_xy = self.conf.ret_xy
        pixel_tol = self.conf.pixel_tol
        ret_basin = self.conf.ret_basin
        dist = self.conf.dist
        block_size = self.conf.block_size
        border = self.conf.remove_borders

        pts0, pts1 = data["descriptors0"], data["descriptors1"]
        pts0, pts1 = pts0[0], pts1[0]  # remove batch dimension
        device = pts0.device

        h0, w0, d0 = pts0.shape
        h1, w1, d1 = pts1.shape
        assert d0 == d1

        pts0 = pts0.reshape(-1, d0)
        pts1 = pts1.reshape(-1, d1)

        if isinstance(subsample_or_initxy0, int) and pixel_tol == 0:
            S = subsample_or_initxy0
            y0, x0 = np.mgrid[S // 2:h0:S, S // 2:w0:S].reshape(2, -1)
            max_iter = 10
        else:
            x0, y0 = subsample_or_initxy0
            if isinstance(x0, torch.Tensor):
                x0 = x0.cpu().numpy()
            if isinstance(y0, torch.Tensor):
                y0 = y0.cpu().numpy()
            max_iter = 1

        xy0 = np.int32(np.unique(x0 + w0 * y0))  # make sure there's no doublons
        xy1 = np.full_like(xy0, -1)
        old_xy0 = xy0.copy()
        old_xy1 = xy1.copy()

        if dist is not None or block_size is not None \
                or (isinstance(device, str) and device.startswith('cuda')) \
                or (isinstance(device, torch.device) and device.type.startswith('cuda')):
            pts0 = pts0.to(device)
            pts1 = pts1.to(device)
            tree0 = cdistMatcher(pts0, self.conf.block_size, self.conf.dist)
            tree1 = cdistMatcher(pts1, self.conf.block_size, self.conf.dist)
        else:
            raise NotImplementedError("KDTree implementation not suported.")

        notyet = np.ones(len(xy0), dtype=bool)
        basin = np.full((h0 * w0 + 1,), -1, dtype=np.int32) if ret_basin else None

        niter = 0
        # n_notyet = [len(notyet)]
        while notyet.any():
            _, xy1[notyet] = to_numpy(tree1.query(pts0[xy0[notyet]]))
            if not ret_basin:
                notyet &= (old_xy1 != xy1)  # remove points that have converged

            _, xy0[notyet] = to_numpy(tree0.query(pts1[xy1[notyet]]))
            if ret_basin:
                basin[old_xy0[notyet]] = xy0[notyet]
            notyet &= (old_xy0 != xy0)  # remove points that have converged

            # n_notyet.append(notyet.sum())
            niter += 1
            if niter >= max_iter:
                break

            old_xy1[:] = xy1
            old_xy0[:] = xy0

        if pixel_tol > 0:
            # in case we only want to match some specific points
            # and still have some way of checking reciprocity
            old_yx0 = np.unravel_index(old_xy0, (h0, w0))[0].base
            new_yx0 = np.unravel_index(xy0, (h0, w0))[0].base
            dis = np.linalg.norm(old_yx0 - new_yx0, axis=-1)
            converged = dis < pixel_tol
            if not isinstance(subsample_or_initxy0, int):
                xy0 = old_xy0  # replace new points by old ones
        else:
            converged = ~notyet  # converged correspondences

        # keep only unique correspondences, and sort on xy0
        keypoints0, keypoints1 = merge_corres(xy0[converged], xy1[converged], (h0, w0), (h1, w1), ret_xy=ret_xy)

        # ignore small border around the edge
        if border is not None and border >= 1:
            valid_keypoints0 = (keypoints0[:, 0] >= border) & (keypoints0[:, 0] < int(w0) - border) & (keypoints0[:, 1] >= border) & (keypoints0[:, 1] < int(h0) - border)
            valid_keypoints1 = (keypoints1[:, 0] >= border) & (keypoints1[:, 0] < int(w1) - border) & (keypoints1[:, 1] >= border) & (keypoints1[:, 1] < int(h1) - border)
            valid_keypoints = valid_keypoints0 & valid_keypoints1
            matches_im0, matches_im1 = keypoints0[valid_keypoints], keypoints1[valid_keypoints]

        keypoints0 = torch.as_tensor(np.ascontiguousarray(keypoints0), dtype=torch.float, device=device).unsqueeze(0)
        keypoints1 = torch.as_tensor(np.ascontiguousarray(keypoints1), dtype=torch.float, device=device).unsqueeze(0)

        b0, k0, c0 = keypoints0.shape
        b1, k1, c1 = keypoints1.shape

        matches0 = torch.arange(k0, device=device).unsqueeze(0)
        matches1 = matches0.clone()

        mscores0 = (matches0 > -1).float()
        mscores1 = (matches1 > -1).float()

        if ret_basin:
            return {
                "keypoints0": keypoints0,  # overwrite
                "keypoints1": keypoints1,  # overwrite
                "matches0": matches0, 
                "matches1": matches1,
                "matching_scores0": mscores0,
                "matching_scores1": mscores1,
                "basin": basin,
            }

        return {
            "keypoints0": keypoints0,  # overwrite
            "keypoints1": keypoints1,  # overwrite
            "matches0": matches0, 
            "matches1": matches1,
            "matching_scores0": mscores0,
            "matching_scores1": mscores1,
        }


    def loss(self, pred, data):
        losses = {}
        if self.conf.loss == "N_pair":
            raise NotImplementedError
        else:
            raise NotImplementedError
        metrics = {} if self.training else matcher_metrics(pred, data)
        return losses, metrics