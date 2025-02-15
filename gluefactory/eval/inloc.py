
from omegaconf import OmegaConf
to_ctr = OmegaConf.to_container

#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# visloc script with support for coarse to fine
# --------------------------------------------------------
import os
import numpy as np
import random
import torch
import torchvision.transforms as tvf
from torch.utils.data._utils.collate import default_collate_fn_map, default_collate_err_msg_format

import argparse
import tqdm
from PIL import Image
import scipy.io
import math
import quaternion
import collections
import cv2
import roma
from typing import Callable, Dict, Optional, Tuple, Type, Union, List

# from ..models.extractors import mast3r
from ..models import get_model

import kapture
from kapture.io.csv import kapture_from_dir
from kapture_localization.utils.pairsfile import get_ordered_pairs_from_file

# from mast3r.model import AsymmetricMASt3R
from packaging import version


def aggregate_stats(info_str, pose_errors, angular_errors):
    stats = collections.Counter()
    median_pos_error = np.median(pose_errors)
    median_angular_error = np.median(angular_errors)
    out_str = f'{info_str}: {len(pose_errors)} images - {median_pos_error=}, {median_angular_error=}'

    for trl_thr, ang_thr in [(0.1, 1), (0.25, 2), (0.5, 5), (5, 10)]:
        for pose_error, angular_error in zip(pose_errors, angular_errors):
            correct_for_this_threshold = (pose_error < trl_thr) and (angular_error < ang_thr)
            stats[trl_thr, ang_thr] += correct_for_this_threshold
    stats = {f'acc@{key[0]:g}m,{key[1]}deg': 100 * val / len(pose_errors) for key, val in stats.items()}
    for metric, perf in stats.items():
        out_str += f'  - {metric:12s}={float(perf):.3f}'
    return out_str

def get_pose_error(pr_camtoworld, gt_cam_to_world):
    abs_transl_error = torch.linalg.norm(torch.tensor(pr_camtoworld[:3, 3]) - torch.tensor(gt_cam_to_world[:3, 3]))
    abs_angular_error = roma.rotmat_geodesic_distance(torch.tensor(pr_camtoworld[:3, :3]),
                                                      torch.tensor(gt_cam_to_world[:3, :3])) * 180 / np.pi
    return abs_transl_error, abs_angular_error

def export_results(output_dir, xp_label, query_names, poses_pred):
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

        lines = ""
        lines_ltvl = ""
        for query_name, pr_querycam_to_world in zip(query_names, poses_pred):
            if pr_querycam_to_world is None:
                pr_world_to_querycam = np.eye(4)
            else:
                pr_world_to_querycam = np.linalg.inv(pr_querycam_to_world)
            query_shortname = os.path.basename(query_name)
            pr_world_to_querycam_q = quaternion.from_rotation_matrix(pr_world_to_querycam[:3, :3])
            pr_world_to_querycam_t = pr_world_to_querycam[:3, 3]

            line_pose = quaternion.as_float_array(pr_world_to_querycam_q).tolist() + \
                pr_world_to_querycam_t.flatten().tolist()

            line_content = [query_name] + line_pose
            lines += ' '.join(str(v) for v in line_content) + '\n'

            line_content_ltvl = [query_shortname] + line_pose
            lines_ltvl += ' '.join(str(v) for v in line_content_ltvl) + '\n'

        with open(os.path.join(output_dir, xp_label + '_results.txt'), 'wt') as f:
            f.write(lines)
        with open(os.path.join(output_dir, xp_label + '_ltvl.txt'), 'wt') as f:
            f.write(lines_ltvl)


try:
    import poselib  # noqa
    HAS_POSELIB = True
except Exception as e:
    HAS_POSELIB = False

try:
    import pycolmap  # noqa
    version_number = pycolmap.__version__
    if version.parse(version_number) < version.parse("0.5.0"):
        HAS_PYCOLMAP = False
    else:
        HAS_PYCOLMAP = True
except Exception as e:
    HAS_PYCOLMAP = False

def run_pnp(pts2D, pts3D, K, distortion = None, mode='cv2', reprojectionError=5, img_size = None):
    """
    use OPENCV model for distortion (4 values)
    """
    assert mode in ['cv2', 'poselib', 'pycolmap']
    try:
        if len(pts2D) > 4 and mode == "cv2":
            confidence = 0.9999
            iterationsCount = 10_000
            if distortion is not None:
                cv2_pts2ds = np.copy(pts2D)
                cv2_pts2ds = cv2.undistortPoints(cv2_pts2ds, K, np.array(distortion), R=None, P=K)
                pts2D = cv2_pts2ds.reshape((-1, 2))

            success, r_pose, t_pose, _ = cv2.solvePnPRansac(pts3D, pts2D, K, None, flags=cv2.SOLVEPNP_SQPNP,
                                                            iterationsCount=iterationsCount,
                                                            reprojectionError=reprojectionError,
                                                            confidence=confidence)
            if not success:
                return False, None
            r_pose = cv2.Rodrigues(r_pose)[0]  # world2cam == world2cam2
            RT = np.r_[np.c_[r_pose, t_pose], [(0,0,0,1)]] # world2cam2
            return True, np.linalg.inv(RT)  # cam2toworld
        elif len(pts2D) > 4 and mode == "poselib":
            assert HAS_POSELIB
            confidence = 0.9999
            iterationsCount = 10_000
            # NOTE: `Camera` struct currently contains `width`/`height` fields,
            # however these are not used anywhere in the code-base and are provided simply to be consistent with COLMAP.
            # so we put garbage in there
            colmap_intrinsics = opencv_to_colmap_intrinsics(K)
            fx = colmap_intrinsics[0, 0]
            fy = colmap_intrinsics[1, 1]
            cx = colmap_intrinsics[0, 2]
            cy = colmap_intrinsics[1, 2]
            width = img_size[0] if img_size is not None else int(cx*2)
            height = img_size[1] if img_size is not None else int(cy*2)

            if distortion is None:
                camera = {'model': 'PINHOLE', 'width': width, 'height': height, 'params': [fx, fy, cx, cy]}
            else:
                camera = {'model': 'OPENCV', 'width': width, 'height': height,
                          'params': [fx, fy, cx, cy] + distortion}
            
            pts2D = np.copy(pts2D)
            pts2D[:, 0] += 0.5
            pts2D[:, 1] += 0.5
            pose, _ = poselib.estimate_absolute_pose(pts2D, pts3D, camera,
                                                        {'max_reproj_error': reprojectionError,
                                                        'max_iterations': iterationsCount,
                                                        'success_prob': confidence}, {})
            if pose is None:
                return False, None
            RT = pose.Rt  # (3x4)
            RT = np.r_[RT, [(0,0,0,1)]]  # world2cam
            return True, np.linalg.inv(RT)  # cam2toworld
        elif len(pts2D) > 4 and mode == "pycolmap":
            assert HAS_PYCOLMAP
            assert img_size is not None
            
            pts2D = np.copy(pts2D)
            pts2D[:, 0] += 0.5
            pts2D[:, 1] += 0.5
            colmap_intrinsics = opencv_to_colmap_intrinsics(K)
            fx = colmap_intrinsics[0, 0]
            fy = colmap_intrinsics[1, 1]
            cx = colmap_intrinsics[0, 2]
            cy = colmap_intrinsics[1, 2]
            width = img_size[0]
            height = img_size[1]
            if distortion is None:
                camera_dict = {'model': 'PINHOLE', 'width': width, 'height': height, 'params': [fx, fy, cx, cy]}
            else:
                camera_dict = {'model': 'OPENCV', 'width': width, 'height': height,
                               'params': [fx, fy, cx, cy] + distortion}

            pycolmap_camera = pycolmap.Camera(
            model=camera_dict['model'], width=camera_dict['width'], height=camera_dict['height'],
            params=camera_dict['params'])

            pycolmap_estimation_options = dict(ransac=dict(max_error=reprojectionError, min_inlier_ratio=0.01,
                                               min_num_trials=1000, max_num_trials=100000,
                                            confidence=0.9999))
            pycolmap_refinement_options=dict(refine_focal_length=False, refine_extra_params=False)
            ret = pycolmap.absolute_pose_estimation(pts2D, pts3D, pycolmap_camera,
                                                    estimation_options=pycolmap_estimation_options,
                                                    refinement_options=pycolmap_refinement_options)
            if ret is None:
                ret = {'success': False}
            else:
                ret['success'] = True
                if callable(ret['cam_from_world'].matrix):
                    retmat = ret['cam_from_world'].matrix()
                else:
                    retmat = ret['cam_from_world'].matrix
                ret['qvec'] = quaternion.from_rotation_matrix(retmat[:3, :3])
                ret['tvec'] = retmat[:3, 3]
                
            if not (ret['success'] and ret['num_inliers'] > 0):
                success = False
                pose = None
            else:
                success = True
                pr_world_to_querycam = np.r_[ret['cam_from_world'].matrix(), [(0,0,0,1)]]
                pose = np.linalg.inv(pr_world_to_querycam)
            return success, pose
        else:
            return False, None
    except Exception as e:
        print(f'error during pnp: {e}')
        return False, None


def mkdir_for(f):
    os.makedirs(os.path.dirname(f), exist_ok=True)
    return f

def cat_collate_tensor_fn(batch, *, collate_fn_map):
    return torch.cat(batch, dim=0)

def cat_collate_list_fn(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
    return [item for bb in batch for item in bb]  # concatenate all lists

cat_collate_fn_map = default_collate_fn_map.copy()
cat_collate_fn_map[torch.Tensor] = cat_collate_tensor_fn
cat_collate_fn_map[List] = cat_collate_list_fn
cat_collate_fn_map[type(None)] = lambda _, **kw: None  # When some Nones, simply return a single None

def cat_collate(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
    r"""Custom collate function that concatenates stuff instead of stacking them, and handles NoneTypes """
    elem = batch[0]
    elem_type = type(elem)

    if collate_fn_map is not None:
        if elem_type in collate_fn_map:
            return collate_fn_map[elem_type](batch, collate_fn_map=collate_fn_map)

        for collate_type in collate_fn_map:
            if isinstance(elem, collate_type):
                return collate_fn_map[collate_type](batch, collate_fn_map=collate_fn_map)

    if isinstance(elem, collections.abc.Mapping):
        try:
            return elem_type({key: cat_collate([d[key] for d in batch], collate_fn_map=collate_fn_map) for key in elem})
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: cat_collate([d[key] for d in batch], collate_fn_map=collate_fn_map) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(cat_collate(samples, collate_fn_map=collate_fn_map) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.

        if isinstance(elem, tuple):
            # Backwards compatibility.
            return [cat_collate(samples, collate_fn_map=collate_fn_map) for samples in transposed]
        else:
            try:
                return elem_type([cat_collate(samples, collate_fn_map=collate_fn_map) for samples in transposed])
            except TypeError:
                # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                return [cat_collate(samples, collate_fn_map=collate_fn_map) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


def _rotation_origin_to_pt(target):
    """ Align the origin (0,0,1) with the target point (x,y,1) in projective space.
    Method: rotate z to put target on (x'+,0,1), then rotate on Y to get (0,0,1) and un-rotate z.
    """
    from scipy.spatial.transform import Rotation
    x, y = target
    rot_z = np.arctan2(y, x)
    rot_y = np.arctan(np.linalg.norm(target))
    R = Rotation.from_euler('ZYZ', [rot_z, rot_y, -rot_z]).as_matrix()
    return R

def _dotmv(Trf, pts, ncol=None, norm=False):
    assert Trf.ndim >= 2
    ncol = ncol or pts.shape[-1]

    # adapt shape if necessary
    output_reshape = pts.shape[:-1]
    if Trf.ndim >= 3:
        n = Trf.ndim - 2
        assert Trf.shape[:n] == pts.shape[:n], 'batch size does not match'
        Trf = Trf.reshape(-1, Trf.shape[-2], Trf.shape[-1])

        if pts.ndim > Trf.ndim:
            # Trf == (B,d,d) & pts == (B,H,W,d) --> (B, H*W, d)
            pts = pts.reshape(Trf.shape[0], -1, pts.shape[-1])
        elif pts.ndim == 2:
            # Trf == (B,d,d) & pts == (B,d) --> (B, 1, d)
            pts = pts[:, None, :]

    if pts.shape[-1] + 1 == Trf.shape[-1]:
        Trf = Trf.swapaxes(-1, -2)  # transpose Trf
        pts = pts @ Trf[..., :-1, :] + Trf[..., -1:, :]

    elif pts.shape[-1] == Trf.shape[-1]:
        Trf = Trf.swapaxes(-1, -2)  # transpose Trf
        pts = pts @ Trf
    else:
        pts = Trf @ pts.T
        if pts.ndim >= 2:
            pts = pts.swapaxes(-1, -2)

    if norm:
        pts = pts / pts[..., -1:]  # DONT DO /= BECAUSE OF WEIRD PYTORCH BUG
        if norm != 1:
            pts *= norm

    res = pts[..., :ncol].reshape(*output_reshape, ncol)
    return res

def crop_to_homography(K, crop, target_size=None):
    """ Given an image and its intrinsics, 
        we want to replicate a rectangular crop with an homography, 
        so that the principal point of the new 'crop' is centered.
    """
    # build intrinsics for the crop
    crop = np.round(crop)
    crop_size = crop[2:] - crop[:2]
    K2 = K.copy()  # same focal
    K2[:2, 2] = crop_size / 2  # new principal point is perfectly centered

    # find which corner is the most far-away from current principal point
    # so that the final homography does not go over the image borders
    corners = crop.reshape(-1, 2)
    corner_idx = np.abs(corners - K[:2, 2]).argmax(0)
    corner = corners[corner_idx, [0, 1]]
    # align with the corresponding corner from the target view
    corner2 = np.c_[[0, 0], crop_size][[0, 1], corner_idx]

    old_pt = _dotmv(np.linalg.inv(K), corner, norm=1)
    new_pt = _dotmv(np.linalg.inv(K2), corner2, norm=1)
    R = _rotation_origin_to_pt(old_pt) @ np.linalg.inv(_rotation_origin_to_pt(new_pt))

    if target_size is not None:
        imsize = target_size
        target_size = np.asarray(target_size)
        scaling = min(target_size / crop_size)
        K2[:2] *= scaling
        K2[:2, 2] = target_size / 2
    else:
        imsize = tuple(np.int32(crop_size).tolist())

    return imsize, K2, R, K @ R @ np.linalg.inv(K2)


def crop_slice(cell):
    return slice(cell[1], cell[3]), slice(cell[0], cell[2])

def multiple_of_16(x):
    return (x // 16) * 16

def _start_pos(total_size, win_size, overlap):
    # we must have AT LEAST overlap between segments
    # first segment starts at 0, last segment starts at total_size-win_size
    assert 0 <= overlap < 1
    assert total_size >= win_size
    spacing = win_size * (1 - overlap)
    last_pt = total_size - win_size
    n_windows = 2 + int((last_pt - 1) // spacing)
    return np.linspace(0, last_pt, n_windows).round().astype(int)

def _make_overlapping_grid(H, W, size, overlap):
    H_win = multiple_of_16(H * size // max(H, W))
    W_win = multiple_of_16(W * size // max(H, W))
    x = _start_pos(W, W_win, overlap)
    y = _start_pos(H, H_win, overlap)
    grid = np.stack(np.meshgrid(x, y, indexing='xy'), axis=-1)
    grid = np.concatenate((grid, grid + (W_win, H_win)), axis=-1)
    return grid.reshape(-1, 4)

def _cell_size(cell2):
    width, height = cell2[:, 2] - cell2[:, 0], cell2[:, 3] - cell2[:, 1]
    assert width.min() >= 0
    assert height.min() >= 0
    return width, height

def _norm_windows(cell2, H2, W2, forced_resolution=None):
    # make sure the window aspect ratio is 3/4, or the output resolution is forced_resolution  if defined
    outcell = cell2.copy()
    width, height = _cell_size(cell2)
    width2, height2 = width.clip(max=W2), height.clip(max=H2)
    if forced_resolution is None:
        width2[width < height] = (height2[width < height] * 3.01 / 4).clip(max=W2)
        height2[width >= height] = (width2[width >= height] * 3.01 / 4).clip(max=H2)
    else:
        forced_H, forced_W = forced_resolution
        width2[:] = forced_W
        height2[:] = forced_H

    half = (width2 - width) / 2
    outcell[:, 0] -= half
    outcell[:, 2] += half
    half = (height2 - height) / 2
    outcell[:, 1] -= half
    outcell[:, 3] += half

    # proj to integers
    outcell = np.floor(outcell).astype(int)
    # Take care of flooring errors
    tmpw, tmph = _cell_size(outcell)
    outcell[:, 0] += tmpw.astype(tmpw.dtype) - width2.astype(tmpw.dtype)
    outcell[:, 1] += tmph.astype(tmpw.dtype) - height2.astype(tmpw.dtype)

    # make sure 0 <= x < W2 and 0 <= y < H2
    outcell[:, 0::2] -= outcell[:, [0]].clip(max=0)
    outcell[:, 1::2] -= outcell[:, [1]].clip(max=0)
    outcell[:, 0::2] -= outcell[:, [2]].clip(min=W2) - W2
    outcell[:, 1::2] -= outcell[:, [3]].clip(min=H2) - H2

    width, height = _cell_size(outcell)
    assert np.all(width == width2.astype(width.dtype)) and np.all(
        height == height2.astype(height.dtype)), "Error, output is not of the expected shape."
    assert np.all(width <= W2)
    assert np.all(height <= H2)
    return outcell

def pos2d_in_rect(p1, cell1):
    x, y = p1.T
    l, t, r, b = cell1
    assigned = (l <= x) & (x < r) & (t <= y) & (y < b)
    return assigned

def _weight_pixels(cell, pix, assigned, gauss_var=2):
    center = cell.reshape(-1, 2, 2).mean(axis=1)
    width, height = _cell_size(cell)

    # square distance between each cell center and each point
    dist = (center[:, None] - pix[None]) / np.c_[width, height][:, None]
    dist2 = np.square(dist).sum(axis=-1)

    assert assigned.shape == dist2.shape
    res = np.where(assigned, np.exp(-gauss_var * dist2), 0)
    return res

def _score_cell(cell1, H2, W2, p1, p2, min_corres=10, forced_resolution=None):
    assert p1.shape == p2.shape

    # compute keypoint assignment
    assigned = pos2d_in_rect(p1, cell1[None].T)
    assert assigned.shape == (len(cell1), len(p1))

    # remove cells without correspondences
    valid_cells = assigned.sum(axis=1) >= min_corres
    cell1 = cell1[valid_cells]
    assigned = assigned[valid_cells]
    if not valid_cells.any():
        return cell1, cell1, assigned

    # fill-in the assigned points in both image
    assigned_p1 = np.empty((len(cell1), len(p1), 2), dtype=np.float32)
    assigned_p2 = np.empty((len(cell1), len(p2), 2), dtype=np.float32)
    assigned_p1[:] = p1[None]
    assigned_p2[:] = p2[None]
    assigned_p1[~assigned] = np.nan
    assigned_p2[~assigned] = np.nan

    # find the median center and scale of assigned points in each cell
    # cell_center1 = np.nanmean(assigned_p1, axis=1)
    cell_center2 = np.nanmean(assigned_p2, axis=1)
    im1_q25, im1_q75 = np.nanquantile(assigned_p1, (0.1, 0.9), axis=1)
    im2_q25, im2_q75 = np.nanquantile(assigned_p2, (0.1, 0.9), axis=1)

    robust_std1 = (im1_q75 - im1_q25).clip(20.)
    robust_std2 = (im2_q75 - im2_q25).clip(20.)

    cell_size1 = (cell1[:, 2:4] - cell1[:, 0:2])
    cell_size2 = cell_size1 * robust_std2 / robust_std1
    cell2 = np.c_[cell_center2 - cell_size2 / 2, cell_center2 + cell_size2 / 2]

    # make sure cell bounds are valid
    cell2 = _norm_windows(cell2, H2, W2, forced_resolution=forced_resolution)

    # compute correspondence weights
    corres_weights = _weight_pixels(cell1, p1, assigned) * _weight_pixels(cell2, p2, assigned)

    # return a list of window pairs and assigned correspondences
    return cell1, cell2, corres_weights

def select_pairs_of_crops(img_q, img_b, pos2d_in_query, pos2d_in_ref, maxdim=512, overlap=.5, forced_resolution=None):
    # prepare the overlapping cells
    grid_q = _make_overlapping_grid(*img_q.shape[:2], maxdim, overlap)
    grid_b = _make_overlapping_grid(*img_b.shape[:2], maxdim, overlap)

    assert forced_resolution is None or len(forced_resolution) == 2
    if isinstance(forced_resolution[0], int) or not len(forced_resolution[0]) == 2:
        forced_resolution1 = forced_resolution2 = forced_resolution
    else:
        assert len(forced_resolution[1]) == 2
        forced_resolution1 = forced_resolution[0]
        forced_resolution2 = forced_resolution[1]

    # Make sure crops respect constraints
    grid_q = _norm_windows(grid_q.astype(float), *img_q.shape[:2], forced_resolution=forced_resolution1)
    grid_b = _norm_windows(grid_b.astype(float), *img_b.shape[:2], forced_resolution=forced_resolution2)

    # score cells
    pairs_q = _score_cell(grid_q, *img_b.shape[:2], pos2d_in_query, pos2d_in_ref, forced_resolution=forced_resolution2)
    pairs_b = _score_cell(grid_b, *img_q.shape[:2], pos2d_in_ref, pos2d_in_query, forced_resolution=forced_resolution1)
    pairs_b = pairs_b[1], pairs_b[0], pairs_b[2]  # cellq, cellb, corres_weights

    # greedy selection until all correspondences are generated
    cell1, cell2, corres_weights = map(np.concatenate, zip(pairs_q, pairs_b))
    if len(corres_weights) == 0:
        return  # tolerated for empty generators
    order = greedy_selection(corres_weights, target=0.9)

    for i in order:
        def pair_tag(qi, bi): return (str(qi) + crop_tag(cell1[i]), str(bi) + crop_tag(cell2[i]))
        yield cell1[i], cell2[i], pair_tag

def colmap_to_opencv_intrinsics(K):
    """
    Modify camera intrinsics to follow a different convention.
    Coordinates of the center of the top-left pixels are by default:
    - (0.5, 0.5) in Colmap
    - (0,0) in OpenCV
    """
    K = K.copy()
    K[0, 2] -= 0.5
    K[1, 2] -= 0.5
    return K

def opencv_to_colmap_intrinsics(K):
    """
    Modify camera intrinsics to follow a different convention.
    Coordinates of the center of the top-left pixels are by default:
    - (0.5, 0.5) in Colmap
    - (0,0) in OpenCV
    """
    K = K.copy()
    K[0, 2] += 0.5
    K[1, 2] += 0.5
    return K

def xy_grid(W, H, device=None, origin=(0, 0), unsqueeze=None, cat_dim=-1, homogeneous=False, **arange_kw):
    """ Output a (H,W,2) array of int32 
        with output[j,i,0] = i + origin[0]
             output[j,i,1] = j + origin[1]
    """
    if device is None:
        # numpy
        arange, meshgrid, stack, ones = np.arange, np.meshgrid, np.stack, np.ones
    else:
        # torch
        arange = lambda *a, **kw: torch.arange(*a, device=device, **kw)
        meshgrid, stack = torch.meshgrid, torch.stack
        ones = lambda *a: torch.ones(*a, device=device)

    tw, th = [arange(o, o + s, **arange_kw) for s, o in zip((W, H), origin)]
    grid = meshgrid(tw, th, indexing='xy')
    if homogeneous:
        grid = grid + (ones((H, W)),)
    if unsqueeze is not None:
        grid = (grid[0].unsqueeze(unsqueeze), grid[1].unsqueeze(unsqueeze))
    if cat_dim is not None:
        grid = stack(grid, cat_dim)
    return grid

def geotrf(Trf, pts, ncol=None, norm=False):
    """ Apply a geometric transformation to a list of 3-D points.

    H: 3x3 or 4x4 projection matrix (typically a Homography)
    p: numpy/torch/tuple of coordinates. Shape must be (...,2) or (...,3)

    ncol: int. number of columns of the result (2 or 3)
    norm: float. if != 0, the resut is projected on the z=norm plane.

    Returns an array of projected 2d points.
    """
    assert Trf.ndim >= 2
    if isinstance(Trf, np.ndarray):
        pts = np.asarray(pts)
    elif isinstance(Trf, torch.Tensor):
        pts = torch.as_tensor(pts, dtype=Trf.dtype)

    # adapt shape if necessary
    output_reshape = pts.shape[:-1]
    ncol = ncol or pts.shape[-1]

    # optimized code
    if (isinstance(Trf, torch.Tensor) and isinstance(pts, torch.Tensor) and
            Trf.ndim == 3 and pts.ndim == 4):
        d = pts.shape[3]
        if Trf.shape[-1] == d:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf, pts)
        elif Trf.shape[-1] == d + 1:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf[:, :d, :d], pts) + Trf[:, None, None, :d, d]
        else:
            raise ValueError(f'bad shape, not ending with 3 or 4, for {pts.shape=}')
    else:
        if Trf.ndim >= 3:
            n = Trf.ndim - 2
            assert Trf.shape[:n] == pts.shape[:n], 'batch size does not match'
            Trf = Trf.reshape(-1, Trf.shape[-2], Trf.shape[-1])

            if pts.ndim > Trf.ndim:
                # Trf == (B,d,d) & pts == (B,H,W,d) --> (B, H*W, d)
                pts = pts.reshape(Trf.shape[0], -1, pts.shape[-1])
            elif pts.ndim == 2:
                # Trf == (B,d,d) & pts == (B,d) --> (B, 1, d)
                pts = pts[:, None, :]

        if pts.shape[-1] + 1 == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf[..., :-1, :] + Trf[..., -1:, :]
        elif pts.shape[-1] == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf
        else:
            pts = Trf @ pts.T
            if pts.ndim >= 2:
                pts = pts.swapaxes(-1, -2)

    if norm:
        pts = pts / pts[..., -1:]  # DONT DO /= BECAUSE OF WEIRD PYTORCH BUG
        if norm != 1:
            pts *= norm

    res = pts[..., :ncol].reshape(*output_reshape, ncol)
    return res


def cam_to_world_from_kapture(kdata, timestamp, camera_id):
    camera_to_world = kdata.trajectories[timestamp, camera_id].inverse()
    camera_pose = np.eye(4, dtype=np.float32)
    camera_pose[:3, :3] = quaternion.as_rotation_matrix(camera_to_world.r)
    camera_pose[:3, 3] = camera_to_world.t_raw
    return camera_pose

ratios_resolutions = {
    224: {1.0: [224, 224]},
    512: {4 / 3: [512, 384], 32 / 21: [512, 336], 16 / 9: [512, 288], 2 / 1: [512, 256], 16 / 5: [512, 160]}
}

def get_resize_function(maxdim, patch_size, H, W, is_mask=False):
    if [max(H, W), min(H, W)] in ratios_resolutions[maxdim].values():
        return lambda x: x, np.eye(3), np.eye(3)
    else:
        target_HW = get_HW_resolution(H, W, maxdim=maxdim, patchsize=patch_size)

        ratio = W / H
        target_ratio = target_HW[1] / target_HW[0]
        to_orig_crop = np.eye(3)
        to_rescaled_crop = np.eye(3)
        if abs(ratio - target_ratio) < np.finfo(np.float32).eps:
            crop_W = W
            crop_H = H
        elif ratio - target_ratio < 0:
            crop_W = W
            crop_H = int(W / target_ratio)
            to_orig_crop[1, 2] = (H - crop_H) / 2.0
            to_rescaled_crop[1, 2] = -(H - crop_H) / 2.0
        else:
            crop_W = int(H * target_ratio)
            crop_H = H
            to_orig_crop[0, 2] = (W - crop_W) / 2.0
            to_rescaled_crop[0, 2] = - (W - crop_W) / 2.0

        crop_op = tvf.CenterCrop([crop_H, crop_W])

        if is_mask:
            resize_op = tvf.Resize(size=target_HW, interpolation=tvf.InterpolationMode.NEAREST_EXACT)
        else:
            resize_op = tvf.Resize(size=target_HW)
        to_orig_resize = np.array([[crop_W / target_HW[1], 0, 0],
                                   [0, crop_H / target_HW[0], 0],
                                   [0, 0, 1]])
        to_rescaled_resize = np.array([[target_HW[1] / crop_W, 0, 0],
                                       [0, target_HW[0] / crop_H, 0],
                                       [0, 0, 1]])

        op = tvf.Compose([crop_op, resize_op])

        return op, to_rescaled_resize @ to_rescaled_crop, to_orig_crop @ to_orig_resize

def get_HW_resolution(H, W, maxdim, patchsize=16):
    assert maxdim in ratios_resolutions, "Error, maxdim can only be 224 or 512 for now. Other maxdims not implemented yet."
    ratios_resolutions_maxdim = ratios_resolutions[maxdim]
    mindims = set([min(res) for res in ratios_resolutions_maxdim.values()])
    ratio = W / H
    ref_ratios = np.array([*(ratios_resolutions_maxdim.keys())])
    islandscape = (W >= H)
    if islandscape:
        diff = np.abs(ratio - ref_ratios)
    else:
        diff = np.abs(ratio - (1 / ref_ratios))
    selkey = ref_ratios[np.argmin(diff)]
    res = ratios_resolutions_maxdim[selkey]
    # check patchsize and make sure output resolution is a multiple of patchsize
    if isinstance(patchsize, tuple):
        assert len(patchsize) == 2 and isinstance(patchsize[0], int) and isinstance(
            patchsize[1], int), "What is your patchsize format? Expected a single int or a tuple of two ints."
        assert patchsize[0] == patchsize[1], "Error, non square patches not managed"
        patchsize = patchsize[0]
    assert max(res) == maxdim
    assert min(res) in mindims
    return res[::-1] if islandscape else res  # return HW
    
def rescale_points3d(pts2d, pts3d, to_resize, HR, WR):
    # rescale pts2d as floats
    # to colmap, so that the image is in [0, D] -> [0, NewD]
    pts2d = pts2d.copy()
    pts2d[:, 0] += 0.5
    pts2d[:, 1] += 0.5

    pts2d_rescaled = geotrf(to_resize, pts2d, norm=True)

    pts2d_rescaled_int = pts2d_rescaled.copy()
    # convert back to cv2 before round [-0.5, 0.5] -> pixel 0
    pts2d_rescaled_int[:, 0] -= 0.5
    pts2d_rescaled_int[:, 1] -= 0.5
    pts2d_rescaled_int = pts2d_rescaled_int.round().astype(np.int64)

    # update valid (remove cropped regions)
    valid_rescaled = (pts2d_rescaled_int[:, 0] >= 0) & (pts2d_rescaled_int[:, 0] < WR) & (
        pts2d_rescaled_int[:, 1] >= 0) & (pts2d_rescaled_int[:, 1] < HR)

    pts2d_rescaled_int = pts2d_rescaled_int[valid_rescaled]

    # rebuild pts3d from rescaled ps2d poses
    pts3d_rescaled = np.full((HR, WR, 3), np.nan, dtype=np.float32)  # pts3d in 512 x something
    pts3d_rescaled[pts2d_rescaled_int[:, 1], pts2d_rescaled_int[:, 0]] = pts3d[valid_rescaled]

    return pts2d_rescaled, pts2d_rescaled_int, pts3d_rescaled, np.isfinite(pts3d_rescaled.sum(axis=-1))


def read_alignments(path_to_alignment):
    aligns = {}
    with open(path_to_alignment, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            if len(line) == 4:
                trans_nr = line[:-1]
                while line != 'After general icp:\n':
                    line = fid.readline()
                line = fid.readline()
                p = []
                for i in range(4):
                    elems = line.split(' ')
                    line = fid.readline()
                    for e in elems:
                        if len(e) != 0:
                            p.append(float(e))
                P = np.array(p).reshape(4, 4)
                aligns[trans_nr] = P
    return aligns

ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

class BaseVislocDataset:
    def __init__(self):
        pass

    def set_resolution(self, model):
        self.maxdim = max(model.patch_embed.img_size)
        self.patch_size = model.patch_embed.patch_size

    def __len__(self):
        raise NotImplementedError()
    
    def __getitem__(self, idx):
        raise NotImplementedError()

class VislocInLoc(BaseVislocDataset):
    def __init__(self, root, pairsfile, topk=1, duc1_only=False):
        super().__init__()
        self.root = root
        self.topk = topk
        self.num_views = self.topk + 1
        self.maxdim = None
        self.patch_size = None

        query_path = os.path.join(self.root, 'query')
        kdata_query = kapture_from_dir(query_path)
        assert kdata_query.records_camera is not None
        kdata_query_searchindex = {kdata_query.records_camera[(timestamp, sensor_id)]: (timestamp, sensor_id)
                                   for timestamp, sensor_id in kdata_query.records_camera.key_pairs()}
        self.query_data = {'path': query_path, 'kdata': kdata_query, 'searchindex': kdata_query_searchindex}

        map_path = os.path.join(self.root, 'mapping')
        kdata_map = kapture_from_dir(map_path)
        assert kdata_map.records_camera is not None and kdata_map.trajectories is not None
        kdata_map_searchindex = {kdata_map.records_camera[(timestamp, sensor_id)]: (timestamp, sensor_id)
                                 for timestamp, sensor_id in kdata_map.records_camera.key_pairs()}
        self.map_data = {'path': map_path, 'kdata': kdata_map, 'searchindex': kdata_map_searchindex}

        try:
            self.pairs = get_ordered_pairs_from_file(os.path.join(self.root, 'pairfiles/query', pairsfile + '.txt'))
        except Exception as e:
            # if using pairs from hloc
            self.pairs = {}
            with open(os.path.join(self.root, 'pairfiles/query', pairsfile + '.txt'), 'r') as fid:
                lines = fid.readlines()
                for line in lines:
                    splits = line.rstrip("\n\r").split(" ")
                    self.pairs.setdefault(splits[0].replace('query/', ''), []).append(
                        (splits[1].replace('database/cutouts/', ''), 1.0)
                    )

        self.scenes = kdata_query.records_camera.data_list()
        if duc1_only:
            self.scenes = self.scenes[:205]  # DUC1 only

        self.aligns_DUC1 = read_alignments(os.path.join(self.root, 'mapping/DUC1_alignment/all_transformations.txt'))
        self.aligns_DUC2 = read_alignments(os.path.join(self.root, 'mapping/DUC2_alignment/all_transformations.txt'))

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        assert self.maxdim is not None and self.patch_size is not None
        query_image = self.scenes[idx]
        map_images = [p[0] for p in self.pairs[query_image][:self.topk]]
        views = []
        dataarray = [(query_image, self.query_data, False)] + [(map_image, self.map_data, True)
                                                               for map_image in map_images]
        for idx, (imgname, data, should_load_depth) in enumerate(dataarray):
            imgpath, kdata, searchindex = map(data.get, ['path', 'kdata', 'searchindex'])

            timestamp, camera_id = searchindex[imgname]

            # for InLoc, SIMPLE_PINHOLE
            camera_params = kdata.sensors[camera_id].camera_params
            W, H, f, cx, cy = camera_params
            distortion = [0, 0, 0, 0]
            intrinsics = np.float32([(f, 0, cx),
                                     (0, f, cy),
                                     (0, 0, 1)])

            if kdata.trajectories is not None and (timestamp, camera_id) in kdata.trajectories:
                cam_to_world = cam_to_world_from_kapture(kdata, timestamp, camera_id)
            else:
                cam_to_world = np.eye(4, dtype=np.float32)

            # Load RGB image
            rgb_image = Image.open(os.path.join(imgpath, 'sensors/records_data', imgname)).convert('RGB')
            rgb_image.load()

            W, H = rgb_image.size
            resize_func, to_resize, to_orig = get_resize_function(self.maxdim, self.patch_size, H, W)

            rgb_tensor = resize_func(ImgNorm(rgb_image))

            view = {
                'intrinsics': intrinsics,
                'distortion': distortion,
                'cam_to_world': cam_to_world,
                'rgb': rgb_image,
                'rgb_rescaled': rgb_tensor,
                'to_orig': to_orig,
                'idx': idx,
                'image_name': imgname
            }

            # Load depthmap
            if should_load_depth:
                depthmap_filename = os.path.join(imgpath, 'sensors/records_data', imgname + '.mat')
                depthmap = scipy.io.loadmat(depthmap_filename)

                pt3d_cut = depthmap['XYZcut']
                scene_id = imgname.replace('\\', '/').split('/')[1]
                if imgname.startswith('DUC1'):
                    pts3d_full = geotrf(self.aligns_DUC1[scene_id], pt3d_cut)
                else:
                    pts3d_full = geotrf(self.aligns_DUC2[scene_id], pt3d_cut)

                pts3d_valid = np.isfinite(pts3d_full.sum(axis=-1))

                pts3d = pts3d_full[pts3d_valid]
                pts2d_int = xy_grid(W, H)[pts3d_valid]
                pts2d = pts2d_int.astype(np.float64)

                # nan => invalid
                pts3d_full[~pts3d_valid] = np.nan
                pts3d_full = torch.from_numpy(pts3d_full)
                view['pts3d'] = pts3d_full
                view["valid"] = pts3d_full.sum(dim=-1).isfinite()

                HR, WR = rgb_tensor.shape[1:]
                _, _, pts3d_rescaled, valid_rescaled = rescale_points3d(pts2d, pts3d, to_resize, HR, WR)
                pts3d_rescaled = torch.from_numpy(pts3d_rescaled)
                valid_rescaled = torch.from_numpy(valid_rescaled)
                view['pts3d_rescaled'] = pts3d_rescaled
                view["valid_rescaled"] = valid_rescaled
            views.append(view)
        return views


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="visloc dataset to eval")
    parser_weights = parser.add_mutually_exclusive_group(required=True)
    parser_weights.add_argument("--weights", type=str, help="path to the model weights", default=None)
    parser_weights.add_argument("--model_name", type=str, help="name of the model weights",
                                choices=["MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"])

    parser.add_argument("--confidence_threshold", type=float, default=1.001,
                        help="confidence values higher than threshold are invalid")
    parser.add_argument('--pixel_tol', default=5, type=int)

    parser.add_argument("--coarse_to_fine", action='store_true', default=False,
                        help="do the matching from coarse to fine")
    parser.add_argument("--max_image_size", type=int, default=None,
                        help="max image size for the fine resolution")
    parser.add_argument("--c2f_crop_with_homography", action='store_true', default=False,
                        help="when using coarse to fine, crop with homographies to keep cx, cy centered")

    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
    parser.add_argument("--pnp_mode", type=str, default="cv2", choices=['cv2', 'poselib', 'pycolmap'],
                        help="pnp lib to use")
    parser_reproj = parser.add_mutually_exclusive_group()
    parser_reproj.add_argument("--reprojection_error", type=float, default=5.0, help="pnp reprojection error")
    parser_reproj.add_argument("--reprojection_error_diag_ratio", type=float, default=None,
                               help="pnp reprojection error as a ratio of the diagonal of the image")

    parser.add_argument("--max_batch_size", type=int, default=48,
                        help="max batch size for inference on crops when using coarse to fine")
    parser.add_argument("--pnp_max_points", type=int, default=100_000, help="pnp maximum number of points kept")
    parser.add_argument("--viz_matches", type=int, default=0, help="debug matches")

    parser.add_argument("--output_dir", type=str, default=None, help="output path")
    parser.add_argument("--output_label", type=str, default='', help="prefix for results files")
    return parser


@torch.no_grad()
def coarse_matching(query_view, map_view, model, device, pixel_tol, fast_nn_params):
    # prepare batch
    imgs = []
    for idx, img in enumerate([query_view['rgb_rescaled'], map_view['rgb_rescaled']]):
        imgs.append(dict(img=img.unsqueeze(0), true_shape=np.int32([img.shape[1:]]),
                         idx=idx, instance=str(idx)))
    output = inference([tuple(imgs)], model, device, batch_size=1, verbose=False)
    pred1, pred2 = output['pred1'], output['pred2']
    conf_list = [pred1['desc_conf'].squeeze(0).cpu().numpy(), pred2['desc_conf'].squeeze(0).cpu().numpy()]
    desc_list = [pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()]

    # find 2D-2D matches between the two images
    PQ, PM = desc_list[0], desc_list[1]
    if len(PQ) == 0 or len(PM) == 0:
        return [], [], [], []

    if pixel_tol == 0:
        fast_reciprocal_NNs = get_model('matchers.fast_nn')(OmegaConf.create({'name': 'matchers.fast_nn', 'subsample_or_initxy0': 8, **fast_nn_params})).to('cuda').eval()
        matches = fast_reciprocal_NNs({"descriptors0": PM.unsqueeze(0).to(device), "descriptors1": PQ.unsqueeze(0).to(device)})
        matches_im_map, matches_im_query = matches['keypoints0'].squeeze(0).int().cpu().detach().numpy(), matches['keypoints1'].squeeze(0).int().cpu().detach().numpy()
        # matches_im_map, matches_im_query = fast_reciprocal_NNs(PM, PQ, subsample_or_initxy1=8, **fast_nn_params)
        HM, WM = map_view['rgb_rescaled'].shape[1:]
        HQ, WQ = query_view['rgb_rescaled'].shape[1:]
        # ignore small border around the edge
        valid_matches_map = (matches_im_map[:, 0] >= 3) & (matches_im_map[:, 0] < WM - 3) & (
            matches_im_map[:, 1] >= 3) & (matches_im_map[:, 1] < HM - 3)
        valid_matches_query = (matches_im_query[:, 0] >= 3) & (matches_im_query[:, 0] < WQ - 3) & (
            matches_im_query[:, 1] >= 3) & (matches_im_query[:, 1] < HQ - 3)
        valid_matches = valid_matches_map & valid_matches_query
        matches_im_map = matches_im_map[valid_matches]
        matches_im_query = matches_im_query[valid_matches]
        valid_pts3d = []
        matches_confs = []
    else:
        yM, xM = torch.where(map_view['valid_rescaled'])
        fast_reciprocal_NNs = get_model('matchers.fast_nn')(OmegaConf.create({'name': 'matchers.fast_nn', 'pixel_tol': pixel_tol, **fast_nn_params})).to('cuda').eval()
        fast_reciprocal_NNs.subsample_or_initxy0 = (xM, yM)
        matches = fast_reciprocal_NNs({"descriptors0": PM.unsqueeze(0).to(device), "descriptors1": PQ.unsqueeze(0).to(device)})
        matches_im_map, matches_im_query = matches['keypoints0'].squeeze(0).int().cpu().detach().numpy(), matches['keypoints1'].squeeze(0).int().cpu().detach().numpy()
        # matches_im_map, matches_im_query = fast_reciprocal_NNs(PM, PQ, (xM, yM), pixel_tol=pixel_tol, **fast_nn_params)
        valid_pts3d = map_view['pts3d_rescaled'].cpu().numpy()[matches_im_map[:, 1], matches_im_map[:, 0]]
        matches_confs = np.minimum(
            conf_list[1][matches_im_map[:, 1], matches_im_map[:, 0]],
            conf_list[0][matches_im_query[:, 1], matches_im_query[:, 0]]
        )
    # from cv2 to colmap
    matches_im_query = matches_im_query.astype(np.float64)
    matches_im_map = matches_im_map.astype(np.float64)
    matches_im_query[:, 0] += 0.5
    matches_im_query[:, 1] += 0.5
    matches_im_map[:, 0] += 0.5
    matches_im_map[:, 1] += 0.5
    # rescale coordinates
    matches_im_query = geotrf(query_view['to_orig'], matches_im_query, norm=True)
    matches_im_map = geotrf(map_view['to_orig'], matches_im_map, norm=True)
    # from colmap back to cv2
    matches_im_query[:, 0] -= 0.5
    matches_im_query[:, 1] -= 0.5
    matches_im_map[:, 0] -= 0.5
    matches_im_map[:, 1] -= 0.5
    return valid_pts3d, matches_im_query, matches_im_map, matches_confs


@torch.no_grad()
def crops_inference(pairs, model, device, batch_size=48, verbose=True):
    assert len(pairs) == 2, "Error, data should be a tuple of dicts containing the batch of image pairs"
    # Forward a possibly big bunch of data, by blocks of batch_size
    B = pairs[0]['img'].shape[0]
    if B < batch_size:
        return loss_of_one_batch(pairs, model, None, device=device, symmetrize_batch=False)
    preds = []
    for ii in range(0, B, batch_size):
        sel = slice(ii, ii + min(B - ii, batch_size))
        temp_data = [{}, {}]
        for di in [0, 1]:
            temp_data[di] = {kk: pairs[di][kk][sel]
                             for kk in pairs[di].keys() if pairs[di][kk] is not None}  # copy chunk for forward
        preds.append(loss_of_one_batch(temp_data, model,
                                       None, device=device, symmetrize_batch=False))  # sequential forward
    # Merge all preds
    return cat_collate(preds, collate_fn_map=cat_collate_fn_map)


@torch.no_grad()
def fine_matching(query_views, map_views, model, device, max_batch_size, pixel_tol, fast_nn_params):
    assert pixel_tol > 0
    output = crops_inference([query_views, map_views],
                             model, device, batch_size=max_batch_size, verbose=False)
    pred1, pred2 = output['pred1'], output['pred2']
    descs1 = pred1['desc'].clone()
    descs2 = pred2['desc'].clone()
    confs1 = pred1['desc_conf'].clone()
    confs2 = pred2['desc_conf'].clone()

    # Compute matches
    valid_pts3d, matches_im_map, matches_im_query, matches_confs = [], [], [], []
    for ppi, (pp1, pp2, cc11, cc21) in enumerate(zip(descs1, descs2, confs1, confs2)):
        valid_ppi = map_views['valid'][ppi]
        pts3d_ppi = map_views['pts3d'][ppi].cpu().numpy()
        conf_list_ppi = [cc11.cpu().numpy(), cc21.cpu().numpy()]

        y_ppi, x_ppi = torch.where(valid_ppi)
        fast_reciprocal_NNs = get_model('matchers.fast_nn')(OmegaConf.create({'name': 'matchers.fast_nn', 'pixel_tol': pixel_tol, **fast_nn_params})).to('cuda').eval()
        fast_reciprocal_NNs.subsample_or_initxy0 = (x_ppi, y_ppi)
        matches = fast_reciprocal_NNs({"descriptors0": pp2.unsqueeze(0).to(device), "descriptors1": pp1.unsqueeze(0).to(device)})
        matches_im_map_ppi, matches_im_query_ppi = matches['keypoints0'].squeeze(0).int().cpu().detach().numpy(), matches['keypoints1'].squeeze(0).int().cpu().detach().numpy()
        # matches_im_map_ppi, matches_im_query_ppi = fast_reciprocal_NNs(pp2, pp1, (x_ppi, y_ppi),
        #                                                                pixel_tol=pixel_tol, **fast_nn_params)

        valid_pts3d_ppi = pts3d_ppi[matches_im_map_ppi[:, 1], matches_im_map_ppi[:, 0]]
        matches_confs_ppi = np.minimum(
            conf_list_ppi[1][matches_im_map_ppi[:, 1], matches_im_map_ppi[:, 0]],
            conf_list_ppi[0][matches_im_query_ppi[:, 1], matches_im_query_ppi[:, 0]]
        )
        # inverse operation where we uncrop pixel coordinates
        matches_im_map_ppi = geotrf(map_views['to_orig'][ppi].cpu().numpy(), matches_im_map_ppi.copy(), norm=True)
        matches_im_query_ppi = geotrf(query_views['to_orig'][ppi].cpu().numpy(), matches_im_query_ppi.copy(), norm=True)

        matches_im_map.append(matches_im_map_ppi)
        matches_im_query.append(matches_im_query_ppi)
        valid_pts3d.append(valid_pts3d_ppi)
        matches_confs.append(matches_confs_ppi)

    if len(valid_pts3d) == 0:
        return [], [], [], []

    matches_im_map = np.concatenate(matches_im_map, axis=0)
    matches_im_query = np.concatenate(matches_im_query, axis=0)
    valid_pts3d = np.concatenate(valid_pts3d, axis=0)
    matches_confs = np.concatenate(matches_confs, axis=0)
    return valid_pts3d, matches_im_query, matches_im_map, matches_confs


def crop(img, mask, pts3d, crop, intrinsics=None):
    out_cropped_img = img.clone()
    if mask is not None:
        out_cropped_mask = mask.clone()
    else:
        out_cropped_mask = None
    if pts3d is not None:
        out_cropped_pts3d = pts3d.clone()
    else:
        out_cropped_pts3d = None
    to_orig = torch.eye(3, device=img.device)

    # If intrinsics available, crop and apply rectifying homography. Otherwise, just crop
    if intrinsics is not None:
        K_old = intrinsics
        imsize, K_new, R, H = crop_to_homography(K_old, crop)
        # apply homography to image
        H /= H[2, 2]
        homo8 = H.ravel().tolist()[:8]
        # From float tensor to uint8 PIL Image
        pilim = Image.fromarray((255 * (img + 1.) / 2).to(torch.uint8).numpy())
        pilout_cropped_img = pilim.transform(imsize, Image.Transform.PERSPECTIVE,
                                             homo8, resample=Image.Resampling.BICUBIC)

        # From uint8 PIL Image to float tensor
        out_cropped_img = 2. * torch.tensor(np.array(pilout_cropped_img)).to(img) / 255. - 1.
        if out_cropped_mask is not None:
            pilmask = Image.fromarray((255 * out_cropped_mask).to(torch.uint8).numpy())
            pilout_cropped_mask = pilmask.transform(
                imsize, Image.Transform.PERSPECTIVE, homo8, resample=Image.Resampling.NEAREST)
            out_cropped_mask = torch.from_numpy(np.array(pilout_cropped_mask) > 0).to(out_cropped_mask.dtype)
        if out_cropped_pts3d is not None:
            out_cropped_pts3d = out_cropped_pts3d.numpy()
            out_cropped_X = np.array(Image.fromarray(out_cropped_pts3d[:, :, 0]).transform(imsize,
                                                                                           Image.Transform.PERSPECTIVE,
                                                                                           homo8,
                                                                                           resample=Image.Resampling.NEAREST))
            out_cropped_Y = np.array(Image.fromarray(out_cropped_pts3d[:, :, 1]).transform(imsize,
                                                                                           Image.Transform.PERSPECTIVE,
                                                                                           homo8,
                                                                                           resample=Image.Resampling.NEAREST))
            out_cropped_Z = np.array(Image.fromarray(out_cropped_pts3d[:, :, 2]).transform(imsize,
                                                                                           Image.Transform.PERSPECTIVE,
                                                                                           homo8,
                                                                                           resample=Image.Resampling.NEAREST))

            out_cropped_pts3d = torch.from_numpy(np.stack([out_cropped_X, out_cropped_Y, out_cropped_Z], axis=-1))

        to_orig = torch.tensor(H, device=img.device)
    else:
        out_cropped_img = img[crop_slice(crop)]
        if out_cropped_mask is not None:
            out_cropped_mask = out_cropped_mask[crop_slice(crop)]
        if out_cropped_pts3d is not None:
            out_cropped_pts3d = out_cropped_pts3d[crop_slice(crop)]
        to_orig[:2, -1] = torch.tensor(crop[:2])

    return out_cropped_img, out_cropped_mask, out_cropped_pts3d, to_orig


def resize_image_to_max(max_image_size, rgb, K):
    W, H = rgb.size
    if max_image_size and max(W, H) > max_image_size:
        islandscape = (W >= H)
        if islandscape:
            WMax = max_image_size
            HMax = int(H * (WMax / W))
        else:
            HMax = max_image_size
            WMax = int(W * (HMax / H))
        resize_op = tvf.Compose([ImgNorm, tvf.Resize(size=[HMax, WMax])])
        rgb_tensor = resize_op(rgb).permute(1, 2, 0)
        to_orig_max = np.array([[W / WMax, 0, 0],
                                [0, H / HMax, 0],
                                [0, 0, 1]])
        to_resize_max = np.array([[WMax / W, 0, 0],
                                  [0, HMax / H, 0],
                                  [0, 0, 1]])

        # Generate new camera parameters
        new_K = opencv_to_colmap_intrinsics(K)
        new_K[0, :] *= WMax / W
        new_K[1, :] *= HMax / H
        new_K = colmap_to_opencv_intrinsics(new_K)
    else:
        rgb_tensor = ImgNorm(rgb).permute(1, 2, 0)
        to_orig_max = np.eye(3)
        to_resize_max = np.eye(3)
        HMax, WMax = H, W
        new_K = K
    return rgb_tensor, new_K, to_orig_max, to_resize_max, (HMax, WMax)

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

def to_cpu(x): return todevice(x, 'cpu')

def collate_with_cat(whatever, lists=False):
    if isinstance(whatever, dict):
        return {k: collate_with_cat(vals, lists=lists) for k, vals in whatever.items()}

    elif isinstance(whatever, (tuple, list)):
        if len(whatever) == 0:
            return whatever
        elem = whatever[0]
        T = type(whatever)

        if elem is None:
            return None
        if isinstance(elem, (bool, float, int, str)):
            return whatever
        if isinstance(elem, tuple):
            return T(collate_with_cat(x, lists=lists) for x in zip(*whatever))
        if isinstance(elem, dict):
            return {k: collate_with_cat([e[k] for e in whatever], lists=lists) for k in elem}

        if isinstance(elem, torch.Tensor):
            return listify(whatever) if lists else torch.cat(whatever)
        if isinstance(elem, np.ndarray):
            return listify(whatever) if lists else torch.cat([torch.from_numpy(x) for x in whatever])

        # otherwise, we just chain lists
        return sum(whatever, T())

def loss_of_one_batch(batch, model, criterion, device, symmetrize_batch=False, use_amp=False, ret=None):
    view1, view2 = batch
    ignore_keys = set(['depthmap', 'dataset', 'label', 'instance', 'idx', 'true_shape', 'rng'])
    for view in batch:
        for name in view.keys():  # pseudo_focal
            if name in ignore_keys:
                continue
            view[name] = view[name].to(device, non_blocking=True)

    # if symmetrize_batch:
    #     view1, view2 = make_batch_symmetric(batch)

    with torch.cuda.amp.autocast(enabled=bool(use_amp)):
        view1['image'], view2['image'] = view1['img'], view2['img']
        pred1, pred2 = model((view1, view2))
        pred1['desc_conf'], pred2['desc_conf'] = pred1['keypoint_scores'], pred2['keypoint_scores']
        pred1['desc'], pred2['desc'] = pred1['descriptors'], pred2['descriptors']
        conf_thr = 1.001  # default value in args
        print((pred1['desc_conf'] >= conf_thr).sum(), (pred2['desc_conf'].max() >= conf_thr).sum())

        # loss is supposed to be symmetric
        with torch.cuda.amp.autocast(enabled=False):
            loss = criterion(view1, view2, pred1, pred2) if criterion is not None else None

    result = dict(view1=view1, view2=view2, pred1=pred1, pred2=pred2, loss=loss)
    return result[ret] if ret else result

def check_if_same_size(pairs):
    shapes1 = [img1['img'].shape[-2:] for img1, img2 in pairs]
    shapes2 = [img2['img'].shape[-2:] for img1, img2 in pairs]
    return all(shapes1[0] == s for s in shapes1) and all(shapes2[0] == s for s in shapes2)

@torch.no_grad()
def inference(pairs, model, device, batch_size=8, verbose=True):
    if verbose:
        print(f'>> Inference with model on {len(pairs)} image pairs')
    result = []

    # first, check if all images have the same size
    multiple_shapes = not (check_if_same_size(pairs))
    if multiple_shapes:  # force bs=1
        batch_size = 1

    for i in tqdm.trange(0, len(pairs), batch_size, disable=not verbose):
        res = loss_of_one_batch(collate_with_cat(pairs[i:i + batch_size]), model, None, device)
        result.append(to_cpu(res))

    result = collate_with_cat(result, lists=multiple_shapes)

    return result


if __name__ == '__main__':
    from .. import logger  # overwrite the logger
    parser = get_args_parser()
    args = parser.parse_args()
    conf_thr = args.confidence_threshold
    device = args.device
    pnp_mode = args.pnp_mode
    assert args.pixel_tol > 0
    reprojection_error = args.reprojection_error
    reprojection_error_diag_ratio = args.reprojection_error_diag_ratio
    pnp_max_points = args.pnp_max_points
    viz_matches = args.viz_matches

    model = get_model('extractors.mast3r')(OmegaConf.create({'name': 'extractors.mast3r'})).to('cuda').eval()
    # fast_nn_params = dict(device=device, dist='dot', block_size=2**13)
    fast_nn_params = dict(dist='dot', block_size=2**13)
    dataset = eval(args.dataset)
    dataset.set_resolution(model)

    query_names = []
    poses_pred = []
    pose_errors = []
    angular_errors = []
    params_str = f'tol_{args.pixel_tol}' + ("_c2f" if args.coarse_to_fine else '')
    if args.max_image_size is not None:
        params_str = params_str + f'_{args.max_image_size}'
    if args.coarse_to_fine and args.c2f_crop_with_homography:
        params_str = params_str + '_with_homography'
    for idx in tqdm.tqdm(range(len(dataset))):
        views = dataset[(idx)]  # 0 is the query
        query_view = views[0]
        map_views = views[1:]
        query_names.append(query_view['image_name'])

        query_pts2d = []
        query_pts3d = []
        maxdim = max(model.patch_embed.img_size)
        query_rgb_tensor, query_K, query_to_orig_max, query_to_resize_max, (HQ, WQ) = resize_image_to_max(
            args.max_image_size, query_view['rgb'], query_view['intrinsics'])

        # pairs of crops have the same resolution
        query_resolution = get_HW_resolution(HQ, WQ, maxdim=maxdim, patchsize=model.patch_embed.patch_size)
        for map_view in map_views:
            if args.output_dir is not None:
                cache_file = os.path.join(args.output_dir, 'matches', params_str,
                                          query_view['image_name'], map_view['image_name'] + '.npz')
            else:
                cache_file = None

            if cache_file is not None and os.path.isfile(cache_file):
                matches = np.load(cache_file)
                valid_pts3d = matches['valid_pts3d']
                matches_im_query = matches['matches_im_query']
                matches_im_map = matches['matches_im_map']
                matches_conf = matches['matches_conf']
            else:
                # coarse matching
                if args.coarse_to_fine and (maxdim < max(WQ, HQ)):
                    # use all points
                    _, coarse_matches_im0, coarse_matches_im1, _ = coarse_matching(query_view, map_view, model, device,
                                                                                   0, fast_nn_params)

                    # visualize a few matches
                    if viz_matches > 0:
                        num_matches = coarse_matches_im1.shape[0]
                        print(f'found {num_matches} matches')

                        viz_imgs = [np.array(query_view['rgb']), np.array(map_view['rgb'])]
                        from matplotlib import pyplot as pl
                        n_viz = viz_matches
                        match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, n_viz)).astype(int)
                        viz_matches_im_query = coarse_matches_im0[match_idx_to_viz]
                        viz_matches_im_map = coarse_matches_im1[match_idx_to_viz]

                        H0, W0, H1, W1 = *viz_imgs[0].shape[:2], *viz_imgs[1].shape[:2]
                        img0 = np.pad(viz_imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)),
                                      'constant', constant_values=0)
                        img1 = np.pad(viz_imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)),
                                      'constant', constant_values=0)
                        img = np.concatenate((img0, img1), axis=1)
                        pl.figure()
                        pl.imshow(img)
                        cmap = pl.get_cmap('jet')
                        for i in range(n_viz):
                            (x0, y0), (x1, y1) = viz_matches_im_query[i].T, viz_matches_im_map[i].T
                            pl.plot([x0, x1 + W0], [y0, y1], '-+',
                                    color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)
                        pl.show(block=True)

                    valid_all = map_view['valid']
                    pts3d = map_view['pts3d']

                    WM_full, HM_full = map_view['rgb'].size
                    map_rgb_tensor, map_K, map_to_orig_max, map_to_resize_max, (HM, WM) = resize_image_to_max(
                        args.max_image_size, map_view['rgb'], map_view['intrinsics'])
                    if WM_full != WM or HM_full != HM:
                        y_full, x_full = torch.where(valid_all)
                        pos2d_cv2 = torch.stack([x_full, y_full], dim=-1).cpu().numpy().astype(np.float64)
                        sparse_pts3d = pts3d[y_full, x_full].cpu().numpy()
                        _, _, pts3d_max, valid_max = rescale_points3d(
                            pos2d_cv2, sparse_pts3d, map_to_resize_max, HM, WM)
                        pts3d = torch.from_numpy(pts3d_max)
                        valid_all = torch.from_numpy(valid_max)

                    coarse_matches_im0 = geotrf(query_to_resize_max, coarse_matches_im0, norm=True)
                    coarse_matches_im1 = geotrf(map_to_resize_max, coarse_matches_im1, norm=True)

                    crops1, crops2 = [], []
                    crops_v1, crops_p1 = [], []
                    to_orig1, to_orig2 = [], []
                    map_resolution = get_HW_resolution(HM, WM, maxdim=maxdim, patchsize=model.patch_embed.patch_size)

                    for crop_q, crop_b, pair_tag in select_pairs_of_crops(map_rgb_tensor,
                                                                          query_rgb_tensor,
                                                                          coarse_matches_im1,
                                                                          coarse_matches_im0,
                                                                          maxdim=maxdim,
                                                                          overlap=.5,
                                                                          forced_resolution=[map_resolution,
                                                                                             query_resolution]):
                        # Per crop processing
                        if not args.c2f_crop_with_homography:
                            map_K = None
                            query_K = None

                        c1, v1, p1, trf1 = crop(map_rgb_tensor, valid_all, pts3d, crop_q, map_K)
                        c2, _, _, trf2 = crop(query_rgb_tensor, None, None, crop_b, query_K)
                        crops1.append(c1)
                        crops2.append(c2)
                        crops_v1.append(v1)
                        crops_p1.append(p1)
                        to_orig1.append(trf1)
                        to_orig2.append(trf2)

                    if len(crops1) == 0 or len(crops2) == 0:
                        valid_pts3d, matches_im_query, matches_im_map, matches_conf = [], [], [], []
                    else:
                        crops1, crops2 = torch.stack(crops1), torch.stack(crops2)
                        if len(crops1.shape) == 3:
                            crops1, crops2 = crops1[None], crops2[None]
                        crops_v1 = torch.stack(crops_v1)
                        crops_p1 = torch.stack(crops_p1)
                        to_orig1, to_orig2 = torch.stack(to_orig1), torch.stack(to_orig2)
                        map_crop_view = dict(img=crops1.permute(0, 3, 1, 2),
                                             instance=['1' for _ in range(crops1.shape[0])],
                                             valid=crops_v1, pts3d=crops_p1,
                                             to_orig=to_orig1)
                        query_crop_view = dict(img=crops2.permute(0, 3, 1, 2),
                                               instance=['2' for _ in range(crops2.shape[0])],
                                               to_orig=to_orig2)

                        # Inference and Matching
                        valid_pts3d, matches_im_query, matches_im_map, matches_conf = fine_matching(query_crop_view,
                                                                                                    map_crop_view,
                                                                                                    model, device,
                                                                                                    args.max_batch_size,
                                                                                                    args.pixel_tol,
                                                                                                    fast_nn_params)
                        matches_im_query = geotrf(query_to_orig_max, matches_im_query, norm=True)
                        matches_im_map = geotrf(map_to_orig_max, matches_im_map, norm=True)
                else:
                    # use only valid 2d points
                    valid_pts3d, matches_im_query, matches_im_map, matches_conf = coarse_matching(query_view, map_view,
                                                                                                  model, device,
                                                                                                  args.pixel_tol,
                                                                                                  fast_nn_params)
                if cache_file is not None:
                    mkdir_for(cache_file)
                    np.savez(cache_file, valid_pts3d=valid_pts3d, matches_im_query=matches_im_query,
                             matches_im_map=matches_im_map, matches_conf=matches_conf)

            # apply conf
            if len(matches_conf) > 0:
                mask = matches_conf >= conf_thr
                valid_pts3d = valid_pts3d[mask]
                matches_im_query = matches_im_query[mask]
                matches_im_map = matches_im_map[mask]
                matches_conf = matches_conf[mask]

            # visualize a few matches
            if viz_matches > 0:
                num_matches = matches_im_map.shape[0]
                print(f'found {num_matches} matches')

                viz_imgs = [np.array(query_view['rgb']), np.array(map_view['rgb'])]
                from matplotlib import pyplot as pl
                n_viz = viz_matches
                match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, n_viz)).astype(int)
                viz_matches_im_query = matches_im_query[match_idx_to_viz]
                viz_matches_im_map = matches_im_map[match_idx_to_viz]

                H0, W0, H1, W1 = *viz_imgs[0].shape[:2], *viz_imgs[1].shape[:2]
                img0 = np.pad(viz_imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
                img1 = np.pad(viz_imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
                img = np.concatenate((img0, img1), axis=1)
                pl.figure()
                pl.imshow(img)
                cmap = pl.get_cmap('jet')
                for i in range(n_viz):
                    (x0, y0), (x1, y1) = viz_matches_im_query[i].T, viz_matches_im_map[i].T
                    pl.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)
                pl.show(block=True)

            if len(valid_pts3d) == 0:
                pass
            else:
                query_pts3d.append(valid_pts3d)
                query_pts2d.append(matches_im_query)

        if len(query_pts2d) == 0:
            success = False
            pr_querycam_to_world = None
        else:
            query_pts2d = np.concatenate(query_pts2d, axis=0).astype(np.float32)
            query_pts3d = np.concatenate(query_pts3d, axis=0)
            if len(query_pts2d) > pnp_max_points:
                idxs = random.sample(range(len(query_pts2d)), pnp_max_points)
                query_pts3d = query_pts3d[idxs]
                query_pts2d = query_pts2d[idxs]

            W, H = query_view['rgb'].size
            if reprojection_error_diag_ratio is not None:
                reprojection_error_img = reprojection_error_diag_ratio * math.sqrt(W**2 + H**2)
            else:
                reprojection_error_img = reprojection_error
            success, pr_querycam_to_world = run_pnp(query_pts2d, query_pts3d,
                                                    query_view['intrinsics'], query_view['distortion'],
                                                    pnp_mode, reprojection_error_img, img_size=[W, H])

        if not success:
            abs_transl_error = float('inf')
            abs_angular_error = float('inf')
        else:
            abs_transl_error, abs_angular_error = get_pose_error(pr_querycam_to_world, query_view['cam_to_world'])

        pose_errors.append(abs_transl_error)
        angular_errors.append(abs_angular_error)
        poses_pred.append(pr_querycam_to_world)

    xp_label = params_str + f'_conf_{conf_thr}'
    if args.output_label:
        xp_label = args.output_label + "_" + xp_label
    if reprojection_error_diag_ratio is not None:
        xp_label = xp_label + f'_reproj_diag_{reprojection_error_diag_ratio}'
    else:
        xp_label = xp_label + f'_reproj_err_{reprojection_error}'
    export_results(args.output_dir, xp_label, query_names, poses_pred)
    out_string = aggregate_stats(f'{args.dataset}', pose_errors, angular_errors)
    print(out_string)
