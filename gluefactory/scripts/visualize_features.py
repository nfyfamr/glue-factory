import argparse
import logging
from pathlib import Path

import torch
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from tqdm import tqdm

from ..datasets import get_dataset
from ..models import get_model
from ..settings import DATA_PATH, VIZ_PATH
from ..utils.export_predictions import export_predictions
from ..utils.tools import fork_rng, PCA
from ..utils.tensor import batch_to_device
from ..utils.patches import get_top_patches
from ..visualization.viz2d import plot_image_grid, plot_keypoints, save_plot, plot_attentions
import numpy as np


dataset_configs = {
    "megadepth": {
        "name": "megadepth",
        "grayscale": False,
        "preprocessing": {
            "resize": 1600,
        },
        "batch_size": 1,
        "num_workers": 1,
        "prefetch_factor": 1,
    },
    "homographies": {
        "name": "homographies",
        "grayscale": False,
        "preprocessing": {
            "resize": 1600,
        },
        "batch_size": 1,
        "num_workers": 1,
        "prefetch_factor": 1,
        "train_size": 150000,
        "val_size": 2000,
    },
}


extractor_configs = {
    "gluefactory_nonfree.superpoint": {
        "name": "gluefactory_nonfree.superpoint",
        "nms_radius": 3,
        "max_num_keypoints": 2048,
        "detection_threshold": 0.000,
    },
    "sift": {
        "name": "sift",
        "max_num_keypoints": 2048,
        "options": {
            "peak_threshold": 0.001,
        },
        "peak_threshold": 0.001,
        "device": "cpu",
    },
    "disk": {
        "name": "disk",
        "max_num_keypoints": 2048,
    },
    "mast3r": {
        "name": "mast3r",
        "max_num_keypoints": 512,
        "sparse_outputs": True,
        "dense_outputs": True,
        "force_num_keypoints": False,
        "randomize_keypoints": True,
        "tiling_keypoints": False,
        "confidence_threshold": 1.001,
    },
}


@torch.no_grad()
def sample_and_render_features(loader, model, device, export_root, num_items, dpi):
    with fork_rng(seed=loader.dataset.conf.seed):
        for it, data in zip(tqdm(range(num_items)), loader):
            data = batch_to_device(data, device, non_blocking=True)
            pred = model(data)
            pred["descriptors_pca0"] = PCA(pred["dense_descriptors0"])
            pred["descriptors_pca1"] = PCA(pred["dense_descriptors1"])

            # Select patches with the highest confidence
            pred["patches0"] = get_top_patches(pred["dense_keypoint_scores0"], num_patches=1, patch_size=model.extractor.conf.patch_size)
            pred["patches1"] = get_top_patches(pred["dense_keypoint_scores1"], num_patches=1, patch_size=model.extractor.conf.patch_size)

            pred = batch_to_device(pred, "cpu", non_blocking=False)
            data = batch_to_device(data, "cpu", non_blocking=False)

            images = []
            kpts = []
            for v in [0, 1]:
                images.append(
                    [
                        data[f"view{v}"]["image"][0].permute(1, 2, 0),
                        pred[f"dense_keypoint_scores{v}"][0],
                        np.tile(np.histogram(pred[f"dense_keypoint_scores{v}"][0].flatten(), bins=50)[0], (50, 1)),
                        pred[f"descriptors_pca{v}"][0],
                        pred[f"pointcloud{v}"][0][..., 2],
                        pred[f"pointcloud_scores{v}"][0],
                    ]
                )
                kpts.append(
                    [
                        pred[f"keypoints{v}"][0],
                        pred[f"keypoints{v}"][0]
                    ]
                )

            fig, axes = plot_image_grid(images, dpi=dpi, cmaps="inferno", return_fig=True)
            [plot_keypoints(kpts[i], axes=axes[i], colors="royalblue") for i in range(len(kpts))]
            save_plot(export_root / f"{it:03}_idx_{data['idx'].item()}.png")
            plt.close("all")

            # Draw new figure for attention map
            img0 = data["view0"]["image"][0].permute(1, 2, 0)
            img1 = data["view1"]["image"][0].permute(1, 2, 0)
            images = [[img0, img1] for _ in range(model.extractor.conf.dec_depth)]
            
            fig, axes = plot_image_grid(images, return_fig=True, set_lim=True)
            for i in range(model.extractor.conf.dec_depth):
                plot_attentions(pred["attentions0"][i][0], patch_indices=pred["patches0"][0], image_size=data[f"view{v}"]["image"][0].shape[1:], patch_size=model.extractor.conf.patch_size, axes=axes[i], reverse=False, cmap="Purples", lw=0.5, ps=0)
                plot_attentions(pred["attentions1"][i][0], patch_indices=pred["patches1"][0], image_size=data[f"view{v}"]["image"][0].shape[1:], patch_size=model.extractor.conf.patch_size, axes=axes[i], reverse=True, cmap="Oranges", lw=0.5, ps=0)

            save_plot(export_root / f"{it:03}_idx_{data['idx'].item()}_attn.png")
            plt.close("all")

def visualize(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device {device}")

    conf = OmegaConf.from_cli(args.dotlist)

    # Load dataset
    data_conf = OmegaConf.create(dataset_configs[args.dataset])
    data_conf = OmegaConf.merge(data_conf, conf.get('dataset', {}))
    dataset = get_dataset(data_conf.name)(data_conf)
    loader = dataset.get_data_loader("train")

    # Load model
    model_cfg = {
        "name": "two_view_pipeline",
        "extractor": extractor_configs[args.extractor],
    }
    model_cfg = OmegaConf.create(model_cfg)
    model_cfg.extractor = OmegaConf.merge(model_cfg.extractor, conf.get('extractor', {}))
    model = get_model(model_cfg.name)(model_cfg).to(device)

    # Prepare image saving
    export_postfix = f"-{args.export_postfix}" if len(args.export_postfix) else ""
    resize = f"-r{dataset.conf.preprocessing.resize}"
    n_kpts = f"-k{model.extractor.conf.max_num_keypoints}" if model.conf.extractor.sparse_outputs else ""

    export_name = f"{args.dataset}{resize}_{args.extractor}{n_kpts}{export_postfix}"
    export_root = Path(VIZ_PATH, export_name)
    export_root.mkdir(parents=True, exist_ok=True)

    sample_and_render_features(loader, model, device, export_root, args.num_items, args.dpi)
    print(f"Images are saved in {export_root}")
    

if __name__ == "__main__":
    # python -m gluefactory.scripts.visualize_features homographies mast3r --export_postfix "sample_kpts-no_tiling" --num_items 100 --dpi 300 dataset.preprocessing.resize=512 dataset.preprocessing.edge_divisible_by=16 extractor.confidence_threshold=1.001 extractor.force_num_keypoints=True extractor.randomize_keypoints=True extractor.tiling_keypoints=False
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, choices=list(dataset_configs.keys()))
    parser.add_argument("extractor", type=str, default="sp", choices=list(extractor_configs.keys()))
    parser.add_argument("--export_postfix", type=str, default="")
    parser.add_argument("--num_items", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("dotlist", nargs="*", help="configurations with namespace: e.g., dataset.preprocessing.resize=512 extractor.confidence_threshold=1.001")
    args = parser.parse_intermixed_args()

    visualize(args)
