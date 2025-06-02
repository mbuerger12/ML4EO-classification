import torch
from torch.utils.data import Dataset
import tifffile
import os
import pathlib


def preprocess_and_save_tiles(prisma_30, s2, lcz_map, patch_size, stride, output_dir="./tiled_dataset"):
    """
    Preprocess images and save tiles to disk.

    Args:
        prisma_30 (str): Path to PRISMA-30 image
        s2 (str): Path to Sentinel-2 image
        lcz_map (str): Path to LCZ map
        patch_size (int): Size of patches
        stride (int): Stride for tiling
        output_dir (str): Directory to save tiles
    """
    # Create output directories
    base_dir = pathlib.Path(output_dir)

    # Extract dataset name from the path
    dataset_name = pathlib.Path(prisma_30).parent.name
    dataset_dir = base_dir / dataset_name

    prisma_dir = dataset_dir / "prisma_tiles"
    s2_dir = dataset_dir / "s2_tiles"
    lcz_dir = dataset_dir / "lcz_tiles"

    # Create directories if they don't exist
    for directory in [prisma_dir, s2_dir, lcz_dir]:
        os.makedirs(directory, exist_ok=True)

    # Load and preprocess images
    H = 1266
    W = 1315

    # Load images
    prisma_image = tifffile.imread(prisma_30)
    s2_image = tifffile.imread(s2)
    lcz_image = tifffile.imread(lcz_map)

    # Crop and resize
    crop_left = 1
    crop_right = 0
    crop_top = 0
    crop_bottom = 1

    H_lcz, W_lcz = lcz_image.shape
    y0, y1 = crop_top, H_lcz - crop_bottom
    x0, x1 = crop_left, W_lcz - crop_right

    prisma_image = torch.from_numpy(prisma_image).float()
    s2_image = torch.from_numpy(s2_image).float()
    lcz_image = torch.from_numpy(lcz_image).float()

    prisma_image_resized = torch.nn.functional.interpolate(
        prisma_image.unsqueeze(0), size=(H, W), mode='bicubic', align_corners=False
    ).squeeze().clone().detach()

    lcz_image = lcz_image[y0:y1, x0:x1]
    s2_image = s2_image.permute(2, 0, 1)

    # Generate tile indices
    tiles = [
        (row, col)
        for row in range(0, H - patch_size + 1, stride)
        for col in range(0, W - patch_size + 1, stride)
    ]

    # Save tiles
    for idx, (row, col) in enumerate(tiles):
        # Extract patches
        pr_patch = prisma_image_resized[:, row:row+patch_size, col:col+patch_size]
        s2_patch = s2_image[:, row:row+patch_size, col:col+patch_size]
        lcz_patch = lcz_image[row:row+patch_size, col:col+patch_size]

        # Save patches as TIFF files
        tifffile.imwrite(prisma_dir / f"tile_{idx:05d}.tif", pr_patch.numpy())
        tifffile.imwrite(s2_dir / f"tile_{idx:05d}.tif", s2_patch.numpy())
        tifffile.imwrite(lcz_dir / f"tile_{idx:05d}.tif", lcz_patch.numpy())

    print(f"Saved {len(tiles)} tiles to {dataset_dir}")
    return dataset_dir

import xarray as xr
import numpy as np
import os

class LCZDataset(Dataset):
    def __init__(self, prisma_30, s2, lcz_map, lst, patch_size, stride, transforms=None):
        self.prisma_30_path = prisma_30
        self.s2_path = s2
        self.lcz_map_path = lcz_map
        self.lst_path = lst
        self.tfms = transforms
        self.patch_size = patch_size
        self.stride = stride
        self.H = 1266
        self.W = 1315
        self.use_tiled_dataset = use_tiled_dataset

        # Check if we should use pre-tiled dataset
        if self.use_tiled_dataset:
            # Extract dataset name from the path
            dataset_name = pathlib.Path(prisma_30).parent.name
            self.dataset_dir = pathlib.Path(tiled_dataset_dir) / dataset_name

            # Check if tiled dataset exists
            self.prisma_dir = self.dataset_dir / "prisma_tiles"
            self.s2_dir = self.dataset_dir / "s2_tiles"
            self.lcz_dir = self.dataset_dir / "lcz_tiles"

            if not all(os.path.exists(d) for d in [self.prisma_dir, self.s2_dir, self.lcz_dir]):
                print(f"Tiled dataset not found at {self.dataset_dir}. Creating it now...")
                preprocess_and_save_tiles(prisma_30, s2, lcz_map, patch_size, stride, tiled_dataset_dir)

            # Count number of tiles (check for both .tif and .pt files for backward compatibility)
            tif_tiles = list(self.prisma_dir.glob("tile_*.tif"))
            pt_tiles = list(self.prisma_dir.glob("tile_*.pt"))

            # Use .tif files if available, otherwise use .pt files
            if tif_tiles:
                self.num_tiles = len(tif_tiles)
                self.file_ext = ".tif"
                print(f"Found {self.num_tiles} .tif tiles in {self.dataset_dir}")
            else:
                self.num_tiles = len(pt_tiles)
                self.file_ext = ".pt"
                print(f"Found {self.num_tiles} .pt tiles in {self.dataset_dir} (legacy format)")
        else:
            # Use the original approach
            self.prisma_30, self.s2, self.lcz_map = self.resize()
            self.tile_indices = self.get_tile_indices()
            print(f"Using on-the-fly tiling with {len(self.tile_indices)} tiles")




    def __len__(self):
        if self.use_tiled_dataset:
            return self.num_tiles
        else:
            return len(self.tile_indices)

    def __getitem__(self, tile_idx):
        if self.use_tiled_dataset:
            # Load tiles from disk
            if self.file_ext == ".tif":
                # Load TIFF files
                pr_patch = torch.from_numpy(tifffile.imread(self.prisma_dir / f"tile_{tile_idx:05d}.tif")).float()
                s2_patch = torch.from_numpy(tifffile.imread(self.s2_dir / f"tile_{tile_idx:05d}.tif")).float()
                lcz_patch = torch.from_numpy(tifffile.imread(self.lcz_dir / f"tile_{tile_idx:05d}.tif")).float()
            else:
                # Load legacy PT files
                pr_patch = torch.load(self.prisma_dir / f"tile_{tile_idx:05d}.pt")
                s2_patch = torch.load(self.s2_dir / f"tile_{tile_idx:05d}.pt")
                lcz_patch = torch.load(self.lcz_dir / f"tile_{tile_idx:05d}.pt")
        else:
            # Use original approach
            pr_patch, s2_patch, lcz_patch = self.get_tile(tile_idx)

        source = torch.cat((s2_patch, pr_patch), dim=0)
        label = lcz_patch
        return {
            "image": source,
            "label": label,
        }


    def resize(self):
        #load paths
        prisma_path = self.prisma_30_path
        s2_path = self.s2_path
        lcz_path = self.lcz_map_path

        #load images
        prisma_image = tifffile.imread(prisma_path)
        s2_image = tifffile.imread(s2_path)
        lcz_image = tifffile.imread(lcz_path)

        crop_left   = 1
        crop_right  = 0
        crop_top    = 0
        crop_bottom = 1

        H, W = lcz_image.shape
        y0, y1 = crop_top,    H - crop_bottom
        x0, x1 = crop_left,   W - crop_right
        prisma_image = torch.from_numpy(prisma_image).float()
        s2_image = torch.from_numpy(s2_image).float()
        lcz_image = torch.from_numpy(lcz_image).float()
        prisma_image_resized = torch.nn.functional.interpolate(prisma_image.unsqueeze(0), size=(self.H, self.W), mode='bicubic',align_corners=False).squeeze().clone().detach()
        lcz_image = lcz_image[y0:y1, x0:x1]
        prisma_image_resized = prisma_image_resized
        s2_image = s2_image.permute(2, 0, 1)
        return prisma_image_resized, s2_image, lcz_image


    def get_tile_indices(self):
        tiles = [
            (row, col)
            for row in range(0, self.H - self.patch_size + 1, self.stride)
            for col in range(0, self.W - self.patch_size + 1, self.stride)
        ]
        return tiles


    def get_tile(self, tile_idx):

        # 1) find the top‐left corner for this tile
        row, col = self.tile_indices[tile_idx]

        # 2) slice out patch for each source
        pr_patch = self.prisma_30[:, row:row+self.patch_size, col:col+self.patch_size]   # Tensor [C_pr, ps, ps]
        s2_patch = self.s2    [:, row:row+self.patch_size, col:col+self.patch_size]   # Tensor [C_s2, ps, ps]
        lcz_patch = self.lcz_map[row:row+self.patch_size, col:col+self.patch_size]      # [ps, ps]
        assert 0 <= tile_idx < len(self.tile_indices), f"Bad idx {tile_idx}"

        # … your code to slice pr_patch, s2_patch, mask_patch …

        # finally:
        assert pr_patch is not None, "pr_patch is None"
        assert s2_patch is not None, "s2_patch is None"
        assert lcz_patch is not None, "mask_patch is None"
        return pr_patch, s2_patch, lcz_patch
