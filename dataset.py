import torch
from torch.utils.data import Dataset
import tifffile
import xarray as xr
import numpy as np
import os

class LCZDataset(Dataset):
    def __init__(self, prisma_30, s2, lcz_map, lst, patch_size, stride, transforms=None, use_tiled_dataset=True, tiled_dataset_dir="./tiled_dataset" ):
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

        source = torch.cat((s2_patch, pr_patch, lst_patch), dim=0)
        label = lcz_patch
        return {
            "image": source,
            "label": label,
        }

    def resize(self):
        prisma_path = self.prisma_30_path
        s2_path = self.s2_path
        lcz_path = self.lcz_map_path
        lst_folder_path = self.lst_path

        prisma_image = tifffile.imread(prisma_path)
        s2_image = tifffile.imread(s2_path)
        lcz_image = tifffile.imread(lcz_path)

        lst_nc_file_path = os.path.join(lst_folder_path, 'LST_in.nc')

        try:
            with xr.open_dataset(lst_nc_file_path) as ds:
                lst_image_xr = ds['LST']
                lst_image = lst_image_xr.values
        except Exception as e:
            print(f"Error loading LST NetCDF file from {lst_nc_file_path}: {e}")
            raise e

        prisma_image = torch.from_numpy(prisma_image).float()
        s2_image = torch.from_numpy(s2_image).float()
        lcz_image = torch.from_numpy(lcz_image).float()
        lst_image = torch.from_numpy(lst_image).float()

        # 1. Handle LST: Ensure it's [C, H, W] for interpolation, where C=1 for a single LST bands
        if lst_image.ndim == 2: # LST is (H, W)
            lst_image = lst_image.unsqueeze(0) # -> (1, H, W)
        elif lst_image.ndim == 3 and lst_image.shape[0] > 1: # LST is (C, H, W)
            lst_image = lst_image[0:1, :, :] # -> (1, H, W)
        elif lst_image.ndim == 1: 
            raise ValueError(f"LST image has unexpected 1D shape: {lst_image.shape}. Expected 2D or 3D.")
        
        lst_image_resized = torch.nn.functional.interpolate(
            lst_image.unsqueeze(0), # Add batch dimension: [1, 1, H_orig, W_orig]
            size=(self.H, self.W),
            mode='bicubic',
            align_corners=False
        ).squeeze(0) # Remove the added batch dimension: [1, H_resized, W_resized]

        # 2. Handle PRISMA: Ensure it's [C, H, W] before interpolation
        if prisma_image.ndim == 3 and prisma_image.shape[2] > 1: # Assuming (H, W, C)
             prisma_image = prisma_image.permute(2, 0, 1) # -> (C, H, W)
        elif prisma_image.ndim == 2: # Single band image (H, W)
             prisma_image = prisma_image.unsqueeze(0) # -> (1, H, W)

        prisma_image_resized = torch.nn.functional.interpolate(
            prisma_image.unsqueeze(0), # Add batch dim for interpolate: [1, C, H, W]
            size=(self.H, self.W),
            mode='bicubic',
            align_corners=False
        ).squeeze(0) 

        # 3. Handle S2: Ensure it's [C, H, W] and potentially resize if its resolution doesn't match
        s2_image = s2_image.permute(2, 0, 1) # Ensure S2 is [C, H, W]

        if s2_image.shape[1] != self.H or s2_image.shape[2] != self.W:
             print(f"Warning: S2 image shape {s2_image.shape} does not match target {self.H}x{self.W}. Interpolating S2.")
             s2_image = torch.nn.functional.interpolate(
                 s2_image.unsqueeze(0), 
                 size=(self.H, self.W),
                 mode='bicubic',
                 align_corners=False
             ).squeeze(0)

        # 4. Handle LCZ Map: Apply crop
        crop_left   = 1
        crop_right  = 0
        crop_top    = 0
        crop_bottom = 1
        H_lcz, W_lcz = lcz_image.shape
        y0, y1 = crop_top,    H_lcz - crop_bottom
        x0, x1 = crop_left,   W_lcz - crop_right
        lcz_image = lcz_image[y0:y1, x0:x1]

        return prisma_image_resized, s2_image, lcz_image, lst_image_resized

    def get_tile_indices(self):
        tiles = [
            (row, col)
            for row in range(0, self.H - self.patch_size + 1, self.stride)
            for col in range(0, self.W - self.patch_size + 1, self.stride)
        ]
        return tiles

    def get_tile(self, tile_idx):
        row, col = self.tile_indices[tile_idx]

        pr_patch = self.prisma_30[:, row:row+self.patch_size, col:col+self.patch_size]
        s2_patch = self.s2    [:, row:row+self.patch_size, col:col+self.patch_size]
        lcz_patch = self.lcz_map[row:row+self.patch_size, col:col+self.patch_size]
        lst_patch = self.lst[:, row:row+self.patch_size, col:col+self.patch_size]

        assert 0 <= tile_idx < len(self.tile_indices), f"Bad idx {tile_idx}"
        assert pr_patch is not None, "pr_patch is None"
        assert s2_patch is not None, "s2_patch is None"
        assert lcz_patch is not None, "mask_patch is None"
        assert lst_patch is not None, "lst_patch is None"
        return pr_patch, s2_patch, lcz_patch, lst_patch