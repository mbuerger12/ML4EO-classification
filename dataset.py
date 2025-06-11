import torch
from torch.utils.data import Dataset
import tifffile
import xarray as xr
import numpy as np
import os
import pathlib
import torch.nn.functional as F

def preprocess_and_save_tiles(prisma_30, s2, lcz_map, lst, patch_size, stride, output_dir="./tiled_dataset",use_layer=False):
    base_dir = pathlib.Path(output_dir)
    dataset_name = pathlib.Path(prisma_30).parent.name
    dataset_dir = base_dir / dataset_name

    prisma_dir = dataset_dir / "prisma_tiles"
    s2_dir = dataset_dir / "s2_tiles"
    lcz_dir = dataset_dir / "lcz_tiles"
    lst_dir = dataset_dir  / "lst_tiles"

    for directory in [prisma_dir, s2_dir, lcz_dir,lst_dir]:
        os.makedirs(directory, exist_ok=True)

    H = 1266
    W = 1315

    prisma_image = tifffile.imread(prisma_30)
    lcz_image = tifffile.imread(lcz_map)
    s2_image = tifffile.imread(s2)
    
    if use_layer:
        lst_nc_file_path = os.path.join(lst, 'LST_in.nc')
        try:
            with xr.open_dataset(lst_nc_file_path) as ds:
                lst_image_xr = ds['LST']
                if '_FillValue' in lst_image_xr.attrs:
                    fill_value = lst_image_xr.attrs['_FillValue']
                    lst_image = lst_image_xr.values.astype(np.float32)
                    lst_image[lst_image == fill_value] = np.nan
                else:
                    lst_image = lst_image_xr.values

        except Exception as e:
            raise e

        if np.isnan(lst_image).any():
            lst_image = np.nan_to_num(lst_image, nan=0.0)

        if lst_image.ndim == 2:
            lst_image = lst_image[np.newaxis, :, :]
        elif lst_image.ndim == 3 and lst_image.shape[0] > 1:
            lst_image = lst_image[0:1, :, :]
        elif lst_image.ndim == 1:
            raise ValueError(f"LST image has unexpected 1D shape: {lst_image.shape}. Expected 2D or 3D.")

        lst_image_resized = F.interpolate(
        torch.from_numpy(lst_image).float().unsqueeze(0),
        size=(H, W),
        mode='bicubic',
        align_corners=False
    ).squeeze(0)

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

    if prisma_image.ndim == 3 and prisma_image.shape[2] > 1:
         prisma_image = prisma_image.permute(2, 0, 1)
    elif prisma_image.ndim == 2:
         prisma_image = prisma_image.unsqueeze(0)


    prisma_image_resized = torch.nn.functional.interpolate(
        prisma_image.unsqueeze(0), size=(H, W), mode='bicubic', align_corners=False
    ).squeeze(0)

    lcz_image = lcz_image[y0:y1, x0:x1]
    s2_image = s2_image.permute(2, 0, 1)

    if s2_image.shape[1] != H or s2_image.shape[2] != W:
         s2_image = F.interpolate(
             s2_image.unsqueeze(0),
             size=(H, W),
             mode='bicubic',
             align_corners=False
         ).squeeze(0)


    tiles = [
        (row, col)
        for row in range(0, H - patch_size + 1, stride)
        for col in range(0, W - patch_size + 1, stride)
    ]

    for idx, (row, col) in enumerate(tiles):
        pr_patch = prisma_image_resized[:, row:row+patch_size, col:col+patch_size]
        s2_patch = s2_image[:, row:row+patch_size, col:col+patch_size]
        lcz_patch = lcz_image[row:row+patch_size, col:col+patch_size]
        if use_layer:
            lst_patch = lst_image_resized[:, row:row+patch_size, col:col+patch_size]
     
        tifffile.imwrite(prisma_dir / f"tile_{idx:05d}.tif", pr_patch.numpy())
        tifffile.imwrite(s2_dir / f"tile_{idx:05d}.tif", s2_patch.numpy())
        tifffile.imwrite(lcz_dir / f"tile_{idx:05d}.tif", lcz_patch.numpy())
        if use_layer:
            tifffile.imwrite(lst_dir / f"tile_{idx:05d}.tif", lst_patch.numpy())

    return dataset_dir


class LCZDataset(Dataset):
    def __init__(self, prisma_30, s2, lcz_map, lst_path, patch_size, stride, transforms=None, use_tiled_dataset=True, tiled_dataset_dir="./tiled_dataset",use_layer=False):
        self.prisma_30_path = prisma_30
        self.s2_path = s2
        self.lcz_map_path = lcz_map
        self.use_layer = use_layer
        self.lst_path = lst_path
        self.tfms = transforms
        self.patch_size = patch_size
        self.stride = stride
        self.H = 1266
        self.W = 1315
        self.use_tiled_dataset = use_tiled_dataset

        if self.use_tiled_dataset:
            dataset_name = pathlib.Path(prisma_30).parent.name
            self.dataset_dir = pathlib.Path(tiled_dataset_dir) / dataset_name

            self.prisma_dir = self.dataset_dir / "prisma_tiles"
            self.s2_dir = self.dataset_dir / "s2_tiles"
            self.lcz_dir = self.dataset_dir / "lcz_tiles"
            self.lst_tile_dir = self.dataset_dir / "lst_tiles"

            dirs_to_check = [self.prisma_dir, self.s2_dir, self.lcz_dir]
            if self.use_layer:
                dirs_to_check.append(self.lst_tile_dir)

            all_dirs_exist_and_not_empty = True
            for d in dirs_to_check:
                if not os.path.exists(d) or not os.listdir(d):
                    all_dirs_exist_and_not_empty = False
                    break

            if not all_dirs_exist_and_not_empty:
                preprocess_and_save_tiles(prisma_30, s2, lcz_map, self.lst_path, patch_size, stride, tiled_dataset_dir, use_layer=self.use_layer)

            tif_tiles = list(self.prisma_dir.glob("tile_*.tif"))
            pt_tiles = list(self.prisma_dir.glob("tile_*.pt"))

            if tif_tiles:
                self.num_tiles = len(tif_tiles)
                self.file_ext = ".tif"
            elif pt_tiles:
                self.num_tiles = len(pt_tiles)
                self.file_ext = ".pt"
            else:
                self.num_tiles = 0
                self.file_ext = None
                raise FileNotFoundError(f"No tiles found in {self.prisma_dir}")
        else:
            self.prisma_30_full, self.s2_full, self.lcz_map_full, self.lst_full = self.resize()
            self.tile_indices = self.get_tile_indices()


    def __len__(self):
        if self.use_tiled_dataset:
            return self.num_tiles
        else:
            return len(self.tile_indices)

    def __getitem__(self, tile_idx):
        pr_patch = None
        s2_patch = None
        lcz_patch = None
        lst_patch = None

        if self.use_tiled_dataset:
            if self.file_ext == ".tif":
                pr_patch = torch.from_numpy(tifffile.imread(self.prisma_dir / f"tile_{tile_idx:05d}.tif")).float()
                s2_patch = torch.from_numpy(tifffile.imread(self.s2_dir / f"tile_{tile_idx:05d}.tif")).float()
                lcz_patch = torch.from_numpy(tifffile.imread(self.lcz_dir / f"tile_{tile_idx:05d}.tif")).float()
                if self.use_layer:
                    lst_patch = torch.from_numpy(tifffile.imread(self.lst_tile_dir / f"tile_{tile_idx:05d}.tif")).float()
            else:
                pr_patch = torch.load(self.prisma_dir / f"tile_{tile_idx:05d}.pt")
                s2_patch = torch.load(self.s2_dir / f"tile_{tile_idx:05d}.pt")
                lcz_patch = torch.load(self.lcz_dir / f"tile_{tile_idx:05d}.pt")
                if self.use_layer:
                    lst_patch = torch.load(self.lst_tile_dir / f"tile_{tile_idx:05d}.pt")
        else:
            pr_patch, s2_patch, lcz_patch, lst_patch = self.get_tile(tile_idx)


        if self.use_layer:
            assert lst_patch is not None, "lst_patch should not be None when use_layer is True"
            source = torch.cat((s2_patch, pr_patch, lst_patch), dim=0)
        else:
            source = torch.cat((s2_patch, pr_patch), dim=0)
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

        prisma_image_np = tifffile.imread(prisma_path)
        s2_image_np = tifffile.imread(s2_path)
        lcz_image_np = tifffile.imread(lcz_path)
        
        lst_nc_file_path = os.path.join(lst_folder_path, 'LST_in.nc')
        lst_image_np = None
        if self.use_layer:
            try:
                with xr.open_dataset(lst_nc_file_path) as ds:
                    lst_image_xr = ds['LST']
                    if '_FillValue' in lst_image_xr.attrs:
                        fill_value = lst_image_xr.attrs['_FillValue']
                        lst_image_np = lst_image_xr.values.astype(np.float32)
                        lst_image_np[lst_image_np == fill_value] = np.nan
                    else:
                        lst_image_np = lst_image_xr.values
            except Exception as e:
                raise e
        else:
            lst_image_np = np.zeros((1, self.H, self.W), dtype=np.float32)

        if self.use_layer and np.isnan(lst_image_np).any():
            lst_image_np = np.nan_to_num(lst_image_np, nan=0.0)

        prisma_image = torch.from_numpy(prisma_image_np).float()
        s2_image = torch.from_numpy(s2_image_np).float()
        lcz_image = torch.from_numpy(lcz_image_np).float()
        lst_image = torch.from_numpy(lst_image_np).float() if lst_image_np is not None else None


        lst_image_resized = None
        if self.use_layer and lst_image is not None:
            if lst_image.ndim == 2:
                lst_image = lst_image.unsqueeze(0)
            elif lst_image.ndim == 3 and lst_image.shape[0] > 1:
                lst_image = lst_image[0:1, :, :]
            elif lst_image.ndim == 1:
                raise ValueError(f"LST image has unexpected 1D shape: {lst_image.shape}. Expected 2D or 3D.")

            lst_image_resized = F.interpolate(
                lst_image.unsqueeze(0),
                size=(self.H, self.W),
                mode='bicubic',
                align_corners=False
            ).squeeze(0)
        elif not self.use_layer:
            lst_image_resized = torch.zeros((1, self.H, self.W), dtype=torch.float32)
        else:
             lst_image_resized = torch.zeros((1, self.H, self.W), dtype=torch.float32)


        if prisma_image.ndim == 3 and prisma_image.shape[2] > 1:
             prisma_image = prisma_image.permute(2, 0, 1)
        elif prisma_image.ndim == 2:
             prisma_image = prisma_image.unsqueeze(0)

        prisma_image_resized = F.interpolate(
            prisma_image.unsqueeze(0),
            size=(self.H, self.W),
            mode='bicubic',
            align_corners=False
        ).squeeze(0)

        s2_image = s2_image.permute(2, 0, 1)

        if s2_image.shape[1] != self.H or s2_image.shape[2] != self.W:
             s2_image = F.interpolate(
                 s2_image.unsqueeze(0),
                 size=(self.H, self.W),
                 mode='bicubic',
                 align_corners=False
             ).squeeze(0)

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

        pr_patch = self.prisma_30_full[:, row:row+self.patch_size, col:col+self.patch_size]
        s2_patch = self.s2_full[:, row:row+self.patch_size, col:col+self.patch_size]
        lcz_patch = self.lcz_map_full[row:row+self.patch_size, col:col+self.patch_size]
        lst_patch = self.lst_full[:, row:row+self.patch_size, col:col+self.patch_size]

        assert 0 <= tile_idx < len(self.tile_indices), f"Bad idx {tile_idx}"
        assert pr_patch is not None, "pr_patch is None"
        assert s2_patch is not None, "s2_patch is None"
        assert lcz_patch is not None, "lcz_patch is None"
        assert lst_patch is not None, "lst_patch is None"
        return pr_patch, s2_patch, lcz_patch, lst_patch