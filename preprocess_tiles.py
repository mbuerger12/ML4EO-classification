import torch
from torch.utils.data import Dataset
import tifffile
import xarray as xr
import numpy as np
import os
import pathlib
import torch.nn.functional as F



def preprocess_tiles(prisma_30, s2, lcz_map, lst, patch_size, stride, output_dir="./tiled_dataset",use_layer=False):
    base_dir = pathlib.Path(output_dir)
    dataset_name = pathlib.Path(prisma_30).parent.name
    dataset_dir = base_dir / dataset_name

    prisma_dir = dataset_dir / "prisma_tiles"
    s2_dir = dataset_dir / "s2_tiles"
    lcz_dir = dataset_dir / "lcz_tiles"
    lst_dir = dataset_dir / "lst_tiles"

    for directory in [prisma_dir, s2_dir, lcz_dir, lst_dir]:
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
        pr_patch = prisma_image_resized[:, row:row + patch_size, col:col + patch_size]
        s2_patch = s2_image[:, row:row + patch_size, col:col + patch_size]
        lcz_patch = lcz_image[row:row + patch_size, col:col + patch_size]
        if use_layer:
            lst_patch = lst_image_resized[:, row:row + patch_size, col:col + patch_size]

        pr = pr_patch.numpy()
        s2 = s2_patch.numpy()
        lcz = lcz_patch.numpy()

        tifffile.imwrite(prisma_dir / f"tile_{idx:05d}.tif", pr, bigtiff=True)
        tifffile.imwrite(s2_dir / f"tile_{idx:05d}.tif", s2, bigtiff=True)
        tifffile.imwrite(lcz_dir / f"tile_{idx:05d}.tif", lcz, bigtiff=True)
        if use_layer:
            tifffile.imwrite(lst_dir / f"tile_{idx:05d}.tif", lst_patch.numpy())

    return dataset_dir