import torch
from torch.utils.data import Dataset
import tifffile


class LCZDataset(Dataset):
    def __init__(self, prisma_30, s2, lcz_map, patch_size, stride, transforms=None):
        self.prisma_30_path = prisma_30
        self.s2_path = s2
        self.lcz_map_path = lcz_map
        self.tfms = transforms
        self.patch_size = patch_size
        self.stride = stride
        self.H = 1266
        self.W = 1315
        self.prisma_30, self.s2, self.lcz_map = self.resize()
        self.tile_indices = self.get_tile_indices()
        print(self.tile_indices)




    def __len__(self):
        return len(self.tile_indices)

    def __getitem__(self, tile_idx):
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
