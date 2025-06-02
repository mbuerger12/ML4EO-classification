import os
import csv
import random
import numpy as np
import time
import matplotlib.pyplot as plt
import torch

def to_device(sample, device):
    if isinstance(sample, list):
        sampleout = []
        for val in sample:
            sampleout.append(val.to(device))
    elif isinstance(sample, dict):
        sampleout = {}
        for key, val in sample.items():
            if isinstance(val, list):
                sampleout[key] = [v.to(device) for v in val]
            else:
                sampleout[key] = val.to(device)
    else:
        sampleout = sample.to(device)
    return sampleout

def seed_all(seed):
    # Fix all random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def new_log(folder_path, args=None):
    os.makedirs(folder_path, exist_ok=True)
    n_exp = len(os.listdir(folder_path))
    randn  = round((time.time()*1000000) % 1000)
    experiment_folder = os.path.join(folder_path, f'experiment_{n_exp}_{randn}')
    os.mkdir(experiment_folder)

    if args is not None:
        args_dict = args.__dict__
        write_params(args_dict, os.path.join(experiment_folder, 'args' + '.csv'))

    return experiment_folder, n_exp, randn


def write_params(params, path):
    with open(path, 'w') as fh:
        writer = csv.writer(fh)
        writer.writerow(['key', 'value'])
        for data in params.items():
            writer.writerow([el for el in data])

def store_images(image_folder, experiment_folder, pred, label, epoch):
    """
    Stores the label and multiple predictions in an epoch-specific subfolder:
    <image_folder>/<experiment_folder>/epoch_{epoch}/

    Args:
        image_folder (str): Base folder where images are stored.
        experiment_folder (str): Name of the specific experiment folder.
        preds (list or tuple of torch.Tensor): Multiple prediction tensors.
        label (torch.Tensor): Label tensor.
        epoch (int): Current epoch number.
    """
    # 1. Create the base experiment folder
    experiment_path = os.path.join(image_folder, experiment_folder)
    os.makedirs(experiment_path, exist_ok=True)

    # 2. Create the epoch-specific directory: e.g., "epoch_0", "epoch_1", etc.
    epoch_dir = os.path.join(experiment_path, f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)

    # 3. Store the label
    index = str(len(os.listdir(epoch_dir)))
    plot_tensor_image(label, epoch_dir, title=f"label_{index}")

    # 4. Loop over predictions and store each one
    plot_tensor_image(pred, epoch_dir, title=f"prediction_{index}")


def plot_tensor_image(img_tensor, path, title="Image", cmap="viridis", slice_idx=0):
    """
    Plots the given image tensor and saves it as a PNG to the given path.
    """
    # Handle batch dimension
    if len(img_tensor.shape) == 4 and img_tensor.shape[0] == 1:
        # (1, C, H, W) => (C, H, W)
        img_tensor = img_tensor[0]

    if len(img_tensor.shape) == 4:
        # For (N, C, H, W) with N>1, pick a slice_idx if needed
        if slice_idx < 0 or slice_idx >= img_tensor.shape[0]:
            raise ValueError(f"Invalid slice_idx {slice_idx} for shape {img_tensor.shape}")
        img_tensor = img_tensor[slice_idx]

    # Convert tensor to numpy on CPU
    img = img_tensor.detach().cpu().numpy()

    # If the shape is (C, H, W) => convert to (H, W, C)
    if len(img.shape) == 3 and img.shape[0] in [1, 3]:
        img = img.transpose(1, 2, 0)  # (C, H, W) => (H, W, C)

        # If single channel, squeeze out
        if img.shape[-1] == 1:
            img = img[..., 0]

    plt.figure(figsize=(6, 6))
    if len(img.shape) == 2:
        plt.imshow(img, cmap=cmap)
    else:
        plt.imshow(img)
    plt.title(title)
    plt.axis("off")

    # Save
    save_path = os.path.join(path, f"{title}.png")
    plt.savefig(save_path)
    plt.close()  # close the figure to free memory


def apply_random_crop_lr_hr(
    lr_tensor: torch.Tensor,
    hr_tensor: torch.Tensor,
    mask_tensor: torch.Tensor,
    lowres_crop_size: int,
    upscale_factor: int
):
    """
    Takes an LR image (lr_tensor) and the corresponding HR image (hr_tensor),
    crops them randomly so they match in spatial location, then returns the cropped versions.

    Args:
        lr_tensor: Low-resolution tensor of shape [C, H, W]
        hr_tensor: High-resolution tensor of shape [C, H, W] (or bigger)
        mask_tensor: Corresponding mask of shape [C, H, W]
        lowres_crop_size: how many pixels to crop from the LR image
        upscale_factor: ratio of HR resolution to LR resolution
    Returns:
        (lr_cropped, hr_cropped, mask_cropped)
    """
    # LR random crop
    i, j, h, w = RandomCrop.get_params(lr_tensor, output_size=(lowres_crop_size, lowres_crop_size))
    lr_cropped = crop(lr_tensor, i, j, h, w)

    # For HR, crop the corresponding region
    # We simply multiply the same offsets by the upscale_factor
    i_hr, j_hr = i * upscale_factor, j * upscale_factor
    h_hr, w_hr = h * upscale_factor, w * upscale_factor
    hr_cropped = crop(hr_tensor, i_hr, j_hr, h_hr, w_hr)
    mask_cropped = crop(mask_tensor, i_hr, j_hr, h_hr, w_hr)

    return lr_cropped, hr_cropped, mask_cropped


def depth_to_colormap(depth_image: torch.Tensor, colormap='viridis') -> torch.Tensor:
    """
    Convert a single-channel depth image to a 3-channel RGB representation using a Matplotlib colormap.
    Expects depth_image in shape [H, W] or [1, H, W]. Returns [3, H, W].
    """
    depth_image_2d = depth_image.squeeze()  # shape => [H, W]
    colormap_func = plt.get_cmap(colormap)
    # Map [H, W] -> [H, W, 4], then slice :3 for RGB
    rgb_image = colormap_func(depth_image_2d.cpu().numpy())[..., :3]
    # Convert to Torch [3, H, W]
    rgb_tensor = torch.from_numpy(rgb_image).permute(2, 0, 1).float()
    return rgb_tensor
