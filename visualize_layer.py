import tifffile
import xarray as xr
import os

# --- Paths to your data ---
s2_path = "../dataset/berlin/S2.tif"
prisma_path = "../dataset/berlin/PRISMA_30.tif"
# The LST data is in a folder, and the file is LST_in.nc inside.
lst_folder_path = "../layer/S3B_SL_2_LST____2025060Berlin.SEN3"
lst_nc_file_path = os.path.join(lst_folder_path, 'LST_in.nc')

# --- Count S2 channels ---
s2_img = tifffile.imread(s2_path)
s2_channels = s2_img.shape[2] if s2_img.ndim == 3 else 1
print(f"S2 Channels: {s2_channels}")

# --- Count PRISMA channels ---
prisma_img = tifffile.imread(prisma_path)
prisma_channels = prisma_img.shape[2] if prisma_img.ndim == 3 else 1
print(f"PRISMA Channels: {prisma_channels}")

# --- Count LST channels ---
# LST is assumed to be 1 channel, but let's confirm programmatically
lst_channels = 0
try:
    with xr.open_dataset(lst_nc_file_path) as ds:
        # Assuming 'LST' is the band you want. If it has multiple dimensions beyond H,W,
        # you might need to adjust how many channels it contributes.
        # Given it's Land Surface Temperature, it's typically a single band.
        lst_data_array = ds['LST']
        # If lst_data_array.ndim is 2 (H, W), it contributes 1 channel.
        # If lst_data_array.ndim is 3 (C, H, W) or (T, H, W), and you take the first band, it contributes 1 channel.
        # If the NetCDF file structure makes a single 'LST' variable have multiple bands, it would be C.
        # Let's verify the shape of the LST variable itself.
        lst_shape = lst_data_array.shape
        if len(lst_shape) == 2: # (H, W) -> 1 channel
            lst_channels = 1
        elif len(lst_shape) == 3: # (C, H, W) or (T, H, W)
            # Assuming first dimension is channel/time dimension, and we're taking the first band
            lst_channels = 1 # We specifically took lst_image[0:1, :, :] earlier, so we add 1 channel.
        else:
            print(f"Warning: Unexpected LST data shape: {lst_shape}. Assuming 1 channel.")
            lst_channels = 1 # Default to 1 if structure is ambiguous
except Exception as e:
    print(f"Error inspecting LST file: {e}. Assuming 1 channel for now.")
    lst_channels = 1 # Fallback if file cannot be read

print(f"LST Channels: {lst_channels}")

# --- Calculate Total Input Channels ---
total_input_channels = s2_channels + prisma_channels + lst_channels
print(f"Calculated Total Input Channels for Model: {total_input_channels}")