import torch
print("Torch version:   ", torch.__version__)
print("CUDA built with: ", torch.version.cuda)        # e.g. “11.7” or None
print("cuDNN enabled:  ", torch.backends.cudnn.enabled)
print("CUDA available: ", torch.cuda.is_available())
print("GPU count:      ", torch.cuda.device_count())
if torch.cuda.device_count()>0:
    print("Device name:   ", torch.cuda.get_device_name(0))


