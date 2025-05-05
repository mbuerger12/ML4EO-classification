import timm
import torch.nn as nn

class TimmSegModel(nn.Module):
    def __init__(self, in_chans, num_classes, arch="segformer_mit_b0"):
        super().__init__()
        # arch options: segformer_mit_b0, segformer_mit_b2,
        #               segmenter_vit_base_patch16_512, etc.
        print("SegFormer variants:", timm.list_models())
        self.model = timm.create_model(
            arch,
            pretrained=True,
            in_chans=in_chans,
            num_classes=num_classes
        )

    def forward(self, x):
        # timm’s segmentation heads outputs [B, num_classes, H, W]
        return self.model(x)