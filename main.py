from dataset import LCZDataset
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from torch import optim
import numpy as np
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,  jaccard_score, classification_report


if __name__ == '__main__':
    num_epochs = 100
    in_chans = 244
    num_classes = 17
    lr = 0.001
    w_decay = 0.0001


    model = smp.Segformer(
        encoder_name="mit_b2",  # backbone size: b0, b1, b2, etc.
        encoder_weights="imagenet",  # pretrained weights
        in_channels=in_chans,  # e.g. 4+10 bands
        classes=18,  # LCZ classes
        activation=None  # we'll use raw logits + CrossEntropyLoss
    )
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=w_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, total_steps=28 * 40000, pct_start=0.1,
                                                   anneal_strategy='cos', cycle_momentum=False)
    train_ds = LCZDataset("./dataset/berlin/PRISMA_30.tif","./dataset/berlin/S2.tif", "./dataset/berlin/LCZ_MAP.tif" , 64, 32, transforms=None)
    val_ds = LCZDataset("./dataset/berlin/PRISMA_30.tif","./dataset/berlin/S2.tif", "./dataset/berlin/LCZ_MAP.tif" , 64, 32, transforms=None)

    # 3) Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(device)
    for epoch in range(num_epochs):
        print(epoch)
        model.train()
        for x, l in train_loader:
            x = x.to(device)  # [B, C_pr, 256,256]
            label = l.to(device)  # [B, 256,256]
            optimizer.zero_grad()
            output = model(x)
            loss = F.cross_entropy(output, label.long())
            print(loss)
            loss.backward()
            print("finished backward")
            optimizer.step()
            scheduler.step()

        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x, l in val_loader:
                x = x.to(device)
                labels = l.to(device)
                logits = model(x)  # [B, 17, H, W]
                preds = logits.argmax(1)  # [B, H, W]

                all_preds.append(preds.cpu().numpy().ravel())
                all_labels.append(labels.cpu().numpy().ravel())

        # flatten your predictions & labels
        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels)

        # figure out which labels actually occur in the ground truth
        present_labels = np.unique(y_true)  # e.g. [0,1,2,5,7,...]

        # 1) Pixel accuracy
        acc = accuracy_score(y_true, y_pred)

        # 2) Per-class Precision / Recall / F1 (only for present_labels)
        prec, rec, f1, sup = precision_recall_fscore_support(
            y_true, y_pred,
            labels=present_labels,
            zero_division=0  # sets any 0/0 to 0 instead of warning
        )

        # 3) Per-class IoU
        ious = jaccard_score(
            y_true, y_pred,
            labels=present_labels,
            average=None,
            zero_division=0
        )

        # 4) Mean IoU
        mean_iou = ious.mean()

        # 5) Optional: a nice text report for present classes
        print(classification_report(
            y_true, y_pred,
            labels=present_labels,
            target_names=[f"LCZ_{i}" for i in present_labels],
            zero_division=0
        ))

        # 6) Print summary
        print(f"Pixel Accuracy: {acc:.4f}")
        print(f"Mean IoU      : {mean_iou:.4f}")
        for lbl, p, r, f, iou in zip(present_labels, prec, rec, f1, ious):
            print(f"LCZ {int(lbl):2d} → P {p:.3f}, R {r:.3f}, F1 {f:.3f}, IoU {iou:.3f}")
