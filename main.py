from tqdm import tqdm
#import wandb
from dataset import LCZDataset
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from torch import optim
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, jaccard_score, classification_report, confusion_matrix
import argparse
from collections import defaultdict
import os
from utils.helper_functions import new_log, to_cuda
import time
from torchvision.models.segmentation import fcn_resnet50
from arguments import train_parser

class Trainer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.model = smp.Segformer(
        encoder_name="mit_b2",  # backbone size: b0, b1, b2, etc.
        encoder_weights="imagenet",  # pretrained weights
        in_channels=244,  # e.g. 4+10 bands
        classes=18,  # LCZ classes
        activation=None  # we'll use raw logits + CrossEntropyLoss
        )

        self.experiment_folder = new_log(os.path.join(args.save_dir, args.dataset),
                                                                          args)[0]
        self.dataloaders = self.get_dataloaders(args)
        wandb.init(project=args.wandb_project, dir=self.experiment_folder)
        wandb.config.update(self.args)
        self.writer = None
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.w_decay = 0.0001

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=self.w_decay)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=0.001, total_steps=self.batch_size*self.num_epochs*152, pct_start=0.1, anneal_strategy='cos', cycle_momentum=False)
        self.epoch = 0
        self.iter = 0
        self.train_stats = defaultdict(lambda: np.nan)
        self.val_stats = defaultdict(lambda: np.nan)
        self.best_optimization_loss = np.inf

    def __del__(self):
        pass

    def train(self):
        if self.args.model == "randomforest":
            self.train_random_forest()
            self.test_random_forest()
            return
        else:
            with tqdm(range(self.epoch, self.args.num_epochs), leave=True) as tnr:
                tnr.set_postfix(training_loss=np.nan, validation_loss=np.nan, best_validation_loss=np.nan)
                for _ in tnr:
                    self.train_epoch(tnr)

                    if (self.epoch + 1) % self.args.val_every_n_epochs == 0:
                        self.validate()

                    if self.args.save_model in ['last', 'both']:
                        self.save_model('last')

                if self.args.lr_scheduler == 'step':
                    self.scheduler.step()
                    if self.use_wandb:
                        wandb.log({'log_lr': np.log10(self.scheduler.get_last_lr())}, self.iter)
                    else:
                        self.writer.add_scalar('log_lr', np.log10(self.scheduler.get_last_lr()), self.epoch)

                self.epoch += 1

                self.test()

    def train_random_forest(self):
        print("Training Random Forest...")
        X_all = []
        y_all = []

        for sample in tqdm(self.dataloaders['train']): # (B, C, H, W)
            image = sample['image'].numpy().transpose(0, 2, 3, 1)   # (B, H, W, C)
            label = sample['label'].numpy()  # (B, H, W)
            X_all.append(image)
            y_all.append(label)

        X_all = np.concatenate(X_all, axis=0)
        y_all = np.concatenate(y_all, axis=0)

        self.model.fit(X_all, y_all)

    def test_random_forest(self):
        print("Testing Random Forest...")
        all_preds, all_labels = [], []

        # Reset metrics
        self.metric_miou.reset()
        self.metric_pixelacc.reset()
        self.metric_dice.reset()

        for sample in tqdm(self.dataloaders['test']):
            image = sample['image'].numpy().transpose(0, 2, 3, 1)  # (B, H, W, C)
            label = sample['label'].numpy()  # (B, H, W)

            preds = self.model.predict(image)  # (B, H, W)

            # Flatten and convert to torch.Tensor
            pred_tensor = torch.tensor(preds, dtype=torch.int64).to(self.device)
            label_tensor = torch.tensor(label, dtype=torch.int64).to(self.device)

            # Update metrics
            self.metric_miou.update(pred_tensor, label_tensor)
            self.metric_pixelacc.update(pred_tensor, label_tensor)
            self.metric_dice.update(pred_tensor, label_tensor)

        # Compute torchmetrics
        miou = self.metric_miou.compute().item()
        pixel_acc = self.metric_pixelacc.compute().item()
        dice = self.metric_dice.compute().item()

        # Log to W&B
        wandb.log({
            "rf/test/mIoU": miou,
            "rf/test/pixel_accuracy": pixel_acc,
            "rf/test/dice": dice,
        })

        print("=== Random Forest Test Results ===")
        print(f"Pixel Accuracy : {pixel_acc:.4f}")
        print(f"Mean IoU       : {miou:.4f}")
        print(f"Dice Score     : {dice:.4f}")



    def train_epoch(self, tnr=None):
        self.train_stats = defaultdict(float)
        self.model.train()
        with tqdm(self.dataloaders['train'], leave=False) as inner_tnr:
            inner_tnr.set_postfix(training_loss=np.nan)
            for i, sample in enumerate(inner_tnr):
                self.optimizer.zero_grad()
                sample = to_cuda(sample)
                output = self.model(sample['image'])
                if self.args.model == "resnet50":
                    output = output['out']
                loss = F.cross_entropy(output, sample['label'].long().to(self.device))
                self.train_stats["loss"] += loss.detach().cpu().item()
                loss.backward()
                optimizer.step()
                scheduler.step()
                self.iter += 1

                if (i + 1) % min(self.args.logstep_train, len(self.dataloaders.datasets['train'])) == 0:
                    self.train_stats = {k: v / self.args.logstep_train for k, v in self.train_stats.items()}

                    inner_tnr.set_postfix(training_loss=avg_loss)
                    if tnr is not None:
                        tnr.set_postfix(training_loss=self.train_stats['loss'],
                                        validation_loss=self.val_stats['loss'],
                                        best_validation_loss=self.best_optimization_loss)

                    if self.use_wandb:
                        wandb.log({k + '/train': v for k, v in self.train_stats.items()}, self.iter)
                    else:
                        for key in self.train_stats:
                            self.writer.add_scalar('train/' + key, self.train_stats[key], self.iter)

                    # reset metrics
                    self.train_stats = defaultdict(float)

    def validate(self):
        pass

    def save_model(self):
        pass

    def get_dataloaders(self, args):
        if args.dataset == 'berlin':
            train_ds = LCZDataset("./dataset/berlin/PRISMA_30.tif", "./dataset/berlin/S2.tif",
                                  "./dataset/berlin/LCZ_MAP.tif", 64, 32, transforms=None)
            val_ds = LCZDataset("./dataset/berlin/PRISMA_30.tif", "./dataset/berlin/S2.tif", "./dataset/berlin/LCZ_MAP.tif",
                                64, 32, transforms=None)

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

            return {'train': train_loader, 'val': val_loader}




if __name__ == '__main__':
    print(torch.cuda.is_available())
    print(torch.version.cuda)
    torch.cuda.empty_cache()  # Clear unused memory in PyTorch's cache
    args = train_parser.parse_args()
    print(train_parser.format_values())

    in_chans = 453 # 245 
    num_classes = 18 
    
    trainer = Trainer(args)
    since = time.time()
    trainer.train()
    time_elapsed = time.time() - since
    print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))




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



