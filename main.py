from tqdm import tqdm
import wandb
from dataset import LCZDataset
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from torch import optim
import numpy as np
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,  jaccard_score, classification_report
import argparse
from collections import defaultdict
import os
from utils.helper_functions import new_log, to_cuda
import time
from torchvision.models.segmentation import fcn_resnet50
from arguments import train_parser
import torch.nn as nn
from iterstrat.ml_stratifiers import (
    MultilabelStratifiedShuffleSplit,
    MultilabelStratifiedKFold
)
import torchmetrics
from torchmetrics.segmentation import DiceScore
from utils.modelCNN1 import ModelCNN1


class Trainer:
    def __init__(self, args: argparse.Namespace):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args

        self.metric_miou = torchmetrics.JaccardIndex(num_classes=18, average='macro', task='multiclass').to(self.device)
        self.metric_pixelacc = torchmetrics.Accuracy(num_classes= 18, task='multiclass').to(self.device)
        self.metric_dice = DiceScore(num_classes=18, average='macro').to(self.device)


        if args.model == "segformer":
            self.model = smp.Segformer(
            encoder_name="mit_b2",
            encoder_weights="imagenet",
            in_channels=244,
            classes=18,
            activation=None
            )

        if args.model == "ownCNN":
            self.model = CNNModel(in_channels=244, num_classes=18)

        elif args.model == "resnet50":
            self.model = fcn_resnet50(pretrained=False, num_classes=18)

            new_conv1 = nn.Conv2d(
                in_channels=244,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            )

            self.model.backbone.conv1 = new_conv1

        elif args.model == 'UNet':
            self.model = smp.Unet(
                encoder_name="resnet34",
                encoder_weights="imagenet",
                in_channels=244,
                classes=18,
                activation=None
            )

        self.model = self.model.to(self.device)
        self.experiment_folder = new_log(os.path.join(args.save_dir, args.dataset),
                                                                          args)[0]
        self.dataloaders = self.get_dataloaders(args)
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            dir=self.experiment_folder,
            name=f"{args.model}-{args.dataset}-{int(time.time())}",
            save_code=True,
            reinit=True
        )
        wandb.watch(self.model, log="all", log_freq=args.logstep_train)
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
        self.all_preds = []
        self.all_labels = []

    def __del__(self):
        pass

    def train(self):
        with tqdm(range(self.epoch, self.args.num_epochs), leave=True) as tnr:
            tnr.set_postfix(training_loss=np.nan, validation_loss=np.nan, best_validation_loss=np.nan)
            for _ in tnr:
                self.train_epoch(tnr)

                if (self.epoch + 1) % self.args.val_every_n_epochs == 0:
                    self.validate()

                self.scheduler.step()
                self.epoch += 1



    def train_epoch(self, tnr=None):
        self.train_stats = defaultdict(float)
        self.model.train()
        with tqdm(self.dataloaders['train'], leave=False) as inner_tnr:
            inner_tnr.set_postfix(training_loss=np.nan)
            for i, sample in enumerate(inner_tnr):
                self.optimizer.zero_grad()
                sample = to_cuda(sample, device=self.device)
                output = self.model(sample['image'])
                if self.args.model == "resnet50":
                    output = output['out']
                loss = F.cross_entropy(output, sample['label'].long().to(self.device))
                self.train_stats["loss"] += loss.detach().cpu().item()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.iter += 1

                if (i + 1) % min(self.args.logstep_train, len(self.dataloaders['train'])) == 0:
                    avg_loss = self.train_stats["loss"] / self.args.logstep_train

                    inner_tnr.set_postfix(training_loss=avg_loss)
                    if tnr is not None:
                        tnr.set_postfix(
                            training_loss=avg_loss,
                            validation_loss=self.val_stats.get('loss', np.nan),
                            best_validation_loss=self.best_optimization_loss
                        )

                    wandb.log({"train/loss": float(avg_loss)}, step=self.iter)

                    # für die nächste Runde zurücksetzen
                    self.train_stats = defaultdict(float)

    def validate(self):
        self.model.eval()
        # reset metrics
        self.metric_pixelacc.reset()
        self.metric_miou.reset()
        self.metric_dice.reset()

        with tqdm(self.dataloaders['val'], leave=False) as inner_tnr:
            inner_tnr.set_postfix(validation_loss=np.nan)
            with torch.no_grad():
                for sample in inner_tnr:
                    sample = to_cuda(sample, self.device)
                    labels = sample['label'].long().to(self.device)
                    output = self.model(sample['image'])
                    if self.args.model == "resnet50":
                        output = output['out']
                    preds = output.argmax(1)

                    self.metric_pixelacc.update(preds, labels)
                    self.metric_miou.update(preds, labels)
                    self.metric_dice.update(preds, labels)

        # extract floats
        oa   = self.metric_pixelacc.compute().item()
        miou = self.metric_miou.compute().item()
        dice = self.metric_dice.compute().item()

        # reset for next epoch
        self.metric_pixelacc.reset()
        self.metric_miou.reset()
        self.metric_dice.reset()

        # log to W&B at the current batch‐step (never regress)
        wandb.log({
            "val/accuracy": float(oa),
            "val/mIoU":     float(miou),
            "val/dice":     float(dice),
        }, step=self.iter)



    def save_model(self):
        pass

    def get_dataloaders(self, args):
        if args.dataset == 'berlin':
            full_dataset = LCZDataset("./dataset/berlin/PRISMA_30.tif", "./dataset/berlin/S2.tif",
                                  "./dataset/berlin/LCZ_MAP.tif", "./", 64, 32, transforms=None, use_tiled_dataset=True, tiled_dataset_dir="./tiled_dataset")
        elif args.dataset == 'athens':
            lst_data_folder_path = "../layer/S3B_SL_2_LST____2025060Athen.SEN3" # Corrected path for Athens's LST data

            full_dataset = LCZDataset(
                "../dataset/Athens/PRISMA_30.tif",
                "../dataset/Athens/S2.tif",
                "../dataset/Athens/LCZ_MAP.tif",
                lst_data_folder_path,
                64, 32, transforms=None
            )
        elif args.dataset == 'milan':
            lst_data_folder_path = "../layer/S3B_SL_2_LST____2025060Milan.SEN3" # Corrected path for Milan's LST data
            full_dataset = LCZDataset(
                "../dataset/Milan/PRISMA_30.tif",
                "../dataset/Milan/S2.tif",
                "../dataset/Milan/LCZ_MAP.tif",
                lst_data_folder_path,
                64, 32, transforms=None
            )

        N = len(full_dataset)
        indices = np.arange(N)
        y_multi = np.zeros((N, 17), dtype=int)

        if args.sampler == "random":
            np.random.seed(42)
            np.random.shuffle(indices)
            split = int(0.8 * len(indices))
            train_idx, val_idx = indices[:split], indices[split:]
            lst_data_folder_path = "../layer/S3B_SL_2_LST____2025060Berlin.SEN3" # Corrected path for Berlin's LST data

            full_dataset = LCZDataset(
                "../dataset/berlin/PRISMA_30.tif",
                "../dataset/berlin/S2.tif",
                "../dataset/berlin/LCZ_MAP.tif",
                lst_data_folder_path,
                64, 32, transforms=None
            )
        elif args.sampler == "stratified":
            msss = MultilabelStratifiedShuffleSplit(
                n_splits=1, test_size=0.2, random_state=42
            )
            train_idx, val_idx = next(msss.split(indices, y_multi))

        elif args.sampler == "skfold":
            mskf = MultilabelStratifiedKFold(
                n_splits=4, shuffle=True, random_state=42
            )
            for fold_idx, (tr, vl) in enumerate(mskf.split(indices, y_multi)):
                if fold_idx == 3:
                    train_idx, val_idx = tr, vl
                    break
        indices = np.arange(len(full_dataset))
        np.random.seed(42)  # for reproducibility
        np.random.shuffle(indices)

        # Define split point
        split = int(0.8 * len(indices))
        train_idx, val_idx = indices[:split], indices[split:]



        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        train_loader = DataLoader(
            full_dataset,
            batch_size=8,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True
        )

        val_loader = DataLoader(
            full_dataset,
            batch_size=8,
            sampler=val_sampler,
            num_workers=4,
            pin_memory=True
        )
        return {'train': train_loader, 'val': val_loader}



if __name__ == '__main__':
    print(torch.cuda.is_available())
    print(torch.version.cuda)
    torch.cuda.empty_cache()
    args = train_parser.parse_args()
    print(train_parser.format_values())
    trainer = Trainer(args)

    since = time.time()
    trainer.train()
    time_elapsed = time.time() - since
    print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
