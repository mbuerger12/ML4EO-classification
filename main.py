from tqdm import tqdm
#import wandb
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
from utils.helper_functions import new_log, to_device
import time
from arguments import train_parser


class Trainer:
    def __init__(self, args: argparse.Namespace):
        self.device = torch.device("cpu")
        print("Forcing device to CPU as CUDA is not available or enabled.") 
        in_chans_for_model = 453#245 
        self.model = smp.Segformer(
            encoder_name="mit_b2",
            encoder_weights="imagenet",
            in_channels=in_chans_for_model, 
            classes=18, 
            activation=None
        ).to(self.device)

        self.args = args
        self.experiment_folder = new_log(os.path.join(args.save_dir, args.dataset),
                                                                          args)[0]
        self.dataloaders = self.get_dataloaders(args)
        #wandb.init(project=args.wandb_project, dir=self.experiment_folder)
        #wandb.config.update(self.args)
        self.writer = None
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.w_decay = 0.0001

        self.optimizer = optim.AdamW(  self.model.parameters(),lr=args.lr,weight_decay=self.w_decay)
        self.total_steps = len(self.dataloaders['train']) * self.num_epochs
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer,max_lr=args.lr, total_steps=self.total_steps,pct_start=0.1,anneal_strategy='cos',cycle_momentum=False)
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
                sample = to_device(sample, self.device)
                output = self.model(sample['image'])
                loss = F.cross_entropy(output, sample['label'].long())
                self.train_stats["loss"] += loss.detach().cpu().item()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.iter += 1

                if (i + 1) % min(self.args.logstep_train, len(self.dataloaders['train'])) == 0:
                    self.train_stats = {k: v / self.args.logstep_train for k, v in self.train_stats.items()}

                    inner_tnr.set_postfix(training_loss=self.train_stats['loss'])
                    if tnr is not None:
                        tnr.set_postfix(training_loss=self.train_stats['loss'],
                                        validation_loss=self.val_stats['loss'],
                                        best_validation_loss=self.best_optimization_loss)


                    #wandb.log({k + '/train': v for k, v in self.train_stats.items()}, self.iter)
                    # reset metrics
                    self.train_stats = defaultdict(float)
    def validate(self):
        self.model.eval()
        self.all_preds = []
        self.all_labels = []
        with tqdm(self.dataloaders['val'], leave=False) as inner_tnr:
            inner_tnr.set_postfix(training_loss=np.nan)
            with torch.no_grad():
                for i, sample in enumerate(inner_tnr):
                    sample = to_device(sample, self.device)
                    labels = sample['label'].long()
                    output = self.model(sample['image'])

                    preds = output.argmax(1)  # [B, H, W]

                    self.all_preds.append(preds.cpu().numpy().ravel())
                    self.all_labels.append(labels.cpu().numpy().ravel())
        # flatten your predictions & labels
        y_pred = np.concatenate(self.all_preds)
        y_true = np.concatenate(self.all_labels)

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
    def save_model(self):
        pass

    def get_dataloaders(self, args):
        if args.dataset == 'berlin':
            lst_data_folder_path = "../layer/S3B_SL_2_LST____2025060Berlin.SEN3" # Corrected path for Berlin's LST data

            full_dataset = LCZDataset(
                "../dataset/berlin/PRISMA_30.tif",
                "../dataset/berlin/S2.tif",
                "../dataset/berlin/LCZ_MAP.tif",
                lst_data_folder_path, 
                64, 32, transforms=None
            )
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
        indices = np.arange(len(full_dataset))
        np.random.seed(42)  # for reproducibility
        np.random.shuffle(indices)

        # Define split point
        split = int(0.8 * len(indices))
        train_idx, val_idx = indices[:split], indices[split:]

        # Create samplers
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        # 3) Create DataLoaders
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

    in_chans = 453 # 245 
    num_classes = 18 
    
    trainer = Trainer(args)
    since = time.time()
    trainer.train()
    time_elapsed = time.time() - since
    print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
