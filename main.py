from tqdm import tqdm
import wandb
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
from preprocess_tiles import preprocess_tiles
from torchvision.models.segmentation import fcn_resnet50
from arguments import train_parser
import torch.nn as nn
from iterstrat.ml_stratifiers import (
    MultilabelStratifiedShuffleSplit,
    MultilabelStratifiedKFold
)
import torchmetrics
from torchmetrics.segmentation import DiceScore
import matplotlib.pyplot as plt
from ModelCNN1 import ModelCNN1
from torch.utils.data import Subset

class Trainer:
    def __init__(self, args: argparse.Namespace):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args

        if self.args.use_layer:
            self.in_chans_for_model = 453
        else:
            self.in_chans_for_model = 244
        self.metric_miou = torchmetrics.JaccardIndex(num_classes=18, average='macro', task='multiclass').to(self.device)
        self.metric_pixelacc = torchmetrics.Accuracy(num_classes= 18, task='multiclass').to(self.device)
        self.metric_dice = DiceScore(num_classes=18, average='macro').to(self.device)


        if args.model == "segformer":
            print("in_chans_for_model", self.in_chans_for_model)
            self.model = smp.Segformer(
            encoder_name="mit_b2",
            encoder_weights="imagenet",
            in_channels=self.in_chans_for_model,
            classes=18,
            activation=None
            )

        if args.model == "ownCNN":
            self.model = ModelCNN1(in_channels=self.in_chans_for_model, num_classes=18)

        elif args.model == "resnet50":
            self.model = fcn_resnet50(pretrained=False, num_classes=18)

            new_conv1 = nn.Conv2d(
                in_channels=self.in_chans_for_model,
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
                in_channels=self.in_chans_for_model,
                classes=18,
                activation=None
            )

        elif args.model == "randomforest":
            from random_forest_model import RandomForestSegmentation
            self.model = RandomForestSegmentation(
                n_estimators=args.rf_n_estimators,
                max_depth=args.rf_max_depth,
                class_weight=args.rf_class_weight,
                random_state=args.rf_random_state
            )

        
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
        if args.model != "randomforest":
            self.model = self.model.to(self.device)
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
        wandb.watch(self.model, log="all", log_freq=args.logstep_train)
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.w_decay = 0.0001

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=self.w_decay)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=0.0001, total_steps=self.batch_size*self.num_epochs*152, pct_start=0.1, anneal_strategy='cos', cycle_momentum=False)
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

                    self.scheduler.step()
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
                sample = to_cuda(sample, device=self.device)
                output = self.model(sample['image'])
                if self.args.model == "resnet50":
                    output = output['out']
                loss = F.cross_entropy(output, sample['label'].long().to(self.device))
                self.train_stats["loss"] += loss.detach().cpu().item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0) # Adjust max_norm as needed
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

        all_preds = []
        all_labels = []

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

                    # Collect predictions and labels for confusion matrix
                    all_preds.append(preds.cpu().numpy().reshape(-1))
                    all_labels.append(labels.cpu().numpy().reshape(-1))

        # extract floats
        oa   = self.metric_pixelacc.compute().item()
        miou = self.metric_miou.compute().item()
        dice = self.metric_dice.compute().item()

        # reset for next epoch
        self.metric_pixelacc.reset()
        self.metric_miou.reset()
        self.metric_dice.reset()

        # Calculate confusion matrix
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        conf_matrix = confusion_matrix(all_labels, all_preds, labels=range(18))

        # Normalize the confusion matrix
        conf_norm = conf_matrix.sum(axis=1)[:, np.newaxis]
        conf_matrix_norm = conf_matrix.astype('float') / conf_norm
        conf_matrix_norm = np.nan_to_num(conf_matrix_norm)  # Replace NaN with 0
        conf_matrix_norm = conf_matrix_norm.T

        # Create a wandb Image with the confusion matrix
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(conf_matrix_norm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
        ax.set_title('Validation Confusion Matrix (Normalized)')
        plt.colorbar(im)

        # Set ticks and labels
        classes = [f"Class {i}" for i in range(18)]  # Replace with actual class names if available
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_yticklabels(classes)

        # Add text annotations
        thresh = conf_matrix_norm.max() / 2.
        for i in range(conf_matrix_norm.shape[0]):
            for j in range(conf_matrix_norm.shape[1]):
                ax.text(j, i, f"{conf_matrix_norm[i, j]:.2f}",
                        ha="center", va="center",
                        fontsize=8)

        ax.set_xlabel('True Label')
        ax.set_ylabel('Predicted Label')
        plt.tight_layout()
        plt.show()

        # Log to W&B at the current batch‐step
        wandb.log({
            "val/accuracy": float(oa),
            "val/mIoU":     float(miou),
            "val/dice":     float(dice),
            "val/confusion_matrix": wandb.Image(fig)
        }, step=self.iter)

        plt.close(fig)

    def test(self):
        """
        Führt eine vollständige Evaluation auf dem Test-Datensatz durch.
        Berechnet Pixel-Accuracy, mIoU, Dice Score sowie
        Precision, Recall, F1-Score und Jaccard (IoU) über alle Klassen.
        Loggt die Ergebnisse in W&B und gibt sie am Ende auf der Konsole aus.
        """
        self.model.eval()
        # TorchMetrics zurücksetzen
        self.metric_pixelacc.reset()
        self.metric_miou.reset()
        self.metric_dice.reset()

        all_preds = []
        all_labels = []

        with tqdm(self.dataloaders['test'], leave=False) as tnr:
            tnr.set_postfix(test_loss=np.nan)
            with torch.no_grad():
                for sample in tnr:
                    sample = to_cuda(sample, self.device)
                    images = sample['image']
                    labels = sample['label'].long().to(self.device)  # [B, H, W]

                    # Vorwärtsdurchlauf
                    output = self.model(images)
                    if self.args.model == "resnet50":
                        output = output['out']
                    preds = output.argmax(dim=1)  # [B, H, W]

                    # TorchMetrics updaten
                    self.metric_pixelacc.update(preds, labels)
                    self.metric_miou.update(preds, labels)
                    self.metric_dice.update(preds, labels)

                    # Für sklearn: flattenen (Batch, H, W) → (N_pixels,)
                    all_preds.append(preds.cpu().numpy().reshape(-1))
                    all_labels.append(labels.cpu().numpy().reshape(-1))

        # TorchMetrics berechnen
        pixel_acc = self.metric_pixelacc.compute().item()
        miou      = self.metric_miou.compute().item()
        dice      = self.metric_dice.compute().item()

        # TorchMetrics zurücksetzen (optional für zukünftige Runs)
        self.metric_pixelacc.reset()
        self.metric_miou.reset()
        self.metric_dice.reset()

        # Arrays zusammenfügen
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # sklearn-Metriken berechnen (multi-class)
        # Hinweis: zero_division=0, damit keine Fehler bei Klassen ohne Vorkommen entstehen.
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )
        # Jaccard-Score (IoU) pro Klasse, dann Durchschnitt
        jaccard = jaccard_score(
            all_labels, all_preds, average='macro', zero_division=0
        )
        # Gesamt-Accuracy (pixelweise)
        acc_sklearn = accuracy_score(all_labels, all_preds)

        # Classification Report (Pro-Klasse) – optional zum Ausdrucken
        class_report = classification_report(
            all_labels, all_preds, zero_division=0
        )

        # Calculate confusion matrix
        conf_matrix = confusion_matrix(all_labels, all_preds, labels=range(18))

        # Normalize the confusion matrix
        conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        conf_matrix_norm = np.nan_to_num(conf_matrix_norm)  # Replace NaN with 0

        # Create a wandb Image with the confusion matrix
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(conf_matrix_norm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
        ax.set_title('Test Confusion Matrix (Normalized)')
        plt.colorbar(im)

        # Set ticks and labels
        classes = [f"Class {i}" for i in range(18)]  # Replace with actual class names if available
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_yticklabels(classes)

        # Add text annotations
        thresh = conf_matrix_norm.max() / 2.
        for i in range(conf_matrix_norm.shape[0]):
            for j in range(conf_matrix_norm.shape[1]):
                ax.text(j, i, f"{conf_matrix_norm[i, j]:.2f}",
                        ha="center", va="center",
                        fontsize=8)

        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        plt.tight_layout()

        # Ergebnisse in W&B loggen
        wandb.log({
            "test/accuracy_pixel": pixel_acc,
            "test/accuracy_sklearn": acc_sklearn,
            "test/mIoU_torchmetrics": miou,
            "test/IoU_sklearn": jaccard,
            "test/dice": dice,
            "test/precision": precision,
            "test/recall": recall,
            "test/f1": f1,
            "test/confusion_matrix": wandb.Image(fig)
        }, step=self.iter)

        plt.close(fig)

        # Ausgabe auf der Konsole
        print("=== Test-Ergebnisse ===")
        print(f"Pixel-Accuracy (TorchMetrics): {pixel_acc:.4f}")
        print(f"Pixel-Accuracy (sklearn)    : {acc_sklearn:.4f}")
        print(f"mIoU  (TorchMetrics)        : {miou:.4f}")
        print(f"IoU   (sklearn)             : {jaccard:.4f}")
        print(f"Dice-Score (TorchMetrics)   : {dice:.4f}")
        print(f"Precision (macro)           : {precision:.4f}")
        print(f"Recall    (macro)           : {recall:.4f}")
        print(f"F1-Score  (macro)           : {f1:.4f}")
        print("\nClassification Report (pro Klasse):")
        print(class_report)

        return {
            "pixel_acc_torch": pixel_acc,
            "pixel_acc_sklearn": acc_sklearn,
            "mIoU_torch": miou,
            "IoU_sklearn": jaccard,
            "dice": dice,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def save_model(self):
        pass

    def get_dataloaders(self, args):
        if args.dataset == 'test':
            print("create milano")
            preprocess_tiles("./dataset/Milan/PRISMA_30.tif", "./dataset/Milan/S2.tif", "./dataset/Milan/LCZ_MAP.tif", "./", 64, 32, output_dir="./tiled_dataset", use_layer=False)
        print("use_layer", self.args.use_layer)
        if args.dataset == 'berlin':
            full_dataset = LCZDataset("./dataset/berlin/PRISMA_30.tif", "./dataset/berlin/S2.tif",
                                  "./dataset/berlin/LCZ_MAP.tif",64, 32, transforms=None, use_tiled_dataset=True, tiled_dataset_dir="./tiled_dataset")
        elif args.dataset == 'athens':
            full_dataset = LCZDataset(
                "./dataset/Athens/PRISMA_30.tif",
                "./dataset/Athens/S2.tif",
                "./dataset/Athens/LCZ_MAP.tif",

                64, 32, transforms=None, use_tiled_dataset=True, tiled_dataset_dir="./tiled_dataset",use_layer=self.args.use_layer
            )
        elif args.dataset == 'milan':
            full_dataset = LCZDataset(
                "./dataset/Milan/PRISMA_30.tif",
                "./dataset/Milan/S2.tif",
                "./dataset/Milan/LCZ_MAP.tif",
                64, 32, transforms=None, use_tiled_dataset=False, tiled_dataset_dir="./tiled_dataset"
            )
        elif args.dataset == "full":
            berlin_dataset = LCZDataset("./dataset/berlin/PRISMA_30.tif",
                                        "./dataset/berlin/S2.tif",
                                    "./dataset/berlin/LCZ_MAP.tif",
                                         64,
                                        32,
                                        use_tiled_dataset=False,
                                        tiled_dataset_dir="./tiled_dataset")
            athens_dataset = LCZDataset(
                "./dataset/Athens/PRISMA_30.tif",
                "./dataset/Athens/S2.tif",
                "./dataset/Athens/LCZ_MAP.tif",
                64, 32, use_tiled_dataset=False, tiled_dataset_dir="./tiled_dataset"
            )
            milan_dataset = LCZDataset(
                "./dataset/Milan/PRISMA_30.tif",
                "./dataset/Milan/S2.tif",
                "./dataset/Milan/LCZ_MAP.tif",
                64, 32,  use_tiled_dataset=False, tiled_dataset_dir="./tiled_dataset"
            )

        if args.dataset == "full":
            full_dataset = torch.utils.data.ConcatDataset([berlin_dataset, athens_dataset])
            test_dataset = milan_dataset
            test_idx = np.arange(len(test_dataset))
            indices = np.arange(len(full_dataset))
            N = len(full_dataset)
            y_multi = np.zeros((N, 17), dtype=int)
        else:
            N = len(full_dataset)
            indices = np.arange(N)
            y_multi = np.zeros((N, 17), dtype=int)

        if args.sampler == "random":
            np.random.seed(42)
            np.random.shuffle(indices)
            split = int(0.8 * len(indices))
            train_idx, val_idx = indices[:split], indices[split:]
            test_idx = indices

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
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        test_sampler = SubsetRandomSampler(test_idx)
        train_loader = DataLoader(
            full_dataset,
            batch_size=8,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True)
        val_loader = DataLoader(
            full_dataset,
            batch_size=8,
            sampler=val_sampler,
            num_workers=4,
            pin_memory=True)
        test_loader = DataLoader(
            full_dataset,
            batch_size=8,
            num_workers=4,
            sampler=test_sampler,
            pin_memory=True
        )
        return {'train': train_loader, 'val': val_loader, 'test': test_loader}





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
