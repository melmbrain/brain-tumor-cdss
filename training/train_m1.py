"""
M1 Model Training Script
MRI Segmentation + Classification using SwinUNETR

Usage:
    python training/train_m1.py --data_dir /path/to/brats --epochs 300
"""

import os
import argparse
import logging
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# MONAI imports
try:
    from monai.networks.nets import SwinUNETR
    from monai.losses import DiceCELoss
    from monai.metrics import DiceMetric
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd,
        Orientationd, Spacingd, ScaleIntensityRanged,
        CropForegroundd, RandCropByPosNegLabeld,
        RandFlipd, RandRotate90d, EnsureTyped
    )
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False
    print("MONAI not installed. Install with: pip install monai")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BraTSDataset(Dataset):
    """BraTS Dataset for MRI Segmentation"""

    def __init__(self, data_dir: str, split: str = 'train', transform=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self):
        """Load sample paths"""
        samples = []
        for patient_dir in sorted(self.data_dir.glob("BraTS*")):
            if patient_dir.is_dir():
                sample = {
                    't1': str(patient_dir / f"{patient_dir.name}_t1.nii.gz"),
                    't1ce': str(patient_dir / f"{patient_dir.name}_t1ce.nii.gz"),
                    't2': str(patient_dir / f"{patient_dir.name}_t2.nii.gz"),
                    'flair': str(patient_dir / f"{patient_dir.name}_flair.nii.gz"),
                    'seg': str(patient_dir / f"{patient_dir.name}_seg.nii.gz"),
                }
                # Check if all files exist
                if all(Path(p).exists() for p in sample.values()):
                    samples.append(sample)

        logger.info(f"Found {len(samples)} samples for {self.split}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        if self.transform:
            data = self.transform(sample)
            return data

        # Basic loading without MONAI transforms
        import nibabel as nib
        volumes = []
        for mod in ['t1', 't1ce', 't2', 'flair']:
            nii = nib.load(sample[mod])
            volumes.append(nii.get_fdata())

        image = np.stack(volumes, axis=0).astype(np.float32)
        label = nib.load(sample['seg']).get_fdata().astype(np.int64)

        return {
            'image': torch.from_numpy(image),
            'label': torch.from_numpy(label)
        }


def get_transforms(split: str = 'train'):
    """Get MONAI transforms for training/validation"""
    if not MONAI_AVAILABLE:
        return None

    keys = ['t1', 't1ce', 't2', 'flair', 'seg']

    if split == 'train':
        return Compose([
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),
            Orientationd(keys=keys, axcodes="RAS"),
            Spacingd(keys=keys, pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "bilinear", "bilinear", "bilinear", "nearest")),
            ScaleIntensityRanged(keys=['t1', 't1ce', 't2', 'flair'], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=keys, source_key='t1'),
            RandCropByPosNegLabeld(
                keys=keys, label_key='seg',
                spatial_size=(128, 128, 128),
                pos=1, neg=1, num_samples=2
            ),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
            RandRotate90d(keys=keys, prob=0.5, max_k=3),
            EnsureTyped(keys=keys),
        ])
    else:
        return Compose([
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),
            Orientationd(keys=keys, axcodes="RAS"),
            Spacingd(keys=keys, pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "bilinear", "bilinear", "bilinear", "nearest")),
            ScaleIntensityRanged(keys=['t1', 't1ce', 't2', 'flair'], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
            EnsureTyped(keys=keys),
        ])


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(loader):
        # Concatenate modalities
        if isinstance(batch, dict):
            image = torch.cat([batch['t1'], batch['t1ce'], batch['t2'], batch['flair']], dim=1)
            label = batch['seg']
        else:
            image = batch['image']
            label = batch['label']

        image = image.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 10 == 0:
            logger.info(f"  Batch {batch_idx}/{len(loader)}, Loss: {loss.item():.4f}")

    return total_loss / len(loader)


def validate(model, loader, metric, device):
    """Validate model"""
    model.eval()
    metric.reset()

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, dict):
                image = torch.cat([batch['t1'], batch['t1ce'], batch['t2'], batch['flair']], dim=1)
                label = batch['seg']
            else:
                image = batch['image']
                label = batch['label']

            image = image.to(device)
            label = label.to(device)

            output = model(image)
            output = torch.argmax(output, dim=1, keepdim=True)

            metric(y_pred=output, y=label)

    return metric.aggregate().item()


def main(args):
    """Main training function"""
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Model
    if MONAI_AVAILABLE:
        model = SwinUNETR(
            img_size=(128, 128, 128),
            in_channels=4,
            out_channels=4,
            feature_size=48,
            use_checkpoint=True
        )
    else:
        # Fallback to custom model
        from models.m1 import MRIMultiTaskModel
        model = MRIMultiTaskModel(in_channels=4, include_segmentation=True)

    model = model.to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Data
    train_transform = get_transforms('train')
    val_transform = get_transforms('val')

    train_dataset = BraTSDataset(args.data_dir, split='train', transform=train_transform)
    val_dataset = BraTSDataset(args.data_dir, split='val', transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Loss and optimizer
    if MONAI_AVAILABLE:
        criterion = DiceCELoss(to_onehot_y=True, softmax=True)
        metric = DiceMetric(include_background=False, reduction="mean")
    else:
        criterion = nn.CrossEntropyLoss()
        metric = None

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_dice = 0
    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.epochs}")

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        logger.info(f"Train Loss: {train_loss:.4f}")

        if metric and len(val_loader) > 0:
            val_dice = validate(model, val_loader, metric, device)
            logger.info(f"Val Dice: {val_dice:.4f}")

            if val_dice > best_dice:
                best_dice = val_dice
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_dice': best_dice,
                }, output_dir / 'best_model.pth')
                logger.info(f"Saved best model with Dice: {best_dice:.4f}")

        scheduler.step()

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, output_dir / f'checkpoint_epoch{epoch+1}.pth')

    logger.info(f"\nTraining completed. Best Dice: {best_dice:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train M1 Model")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to BraTS data')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/m1', help='Output directory')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--save_every', type=int, default=50, help='Save checkpoint every N epochs')

    args = parser.parse_args()
    main(args)
