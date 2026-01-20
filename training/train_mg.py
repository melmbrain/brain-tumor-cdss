"""
MG Model Training Script
Gene Expression VAE Encoder

Usage:
    python training/train_mg.py --data_dir /path/to/cgga --epochs 200
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
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeneExpressionDataset(Dataset):
    """Gene Expression Dataset for VAE Training"""

    def __init__(
        self,
        expression_file: str,
        clinical_file: str = None,
        n_genes: int = 500,
        split: str = 'train',
        train_ratio: float = 0.8
    ):
        self.n_genes = n_genes
        self.split = split

        # Load expression data
        logger.info(f"Loading expression data from {expression_file}")
        self.expression_df = pd.read_csv(expression_file, index_col=0)

        # Select top variable genes
        gene_var = self.expression_df.var(axis=0)
        top_genes = gene_var.nlargest(n_genes).index
        self.expression_df = self.expression_df[top_genes]
        self.gene_names = list(top_genes)

        # Normalize (Z-score per gene)
        self.expression_df = (self.expression_df - self.expression_df.mean()) / (self.expression_df.std() + 1e-8)

        # Load clinical data if available
        self.clinical_df = None
        if clinical_file and Path(clinical_file).exists():
            self.clinical_df = pd.read_csv(clinical_file, index_col=0)
            # Align indices
            common_idx = self.expression_df.index.intersection(self.clinical_df.index)
            self.expression_df = self.expression_df.loc[common_idx]
            self.clinical_df = self.clinical_df.loc[common_idx]

        # Train/val split
        n_samples = len(self.expression_df)
        n_train = int(n_samples * train_ratio)

        indices = np.random.permutation(n_samples)
        if split == 'train':
            self.indices = indices[:n_train]
        else:
            self.indices = indices[n_train:]

        logger.info(f"Loaded {len(self.indices)} samples for {split}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        expression = self.expression_df.iloc[real_idx].values.astype(np.float32)

        sample = {
            'expression': torch.from_numpy(expression),
            'patient_id': self.expression_df.index[real_idx]
        }

        # Add clinical labels if available
        if self.clinical_df is not None:
            patient_id = self.expression_df.index[real_idx]
            if patient_id in self.clinical_df.index:
                clinical = self.clinical_df.loc[patient_id]

                # Survival time (log-transformed)
                if 'OS_time' in clinical:
                    surv_time = np.log1p(max(0, clinical['OS_time']))
                    sample['survival_time'] = torch.tensor(surv_time, dtype=torch.float32)

                # Event indicator
                if 'OS_status' in clinical:
                    sample['event'] = torch.tensor(int(clinical['OS_status']), dtype=torch.long)

                # Grade
                if 'Grade' in clinical:
                    grade_map = {'II': 0, 'III': 1, 'IV': 2, 'G2': 0, 'G3': 1, 'G4': 2}
                    grade = grade_map.get(str(clinical['Grade']), 1)
                    sample['grade'] = torch.tensor(grade, dtype=torch.long)

        return sample


class VAELoss(nn.Module):
    """VAE Loss: Reconstruction + KL Divergence"""

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta
        self.mse = nn.MSELoss(reduction='mean')

    def forward(self, recon_x, x, mu, log_var):
        # Reconstruction loss
        recon_loss = self.mse(recon_x, x)

        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

        return recon_loss + self.beta * kl_loss, recon_loss, kl_loss


class GeneVAE(nn.Module):
    """Simple VAE for Gene Expression"""

    def __init__(self, input_dim: int = 500, latent_dim: int = 64, hidden_dims: list = [256, 128]):
        super().__init__()

        self.latent_dim = latent_dim

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = h_dim

        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = h_dim

        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        # Task heads
        self.survival_head = nn.Linear(latent_dim, 1)
        self.grade_head = nn.Linear(latent_dim, 3)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)

        return {
            'recon': recon,
            'mu': mu,
            'log_var': log_var,
            'z': z,
            'survival': self.survival_head(z).squeeze(-1),
            'grade': self.grade_head(z)
        }


def train_epoch(model, loader, optimizer, vae_criterion, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0

    for batch_idx, batch in enumerate(loader):
        expression = batch['expression'].to(device)

        optimizer.zero_grad()
        outputs = model(expression)

        # VAE loss
        vae_loss, recon_loss, kl_loss = vae_criterion(
            outputs['recon'], expression, outputs['mu'], outputs['log_var']
        )

        # Task losses (if labels available)
        task_loss = 0
        if 'survival_time' in batch:
            surv_target = batch['survival_time'].to(device)
            task_loss += nn.MSELoss()(outputs['survival'], surv_target)

        if 'grade' in batch:
            grade_target = batch['grade'].to(device)
            task_loss += nn.CrossEntropyLoss()(outputs['grade'], grade_target)

        loss = vae_loss + 0.5 * task_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()

    n_batches = len(loader)
    return total_loss / n_batches, total_recon / n_batches, total_kl / n_batches


def validate(model, loader, device):
    """Validate model"""
    model.eval()
    total_recon = 0

    with torch.no_grad():
        for batch in loader:
            expression = batch['expression'].to(device)
            outputs = model(expression)
            recon_loss = nn.MSELoss()(outputs['recon'], expression)
            total_recon += recon_loss.item()

    return total_recon / len(loader)


def main(args):
    """Main training function"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Data
    train_dataset = GeneExpressionDataset(
        args.expression_file,
        args.clinical_file,
        n_genes=args.n_genes,
        split='train'
    )
    val_dataset = GeneExpressionDataset(
        args.expression_file,
        args.clinical_file,
        n_genes=args.n_genes,
        split='val'
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model
    model = GeneVAE(input_dim=args.n_genes, latent_dim=args.latent_dim)
    model = model.to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    vae_criterion = VAELoss(beta=args.beta)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        train_loss, recon_loss, kl_loss = train_epoch(
            model, train_loader, optimizer, vae_criterion, device, epoch
        )

        val_loss = validate(model, val_loader, device)

        logger.info(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train: {train_loss:.4f} (Recon: {recon_loss:.4f}, KL: {kl_loss:.4f}) | "
            f"Val: {val_loss:.4f}"
        )

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'gene_names': train_dataset.gene_names,
                'latent_dim': args.latent_dim,
                'best_val_loss': best_val_loss,
            }, output_dir / 'best_model.pth')
            logger.info(f"Saved best model with Val Loss: {best_val_loss:.4f}")

    logger.info(f"\nTraining completed. Best Val Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MG VAE Model")
    parser.add_argument('--expression_file', type=str, required=True, help='Gene expression CSV')
    parser.add_argument('--clinical_file', type=str, default=None, help='Clinical data CSV')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/mg', help='Output directory')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--n_genes', type=int, default=500, help='Number of genes')
    parser.add_argument('--latent_dim', type=int, default=64, help='Latent dimension')
    parser.add_argument('--beta', type=float, default=1.0, help='KL weight')

    args = parser.parse_args()
    main(args)
