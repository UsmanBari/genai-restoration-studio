"""
Training and validation routines for Universal Denoising Autoencoder (Task 1).
Supports mixed precision training, learning rate scheduling, MLflow logging,
and frequent checkpointing for Google Colab sessions.
"""

import os
import time
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import mlflow

from training.losses import RestorationLoss, ssim_tensor
from evaluation.metrics import compute_psnr, compute_ssim, compute_l1


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: RestorationLoss,
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: str
) -> Dict[str, float]:
    """Runs one training epoch over runtime corrupted batches."""
    model.train()
    total_loss = 0.0
    total_l1 = 0.0
    total_ssim = 0.0
    num_batches = len(dataloader)

    for batch in dataloader:
        corrupted = batch['corrupted'].to(device)
        clean = batch['clean'].to(device)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None and device.startswith('cuda'):
            with torch.cuda.amp.autocast():
                pred = model(corrupted)
                loss, breakdown = criterion(pred, clean)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(corrupted)
            loss, breakdown = criterion(pred, clean)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += breakdown['loss']
        total_l1 += breakdown['l1']
        total_ssim += breakdown['ssim']

    return {
        'loss': total_loss / max(1, num_batches),
        'l1': total_l1 / max(1, num_batches),
        'ssim': total_ssim / max(1, num_batches)
    }


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: RestorationLoss,
    device: str
) -> Dict[str, float]:
    """Evaluates model on validation set with deterministic corruptions."""
    model.eval()
    total_loss = 0.0
    total_l1 = 0.0
    total_ssim = 0.0
    total_psnr = 0.0
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            corrupted = batch['corrupted'].to(device)
            clean = batch['clean'].to(device)

            pred = model(corrupted)
            pred = torch.clamp(pred, 0.0, 1.0)
            loss, breakdown = criterion(pred, clean)

            bs = corrupted.size(0)
            total_loss += breakdown['loss'] * bs
            total_l1 += breakdown['l1'] * bs
            total_ssim += breakdown['ssim'] * bs

            # Compute PSNR over batch
            pred_np = pred.permute(0, 2, 3, 1).cpu().numpy()
            clean_np = clean.permute(0, 2, 3, 1).cpu().numpy()
            for i in range(bs):
                total_psnr += compute_psnr(pred_np[i], clean_np[i])

            count += bs

    return {
        'loss': total_loss / max(1, count),
        'l1': total_l1 / max(1, count),
        'ssim': total_ssim / max(1, count),
        'psnr': total_psnr / max(1, count)
    }


def train_universal_autoencoder(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 25,
    lr: float = 0.0002,
    alpha: float = 0.8,
    weight_decay: float = 1e-5,
    checkpoints_dir: str = "checkpoints",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    use_mlflow: bool = True
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Complete training loop for Universal Autoencoder with checkpoints and MLflow tracking.
    """
    os.makedirs(checkpoints_dir, exist_ok=True)
    best_ckpt_path = os.path.join(checkpoints_dir, "task1_universal_best.pth")
    latest_ckpt_path = os.path.join(checkpoints_dir, "task1_universal_latest.pth")

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = RestorationLoss(alpha=alpha)
    scaler = torch.cuda.amp.GradScaler() if device.startswith('cuda') else None

    best_val_loss = float('inf')
    best_metrics = {}
    history = []

    print(f"Starting Universal Autoencoder Training on {device}...")
    print(f"  Epochs: {epochs} | LR: {lr} | Alpha: {alpha} | Batch Size: {train_loader.batch_size}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        curr_lr = scheduler.get_last_lr()[0]

        epoch_record = {
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_l1': train_metrics['l1'],
            'train_ssim': train_metrics['ssim'],
            'val_loss': val_metrics['loss'],
            'val_l1': val_metrics['l1'],
            'val_ssim': val_metrics['ssim'],
            'val_psnr': val_metrics['psnr'],
            'lr': curr_lr,
            'time_s': elapsed
        }
        history.append(epoch_record)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) "
            f"Train Loss: {train_metrics['loss']:.4f} (L1: {train_metrics['l1']:.4f}, SSIM: {train_metrics['ssim']:.3f}) | "
            f"Val Loss: {val_metrics['loss']:.4f} (PSNR: {val_metrics['psnr']:.2f}dB, SSIM: {val_metrics['ssim']:.3f})"
        )

        if use_mlflow and mlflow.active_run():
            mlflow.log_metric("train_loss", train_metrics['loss'], step=epoch)
            mlflow.log_metric("train_l1", train_metrics['l1'], step=epoch)
            mlflow.log_metric("train_ssim", train_metrics['ssim'], step=epoch)
            mlflow.log_metric("val_loss", val_metrics['loss'], step=epoch)
            mlflow.log_metric("val_l1", val_metrics['l1'], step=epoch)
            mlflow.log_metric("val_ssim", val_metrics['ssim'], step=epoch)
            mlflow.log_metric("val_psnr", val_metrics['psnr'], step=epoch)
            mlflow.log_metric("lr", curr_lr, step=epoch)

        # Save latest checkpoint every epoch (resilience against Colab disconnects)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_metrics': val_metrics,
            'config': {
                'base_channels': model.base_channels,
                'bottleneck_dim': model.bottleneck_dim,
                'dropout_rate': model.dropout_rate,
                'lr': lr,
                'alpha': alpha
            }
        }, latest_ckpt_path)

        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_metrics = dict(val_metrics)
            best_metrics['epoch'] = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'config': {
                    'base_channels': model.base_channels,
                    'bottleneck_dim': model.bottleneck_dim,
                    'dropout_rate': model.dropout_rate,
                    'lr': lr,
                    'alpha': alpha
                }
            }, best_ckpt_path)
            print(f"  ⭐ New best checkpoint saved (Val Loss: {best_val_loss:.4f})")

    # Load best weights before returning
    if os.path.exists(best_ckpt_path):
        checkpoint = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    return model, {
        'best_val_metrics': best_metrics,
        'history': history,
        'best_checkpoint_path': best_ckpt_path,
        'latest_checkpoint_path': latest_ckpt_path
    }
