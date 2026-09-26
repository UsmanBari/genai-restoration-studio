"""
Training and validation routines for Specialist Autoencoders (Task 2).
Specialists are trained on single 100% targeted corruption streams:
  - salt_and_pepper (label 1)
  - gaussian_blur (label 2)
  - rectangular_occlusion (label 3)

Includes explicit anti-contamination batch-level assertions to mathematically guarantee
zero cross-corruption or clean data leakage during specialist training.
"""

import os
import time
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import mlflow

from training.losses import RestorationLoss
from evaluation.metrics import compute_psnr, compute_ssim, compute_l1
from data.corruptions import CORRUPTION_NAMES, NAME_TO_CLASS


def train_one_epoch_specialist(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: RestorationLoss,
    scaler: Optional[torch.cuda.amp.GradScaler],
    target_corruption: str,
    device: str
) -> Dict[str, float]:
    """
    Runs one training epoch for a specialist autoencoder.
    Includes strict batch assertion ensuring 100% of samples match target_corruption.
    """
    model.train()
    total_loss = 0.0
    total_l1 = 0.0
    total_ssim = 0.0
    num_batches = len(dataloader)

    expected_label = NAME_TO_CLASS[target_corruption.lower()]

    for batch in dataloader:
        corrupted = batch['corrupted'].to(device)
        clean = batch['clean'].to(device)
        labels = batch['label'].to(device)

        # STRICT ANTI-CONTAMINATION ASSERTION:
        # Guarantee zero cross-corruption or clean leakage into specialist training streams
        if not torch.all(labels == expected_label):
            unique_labels = torch.unique(labels).cpu().tolist()
            raise AssertionError(
                f"CRITICAL CORRUPTION CONTAMINATION DETECTED: Specialist '{target_corruption}' "
                f"(expected label {expected_label}) received a training batch containing illegal labels: {unique_labels}. "
                f"Verify corruption_mode filtering in OxfordPetDataset."
            )

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


def evaluate_specialist(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: RestorationLoss,
    target_corruption: str,
    device: str
) -> Dict[str, float]:
    """
    Evaluates specialist autoencoder on validation set filtered to target_corruption.
    """
    model.eval()
    total_loss = 0.0
    total_l1 = 0.0
    total_ssim = 0.0
    total_psnr = 0.0
    count = 0

    expected_label = NAME_TO_CLASS[target_corruption.lower()]

    with torch.no_grad():
        for batch in dataloader:
            corrupted = batch['corrupted'].to(device)
            clean = batch['clean'].to(device)
            labels = batch['label'].to(device)

            # Check that validation stream is also properly matched
            if not torch.all(labels == expected_label):
                unique_labels = torch.unique(labels).cpu().tolist()
                raise AssertionError(
                    f"Specialist validation stream for '{target_corruption}' contains mismatched labels: {unique_labels}"
                )

            pred = model(corrupted)
            pred = torch.clamp(pred, 0.0, 1.0)
            loss, breakdown = criterion(pred, clean)

            bs = corrupted.size(0)
            total_loss += breakdown['loss'] * bs
            total_l1 += breakdown['l1'] * bs
            total_ssim += breakdown['ssim'] * bs

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


def train_specialist_full(
    target_corruption: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 40,
    lr: float = 2.33e-4,
    alpha: float = 0.90,
    checkpoint_dir: str = "checkpoints/specialists",
    experiment_name: str = "Task2_Specialist_Training",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, Any]:
    """
    Full training loop for a specialist autoencoder.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = RestorationLoss(alpha=alpha)
    scaler = torch.cuda.amp.GradScaler() if device.startswith('cuda') else None

    mlflow.set_experiment(experiment_name)
    best_val_score = float('inf')  # minimize unweighted L1 + (1 - SSIM)
    best_val_psnr = 0.0
    best_val_ssim = 0.0
    checkpoint_filename = f"best_specialist_{target_corruption}.pth"
    best_checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
    history = []

    print(f"\n=== Starting Specialist Training: '{target_corruption}' ({epochs} epochs, device: {device}) ===")

    with mlflow.start_run(run_name=f"specialist_{target_corruption}"):
        mlflow.log_params({
            "target_corruption": target_corruption,
            "epochs": epochs,
            "lr": lr,
            "alpha": alpha,
            "base_channels": getattr(model, "base_channels", 64),
            "bottleneck_dim": getattr(model, "bottleneck_dim", 96),
            "device": device
        })

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_metrics = train_one_epoch_specialist(
                model, train_loader, optimizer, criterion, scaler, target_corruption, device
            )
            val_metrics = evaluate_specialist(
                model, val_loader, criterion, target_corruption, device
            )
            scheduler.step()
            elapsed = time.time() - t0

            # Independent selection metric: unweighted score
            eval_score = val_metrics['l1'] + (1.0 - val_metrics['ssim'])

            mlflow.log_metric("train_loss", train_metrics['loss'], step=epoch)
            mlflow.log_metric("train_l1", train_metrics['l1'], step=epoch)
            mlflow.log_metric("train_ssim", train_metrics['ssim'], step=epoch)
            mlflow.log_metric("val_loss", val_metrics['loss'], step=epoch)
            mlflow.log_metric("val_l1", val_metrics['l1'], step=epoch)
            mlflow.log_metric("val_ssim", val_metrics['ssim'], step=epoch)
            mlflow.log_metric("val_psnr", val_metrics['psnr'], step=epoch)
            mlflow.log_metric("lr", optimizer.param_groups[0]['lr'], step=epoch)

            is_best = eval_score < best_val_score
            if is_best:
                best_val_score = eval_score
                best_val_psnr = val_metrics['psnr']
                best_val_ssim = val_metrics['ssim']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_score': best_val_score,
                    'val_psnr': best_val_psnr,
                    'val_ssim': best_val_ssim,
                    'target_corruption': target_corruption,
                    'base_channels': getattr(model, "base_channels", 64),
                    'bottleneck_dim': getattr(model, "bottleneck_dim", 96),
                    'alpha': alpha
                }, best_checkpoint_path)

            status = (
                f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val PSNR: {val_metrics['psnr']:.2f}dB SSIM: {val_metrics['ssim']:.4f}"
            )
            if is_best:
                status += f" -> [BEST SAVED (PSNR: {best_val_psnr:.2f}dB)]"
            print(status)

            history.append({
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'val_psnr': val_metrics['psnr'],
                'val_ssim': val_metrics['ssim'],
                'is_best': is_best
            })

    print(f"=== Specialist '{target_corruption}' Complete. Best Val PSNR: {best_val_psnr:.2f}dB SSIM: {best_val_ssim:.4f} ===")
    return {
        'target_corruption': target_corruption,
        'best_val_psnr': best_val_psnr,
        'best_val_ssim': best_val_ssim,
        'best_checkpoint_path': best_checkpoint_path,
        'history': history
    }
