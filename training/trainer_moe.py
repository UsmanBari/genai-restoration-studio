"""
Two-Phase Training Pipeline for Soft Mixture-of-Experts (SoftMoERestorationNetwork).

Implements:
  Phase 1 (Warm-Up):
    - Specialist experts frozen.
    - Only the Gate Network is trained to align routing distributions with warm-start representations.
  Phase 2 (Joint Fine-Tuning):
    - All 4 branches unfrozen (Identity bypass + 3 Specialist Autoencoders).
    - End-to-end backpropagation through the differentiable blended reconstruction.
    - Smaller joint learning rate with CosineAnnealingLR.
  MLflow Integration:
    - Logs train/val composite loss, reconstruction L1/SSIM, PSNR, classification loss, balance penalty, and entropy regularizer.
"""

from typing import Dict, Any, Optional, Tuple, List
import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.moe import SoftMoERestorationNetwork
from training.losses import SoftMoECompositeLoss
from evaluation.metrics import compute_psnr, compute_ssim, compute_l1


def train_one_epoch_moe(
    model: SoftMoERestorationNetwork,
    dataloader: DataLoader,
    criterion: SoftMoECompositeLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Dict[str, float]:
    """Train Soft MoE for one epoch."""
    model.train()
    running_loss = 0.0
    running_l_recon = 0.0
    running_l_class = 0.0
    running_l_balance = 0.0
    running_l_entropy = 0.0
    running_l1 = 0.0
    running_ssim = 0.0
    running_psnr = 0.0
    total_samples = 0

    for batch in dataloader:
        corrupted = batch['corrupted'].to(device, non_blocking=True)
        clean = batch['clean'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)
        batch_size = corrupted.size(0)

        optimizer.zero_grad()
        reconstructed, weights, logits = model(corrupted)
        loss, breakdown = criterion(reconstructed, clean, logits, weights, labels)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Metrics
        with torch.no_grad():
            mse = torch.mean((reconstructed.detach() - clean.detach()) ** 2).item()
            psnr_val = 100.0 if mse <= 0 else 10.0 * math.log10(1.0 / mse)
            ssim_val = breakdown['ssim']

        running_loss += breakdown['loss'] * batch_size
        running_l_recon += breakdown['l_recon'] * batch_size
        running_l_class += breakdown['l_class'] * batch_size
        running_l_balance += breakdown['l_balance'] * batch_size
        running_l_entropy += breakdown['l_entropy'] * batch_size
        running_l1 += breakdown['l1'] * batch_size
        running_ssim += ssim_val * batch_size
        running_psnr += psnr_val * batch_size
        total_samples += batch_size

    n = max(total_samples, 1)
    return {
        'loss': running_loss / n,
        'l_recon': running_l_recon / n,
        'l_class': running_l_class / n,
        'l_balance': running_l_balance / n,
        'l_entropy': running_l_entropy / n,
        'l1': running_l1 / n,
        'ssim': running_ssim / n,
        'psnr': running_psnr / n
    }


@torch.no_grad()
def evaluate_moe(
    model: SoftMoERestorationNetwork,
    dataloader: DataLoader,
    criterion: SoftMoECompositeLoss,
    device: torch.device
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Evaluate Soft MoE on validation/test set.
    Returns:
      (metrics_dict, average_weights_per_class [4, 4])
    """
    model.eval()
    running_loss = 0.0
    running_l_recon = 0.0
    running_l_class = 0.0
    running_l_balance = 0.0
    running_l_entropy = 0.0
    running_l1 = 0.0
    running_ssim = 0.0
    running_psnr = 0.0
    total_samples = 0

    class_weights_sum = np.zeros((4, 4), dtype=np.float64)
    class_counts = np.zeros(4, dtype=np.int64)

    for batch in dataloader:
        corrupted = batch['corrupted'].to(device, non_blocking=True)
        clean = batch['clean'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)
        batch_size = corrupted.size(0)

        reconstructed, weights, logits = model(corrupted)
        loss, breakdown = criterion(reconstructed, clean, logits, weights, labels)

        mse = torch.mean((reconstructed.detach() - clean.detach()) ** 2).item()
        psnr_val = 100.0 if mse <= 0 else 10.0 * math.log10(1.0 / mse)
        ssim_val = breakdown['ssim']

        running_loss += breakdown['loss'] * batch_size
        running_l_recon += breakdown['l_recon'] * batch_size
        running_l_class += breakdown['l_class'] * batch_size
        running_l_balance += breakdown['l_balance'] * batch_size
        running_l_entropy += breakdown['l_entropy'] * batch_size
        running_l1 += breakdown['l1'] * batch_size
        running_ssim += ssim_val * batch_size
        running_psnr += psnr_val * batch_size
        total_samples += batch_size

        # Track expert weight distribution per true corruption class
        weights_np = weights.cpu().numpy()
        labels_np = labels.cpu().numpy()
        for i in range(batch_size):
            c = labels_np[i]
            if 0 <= c < 4:
                class_weights_sum[c] += weights_np[i]
                class_counts[c] += 1

    n = max(total_samples, 1)
    metrics = {
        'loss': running_loss / n,
        'l_recon': running_l_recon / n,
        'l_class': running_l_class / n,
        'l_balance': running_l_balance / n,
        'l_entropy': running_l_entropy / n,
        'l1': running_l1 / n,
        'ssim': running_ssim / n,
        'psnr': running_psnr / n
    }

    avg_weights_per_class = np.zeros((4, 4), dtype=np.float64)
    for c in range(4):
        if class_counts[c] > 0:
            avg_weights_per_class[c] = class_weights_sum[c] / class_counts[c]

    return metrics, avg_weights_per_class


def train_soft_moe(
    model: SoftMoERestorationNetwork,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    save_dir: str = "checkpoints/task3",
    warmup_epochs: int = 5,
    joint_epochs: int = 30,
    lr_warmup: float = 5e-4,
    lr_joint: float = 1e-4,
    temperature: float = 1.0,
    lambda_recon: float = 0.8,
    lambda_class: float = 0.2,
    lambda_balance: float = 0.1,
    lambda_entropy: float = 0.01,
    alpha: float = 0.90,
    mlflow_tracking: bool = True
) -> Dict[str, Any]:
    """
    Complete 2-Phase Training Workflow for SoftMoERestorationNetwork.
    """
    os.makedirs(save_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(save_dir, "best_soft_moe.pth")
    model.set_temperature(temperature)
    model.to(device)

    criterion = SoftMoECompositeLoss(
        lambda_recon=lambda_recon,
        lambda_class=lambda_class,
        lambda_balance=lambda_balance,
        lambda_entropy=lambda_entropy,
        alpha=alpha
    ).to(device)

    history: Dict[str, List[float]] = {
        'train_loss': [], 'val_loss': [],
        'train_psnr': [], 'val_psnr': [],
        'train_ssim': [], 'val_ssim': [],
        'val_l_balance': [], 'val_l_entropy': []
    }

    best_val_psnr = -float('inf')
    best_epoch = 0

    print("=" * 70)
    print(f"Starting Soft MoE 2-Phase Training")
    print(f"  Warm-Up Phase:       {warmup_epochs} epochs (lr={lr_warmup:.2e}, Experts Frozen)")
    print(f"  Joint Fine-Tuning:   {joint_epochs} epochs (lr={lr_joint:.2e}, End-to-End)")
    print(f"  Temperature tau:     {temperature:.2f}")
    print(f"  Loss Weights:        recon={lambda_recon}, class={lambda_class}, balance={lambda_balance}, entropy={lambda_entropy}, alpha={alpha}")
    print("=" * 70)

    # ----------------------------------------------------
    # PHASE 1: WARM-UP (Experts Frozen, Gate Only)
    # ----------------------------------------------------
    if warmup_epochs > 0:
        print("\n--- PHASE 1: Gate Network Warm-Up ---")
        model.freeze_experts()
        model.unfreeze_gate()
        optimizer_warmup = torch.optim.AdamW(model.gate.parameters(), lr=lr_warmup, weight_decay=1e-4)

        for epoch in range(1, warmup_epochs + 1):
            t0 = time.time()
            train_m = train_one_epoch_moe(model, train_loader, criterion, optimizer_warmup, device)
            val_m, _ = evaluate_moe(model, val_loader, criterion, device)
            elapsed = time.time() - t0

            print(f"[Phase 1: WarmUp {epoch:02d}/{warmup_epochs:02d}] "
                  f"Train Loss: {train_m['loss']:.4f} | Val Loss: {val_m['loss']:.4f} | "
                  f"Val PSNR: {val_m['psnr']:.2f} dB | Val SSIM: {val_m['ssim']:.4f} | Time: {elapsed:.1f}s")

    # ----------------------------------------------------
    # PHASE 2: JOINT END-TO-END FINE-TUNING
    # ----------------------------------------------------
    print("\n--- PHASE 2: Joint End-to-End Fine-Tuning ---")
    model.unfreeze_experts()
    model.unfreeze_gate()

    optimizer_joint = torch.optim.AdamW([
        {'params': model.gate.parameters(), 'lr': lr_joint},
        {'params': model.specialist_sp.parameters(), 'lr': lr_joint * 0.5},
        {'params': model.specialist_blur.parameters(), 'lr': lr_joint * 0.5},
        {'params': model.specialist_occlusion.parameters(), 'lr': lr_joint * 0.5},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_joint, T_max=joint_epochs, eta_min=1e-6)

    for epoch in range(1, joint_epochs + 1):
        t0 = time.time()
        train_m = train_one_epoch_moe(model, train_loader, criterion, optimizer_joint, device)
        val_m, avg_weights = evaluate_moe(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        history['train_loss'].append(train_m['loss'])
        history['val_loss'].append(val_m['loss'])
        history['train_psnr'].append(train_m['psnr'])
        history['val_psnr'].append(val_m['psnr'])
        history['train_ssim'].append(train_m['ssim'])
        history['val_ssim'].append(val_m['ssim'])
        history['val_l_balance'].append(val_m['l_balance'])
        history['val_l_entropy'].append(val_m['l_entropy'])

        is_best = val_m['psnr'] > best_val_psnr
        if is_best:
            best_val_psnr = val_m['psnr']
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_psnr': val_m['psnr'],
                'val_ssim': val_m['ssim'],
                'val_loss': val_m['loss'],
                'temperature': temperature,
                'avg_weights_per_class': avg_weights
            }, best_checkpoint_path)

        star = " ★ BEST" if is_best else ""
        print(f"[Phase 2: Joint {epoch:02d}/{joint_epochs:02d}] "
              f"Train Loss: {train_m['loss']:.4f} | Val Loss: {val_m['loss']:.4f} | "
              f"Val PSNR: {val_m['psnr']:.2f} dB | Val SSIM: {val_m['ssim']:.4f}{star} | Time: {elapsed:.1f}s")

    print("\n" + "=" * 70)
    print(f"Training Complete! Best Validation PSNR: {best_val_psnr:.2f} dB at Epoch {best_epoch}")
    print(f"Saved checkpoint to: {best_checkpoint_path}")
    print("=" * 70)

    # Load best weights before returning
    if os.path.exists(best_checkpoint_path):
        ckpt = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])

    return {
        'best_val_psnr': best_val_psnr,
        'best_epoch': best_epoch,
        'checkpoint_path': best_checkpoint_path,
        'history': history
    }
