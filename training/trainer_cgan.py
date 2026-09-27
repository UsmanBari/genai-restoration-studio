"""
Training Pipeline for Style-Conditioned Conditional GAN (Task 4).
Implements:
  1. Two-player adversarial training step (D step with detached fakes, G step with L1 penalty).
  2. Training stability metrics: D real/fake accuracy, G adv loss, G L1 loss.
  3. Validation metrics: L1, SSIM, PSNR.
  4. Visual sample generation for qualitative progression tracking.
  5. MLflow experiment tracking and best checkpoint preservation.
"""

from typing import Dict, Any, Tuple, Optional
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

from models.cgan import StyleConditionedUNetGenerator, ConditionalPatchGANDiscriminator
from training.losses_cgan import ConditionalGANLoss
from evaluation.metrics import compute_psnr, compute_ssim, compute_l1


def unnormalize_to_0_1(tensor: torch.Tensor) -> np.ndarray:
    """Converts (B, C, H, W) tensor in [-1.0, 1.0] to (B, H, W, C) numpy array in [0.0, 1.0]."""
    np_img = tensor.detach().cpu().permute(0, 2, 3, 1).numpy()
    np_img = np.clip((np_img * 0.5) + 0.5, 0.0, 1.0)
    return np_img


def save_sample_grid(
    net_g: StyleConditionedUNetGenerator,
    fixed_batch: Dict[str, torch.Tensor],
    save_path: str,
    device: torch.device
):
    """Generates and saves a side-by-side comparison grid: Photo | Real Sketch | Generated Sketch."""
    net_g.eval()
    with torch.no_grad():
        photos = fixed_batch['photo'].to(device)
        sketches = fixed_batch['sketch'].to(device)
        styles = fixed_batch['style'].to(device)
        
        fakes = net_g(photos, styles)

        photos_np = (unnormalize_to_0_1(photos) * 255.0).astype(np.uint8)
        sketches_np = (unnormalize_to_0_1(sketches) * 255.0).astype(np.uint8)
        fakes_np = (unnormalize_to_0_1(fakes) * 255.0).astype(np.uint8)

        num_samples = min(photos.size(0), 4)
        h, w = photos_np.shape[1], photos_np.shape[2]
        grid = np.zeros((num_samples * h, 3 * w, 3), dtype=np.uint8)

        for i in range(num_samples):
            grid[i*h:(i+1)*h, 0:w] = photos_np[i]
            grid[i*h:(i+1)*h, w:2*w] = sketches_np[i]
            grid[i*h:(i+1)*h, 2*w:3*w] = fakes_np[i]

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        Image.fromarray(grid).save(save_path)


def train_one_epoch_cgan(
    net_g: StyleConditionedUNetGenerator,
    net_d: ConditionalPatchGANDiscriminator,
    dataloader: DataLoader,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    criterion: ConditionalGANLoss,
    device: torch.device,
    d_update_freq: int = 2,
    d_max_acc_throttle: float = 0.92
) -> Dict[str, float]:
    """
    Executes one training epoch of conditional GAN with discriminator rebalancing.
    
    Args:
        d_update_freq: Update D once every `d_update_freq` generator updates (default: 2).
        d_max_acc_throttle: If D accuracy exceeds this threshold, skip D update on that step.
    """
    net_g.train()
    net_d.train()

    total_loss_d = 0.0
    total_loss_g = 0.0
    total_loss_g_adv = 0.0
    total_loss_g_l1 = 0.0
    total_d_acc_real = 0.0
    total_d_acc_fake = 0.0
    total_d_acc = 0.0
    num_batches = 0
    d_updates_count = 0

    last_metrics_d = {
        'loss_d': 0.5,
        'd_acc_real': 0.5,
        'd_acc_fake': 0.5,
        'd_acc_total': 0.5
    }

    for batch_idx, batch in enumerate(dataloader):
        photos = batch['photo'].to(device, non_blocking=True)
        sketches = batch['sketch'].to(device, non_blocking=True)
        styles = batch['style'].to(device, non_blocking=True)

        # ----------------------------------------------------
        # 1. Forward Generator to produce current fakes
        # ----------------------------------------------------
        fake_sketches = net_g(photos, styles)

        # ----------------------------------------------------
        # 2. Train Discriminator (only on scheduled frequency steps)
        # ----------------------------------------------------
        should_update_d = (batch_idx % d_update_freq == 0)

        # Real and fake forward passes for D
        d_real_logits = net_d(photos, sketches, styles)
        d_fake_logits = net_d(photos, fake_sketches.detach(), styles)
        loss_d, metrics_d = criterion.discriminator_loss(d_real_logits, d_fake_logits)
        last_metrics_d = metrics_d

        if should_update_d:
            # Check if D is already overpowering G in this batch
            if metrics_d['d_acc_total'] < d_max_acc_throttle:
                opt_d.zero_grad()
                loss_d.backward()
                opt_d.step()
                d_updates_count += 1

        # ----------------------------------------------------
        # 3. Train Generator: maximize adversarial + lambda_L1 * L1
        # ----------------------------------------------------
        opt_g.zero_grad()
        d_fake_logits_for_g = net_d(photos, fake_sketches, styles)
        loss_g, metrics_g = criterion.generator_loss(d_fake_logits_for_g, fake_sketches, sketches)
        loss_g.backward()
        opt_g.step()

        # Accumulate metrics
        total_loss_d += metrics_d['loss_d']
        total_loss_g += metrics_g['loss_g']
        total_loss_g_adv += metrics_g['loss_g_adv']
        total_loss_g_l1 += metrics_g['loss_g_l1']
        total_d_acc_real += metrics_d['d_acc_real']
        total_d_acc_fake += metrics_d['d_acc_fake']
        total_d_acc += metrics_d['d_acc_total']
        num_batches += 1

    n = max(1, num_batches)
    return {
        'loss_d': total_loss_d / n,
        'loss_g': total_loss_g / n,
        'loss_g_adv': total_loss_g_adv / n,
        'loss_g_l1': total_loss_g_l1 / n,
        'd_acc_real': total_d_acc_real / n,
        'd_acc_fake': total_d_acc_fake / n,
        'd_acc_total': total_d_acc / n,
        'd_updates_ratio': d_updates_count / n
    }


@torch.no_grad()
def evaluate_cgan(
    net_g: StyleConditionedUNetGenerator,
    dataloader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Evaluates Generator reconstruction quality on validation/test dataloader."""
    net_g.eval()
    total_l1 = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0

    for batch in dataloader:
        photos = batch['photo'].to(device, non_blocking=True)
        sketches = batch['sketch'].to(device, non_blocking=True)
        styles = batch['style'].to(device, non_blocking=True)
        bs = photos.size(0)

        fakes = net_g(photos, styles)

        # Convert to numpy in [0.0, 1.0] for standard metric evaluation
        pred_np = unnormalize_to_0_1(fakes)
        real_np = unnormalize_to_0_1(sketches)

        for i in range(bs):
            total_l1 += compute_l1(pred_np[i], real_np[i])
            total_psnr += compute_psnr(pred_np[i], real_np[i])
            total_ssim += compute_ssim(pred_np[i], real_np[i])
            count += 1

    n = max(1, count)
    return {
        'l1': total_l1 / n,
        'psnr': total_psnr / n,
        'ssim': total_ssim / n
    }


def train_cgan_full(
    net_g: StyleConditionedUNetGenerator,
    net_d: ConditionalPatchGANDiscriminator,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    save_dir: str,
    epochs: int = 40,
    lr_g: float = 4e-4,
    lr_d: float = 1.5e-4,
    beta1: float = 0.5,
    beta2: float = 0.999,
    lambda_l1: float = 100.0,
    d_update_freq: int = 2,
    patience: int = 12,
    experiment_name: str = "Task4_cGAN_Face_To_Sketch"
) -> Dict[str, Any]:
    """
    Executes full multi-epoch cGAN training workflow with equilibrium monitoring.
    """
    os.makedirs(save_dir, exist_ok=True)
    samples_dir = os.path.join(save_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    criterion = ConditionalGANLoss(lambda_l1=lambda_l1, real_label=0.9, fake_label=0.0)
    opt_g = torch.optim.Adam(net_g.parameters(), lr=lr_g, betas=(beta1, beta2))
    opt_d = torch.optim.Adam(net_d.parameters(), lr=lr_d, betas=(beta1, beta2))

    scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=epochs, eta_min=1e-6)
    scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=epochs, eta_min=1e-6)

    # Grab a fixed validation batch for progress visualization
    fixed_val_batch = next(iter(val_loader))

    if HAS_MLFLOW:
        try:
            mlflow.set_experiment(experiment_name)
        except Exception:
            pass

    best_val_score = float('inf')  # Minimize L1 + (1 - SSIM)
    best_val_metrics = {}
    best_epoch = 0
    best_checkpoint_path = os.path.join(save_dir, "best_cgan_generator.pth")

    consecutive_high_d_acc = 0
    epochs_without_improvement = 0

    print(f"\n=== Starting cGAN Training ({epochs} epochs, lr_g={lr_g:.1e}, lr_d={lr_d:.1e}, lambda_l1={lambda_l1}, d_freq={d_update_freq}) ===")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_m = train_one_epoch_cgan(
            net_g, net_d, train_loader, opt_g, opt_d, criterion, device,
            d_update_freq=d_update_freq
        )
        val_m = evaluate_cgan(net_g, val_loader, device)
        elapsed = time.time() - t0

        scheduler_g.step()
        scheduler_d.step()

        # Combined validation quality score: L1 + (1 - SSIM)
        val_score = val_m['l1'] + (1.0 - val_m['ssim'])
        is_best = val_score < best_val_score

        if is_best:
            best_val_score = val_score
            best_val_metrics = val_m
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({
                'epoch': epoch,
                'generator_state_dict': net_g.state_dict(),
                'discriminator_state_dict': net_d.state_dict(),
                'opt_g_state_dict': opt_g.state_dict(),
                'opt_d_state_dict': opt_d.state_dict(),
                'val_metrics': val_m,
                'lambda_l1': lambda_l1,
                'emb_dim': net_g.emb_dim,
                'base_channels_g': net_g.base_channels,
                'base_channels_d': net_d.base_channels
            }, best_checkpoint_path)
        else:
            epochs_without_improvement += 1

        # Save sample visual grid periodically (every 5 epochs and on best)
        if epoch % 5 == 0 or is_best or epoch == epochs:
            sample_path = os.path.join(samples_dir, f"epoch_{epoch:03d}.png")
            save_sample_grid(net_g, fixed_val_batch, sample_path, device)

        # Discriminator dominance diagnostic check
        if train_m['d_acc_total'] >= 0.90:
            consecutive_high_d_acc += 1
            if consecutive_high_d_acc >= 4:
                print(
                    f"  >> [WARNING] Discriminator accuracy sustained at {train_m['d_acc_total']*100:.1f}% "
                    f"for {consecutive_high_d_acc} consecutive epochs! Throttling D to maintain gradient flow to G."
                )
        else:
            consecutive_high_d_acc = 0

        # Logging to MLflow
        if HAS_MLFLOW:
            try:
                mlflow.log_metric("train_loss_d", train_m['loss_d'], step=epoch)
                mlflow.log_metric("train_loss_g", train_m['loss_g'], step=epoch)
                mlflow.log_metric("train_loss_g_adv", train_m['loss_g_adv'], step=epoch)
                mlflow.log_metric("train_loss_g_l1", train_m['loss_g_l1'], step=epoch)
                mlflow.log_metric("d_acc_real", train_m['d_acc_real'], step=epoch)
                mlflow.log_metric("d_acc_fake", train_m['d_acc_fake'], step=epoch)
                mlflow.log_metric("d_acc_total", train_m['d_acc_total'], step=epoch)
                mlflow.log_metric("val_l1", val_m['l1'], step=epoch)
                mlflow.log_metric("val_psnr", val_m['psnr'], step=epoch)
                mlflow.log_metric("val_ssim", val_m['ssim'], step=epoch)
            except Exception:
                pass

        best_flag = " [BEST]" if is_best else ""
        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
            f"Loss D: {train_m['loss_d']:.4f} (Acc: {train_m['d_acc_total']*100:.1f}%) | "
            f"Loss G: {train_m['loss_g']:.2f} (L1: {train_m['loss_g_l1']:.4f}) | "
            f"Val PSNR: {val_m['psnr']:.2f}dB SSIM: {val_m['ssim']:.4f}{best_flag}"
        )

        # Early stopping if sustained degradation occurs late in training
        if epoch >= 20 and epochs_without_improvement >= patience:
            print(f"\n[EARLY STOPPING] Validation metrics peaked at Epoch {best_epoch} ({best_val_metrics.get('psnr', 0):.2f}dB). Stopping to prevent mode collapse.")
            break

    print(f"\n[COMPLETE] Best Validation: PSNR {best_val_metrics.get('psnr', 0):.2f}dB | SSIM {best_val_metrics.get('ssim', 0):.4f} at Epoch {best_epoch}")
    print(f"Best checkpoint preserved at: {best_checkpoint_path}")

    return {
        'best_val_metrics': best_val_metrics,
        'best_epoch': best_epoch,
        'checkpoint_path': best_checkpoint_path
    }

