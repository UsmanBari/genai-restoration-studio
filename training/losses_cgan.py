"""
Loss Functions for Style-Conditioned Conditional GAN (Task 4).
Combines:
  1. Adversarial Loss: BCEWithLogitsLoss for real vs fake patch discrimination.
  2. Reconstruction Loss: L1 pixel distance between generated and ground-truth sketches.
  3. Real/Fake Discriminator Accuracy tracking for training equilibrium diagnostics.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalGANLoss(nn.Module):
    """
    Pix2Pix-style Conditional GAN Composite Loss.
    
    Args:
        lambda_l1: Weight for L1 paired sketch reconstruction loss (default: 100.0).
    """

    def __init__(self, lambda_l1: float = 100.0):
        super().__init__()
        self.lambda_l1 = float(lambda_l1)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()

    def set_lambda_l1(self, lambda_l1: float):
        """Update L1 reconstruction loss weight."""
        if lambda_l1 < 0:
            raise ValueError(f"lambda_l1 must be non-negative, got {lambda_l1}")
        self.lambda_l1 = float(lambda_l1)

    def discriminator_loss(
        self,
        d_real_logits: torch.Tensor,
        d_fake_logits: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute Discriminator loss and real/fake classification accuracy.
        
        Args:
            d_real_logits: Patch logits from real pairs (B, 1, H, W).
            d_fake_logits: Patch logits from generated pairs (B, 1, H, W).
            
        Returns:
            (loss_d, metrics_dict)
        """
        real_targets = torch.ones_like(d_real_logits)
        fake_targets = torch.zeros_like(d_fake_logits)

        loss_d_real = self.bce_loss(d_real_logits, real_targets)
        loss_d_fake = self.bce_loss(d_fake_logits, fake_targets)
        loss_d = 0.5 * (loss_d_real + loss_d_fake)

        # Classification accuracy on local patches
        with torch.no_grad():
            acc_real = (d_real_logits > 0.0).float().mean().item()
            acc_fake = (d_fake_logits < 0.0).float().mean().item()
            acc_total = 0.5 * (acc_real + acc_fake)

        metrics = {
            'loss_d': loss_d.item(),
            'loss_d_real': loss_d_real.item(),
            'loss_d_fake': loss_d_fake.item(),
            'd_acc_real': acc_real,
            'd_acc_fake': acc_fake,
            'd_acc_total': acc_total
        }
        return loss_d, metrics

    def generator_loss(
        self,
        d_fake_logits: torch.Tensor,
        fake_sketches: torch.Tensor,
        real_sketches: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute Generator composite loss: Adversarial Loss + lambda_L1 * L1 Loss.
        
        Args:
            d_fake_logits: Patch logits from generated pairs (B, 1, H, W).
            fake_sketches: Generated sketch images (B, 3, 128, 128) in [-1, 1].
            real_sketches: Target ground-truth sketches (B, 3, 128, 128) in [-1, 1].
            
        Returns:
            (loss_g, metrics_dict)
        """
        # Generator wants Discriminator to predict 1.0 (real) for generated samples
        real_targets = torch.ones_like(d_fake_logits)
        loss_g_adv = self.bce_loss(d_fake_logits, real_targets)

        # L1 Pixel Reconstruction Loss
        loss_g_l1 = self.l1_loss(fake_sketches, real_sketches)

        # Total Generator Loss
        loss_g = loss_g_adv + self.lambda_l1 * loss_g_l1

        metrics = {
            'loss_g': loss_g.item(),
            'loss_g_adv': loss_g_adv.item(),
            'loss_g_l1': loss_g_l1.item(),
            'lambda_l1': self.lambda_l1
        }
        return loss_g, metrics
