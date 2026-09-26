"""
Soft Mixture-of-Experts (SoftMoERestorationNetwork) Architecture for Task 3.

Combines:
  1. Gating Network: Lightweight 4-way convolutional classifier initialized from Task 2 weights.
  2. Four Expert Branches:
     - Branch 0: Clean Identity Pass-Through (f0(x) = x)
     - Branch 1: Salt & Pepper Specialist Autoencoder (UniversalAutoencoder)
     - Branch 2: Gaussian Blur Specialist Autoencoder (UniversalAutoencoder)
     - Branch 3: Rectangular Occlusion Specialist Autoencoder (UniversalAutoencoder)
  3. Soft Blending:
     - Continuous routing weights w = softmax(z / tau), sum(w) = 1.0
     - Differentiable reconstructed output y_hat = sum_k (w_k * f_k(x))
"""

from typing import Tuple, Dict, Any, Optional
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.classifiers import CorruptionClassifier
from models.autoencoders import UniversalAutoencoder
from data.corruptions import CORRUPTION_NAMES


class SoftMoERestorationNetwork(nn.Module):
    """
    Soft Mixture-of-Experts Restoration Network (Task 3).
    
    Args:
      gate_base_channels: Base channels for the CorruptionClassifier gate (default: 32).
      gate_dropout: Dropout rate for the gate head (default: 0.2).
      specialist_base_channels: Base channels for specialist autoencoders (default: 48).
      specialist_bottleneck_dim: Bottleneck dimension for specialists (default: 96).
      use_gated_skip: Whether specialists use gated high-resolution skips (default: True).
      temperature: Softmax temperature parameter tau controlling routing sharpness (default: 1.0).
    """

    def __init__(
        self,
        gate_base_channels: int = 32,
        gate_dropout: float = 0.2,
        specialist_base_channels: int = 48,
        specialist_bottleneck_dim: int = 96,
        specialist_dropout: float = 0.2,
        temperature: float = 1.0
    ):
        super().__init__()
        self.temperature = float(temperature)
        self.expert_names = ["Clean (Identity)", "Salt & Pepper", "Gaussian Blur", "Rectangular Occlusion"]

        # 1. Gate Network
        self.gate = CorruptionClassifier(
            in_channels=3,
            num_classes=4,
            base_channels=gate_base_channels,
            dropout_rate=gate_dropout
        )

        # 2. Specialist Experts (Branch 1, 2, 3)
        self.specialist_sp = UniversalAutoencoder(
            in_channels=3,
            out_channels=3,
            base_channels=specialist_base_channels,
            bottleneck_dim=specialist_bottleneck_dim,
            dropout_rate=specialist_dropout
        )
        self.specialist_blur = UniversalAutoencoder(
            in_channels=3,
            out_channels=3,
            base_channels=specialist_base_channels,
            bottleneck_dim=specialist_bottleneck_dim,
            dropout_rate=specialist_dropout
        )
        self.specialist_occlusion = UniversalAutoencoder(
            in_channels=3,
            out_channels=3,
            base_channels=specialist_base_channels,
            bottleneck_dim=specialist_bottleneck_dim,
            dropout_rate=specialist_dropout
        )

    def set_temperature(self, temperature: float):
        """Update softmax temperature tau."""
        if temperature <= 0.0:
            raise ValueError(f"Temperature must be strictly positive, got {temperature}")
        self.temperature = float(temperature)

    def freeze_experts(self):
        """Freeze all specialist autoencoder weights (for Phase 1 warm-up)."""
        for specialist in (self.specialist_sp, self.specialist_blur, self.specialist_occlusion):
            for param in specialist.parameters():
                param.requires_grad = False

    def unfreeze_experts(self):
        """Unfreeze all specialist autoencoder weights (for Phase 2 joint fine-tuning)."""
        for specialist in (self.specialist_sp, self.specialist_blur, self.specialist_occlusion):
            for param in specialist.parameters():
                param.requires_grad = True

    def freeze_gate(self):
        """Freeze gate network parameters."""
        for param in self.gate.parameters():
            param.requires_grad = False

    def unfreeze_gate(self):
        """Unfreeze gate network parameters."""
        for param in self.gate.parameters():
            param.requires_grad = True

    def get_gating_weights(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute raw logits and softmax-normalized gating weights for input batch x.
        Returns:
          (weights [B, 4], logits [B, 4])
        """
        logits = self.gate(x)
        weights = F.softmax(logits / max(self.temperature, 1e-6), dim=1)
        return weights, logits

    def forward(
        self,
        x: torch.Tensor,
        return_all_experts: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with soft blending over all 4 expert branches.
        
        Args:
          x: Corrupted input images [B, 3, 128, 128] in [0.0, 1.0]
          return_all_experts: If True, also return individual expert reconstructions.
          
        Returns:
          (reconstructed [B, 3, 128, 128], weights [B, 4], logits [B, 4])
        """
        # 1. Gate logits and softmax weights
        logits = self.gate(x)
        weights = F.softmax(logits / max(self.temperature, 1e-6), dim=1)  # [B, 4]

        # 2. Expert outputs
        # Branch 0: Identity bypass for clean inputs
        out_0 = x
        # Branch 1, 2, 3: Specialist autoencoders
        out_1 = self.specialist_sp(x)
        out_2 = self.specialist_blur(x)
        out_3 = self.specialist_occlusion(x)

        # 3. Soft blending: sum_k (w_k * f_k(x))
        w0 = weights[:, 0].view(-1, 1, 1, 1)
        w1 = weights[:, 1].view(-1, 1, 1, 1)
        w2 = weights[:, 2].view(-1, 1, 1, 1)
        w3 = weights[:, 3].view(-1, 1, 1, 1)

        reconstructed = w0 * out_0 + w1 * out_1 + w2 * out_2 + w3 * out_3

        if return_all_experts:
            expert_outputs = torch.stack([out_0, out_1, out_2, out_3], dim=1)  # [B, 4, 3, 128, 128]
            return reconstructed, weights, logits, expert_outputs

        return reconstructed, weights, logits

    def load_warmstart_weights(
        self,
        classifier_path: Optional[str] = None,
        sp_path: Optional[str] = None,
        blur_path: Optional[str] = None,
        occlusion_path: Optional[str] = None,
        device: str = "cpu"
    ) -> Dict[str, bool]:
        """
        Load warm-start pretrained weights from Task 2 into Gate and Specialists.
        """
        status = {"gate": False, "sp": False, "blur": False, "occlusion": False}

        def _load_weights(model: nn.Module, path: Optional[str]) -> bool:
            if not path or not os.path.exists(path):
                return False
            try:
                try:
                    ckpt = torch.load(path, map_location=device, weights_only=False)
                except TypeError:
                    ckpt = torch.load(path, map_location=device)
                state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
                model.load_state_dict(state_dict, strict=True)
                return True
            except Exception:
                try:
                    # Fallback to non-strict loading if minor prefix differences exist
                    model.load_state_dict(state_dict, strict=False)
                    return True
                except Exception:
                    return False

        status["gate"] = _load_weights(self.gate, classifier_path)
        status["sp"] = _load_weights(self.specialist_sp, sp_path)
        status["blur"] = _load_weights(self.specialist_blur, blur_path)
        status["occlusion"] = _load_weights(self.specialist_occlusion, occlusion_path)

        return status
