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
        specialist_base_channels: int = 64,
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
        Automatically inspects checkpoint state_dicts to ensure channel & bottleneck
        compatibility, re-instantiating submodules if necessary, and surfacing detailed
        load diagnostics.
        """
        status = {"gate": False, "sp": False, "blur": False, "occlusion": False}

        def _load_component(name: str, path: Optional[str], is_specialist: bool = False) -> bool:
            if not path:
                print(f"[MoE Warmstart] No checkpoint path provided for '{name}'.")
                return False
            if not os.path.exists(path):
                print(f"[MoE Warmstart ERROR] Checkpoint path for '{name}' does not exist: {path}")
                return False

            try:
                try:
                    ckpt = torch.load(path, map_location=device, weights_only=False)
                except TypeError:
                    ckpt = torch.load(path, map_location=device)

                state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
                if not isinstance(state_dict, dict):
                    print(f"[MoE Warmstart ERROR] Invalid state_dict format in {path}: {type(state_dict)}")
                    return False

                # Clean any unexpected prefix like 'module.'
                cleaned_sd = {}
                for k, v in state_dict.items():
                    key = k[7:] if k.startswith("module.") else k
                    cleaned_sd[key] = v

                if is_specialist:
                    # Determine base_channels and bottleneck_dim from checkpoint
                    ckpt_base_channels = ckpt.get("base_channels")
                    if ckpt_base_channels is None and "enc_init.block.0.weight" in cleaned_sd:
                        ckpt_base_channels = cleaned_sd["enc_init.block.0.weight"].shape[0]

                    ckpt_bottleneck_dim = ckpt.get("bottleneck_dim")
                    if ckpt_bottleneck_dim is None and "bottleneck_conv.0.weight" in cleaned_sd:
                        ckpt_bottleneck_dim = cleaned_sd["bottleneck_conv.0.weight"].shape[0]

                    target_module: UniversalAutoencoder = getattr(self, f"specialist_{name}")

                    # Adapt architecture if checkpoint dimensions differ from current module
                    need_reinit = False
                    if ckpt_base_channels is not None and target_module.base_channels != ckpt_base_channels:
                        need_reinit = True
                    if ckpt_bottleneck_dim is not None and target_module.bottleneck_dim != ckpt_bottleneck_dim:
                        need_reinit = True

                    if need_reinit:
                        base_c = ckpt_base_channels if ckpt_base_channels is not None else target_module.base_channels
                        b_dim = ckpt_bottleneck_dim if ckpt_bottleneck_dim is not None else target_module.bottleneck_dim
                        print(f"[MoE Warmstart] Adapting specialist_{name} architecture to match checkpoint: base_channels={base_c}, bottleneck_dim={b_dim}")
                        new_spec = UniversalAutoencoder(
                            in_channels=3,
                            out_channels=3,
                            base_channels=base_c,
                            bottleneck_dim=b_dim,
                            dropout_rate=target_module.dropout_rate
                        ).to(device)
                        setattr(self, f"specialist_{name}", new_spec)
                        target_module = new_spec

                    target_module.load_state_dict(cleaned_sd, strict=True)
                    print(f"[MoE Warmstart] Successfully loaded specialist '{name}' (base_channels={target_module.base_channels}, bottleneck_dim={target_module.bottleneck_dim}) from {path}")
                    return True
                else:
                    # Loading gate classifier
                    ckpt_base_channels = ckpt.get("base_channels")
                    if ckpt_base_channels is None and "features.0.block.0.weight" in cleaned_sd:
                        ckpt_base_channels = cleaned_sd["features.0.block.0.weight"].shape[0]

                    if ckpt_base_channels is not None and self.gate.base_channels != ckpt_base_channels:
                        print(f"[MoE Warmstart] Adapting gate architecture to match checkpoint: base_channels={ckpt_base_channels}")
                        self.gate = CorruptionClassifier(
                            in_channels=3,
                            num_classes=4,
                            base_channels=ckpt_base_channels,
                            dropout_rate=self.gate.dropout_rate
                        ).to(device)

                    self.gate.load_state_dict(cleaned_sd, strict=True)
                    print(f"[MoE Warmstart] Successfully loaded gate classifier (base_channels={self.gate.base_channels}) from {path}")
                    return True

            except Exception as e:
                import traceback
                print(f"[MoE Warmstart ERROR] Failed to load '{name}' from {path}: {type(e).__name__}: {e}")
                print(f"[MoE Warmstart ERROR] Traceback:\n{traceback.format_exc()}")
                return False

        status["gate"] = _load_component("gate", classifier_path, is_specialist=False)
        status["sp"] = _load_component("sp", sp_path, is_specialist=True)
        status["blur"] = _load_component("blur", blur_path, is_specialist=True)
        status["occlusion"] = _load_component("occlusion", occlusion_path, is_specialist=True)

        return status
