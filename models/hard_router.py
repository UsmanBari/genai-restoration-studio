"""
Hard Routing Restoration Pipeline for Task 2.
Implements dynamic routing based on corruption classifier predictions:
  - Class 0 (Clean): Identity Bypass (returns input unmodified with 0 degradation)
  - Class 1 (Salt & Pepper): Routes to Salt & Pepper Specialist Autoencoder
  - Class 2 (Gaussian Blur): Routes to Gaussian Blur Specialist Autoencoder
  - Class 3 (Rectangular Occlusion): Routes to Occlusion Specialist Autoencoder

Also supports Oracle Hard Routing (routing with ground-truth corruption labels)
to establish the theoretical upper-bound performance of specialist architectures.
"""

from typing import Dict, Any, Optional, Tuple, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.corruptions import CORRUPTION_NAMES, NAME_TO_CLASS


class HardRoutingRestorationPipeline(nn.Module):
    """
    Hard Routing Restoration Pipeline combining a 4-way Corruption Classifier
    and 3 Specialist Autoencoders with an Identity Bypass for clean images.
    """

    def __init__(
        self,
        classifier: Optional[nn.Module] = None,
        specialists: Optional[Dict[str, nn.Module]] = None,
        device: str = "cpu"
    ):
        super().__init__()
        self.device = device
        self.classifier = classifier
        self.specialists = nn.ModuleDict()

        if specialists is not None:
            for k, v in specialists.items():
                # Normalize key names
                norm_k = k.lower().replace(" ", "_")
                self.specialists[norm_k] = v

    def set_classifier(self, classifier: nn.Module):
        self.classifier = classifier

    def set_specialist(self, corruption_name: str, model: nn.Module):
        norm_k = corruption_name.lower().replace(" ", "_")
        self.specialists[norm_k] = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Default forward executes predicted routing and returns restored tensor."""
        restored, _, _ = self.route_predicted(x)
        return restored

    def route_predicted(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Routes each sample in batch based on classifier prediction.
        
        Args:
            x: (B, 3, 128, 128) float32 image tensor in [0.0, 1.0]
            
        Returns:
            restored: (B, 3, 128, 128) restored tensor in [0.0, 1.0]
            pred_classes: (B,) int64 tensor of chosen routes
            probs: (B, 4) float32 tensor of classification probabilities
        """
        if self.classifier is None:
            raise RuntimeError("Classifier must be set to run route_predicted().")

        self.classifier.eval()
        with torch.no_grad():
            logits = self.classifier(x)
            probs = F.softmax(logits, dim=1)
            pred_classes = torch.argmax(probs, dim=1)

        restored = self._dispatch_batch(x, pred_classes)
        return restored, pred_classes, probs

    def route_oracle(
        self,
        x: torch.Tensor,
        true_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Routes each sample in batch strictly according to ground-truth labels.
        Establishes theoretical ceiling of the specialist routing paradigm.
        """
        with torch.no_grad():
            return self._dispatch_batch(x, true_labels)

    def _dispatch_batch(
        self,
        x: torch.Tensor,
        route_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Dispatches samples in batch to appropriate specialist or identity bypass.
        Handles mixed batches efficiently by grouping samples by target specialist.
        """
        B, C, H, W = x.shape
        restored = torch.empty_like(x)

        for c_idx in range(4):
            mask = (route_labels == c_idx)
            if not torch.any(mask):
                continue

            indices = torch.nonzero(mask, as_tuple=True)[0]
            sub_batch = x[indices]

            if c_idx == 0:
                # Class 0: Clean -> Identity Bypass (zero modification)
                restored[indices] = sub_batch
            elif c_idx == 1:
                # Class 1: Salt & Pepper
                if "salt_and_pepper" in self.specialists:
                    sp_model = self.specialists["salt_and_pepper"]
                    sp_model.eval()
                    restored[indices] = sp_model(sub_batch)
                else:
                    restored[indices] = sub_batch
            elif c_idx == 2:
                # Class 2: Gaussian Blur
                if "gaussian_blur" in self.specialists:
                    blur_model = self.specialists["gaussian_blur"]
                    blur_model.eval()
                    restored[indices] = blur_model(sub_batch)
                else:
                    restored[indices] = sub_batch
            elif c_idx == 3:
                # Class 3: Rectangular Occlusion
                if "rectangular_occlusion" in self.specialists:
                    occ_model = self.specialists["rectangular_occlusion"]
                    occ_model.eval()
                    restored[indices] = occ_model(sub_batch)
                else:
                    restored[indices] = sub_batch

        return torch.clamp(restored, 0.0, 1.0)
