"""
Comparative Benchmark Module for Task 2: Hard-Routing Inference Pipeline.
Compares:
  1. Universal Autoencoder (Task 1 Baseline)
  2. Oracle Hard-Routed Specialists (Ceiling: routing by ground-truth labels)
  3. Predicted Hard-Routed Pipeline (Classifier + Specialists + Clean Identity Bypass)

Evaluates on the full test set with per-corruption and per-severity tier breakdowns.
"""

import os
import json
import time
from typing import Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.oxford_pet import OxfordPetDataset, collate_oxford
from data.corruptions import CORRUPTION_NAMES
from models.autoencoders import UniversalAutoencoder
from models.classifiers import CorruptionClassifier
from models.hard_router import HardRoutingRestorationPipeline
from evaluation.metrics import compute_psnr, compute_ssim, compute_l1


def run_hard_routing_benchmark(
    universal_model: nn.Module,
    hard_router: HardRoutingRestorationPipeline,
    test_loader: DataLoader,
    output_json_path: str = "configs/benchmarks/task2_hard_routing_benchmark_results.json",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, Any]:
    """
    Runs comprehensive 3-way evaluation across test set.
    """
    universal_model.to(device).eval()
    hard_router.to(device).eval()

    systems = ["universal", "oracle_hard_routing", "predicted_hard_routing"]
    results_by_system = {
        s: {
            "overall": {"psnr": [], "ssim": [], "l1": []},
            "by_type": {name: {"psnr": [], "ssim": [], "l1": []} for name in CORRUPTION_NAMES.values()},
            "by_severity": {
                "mild": {"psnr": [], "ssim": [], "l1": []},
                "medium": {"psnr": [], "ssim": [], "l1": []},
                "severe": {"psnr": [], "ssim": [], "l1": []}
            }
        }
        for s in systems
    }

    # Classifier confusion matrix tracking
    num_classes = 4
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_samples = 0

    print("=== Running 3-Way Comparative Benchmark (Task 2) ===")
    t0 = time.time()

    with torch.no_grad():
        for batch in test_loader:
            corrupted = batch['corrupted'].to(device)
            clean = batch['clean'].to(device)
            labels = batch['label'].to(device)
            metadata = batch['metadata']
            bs = corrupted.size(0)
            total_samples += bs

            # 1. Universal Autoencoder inference
            pred_universal = torch.clamp(universal_model(corrupted), 0.0, 1.0)

            # 2. Oracle Hard-Routing inference
            pred_oracle = hard_router.route_oracle(corrupted, labels)

            # 3. Predicted Hard-Routing inference
            pred_predicted, predicted_classes, _ = hard_router.route_predicted(corrupted)

            # Update classifier confusion matrix
            preds_np = predicted_classes.cpu().numpy()
            labels_np = labels.cpu().numpy()
            for p, y in zip(preds_np, labels_np):
                if 0 <= y < num_classes and 0 <= p < num_classes:
                    conf_matrix[y, p] += 1

            # Convert tensors to numpy for metric computation
            clean_np = clean.permute(0, 2, 3, 1).cpu().numpy()
            preds_dict = {
                "universal": pred_universal.permute(0, 2, 3, 1).cpu().numpy(),
                "oracle_hard_routing": pred_oracle.permute(0, 2, 3, 1).cpu().numpy(),
                "predicted_hard_routing": pred_predicted.permute(0, 2, 3, 1).cpu().numpy()
            }

            for i in range(bs):
                meta = metadata[i]
                c_type = meta.get('corruption_type', 'clean')
                tier = meta.get('severity_tier', 'mild')
                target_img = clean_np[i]

                for sys_name in systems:
                    recon_img = preds_dict[sys_name][i]
                    p_val = compute_psnr(recon_img, target_img)
                    s_val = compute_ssim(recon_img, target_img)
                    l_val = compute_l1(recon_img, target_img)

                    results_by_system[sys_name]["overall"]["psnr"].append(p_val)
                    results_by_system[sys_name]["overall"]["ssim"].append(s_val)
                    results_by_system[sys_name]["overall"]["l1"].append(l_val)

                    if c_type in results_by_system[sys_name]["by_type"]:
                        results_by_system[sys_name]["by_type"][c_type]["psnr"].append(p_val)
                        results_by_system[sys_name]["by_type"][c_type]["ssim"].append(s_val)
                        results_by_system[sys_name]["by_type"][c_type]["l1"].append(l_val)

                    if tier in results_by_system[sys_name]["by_severity"]:
                        results_by_system[sys_name]["by_severity"][tier]["psnr"].append(p_val)
                        results_by_system[sys_name]["by_severity"][tier]["ssim"].append(s_val)
                        results_by_system[sys_name]["by_severity"][tier]["l1"].append(l_val)

    elapsed = time.time() - t0
    print(f"Benchmark completed in {elapsed:.2f}s over {total_samples} test samples.")

    # Compute summary aggregates
    summary = {}
    for sys_name in systems:
        summary[sys_name] = {
            "overall": {
                "psnr": float(np.mean(results_by_system[sys_name]["overall"]["psnr"])),
                "ssim": float(np.mean(results_by_system[sys_name]["overall"]["ssim"])),
                "l1": float(np.mean(results_by_system[sys_name]["overall"]["l1"]))
            },
            "by_type": {
                c_type: {
                    "psnr": float(np.mean(vals["psnr"])) if len(vals["psnr"]) > 0 else 0.0,
                    "ssim": float(np.mean(vals["ssim"])) if len(vals["ssim"]) > 0 else 0.0,
                    "l1": float(np.mean(vals["l1"])) if len(vals["l1"]) > 0 else 0.0,
                    "count": len(vals["psnr"])
                }
                for c_type, vals in results_by_system[sys_name]["by_type"].items()
            },
            "by_severity": {
                tier: {
                    "psnr": float(np.mean(vals["psnr"])) if len(vals["psnr"]) > 0 else 0.0,
                    "ssim": float(np.mean(vals["ssim"])) if len(vals["ssim"]) > 0 else 0.0,
                    "l1": float(np.mean(vals["l1"])) if len(vals["l1"]) > 0 else 0.0,
                    "count": len(vals["psnr"])
                }
                for tier, vals in results_by_system[sys_name]["by_severity"].items()
            }
        }

    # Classifier summary
    classifier_acc = (float(np.trace(conf_matrix)) / max(1, total_samples)) * 100.0
    summary["classifier_evaluation"] = {
        "test_accuracy": classifier_acc,
        "confusion_matrix": conf_matrix.tolist()
    }

    # Save to JSON
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"Benchmark results saved to: {output_json_path}")
    _print_comparative_table(summary)

    return summary


def _print_comparative_table(summary: Dict[str, Any]):
    """Prints a clean ASCII markdown comparative table."""
    print("\n" + "=" * 80)
    print(f"{'Condition':<24} | {'Universal':<16} | {'Oracle Hard-Route':<18} | {'Predicted Hard-Route':<20}")
    print("-" * 80)

    for c_type in ["clean", "salt_and_pepper", "gaussian_blur", "rectangular_occlusion"]:
        u = summary["universal"]["by_type"].get(c_type, {})
        o = summary["oracle_hard_routing"]["by_type"].get(c_type, {})
        p = summary["predicted_hard_routing"]["by_type"].get(c_type, {})

        u_str = f"{u.get('psnr', 0.0):.2f}dB / {u.get('ssim', 0.0):.4f}"
        o_str = f"{o.get('psnr', 0.0):.2f}dB / {o.get('ssim', 0.0):.4f}"
        p_str = f"{p.get('psnr', 0.0):.2f}dB / {p.get('ssim', 0.0):.4f}"

        print(f"{c_type.replace('_', ' ').title():<24} | {u_str:<16} | {o_str:<18} | {p_str:<20}")

    print("-" * 80)
    u_all = summary["universal"]["overall"]
    o_all = summary["oracle_hard_routing"]["overall"]
    p_all = summary["predicted_hard_routing"]["overall"]

    u_all_str = f"{u_all.get('psnr', 0.0):.2f}dB / {u_all.get('ssim', 0.0):.4f}"
    o_all_str = f"{o_all.get('psnr', 0.0):.2f}dB / {o_all.get('ssim', 0.0):.4f}"
    p_all_str = f"{p_all.get('psnr', 0.0):.2f}dB / {p_all.get('ssim', 0.0):.4f}"

    print(f"{'OVERALL AVERAGE':<24} | {u_all_str:<16} | {o_all_str:<18} | {p_all_str:<20}")
    print("=" * 80 + "\n")
