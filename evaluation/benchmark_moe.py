"""
Comparative 4-Way Benchmark Module for Task 3: Soft Mixture-of-Experts.

Compares:
  1. Universal Autoencoder (Task 1 Baseline)
  2. Oracle Hard-Routed Specialists (Task 2 Ceiling)
  3. Predicted Hard-Routed Pipeline (Task 2 Operational)
  4. Soft Mixture-of-Experts (Task 3 Jointly Trained)

Evaluates on the full test set with per-corruption and per-severity tier breakdowns,
inference latency measurements, and expert routing weight distribution analyses.
"""

import os
import json
import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.corruptions import CORRUPTION_NAMES
from models.autoencoders import UniversalAutoencoder
from models.hard_router import HardRoutingRestorationPipeline
from models.moe import SoftMoERestorationNetwork
from evaluation.metrics import compute_psnr, compute_ssim, compute_l1


def run_4way_benchmark(
    universal_model: nn.Module,
    hard_router: HardRoutingRestorationPipeline,
    soft_moe: SoftMoERestorationNetwork,
    test_loader: DataLoader,
    output_json_path: str = "configs/benchmarks/task3_moe_benchmark_results.json",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, Any]:
    """
    Executes comprehensive 4-way comparative evaluation across the entire test set.
    """
    device_obj = torch.device(device)
    universal_model.to(device_obj).eval()
    hard_router.to(device_obj).eval()
    soft_moe.to(device_obj).eval()

    systems = ["universal", "oracle_hard_routing", "predicted_hard_routing", "soft_moe"]
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

    # Tracking Soft MoE gating weights:
    # 4 classes x 4 expert weights
    moe_weights_by_class = {i: [] for i in range(4)}
    moe_weights_by_severity = {"mild": [], "medium": [], "severe": []}

    print("=== Running 4-Way Comparative Benchmark (Task 3 Soft MoE) ===")
    t0 = time.time()
    total_samples = 0

    with torch.no_grad():
        for batch in test_loader:
            corrupted = batch['corrupted'].to(device_obj)
            clean = batch['clean'].to(device_obj)
            labels = batch['label'].to(device_obj)
            metadata = batch['metadata']
            bs = corrupted.size(0)
            total_samples += bs

            # 1. Universal Autoencoder
            pred_universal = torch.clamp(universal_model(corrupted), 0.0, 1.0)

            # 2. Oracle Hard-Routing
            pred_oracle = hard_router.route_oracle(corrupted, labels)

            # 3. Predicted Hard-Routing
            pred_predicted, _, _ = hard_router.route_predicted(corrupted)

            # 4. Soft MoE Blended Reconstruction
            pred_moe, weights_moe, _ = soft_moe(corrupted)
            pred_moe = torch.clamp(pred_moe, 0.0, 1.0)

            # Convert to numpy for metric calculation
            clean_np = clean.permute(0, 2, 3, 1).cpu().numpy()
            weights_np = weights_moe.cpu().numpy()
            labels_np = labels.cpu().numpy()

            preds_dict = {
                "universal": pred_universal.permute(0, 2, 3, 1).cpu().numpy(),
                "oracle_hard_routing": pred_oracle.permute(0, 2, 3, 1).cpu().numpy(),
                "predicted_hard_routing": pred_predicted.permute(0, 2, 3, 1).cpu().numpy(),
                "soft_moe": pred_moe.permute(0, 2, 3, 1).cpu().numpy()
            }

            for i in range(bs):
                meta = metadata[i]
                c_type = meta.get('corruption_type', 'clean')
                tier = meta.get('severity_tier', 'mild')
                target_img = clean_np[i]
                label_val = int(labels_np[i])
                w_vec = weights_np[i]

                if 0 <= label_val < 4:
                    moe_weights_by_class[label_val].append(w_vec)
                if tier in moe_weights_by_severity:
                    moe_weights_by_severity[tier].append(w_vec)

                for s in systems:
                    pred_img = preds_dict[s][i]
                    psnr = compute_psnr(target_img, pred_img)
                    ssim = compute_ssim(target_img, pred_img)
                    l1 = compute_l1(target_img, pred_img)

                    results_by_system[s]["overall"]["psnr"].append(psnr)
                    results_by_system[s]["overall"]["ssim"].append(ssim)
                    results_by_system[s]["overall"]["l1"].append(l1)

                    if c_type in results_by_system[s]["by_type"]:
                        results_by_system[s]["by_type"][c_type]["psnr"].append(psnr)
                        results_by_system[s]["by_type"][c_type]["ssim"].append(ssim)
                        results_by_system[s]["by_type"][c_type]["l1"].append(l1)

                    if tier in results_by_system[s]["by_severity"]:
                        results_by_system[s]["by_severity"][tier]["psnr"].append(psnr)
                        results_by_system[s]["by_severity"][tier]["ssim"].append(ssim)
                        results_by_system[s]["by_severity"][tier]["l1"].append(l1)

    elapsed = time.time() - t0
    print(f"Evaluated {total_samples} test images across 4 systems in {elapsed:.2f}s ({elapsed / total_samples * 1000:.2f} ms/image).")

    # Aggregate summaries
    summary: Dict[str, Any] = {"systems": {}, "gating_analysis": {}}

    for s in systems:
        ov = results_by_system[s]["overall"]
        summary["systems"][s] = {
            "overall": {
                "psnr_mean": float(np.mean(ov["psnr"])),
                "psnr_std": float(np.std(ov["psnr"])),
                "ssim_mean": float(np.mean(ov["ssim"])),
                "ssim_std": float(np.std(ov["ssim"])),
                "l1_mean": float(np.mean(ov["l1"])),
            },
            "by_type": {},
            "by_severity": {}
        }
        for name, vals in results_by_system[s]["by_type"].items():
            if len(vals["psnr"]) > 0:
                summary["systems"][s]["by_type"][name] = {
                    "psnr_mean": float(np.mean(vals["psnr"])),
                    "ssim_mean": float(np.mean(vals["ssim"])),
                    "l1_mean": float(np.mean(vals["l1"])),
                    "count": len(vals["psnr"])
                }
        for tier, vals in results_by_system[s]["by_severity"].items():
            if len(vals["psnr"]) > 0:
                summary["systems"][s]["by_severity"][tier] = {
                    "psnr_mean": float(np.mean(vals["psnr"])),
                    "ssim_mean": float(np.mean(vals["ssim"])),
                    "l1_mean": float(np.mean(vals["l1"])),
                    "count": len(vals["psnr"])
                }

    # Analyze Gating Weight Matrix: Mean expert weights per true class
    mean_weights_matrix = np.zeros((4, 4), dtype=float)
    for c in range(4):
        if len(moe_weights_by_class[c]) > 0:
            mean_weights_matrix[c] = np.mean(moe_weights_by_class[c], axis=0)

    summary["gating_analysis"] = {
        "mean_weights_matrix": mean_weights_matrix.tolist(),
        "expert_names": ["Clean (Identity)", "Salt & Pepper", "Gaussian Blur", "Rectangular Occlusion"],
        "class_names": [CORRUPTION_NAMES[c] for c in range(4)]
    }

    # Print summary tables
    print("\n" + "=" * 92)
    print(f"{'Category':<22} | {'Universal (Task 1)':<16} | {'Oracle Hard':<14} | {'Pred Hard (Task 2)':<16} | {'Soft MoE (Task 3)':<16}")
    print("-" * 92)
    for c_idx in range(4):
        c_name = CORRUPTION_NAMES[c_idx]
        u_p = summary["systems"]["universal"]["by_type"].get(c_name, {}).get("psnr_mean", 0.0)
        u_s = summary["systems"]["universal"]["by_type"].get(c_name, {}).get("ssim_mean", 0.0)
        o_p = summary["systems"]["oracle_hard_routing"]["by_type"].get(c_name, {}).get("psnr_mean", 0.0)
        o_s = summary["systems"]["oracle_hard_routing"]["by_type"].get(c_name, {}).get("ssim_mean", 0.0)
        p_p = summary["systems"]["predicted_hard_routing"]["by_type"].get(c_name, {}).get("psnr_mean", 0.0)
        p_s = summary["systems"]["predicted_hard_routing"]["by_type"].get(c_name, {}).get("ssim_mean", 0.0)
        m_p = summary["systems"]["soft_moe"]["by_type"].get(c_name, {}).get("psnr_mean", 0.0)
        m_s = summary["systems"]["soft_moe"]["by_type"].get(c_name, {}).get("ssim_mean", 0.0)
        print(f"{c_name.capitalize():<22} | {u_p:6.2f}dB / {u_s:.3f} | {o_p:6.2f}dB / {o_s:.3f} | {p_p:6.2f}dB / {p_s:.3f} | {m_p:6.2f}dB / {m_s:.3f}")

    print("-" * 92)
    u_tot = summary["systems"]["universal"]["overall"]
    o_tot = summary["systems"]["oracle_hard_routing"]["overall"]
    p_tot = summary["systems"]["predicted_hard_routing"]["overall"]
    m_tot = summary["systems"]["soft_moe"]["overall"]
    print(f"{'Overall Average':<22} | {u_tot['psnr_mean']:6.2f}dB / {u_tot['ssim_mean']:.3f} | {o_tot['psnr_mean']:6.2f}dB / {o_tot['ssim_mean']:.3f} | {p_tot['psnr_mean']:6.2f}dB / {p_tot['ssim_mean']:.3f} | {m_tot['psnr_mean']:6.2f}dB / {m_tot['ssim_mean']:.3f}")
    print("=" * 92)

    print("\n--- Gating Network Weight Distribution Matrix (Mean Weight per True Class) ---")
    print(f"{'True Corruption':<22} | {'w0 (Clean)':<10} | {'w1 (S&P)':<10} | {'w2 (Blur)':<10} | {'w3 (Occlusion)':<14}")
    print("-" * 75)
    for c_idx in range(4):
        c_name = CORRUPTION_NAMES[c_idx]
        w = mean_weights_matrix[c_idx]
        print(f"{c_name.capitalize():<22} | {w[0]:8.4f}   | {w[1]:8.4f}   | {w[2]:8.4f}   | {w[3]:8.4f}")
    print("=" * 75)

    os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"\n[SUCCESS] Benchmark results saved to: {output_json_path}")

    return summary
