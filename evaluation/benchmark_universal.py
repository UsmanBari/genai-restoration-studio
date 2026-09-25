"""
Universal Autoencoder Benchmark & Comprehensive Evaluation (Task 1).
Evaluates models against the deterministic fixed-tier test manifest.
Computes PSNR, SSIM, and L1 broken down by corruption type and severity tier.
Generates 12 representative comparison panels and extracts 4 worst failure cases.
"""

import os
import json
from typing import Dict, List, Any, Optional
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import torch
from torch.utils.data import DataLoader

from evaluation.metrics import evaluate_image_pair, compute_psnr, compute_ssim, compute_l1
from data.oxford_pet import OxfordPetDataset, collate_oxford


def run_universal_benchmark(
    model: torch.nn.Module,
    manifest_path: str,
    images_dir: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_dir: str = "artifacts/evaluation_task1",
    num_visualizations: int = 12
) -> Dict[str, Any]:
    """
    Runs comprehensive evaluation on the fixed-tier test manifest.
    
    Returns structured results dict:
      {
        'summary': {'mean_psnr': ..., 'mean_ssim': ..., 'mean_l1': ...},
        'by_category': {
           'clean': {'psnr': ..., 'ssim': ..., 'l1': ...},
           'salt_and_pepper_low': ...,
           'salt_and_pepper_med': ...,
           'salt_and_pepper_high': ...,
           ...
        },
        'failure_cases': [...]
      }
    """
    os.makedirs(output_dir, exist_ok=True)
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    dataset = OxfordPetDataset(
        manifest_path=manifest_path,
        images_dir=images_dir,
        split='test'
    )
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_oxford
    )

    model.to(device)
    model.eval()

    all_results = []
    # Buckets for breakdown
    category_buckets = {
        'clean_tier_0': [],
        'salt_and_pepper_tier_1': [],
        'salt_and_pepper_tier_2': [],
        'salt_and_pepper_tier_3': [],
        'gaussian_blur_tier_1': [],
        'gaussian_blur_tier_2': [],
        'gaussian_blur_tier_3': [],
        'rectangular_occlusion_tier_1': [],
        'rectangular_occlusion_tier_2': [],
        'rectangular_occlusion_tier_3': []
    }

    with torch.no_grad():
        for batch in dataloader:
            corrupted = batch['corrupted'].to(device)
            clean = batch['clean'].to(device)
            metadata = batch['metadata']  # Plain Python list of dicts

            recon = model(corrupted)
            recon = torch.clamp(recon, 0.0, 1.0)

            # Move to CPU numpy
            recon_np = recon.permute(0, 2, 3, 1).cpu().numpy()
            clean_np = clean.permute(0, 2, 3, 1).cpu().numpy()
            corr_np = corrupted.permute(0, 2, 3, 1).cpu().numpy()

            batch_size = len(recon_np)
            for i in range(batch_size):
                meta_item = metadata[i]
                c_type = meta_item.get('corruption_type', 'clean')
                tier = meta_item.get('severity_tier', 0)
                bucket_key = f"{c_type}_tier_{tier}"

                metrics = evaluate_image_pair(recon_np[i], clean_np[i])
                item_record = {
                    'image_id': meta_item.get('image_id', f"img_{len(all_results)}"),
                    'corruption_type': c_type,
                    'severity_tier': tier,
                    'bucket_key': bucket_key,
                    'metrics': metrics,
                    'pred_np': recon_np[i],
                    'clean_np': clean_np[i],
                    'corr_np': corr_np[i],
                    'metadata': meta_item
                }
                all_results.append(item_record)
                if bucket_key in category_buckets:
                    category_buckets[bucket_key].append(metrics)

    # Compute aggregate statistics
    def aggregate_metrics(metric_list):
        if not metric_list:
            return {'psnr': 0.0, 'ssim': 0.0, 'l1': 0.0, 'count': 0}
        return {
            'psnr': round(float(np.mean([m['psnr'] for m in metric_list])), 2),
            'ssim': round(float(np.mean([m['ssim'] for m in metric_list])), 4),
            'l1': round(float(np.mean([m['l1'] for m in metric_list])), 5),
            'count': len(metric_list)
        }

    overall_metrics = aggregate_metrics([r['metrics'] for r in all_results])
    category_summary = {k: aggregate_metrics(v) for k, v in category_buckets.items()}

    # Group high-level categories (clean, s&p, blur, occlusion)
    high_level_groups = {
        'Clean': [r['metrics'] for r in all_results if r['corruption_type'] == 'clean'],
        'Salt & Pepper (All Tiers)': [r['metrics'] for r in all_results if r['corruption_type'] == 'salt_and_pepper'],
        'Gaussian Blur (All Tiers)': [r['metrics'] for r in all_results if r['corruption_type'] == 'gaussian_blur'],
        'Occlusion (All Tiers)': [r['metrics'] for r in all_results if r['corruption_type'] == 'rectangular_occlusion']
    }
    grouped_summary = {k: aggregate_metrics(v) for k, v in high_level_groups.items()}

    # Generate 12 representative visualizations
    print(f"Generating {num_visualizations} representative visualization panels...")
    selected_indices = np.linspace(0, len(all_results) - 1, num_visualizations, dtype=int)

    fig, axes = plt.subplots(num_visualizations, 4, figsize=(16, 3.5 * num_visualizations))
    for row, idx in enumerate(selected_indices):
        item = all_results[idx]
        corr_img = item['corr_np']
        clean_img = item['clean_np']
        recon_img = item['pred_np']
        error_map = np.abs(recon_img - clean_img).mean(axis=2)  # Heatmap of L1 error

        # 1. Clean Target
        axes[row, 0].imshow(clean_img)
        axes[row, 0].set_title(f"Target Clean ({item['image_id']})", fontsize=10)
        axes[row, 0].axis('off')

        # 2. Corrupted Input
        axes[row, 1].imshow(corr_img)
        axes[row, 1].set_title(f"Input: {item['corruption_type']} (Tier {item['severity_tier']})", fontsize=10)
        axes[row, 1].axis('off')

        # 3. Model Reconstruction
        axes[row, 2].imshow(recon_img)
        axes[row, 2].set_title(f"Reconstruction (PSNR: {item['metrics']['psnr']:.1f}dB, SSIM: {item['metrics']['ssim']:.3f})", fontsize=10)
        axes[row, 2].axis('off')

        # 4. Absolute Error Map
        im = axes[row, 3].imshow(error_map, cmap='inferno', vmin=0.0, vmax=0.4)
        axes[row, 3].set_title(f"L1 Error Map (Mean: {item['metrics']['l1']:.4f})", fontsize=10)
        axes[row, 3].axis('off')

    plt.tight_layout()
    grid_path = os.path.join(figures_dir, "task1_12_representative_examples.png")
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Identify top 4 failure cases (lowest PSNR / highest L1 error)
    sorted_failures = sorted(all_results, key=lambda x: x['metrics']['l1'], reverse=True)
    # Pick top 1 from each corruption category to ensure diversity
    failure_cases = []
    seen_types = set()
    for item in sorted_failures:
        if item['corruption_type'] not in seen_types and len(seen_types) < 4:
            seen_types.add(item['corruption_type'])
            failure_cases.append({
                'image_id': item['image_id'],
                'corruption_type': item['corruption_type'],
                'severity_tier': item['severity_tier'],
                'psnr': item['metrics']['psnr'],
                'ssim': item['metrics']['ssim'],
                'l1_error': item['metrics']['l1'],
                'notes': f"Worst reconstruction for {item['corruption_type']} under fixed test set."
            })

    # Plot the 4 failure cases
    fig, axes = plt.subplots(4, 4, figsize=(16, 14))
    for row, item in enumerate([f for f in sorted_failures if f['corruption_type'] in seen_types][:4]):
        corr_img = item['corr_np']
        clean_img = item['clean_np']
        recon_img = item['pred_np']
        error_map = np.abs(recon_img - clean_img).mean(axis=2)

        axes[row, 0].imshow(clean_img)
        axes[row, 0].set_title(f"Target: {item['image_id']}", fontsize=10)
        axes[row, 0].axis('off')

        axes[row, 1].imshow(corr_img)
        axes[row, 1].set_title(f"Corrupted ({item['corruption_type']}, Tier {item['severity_tier']})", fontsize=10)
        axes[row, 1].axis('off')

        axes[row, 2].imshow(recon_img)
        axes[row, 2].set_title(f"Reconstruction (PSNR: {item['metrics']['psnr']:.1f}dB, SSIM: {item['metrics']['ssim']:.3f})", fontsize=10)
        axes[row, 2].axis('off')

        axes[row, 3].imshow(error_map, cmap='inferno', vmin=0.0, vmax=0.5)
        axes[row, 3].set_title(f"Error Map (L1: {item['metrics']['l1']:.4f})", fontsize=10)
        axes[row, 3].axis('off')

    plt.tight_layout()
    fail_path = os.path.join(figures_dir, "task1_4_failure_cases.png")
    plt.savefig(fail_path, dpi=150, bbox_inches='tight')
    plt.close()

    results_export = {
        'overall_summary': overall_metrics,
        'grouped_by_corruption': grouped_summary,
        'detailed_by_severity_tier': category_summary,
        'visualizations_path': grid_path,
        'failure_cases_path': fail_path,
        'failure_cases_analysis': failure_cases
    }

    json_path = os.path.join(output_dir, "task1_benchmark_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_export, f, indent=2)

    print(f"Benchmark completed! Results saved to {json_path}")
    return results_export
