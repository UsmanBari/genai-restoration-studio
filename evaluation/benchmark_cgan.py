"""
Style-Conditioned cGAN Benchmark & Comprehensive Evaluation (Task 4).
Evaluates StyleConditionedUNetGenerator on the FS2K test split (1,046 test pairs).
Computes L1 distance, PSNR, SSIM, and Pixel-FD broken down by style category (Style 0, Style 1, Style 2, Overall).
Includes 3 reference baselines:
  (a) All-white image (1.0)
  (b) Mean training sketch
  (c) Grayscale version of input photo
Generates visual comparison panels, error maps (|Fake - Real|), and failure cases.
"""

import os
import json
import time
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from evaluation.metrics import compute_psnr, compute_ssim, compute_l1, evaluate_image_pair
from data.fs2k import FS2KDataset, collate_fs2k
from models.cgan import StyleConditionedUNetGenerator


def unnormalize_to_0_1(tensor: torch.Tensor) -> np.ndarray:
    """Converts (B, C, H, W) tensor in [-1.0, 1.0] to (B, H, W, C) numpy array in [0.0, 1.0]."""
    np_img = tensor.detach().cpu().permute(0, 2, 3, 1).numpy()
    np_img = np.clip((np_img * 0.5) + 0.5, 0.0, 1.0)
    return np_img


def compute_pixel_frechet_distance(
    real_images: np.ndarray,
    fake_images: np.ndarray,
    seed: int = 42
) -> float:
    """
    Computes Pixel-Space Fréchet Distance (Pixel-FD) on 49,152-dim raw pixel distribution statistics:
    ||mu_1 - mu_2||^2 + Tr(C1 + C2 - 2 * sqrt(C1 * C2)) over flattened image vectors.
    Uses deterministic seeded subsampling (seed=42) to ensure exact reproducibility.
    """
    try:
        b = len(real_images)
        if b < 2:
            return 0.0

        # Subsample if dataset is large with deterministic RandomState
        sub_n = min(b, 500)
        rng = np.random.RandomState(seed)
        idx = rng.choice(b, sub_n, replace=False)
        r_flat = real_images[idx].reshape(sub_n, -1).astype(np.float32)
        f_flat = fake_images[idx].reshape(sub_n, -1).astype(np.float32)

        mu_r = np.mean(r_flat, axis=0)
        mu_f = np.mean(f_flat, axis=0)
        diff = mu_r - mu_f

        # Tr(Cov) = sum(Var(x))
        tr_r = float(np.var(r_flat, axis=0).sum())
        tr_f = float(np.var(f_flat, axis=0).sum())

        fd = float(np.dot(diff, diff) + tr_r + tr_f - 2.0 * np.sqrt(max(0.0, tr_r * tr_f)))
        return round(float(max(0.0, fd)), 2)
    except Exception:
        return 0.0


def _extract_image_name(batch: Dict[str, Any], idx: int, fallback_idx: int) -> str:
    """Safely extracts image_name from batch dictionary regardless of collate structure."""
    if 'image_name' in batch:
        val = batch['image_name']
        if isinstance(val, (list, tuple)) and idx < len(val):
            return str(val[idx])
        elif isinstance(val, str):
            return val
    if 'metadata' in batch:
        meta = batch['metadata']
        if isinstance(meta, dict) and 'image_name' in meta:
            val = meta['image_name']
            if isinstance(val, (list, tuple)) and idx < len(val):
                return str(val[idx])
            elif isinstance(val, str):
                return val
        elif isinstance(meta, list) and idx < len(meta) and isinstance(meta[idx], dict):
            return str(meta[idx].get('image_name', f"img_{fallback_idx}"))
    return f"img_{fallback_idx}"


def run_cgan_benchmark(
    generator: StyleConditionedUNetGenerator,
    manifest_path: str,
    fs2k_root: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_dir: str = "artifacts/evaluation_task4",
    num_visualizations: int = 12
) -> Dict[str, Any]:
    """
    Runs full benchmark evaluation of the trained cGAN generator on the FS2K test set.
    Includes comparisons against 3 reference baselines (all-white, mean sketch, grayscale photo).
    """
    os.makedirs(output_dir, exist_ok=True)
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    dataset = FS2KDataset(
        manifest_path=manifest_path,
        fs2k_root=fs2k_root,
        split='test',
        normalize_gan=True
    )
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fs2k
    )

    generator.to(device)
    generator.eval()

    style_buckets: Dict[int, List[Dict[str, Any]]] = {0: [], 1: [], 2: []}
    all_results: List[Dict[str, Any]] = []
    all_real_imgs: List[np.ndarray] = []
    all_fake_imgs: List[np.ndarray] = []
    all_photo_imgs: List[np.ndarray] = []

    total_inference_time = 0.0
    total_images = 0

    print(f"[BENCHMARK] Evaluating cGAN Generator on FS2K test split ({len(dataset)} pairs)...")

    with torch.no_grad():
        for batch in dataloader:
            photos = batch['photo'].to(device)
            sketches = batch['sketch'].to(device)
            styles = batch['style'].to(device)

            t0 = time.perf_counter()
            fakes = generator(photos, styles)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            total_inference_time += (t1 - t0)

            fakes_np = unnormalize_to_0_1(fakes)
            reals_np = unnormalize_to_0_1(sketches)
            photos_np = unnormalize_to_0_1(photos)

            bs = photos.size(0)
            total_images += bs

            for i in range(bs):
                img_name = _extract_image_name(batch, i, total_images - bs + i)
                st_id = int(styles[i].item())
                p_img = photos_np[i]
                r_img = reals_np[i]
                f_img = fakes_np[i]

                # Model metrics
                l1_val = compute_l1(f_img, r_img)
                psnr_val = compute_psnr(f_img, r_img)
                ssim_val = compute_ssim(f_img, r_img)

                item_res = {
                    'image_name': img_name,
                    'style': st_id,
                    'l1': l1_val,
                    'psnr': psnr_val,
                    'ssim': ssim_val,
                    'photo': p_img,
                    'real_sketch': r_img,
                    'gen_sketch': f_img
                }

                if st_id in style_buckets:
                    style_buckets[st_id].append(item_res)
                all_results.append(item_res)
                all_real_imgs.append(r_img)
                all_fake_imgs.append(f_img)
                all_photo_imgs.append(p_img)

    # -------------------------------------------------------------------------
    # Baseline Calculations:
    # (a) All-white image (1.0)
    # (b) Mean training sketch (computed across dataset or test ground truths)
    # (c) Grayscale version of input photo
    # -------------------------------------------------------------------------
    mean_sketch_img = np.mean(all_real_imgs, axis=0) if all_real_imgs else np.ones((128, 128, 3), dtype=np.float32)

    baseline_metrics = {
        'all_white': {'psnr': [], 'ssim': [], 'l1': []},
        'mean_sketch': {'psnr': [], 'ssim': [], 'l1': []},
        'grayscale_photo': {'psnr': [], 'ssim': [], 'l1': []}
    }
    baseline_by_style = {
        st_id: {
            'all_white': {'psnr': [], 'ssim': [], 'l1': []},
            'mean_sketch': {'psnr': [], 'ssim': [], 'l1': []},
            'grayscale_photo': {'psnr': [], 'ssim': [], 'l1': []}
        } for st_id in [0, 1, 2]
    }

    for idx, item in enumerate(all_results):
        r_img = item['real_sketch']
        p_img = item['photo']
        st_id = item['style']

        # Baseline (a): All-white
        white_img = np.ones_like(r_img, dtype=np.float32)
        psnr_w = compute_psnr(white_img, r_img)
        ssim_w = compute_ssim(white_img, r_img)
        l1_w = compute_l1(white_img, r_img)

        # Baseline (b): Mean sketch
        psnr_m = compute_psnr(mean_sketch_img, r_img)
        ssim_m = compute_ssim(mean_sketch_img, r_img)
        l1_m = compute_l1(mean_sketch_img, r_img)

        # Baseline (c): Grayscale input photo
        gray_1c = 0.2989 * p_img[..., 0] + 0.5870 * p_img[..., 1] + 0.1140 * p_img[..., 2]
        gray_img = np.stack([gray_1c, gray_1c, gray_1c], axis=-1).astype(np.float32)
        psnr_g = compute_psnr(gray_img, r_img)
        ssim_g = compute_ssim(gray_img, r_img)
        l1_g = compute_l1(gray_img, r_img)

        # Record overall
        baseline_metrics['all_white']['psnr'].append(psnr_w)
        baseline_metrics['all_white']['ssim'].append(ssim_w)
        baseline_metrics['all_white']['l1'].append(l1_w)

        baseline_metrics['mean_sketch']['psnr'].append(psnr_m)
        baseline_metrics['mean_sketch']['ssim'].append(ssim_m)
        baseline_metrics['mean_sketch']['l1'].append(l1_m)

        baseline_metrics['grayscale_photo']['psnr'].append(psnr_g)
        baseline_metrics['grayscale_photo']['ssim'].append(ssim_g)
        baseline_metrics['grayscale_photo']['l1'].append(l1_g)

        # Record per style
        if st_id in baseline_by_style:
            baseline_by_style[st_id]['all_white']['psnr'].append(psnr_w)
            baseline_by_style[st_id]['all_white']['ssim'].append(ssim_w)
            baseline_by_style[st_id]['all_white']['l1'].append(l1_w)

            baseline_by_style[st_id]['mean_sketch']['psnr'].append(psnr_m)
            baseline_by_style[st_id]['mean_sketch']['ssim'].append(ssim_m)
            baseline_by_style[st_id]['mean_sketch']['l1'].append(l1_m)

            baseline_by_style[st_id]['grayscale_photo']['psnr'].append(psnr_g)
            baseline_by_style[st_id]['grayscale_photo']['ssim'].append(ssim_g)
            baseline_by_style[st_id]['grayscale_photo']['l1'].append(l1_g)

    # Compute Model & Baseline Aggregate Metrics
    avg_l1 = float(np.mean([r['l1'] for r in all_results])) if all_results else 0.0
    avg_psnr = float(np.mean([r['psnr'] for r in all_results])) if all_results else 0.0
    avg_ssim = float(np.mean([r['ssim'] for r in all_results])) if all_results else 0.0

    real_arr = np.array(all_real_imgs) if all_real_imgs else np.zeros((0, 128, 128, 3))
    fake_arr = np.array(all_fake_imgs) if all_fake_imgs else np.zeros((0, 128, 128, 3))
    photo_arr = np.array(all_photo_imgs) if all_photo_imgs else np.zeros((0, 128, 128, 3))
    white_arr = np.ones_like(real_arr) if len(real_arr) > 0 else np.zeros((0, 128, 128, 3))
    mean_arr = np.tile(mean_sketch_img, (len(all_real_imgs), 1, 1, 1)) if len(all_real_imgs) > 0 else np.zeros((0, 128, 128, 3))

    pixel_fd = compute_pixel_frechet_distance(real_arr, fake_arr) if len(real_arr) > 0 else 0.0
    pixel_fd_white = compute_pixel_frechet_distance(real_arr, white_arr) if len(real_arr) > 0 else 0.0
    pixel_fd_mean = compute_pixel_frechet_distance(real_arr, mean_arr) if len(real_arr) > 0 else 0.0
    pixel_fd_gray = compute_pixel_frechet_distance(real_arr, photo_arr) if len(real_arr) > 0 else 0.0

    fps = total_images / total_inference_time if total_inference_time > 0 else 0.0
    latency_ms = (total_inference_time / total_images) * 1000.0 if total_images > 0 else 0.0

    # Summary of baselines
    baseline_summary = {
        'all_white': {
            'mean_psnr': round(float(np.mean(baseline_metrics['all_white']['psnr'])), 2) if baseline_metrics['all_white']['psnr'] else 0.0,
            'mean_ssim': round(float(np.mean(baseline_metrics['all_white']['ssim'])), 4) if baseline_metrics['all_white']['ssim'] else 0.0,
            'mean_l1': round(float(np.mean(baseline_metrics['all_white']['l1'])), 4) if baseline_metrics['all_white']['l1'] else 0.0,
            'pixel_frechet_distance': pixel_fd_white,
        },
        'mean_sketch': {
            'mean_psnr': round(float(np.mean(baseline_metrics['mean_sketch']['psnr'])), 2) if baseline_metrics['mean_sketch']['psnr'] else 0.0,
            'mean_ssim': round(float(np.mean(baseline_metrics['mean_sketch']['ssim'])), 4) if baseline_metrics['mean_sketch']['ssim'] else 0.0,
            'mean_l1': round(float(np.mean(baseline_metrics['mean_sketch']['l1'])), 4) if baseline_metrics['mean_sketch']['l1'] else 0.0,
            'pixel_frechet_distance': pixel_fd_mean,
        },
        'grayscale_photo': {
            'mean_psnr': round(float(np.mean(baseline_metrics['grayscale_photo']['psnr'])), 2) if baseline_metrics['grayscale_photo']['psnr'] else 0.0,
            'mean_ssim': round(float(np.mean(baseline_metrics['grayscale_photo']['ssim'])), 4) if baseline_metrics['grayscale_photo']['ssim'] else 0.0,
            'mean_l1': round(float(np.mean(baseline_metrics['grayscale_photo']['l1'])), 4) if baseline_metrics['grayscale_photo']['l1'] else 0.0,
            'pixel_frechet_distance': pixel_fd_gray,
        }
    }

    by_style_summary = {}
    for st_id in sorted(style_buckets.keys()):
        bucket = style_buckets[st_id]
        cnt = len(bucket)
        is_low_sample = (cnt < 100)  # Style 2 has only 46 test images
        if bucket:
            by_style_summary[f"style_{st_id}"] = {
                'count': cnt,
                'low_sample_warning': is_low_sample,
                'l1': round(float(np.mean([r['l1'] for r in bucket])), 4),
                'psnr': round(float(np.mean([r['psnr'] for r in bucket])), 2),
                'ssim': round(float(np.mean([r['ssim'] for r in bucket])), 4),
                'baselines': {
                    'all_white': {
                        'psnr': round(float(np.mean(baseline_by_style[st_id]['all_white']['psnr'])), 2),
                        'ssim': round(float(np.mean(baseline_by_style[st_id]['all_white']['ssim'])), 4),
                        'l1': round(float(np.mean(baseline_by_style[st_id]['all_white']['l1'])), 4),
                    },
                    'mean_sketch': {
                        'psnr': round(float(np.mean(baseline_by_style[st_id]['mean_sketch']['psnr'])), 2),
                        'ssim': round(float(np.mean(baseline_by_style[st_id]['mean_sketch']['ssim'])), 4),
                        'l1': round(float(np.mean(baseline_by_style[st_id]['mean_sketch']['l1'])), 4),
                    },
                    'grayscale_photo': {
                        'psnr': round(float(np.mean(baseline_by_style[st_id]['grayscale_photo']['psnr'])), 2),
                        'ssim': round(float(np.mean(baseline_by_style[st_id]['grayscale_photo']['ssim'])), 4),
                        'l1': round(float(np.mean(baseline_by_style[st_id]['grayscale_photo']['l1'])), 4),
                    }
                }
            }
        else:
            by_style_summary[f"style_{st_id}"] = {
                'count': 0, 'low_sample_warning': True, 'l1': 0.0, 'psnr': 0.0, 'ssim': 0.0, 'baselines': {}
            }

    # -------------------------------------------------------------------------
    # Stratified Visual Comparison Panels with Absolute Error Maps (|Fake - Real|)
    # Selects a fixed, stratified set of 4 test pairs per style (seeded for repeatability)
    # -------------------------------------------------------------------------
    print(f"[BENCHMARK] Generating style-stratified visual comparison panels and error maps (4 pairs per style)...")
    
    stratified_samples: List[Dict[str, Any]] = []
    rng_vis = np.random.RandomState(42)
    per_style_k = max(1, num_visualizations // 3) if num_visualizations >= 3 else 1

    for st_id in sorted(style_buckets.keys()):
        bucket = style_buckets[st_id]
        if bucket:
            sample_count = min(per_style_k, len(bucket))
            chosen_idxs = sorted(rng_vis.choice(len(bucket), sample_count, replace=False))
            for c_idx in chosen_idxs:
                stratified_samples.append(bucket[c_idx])

    # Fallback to general list if stratified buckets are empty
    if not stratified_samples:
        stratified_samples = all_results[:min(num_visualizations, len(all_results))]

    vis_count = len(stratified_samples)
    if vis_count > 0:
        fig, axes = plt.subplots(vis_count, 4, figsize=(15, 3.8 * vis_count))
        if vis_count == 1:
            axes = np.expand_dims(axes, 0)

        for idx, r in enumerate(stratified_samples):
            p_img = r['photo']
            r_img = r['real_sketch']
            f_img = r['gen_sketch']
            err_map = np.mean(np.abs(f_img - r_img), axis=-1)  # (H, W) in [0, 1]

            # 1. Input Photo
            axes[idx, 0].imshow(p_img)
            axes[idx, 0].set_title(f"[Style {r['style']}] Input Photo", fontsize=11, fontweight='bold')
            axes[idx, 0].axis('off')

            # 2. Ground Truth Sketch
            axes[idx, 1].imshow(r_img)
            axes[idx, 1].set_title(f"[Style {r['style']}] Ground Truth ({r['image_name']})", fontsize=11, fontweight='bold')
            axes[idx, 1].axis('off')

            # 3. cGAN Synthesis
            axes[idx, 2].imshow(f_img)
            axes[idx, 2].set_title(f"cGAN Gen (PSNR: {r['psnr']:.2f} dB, SSIM: {r['ssim']:.4f})", fontsize=11, fontweight='bold')
            axes[idx, 2].axis('off')

            # 4. Absolute Error Map
            axes[idx, 3].imshow(err_map, cmap='inferno', vmin=0.0, vmax=0.5)
            axes[idx, 3].set_title(f"|Fake - Real| (L1: {r['l1']:.4f})", fontsize=11, fontweight='bold')
            axes[idx, 3].axis('off')

        plt.tight_layout()
        comparison_plot_path = os.path.join(figures_dir, "test_synthesis_comparisons.png")
        plt.savefig(comparison_plot_path, dpi=180, bbox_inches='tight')
        plt.close()
        print(f"[BENCHMARK] Saved stratified 12-sample comparison grid to: {comparison_plot_path}")

    # -------------------------------------------------------------------------
    # Multi-Style Transfer Matrix Figure (4 subjects x 3 synthesized styles)
    # -------------------------------------------------------------------------
    print("[BENCHMARK] Generating multi-style transfer matrix figure across all 3 styles...")
    matrix_samples = []
    rng_matrix = np.random.RandomState(1337)
    if all_results:
        n_matrix = min(4, len(all_results))
        m_idxs = sorted(rng_matrix.choice(len(all_results), n_matrix, replace=False))
        matrix_samples = [all_results[i] for i in m_idxs]

    if matrix_samples:
        fig_m, axes_m = plt.subplots(len(matrix_samples), 4, figsize=(14, 3.6 * len(matrix_samples)))
        if len(matrix_samples) == 1:
            axes_m = np.expand_dims(axes_m, 0)

        generator.eval()
        with torch.no_grad():
            for row_idx, r_item in enumerate(matrix_samples):
                p_np = r_item['photo']
                # Convert back to normalized GAN tensor [-1, 1]
                p_tensor = torch.from_numpy(p_np).permute(2, 0, 1).unsqueeze(0).float()
                p_tensor = (p_tensor - 0.5) / 0.5
                p_tensor = p_tensor.to(device)

                # Col 0: Input Photo
                axes_m[row_idx, 0].imshow(p_np)
                axes_m[row_idx, 0].set_title(f"Input ({r_item['image_name']})", fontsize=11, fontweight='bold')
                axes_m[row_idx, 0].axis('off')

                # Cols 1-3: Styles 0, 1, 2
                for s_id in range(3):
                    s_tensor = torch.tensor([s_id], dtype=torch.long, device=device)
                    fake_out = generator(p_tensor, s_tensor)
                    fake_img = unnormalize_to_0_1(fake_out)[0]

                    axes_m[row_idx, s_id + 1].imshow(fake_img)
                    axes_m[row_idx, s_id + 1].set_title(f"Synthesized Style {s_id}", fontsize=11, fontweight='bold')
                    axes_m[row_idx, s_id + 1].axis('off')

        plt.suptitle("Task 4 Style-Conditioned Synthesis: Multi-Style Matrix (Styles 0, 1, 2)", fontsize=13, fontweight='bold', y=1.01)
        plt.tight_layout()
        matrix_plot_path = os.path.join(figures_dir, "multistyle_transfer_matrix.png")
        plt.savefig(matrix_plot_path, dpi=180, bbox_inches='tight')
        plt.close()
        print(f"[BENCHMARK] Saved multi-style transfer matrix to: {matrix_plot_path}")

    # Identify Worst Failure Cases (lowest PSNR)
    sorted_by_psnr = sorted(all_results, key=lambda x: x['psnr'])
    failure_cases = []
    for fc in sorted_by_psnr[:4]:
        failure_cases.append({
            'image_name': fc['image_name'],
            'style': fc['style'],
            'psnr': round(fc['psnr'], 2),
            'ssim': round(fc['ssim'], 4),
            'l1': round(fc['l1'], 4)
        })

    structured_results = {
        'summary': {
            'total_test_samples': total_images,
            'mean_l1': round(avg_l1, 4),
            'mean_psnr': round(avg_psnr, 2),
            'mean_ssim': round(avg_ssim, 4),
            'pixel_frechet_distance': pixel_fd,
            'latency_ms_per_image': round(latency_ms, 2),
            'throughput_fps': round(fps, 1)
        },
        'baselines': baseline_summary,
        'by_style': by_style_summary,
        'failure_cases': failure_cases
    }

    # Save JSON summary report
    report_path = os.path.join(output_dir, "cgan_benchmark_results.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(structured_results, f, indent=2)

    print("\n" + "=" * 76)
    print("TASK 4 CONDITIONAL GAN (FS2K) BENCHMARK SUMMARY & BASELINES")
    print("=" * 76)
    print(f"cGAN Generator Model:     PSNR = {structured_results['summary']['mean_psnr']} dB | SSIM = {structured_results['summary']['mean_ssim']} | L1 = {structured_results['summary']['mean_l1']} | Pixel-FD = {structured_results['summary']['pixel_frechet_distance']}")
    print(f"Baseline (All-White):     PSNR = {baseline_summary['all_white']['mean_psnr']} dB | SSIM = {baseline_summary['all_white']['mean_ssim']} | L1 = {baseline_summary['all_white']['mean_l1']} | Pixel-FD = {baseline_summary['all_white']['pixel_frechet_distance']}")
    print(f"Baseline (Mean Sketch):   PSNR = {baseline_summary['mean_sketch']['mean_psnr']} dB | SSIM = {baseline_summary['mean_sketch']['mean_ssim']} | L1 = {baseline_summary['mean_sketch']['mean_l1']} | Pixel-FD = {baseline_summary['mean_sketch']['pixel_frechet_distance']}")
    print(f"Baseline (Gray Photo):    PSNR = {baseline_summary['grayscale_photo']['mean_psnr']} dB | SSIM = {baseline_summary['grayscale_photo']['mean_ssim']} | L1 = {baseline_summary['grayscale_photo']['mean_l1']} | Pixel-FD = {baseline_summary['grayscale_photo']['pixel_frechet_distance']}")
    print(f"Inference Latency:        {structured_results['summary']['latency_ms_per_image']} ms/image ({structured_results['summary']['throughput_fps']} FPS)")
    print("-" * 76)
    print("STYLE-STRATIFIED BREAKDOWN & BASELINES (Styles 0, 1, 2):")
    for st_k, st_v in by_style_summary.items():
        warn = f" [LOW-SAMPLE N={st_v['count']} < 100]" if st_v['low_sample_warning'] else ""
        b_w = st_v['baselines']['all_white']['psnr'] if 'all_white' in st_v.get('baselines', {}) else 'N/A'
        b_m = st_v['baselines']['mean_sketch']['psnr'] if 'mean_sketch' in st_v.get('baselines', {}) else 'N/A'
        b_g = st_v['baselines']['grayscale_photo']['psnr'] if 'grayscale_photo' in st_v.get('baselines', {}) else 'N/A'
        print(f"  {st_k.upper()} (N={st_v['count']}){warn}:")
        print(f"    - cGAN Generator: PSNR = {st_v['psnr']} dB | SSIM = {st_v['ssim']} | L1 = {st_v['l1']}")
        print(f"    - Baselines:      All-White: {b_w} dB | Mean Sketch: {b_m} dB | Gray Photo: {b_g} dB")
    print("=" * 76)

    return structured_results
