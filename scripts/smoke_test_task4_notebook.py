"""
Comprehensive verification and end-to-end smoke test for Task 4:
1. json.load validation of ALL notebooks in notebooks/
2. Static AST / compile() check for all code cells in notebooks/05_task4_cgan_sketch.ipynb
3. End-to-end CPU training, checkpoint saving, strict checkpoint loading, benchmark run,
   training curve visualization, and ONNX parity check using authentic FS2K disk layout
   and get_fs2k_dataloaders (with batch_size > 2, real metadata structure, .JPG photo, .png sketch).
"""

import os
import sys
import json
import ast
import tempfile
import shutil
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.fs2k import get_fs2k_dataloaders, FS2KDataset
from models.cgan import StyleConditionedUNetGenerator, ConditionalPatchGANDiscriminator
from training.trainer_cgan import train_cgan_full
from evaluation.benchmark_cgan import run_cgan_benchmark
from models.onnx_export_cgan import export_cgan_generator_to_onnx, verify_cgan_onnx_numerical_equivalence


def test_1_json_and_compile_validation_all_notebooks():
    print("\n--- TEST 1: json.load & compile() validation of ALL notebooks ---")
    notebook_dir = "notebooks"
    files = [f for f in os.listdir(notebook_dir) if f.endswith(".ipynb")]
    assert len(files) > 0, "No notebooks found in notebooks/"

    for nb_file in sorted(files):
        path = os.path.join(notebook_dir, nb_file)
        with open(path, "r", encoding="utf-8") as f:
            nb = json.load(f)
        assert "cells" in nb, f"Malformed notebook (no cells): {nb_file}"

        code_cells = [c for c in nb["cells"] if c.get("cell_type") == "code"]
        for idx, cell in enumerate(code_cells, start=1):
            src_lines = cell.get("source", [])
            clean_lines = []
            for line in src_lines:
                l_strip = line.strip()
                indent = len(line) - len(line.lstrip())
                if l_strip.startswith("!") or l_strip.startswith("%"):
                    clean_lines.append(" " * indent + f"pass # {l_strip}\n")
                else:
                    clean_lines.append(line)
            clean_code = "".join(clean_lines)
            try:
                compile(clean_code, f"<{nb_file}_cell_{idx}>", "exec")
            except Exception as e:
                print(f"  [FAIL] {nb_file} Code cell {idx} failed compilation: {e}")
                raise e

        print(f"  [PASS] {nb_file}: valid JSON and all {len(code_cells)} code cells compiled successfully ({os.path.getsize(path)/1024:.1f} KB)")


def test_2_static_compile_task4_notebook():
    print("\n--- TEST 2: Detailed compile() check on Task 4 notebook code cells ---")
    nb_path = "notebooks/05_task4_cgan_sketch.ipynb"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    code_cell_idx = 0
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            code_cell_idx += 1
            src_lines = cell.get("source", [])
            clean_lines = []
            for line in src_lines:
                l_strip = line.strip()
                indent = len(line) - len(line.lstrip())
                if l_strip.startswith("!") or l_strip.startswith("%"):
                    clean_lines.append(" " * indent + f"pass # {l_strip}\n")
                else:
                    clean_lines.append(line)
            clean_code = "".join(clean_lines)
            try:
                compile(clean_code, f"<cell_{code_cell_idx}>", "exec")
                print(f"  [PASS] Task 4 Code cell {code_cell_idx} compiled successfully ({len(src_lines)} lines)")
            except Exception as e:
                print(f"  [FAIL] Task 4 Code cell {code_cell_idx} failed compilation: {e}")
                raise e


def _create_synthetic_fs2k_tree(root_dir: str, manifest_dir: str):
    """
    Creates an authentic FS2K directory structure with:
      - photo/photo1, photo/photo3
      - sketch/sketch1, sketch/sketch3
      - One uppercase .JPG photo and one lowercase .png sketch
      - Realistic annotation fields (skin_color [2], lip_color [3], hair_color [3])
      - fs2k_train_manifest.json, fs2k_val_manifest.json, fs2k_test_manifest.json
    """
    os.makedirs(os.path.join(root_dir, "photo", "photo1"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, "photo", "photo3"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, "sketch", "sketch1"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, "sketch", "sketch3"), exist_ok=True)
    os.makedirs(manifest_dir, exist_ok=True)

    def _make_split(n_samples: int, split_name: str, has_uppercase_jpg: bool = False):
        items = []
        for i in range(n_samples):
            st = i % 3
            folder_id = "photo3" if (st == 2 or i % 2 == 1) else "photo1"
            sketch_folder = folder_id.replace("photo", "sketch")
            base_name = f"image_{split_name}_{i:04d}"

            if has_uppercase_jpg and i == 0:
                # Authentic corner case: uppercase .JPG photo and lowercase .png sketch
                p_path = os.path.join(root_dir, "photo", folder_id, f"{base_name}.JPG")
                s_path = os.path.join(root_dir, "sketch", sketch_folder, f"{base_name.replace('image', 'sketch')}.png")
            else:
                p_path = os.path.join(root_dir, "photo", folder_id, f"{base_name}.jpg")
                s_path = os.path.join(root_dir, "sketch", sketch_folder, f"{base_name.replace('image', 'sketch')}.jpg")

            # Create dummy 128x128 images
            img_p = Image.fromarray((np.random.RandomState(i).uniform(0, 255, (128, 128, 3))).astype(np.uint8))
            img_s = Image.fromarray((np.random.RandomState(i + 100).uniform(0, 255, (128, 128, 3))).astype(np.uint8))
            img_p.save(p_path)
            img_s.save(s_path)

            # Realistic metadata with variable length list fields (the exact cause of earlier collate crash)
            items.append({
                "image_name": f"{folder_id}/{base_name}",
                "style": st,
                "skin_color": [156, 137],
                "lip_color": [197, 125, 109],
                "hair_color": [42, 33, 29],
                "gender": 1 if i % 2 == 0 else 0,
                "split": split_name
            })

        manifest_file = os.path.join(manifest_dir, f"fs2k_{split_name}_manifest.json")
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2)
        return items

    _make_split(8, "train", has_uppercase_jpg=False)
    _make_split(4, "val", has_uppercase_jpg=False)
    _make_split(6, "test", has_uppercase_jpg=True)


def test_3_end_to_end_smoke_test():
    print("\n--- TEST 3: End-to-End CPU Smoke Test (Steps 6, 7, 8, 9 Simulation on Real Layout) ---")
    device = torch.device('cpu')
    tmp_dir = tempfile.mkdtemp(prefix="task4_smoke_real_")

    try:
        local_ckpt_dir = os.path.join(tmp_dir, "checkpoints/task4")
        drive_ckpt_dir = os.path.join(tmp_dir, "drive/checkpoints/task4")
        eval_out_dir = os.path.join(tmp_dir, "artifacts/evaluation_task4")
        drive_eval_dir = os.path.join(tmp_dir, "drive/artifacts/evaluation_task4")
        manifest_dir = os.path.join(tmp_dir, "configs/manifests")
        data_root = os.path.join(tmp_dir, "local_data/FS2K")

        os.makedirs(local_ckpt_dir, exist_ok=True)
        os.makedirs(drive_ckpt_dir, exist_ok=True)
        os.makedirs(eval_out_dir, exist_ok=True)
        os.makedirs(drive_eval_dir, exist_ok=True)

        # 1. Build genuine on-disk FS2K tree
        _create_synthetic_fs2k_tree(root_dir=data_root, manifest_dir=manifest_dir)
        print(f"  [Data Setup] Created authentic FS2K disk tree in {data_root} with manifests in {manifest_dir}")

        # 2. Build DataLoaders via get_fs2k_dataloaders with batch_size=4 (> 2)
        train_loader, val_loader, test_loader = get_fs2k_dataloaders(
            manifest_dir=manifest_dir,
            fs2k_root=data_root,
            batch_size=4,
            num_workers=0
        )
        assert len(train_loader) >= 2, "Train loader must have >= 2 batches"
        assert len(val_loader) >= 1, "Val loader must have >= 1 batch"
        assert len(test_loader) >= 2, "Test loader must have >= 2 batches"

        # Verify batch structure contains flat metadata and tensors
        sample_batch = next(iter(train_loader))
        assert sample_batch['photo'].shape == (4, 3, 128, 128)
        assert sample_batch['sketch'].shape == (4, 3, 128, 128)
        assert sample_batch['style'].shape == (4,)
        print(f"  [Data Setup] Dataloaders verified with batch_size=4. Photo shape: {sample_batch['photo'].shape}")

        # 3. Simulate Step 5: Save Optuna best_params JSON
        best_params = {
            'lr_g': 0.0004,
            'lr_d': 0.00008,
            'batch_size': 4,
            'lambda_l1': 100.0,
            'base_channels_g': 32,
            'emb_dim': 16,
            'dropout_rate': 0.2
        }
        with open(os.path.join(local_ckpt_dir, "optuna_best_params.json"), "w") as f:
            json.dump(best_params, f)
        with open(os.path.join(drive_ckpt_dir, "optuna_best_params.json"), "w") as f:
            json.dump(best_params, f)
        print("  [Step 5 Sim] Saved optuna_best_params.json to local and Drive.")

        # 4. Simulate Step 6: Full Training with drive_save_dir
        net_g = StyleConditionedUNetGenerator(
            in_channels=3, out_channels=3, num_styles=3,
            emb_dim=best_params['emb_dim'],
            emb_channels=16,
            base_channels=best_params['base_channels_g'],
            dropout_rate=best_params['dropout_rate']
        ).to(device)

        net_d = ConditionalPatchGANDiscriminator(
            in_channels=3, num_styles=3,
            emb_dim=best_params['emb_dim'],
            emb_channels=16,
            base_channels=32
        ).to(device)

        train_res = train_cgan_full(
            net_g=net_g,
            net_d=net_d,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            save_dir=local_ckpt_dir,
            drive_save_dir=drive_ckpt_dir,
            epochs=2,
            lr_g=best_params['lr_g'],
            lr_d=best_params['lr_d'],
            lambda_l1=best_params['lambda_l1'],
            patience=2
        )

        assert os.path.exists(os.path.join(local_ckpt_dir, "best_cgan_generator.pth")), "Local checkpoint missing!"
        assert os.path.exists(os.path.join(drive_ckpt_dir, "best_cgan_generator.pth")), "Drive checkpoint missing!"
        assert os.path.exists(os.path.join(local_ckpt_dir, "training_history.json")), "Local training_history.json missing!"
        assert os.path.exists(os.path.join(drive_ckpt_dir, "training_history.json")), "Drive training_history.json missing!"

        # Verify all 4 required losses recorded separately
        assert 'history' in train_res
        assert len(train_res['history']) == 2
        for h_row in train_res['history']:
            for key in ['loss_d_real', 'loss_d_fake', 'loss_g_adv', 'loss_g_recon_l1', 'val_psnr', 'val_ssim', 'val_l1', 'd_updates_ratio']:
                assert key in h_row, f"Required metric key '{key}' missing from training history row!"
        print(f"  [Step 6 Sim] Training complete. Checkpoints & 4-loss history verified in local and Drive.")

        # 5. Simulate Step 7: Load from Drive checkpoint & Run Benchmark with baselines
        rebuilt_g = StyleConditionedUNetGenerator(
            in_channels=3, out_channels=3, num_styles=3,
            emb_dim=best_params['emb_dim'],
            emb_channels=16,
            base_channels=best_params['base_channels_g'],
            dropout_rate=best_params['dropout_rate']
        ).to(device)

        drive_ckpt_file = os.path.join(drive_ckpt_dir, "best_cgan_generator.pth")
        ckpt = torch.load(drive_ckpt_file, map_location=device)
        rebuilt_g.load_state_dict(ckpt['generator_state_dict'], strict=True)
        print(f"  [Step 7 Sim] Strict load_state_dict passed from Drive checkpoint (Epoch {ckpt['epoch']}).")

        bench_res = run_cgan_benchmark(
            generator=rebuilt_g,
            manifest_path=os.path.join(manifest_dir, "fs2k_test_manifest.json"),
            fs2k_root=data_root,
            device="cpu",
            output_dir=eval_out_dir,
            num_visualizations=4
        )
        assert 'summary' in bench_res
        assert 'baselines' in bench_res
        assert 'all_white' in bench_res['baselines']
        assert 'mean_sketch' in bench_res['baselines']
        assert 'grayscale_photo' in bench_res['baselines']
        assert os.path.exists(os.path.join(eval_out_dir, "cgan_benchmark_results.json"))
        assert os.path.exists(os.path.join(eval_out_dir, "figures/test_synthesis_comparisons.png"))
        print(f"  [Step 7 Sim] Benchmark executed successfully with 3 baselines and absolute error maps.")

        # 6. Simulate Step 8: Plot training & validation curves from training_history.json
        with open(os.path.join(local_ckpt_dir, "training_history.json"), "r") as f:
            hist = json.load(f)
        eps = [h['epoch'] for h in hist]
        d_loss = [h['loss_d_total'] for h in hist]
        g_adv = [h['loss_g_adv'] for h in hist]
        g_l1 = [h['loss_g_recon_l1'] for h in hist]
        val_psnr = [h['val_psnr'] for h in hist]

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(eps, d_loss, label='D Loss')
        ax[0].plot(eps, g_adv, label='G Adv Loss')
        ax[0].plot(eps, g_l1, label='G L1 Loss')
        ax[0].legend()
        ax[1].plot(eps, val_psnr, label='Val PSNR (dB)')
        ax[1].legend()
        plt.tight_layout()
        curves_path = os.path.join(eval_out_dir, "figures", "training_curves.png")
        plt.savefig(curves_path)
        plt.close()
        assert os.path.exists(curves_path)
        print(f"  [Step 8 Sim] Training & validation curves plotted successfully.")

        # 7. Simulate Step 9: Export to ONNX & Parity check
        onnx_file = os.path.join(local_ckpt_dir, "cgan_generator.onnx")
        export_cgan_generator_to_onnx(rebuilt_g, onnx_file, opset_version=18)
        parity = verify_cgan_onnx_numerical_equivalence(rebuilt_g, onnx_file, atol=1e-4)
        assert parity['equivalent'] is True, "ONNX parity failed!"
        print(f"  [Step 9 Sim] ONNX export and numerical parity passed (Size: {parity['onnx_size_mb']:.2f} MB).")

        print("\n>>> ALL SMOKE TESTS PASSED END-TO-END WITH AUTHENTIC FS2K DATASET STRUCTURE! <<<")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == '__main__':
    test_1_json_and_compile_validation_all_notebooks()
    test_2_static_compile_task4_notebook()
    test_3_end_to_end_smoke_test()
