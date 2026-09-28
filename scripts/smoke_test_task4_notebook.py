"""
Comprehensive verification and end-to-end smoke test for Task 4:
1. json.load validation of ALL notebooks in notebooks/
2. Static AST / compile() check for all code cells in notebooks/05_task4_cgan_sketch.ipynb
3. End-to-end CPU training, checkpoint saving, strict checkpoint loading, benchmark run, and ONNX parity check
"""

import os
import sys
import json
import ast
import tempfile
import shutil
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.cgan import StyleConditionedUNetGenerator, ConditionalPatchGANDiscriminator
from training.trainer_cgan import train_cgan_full
from evaluation.benchmark_cgan import run_cgan_benchmark
from models.onnx_export_cgan import export_cgan_generator_to_onnx, verify_cgan_onnx_numerical_equivalence


def test_1_json_validation_all_notebooks():
    print("\n--- TEST 1: json.load validation of ALL notebooks ---")
    notebook_dir = "notebooks"
    files = [f for f in os.listdir(notebook_dir) if f.endswith(".ipynb")]
    assert len(files) > 0, "No notebooks found in notebooks/"

    for nb_file in sorted(files):
        path = os.path.join(notebook_dir, nb_file)
        with open(path, "r", encoding="utf-8") as f:
            nb = json.load(f)
        assert "cells" in nb, f"Malformed notebook (no cells): {nb_file}"
        print(f"  [PASS] {nb_file} is valid JSON ({len(nb['cells'])} cells, {os.path.getsize(path)/1024:.1f} KB)")


def test_2_static_compile_task4_notebook():
    print("\n--- TEST 2: Static compile() check on Task 4 notebook code cells ---")
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
                # Strip shell / colab magics for pure python syntax compilation
                l_strip = line.strip()
                if l_strip.startswith("!") or l_strip.startswith("%"):
                    clean_lines.append(f"# {line}")
                else:
                    clean_lines.append(line)
            clean_code = "".join(clean_lines)
            try:
                compile(clean_code, f"<cell_{code_cell_idx}>", "exec")
                print(f"  [PASS] Code cell {code_cell_idx} compiled successfully ({len(src_lines)} lines)")
            except Exception as e:
                print(f"  [FAIL] Code cell {code_cell_idx} failed compilation: {e}")
                raise e


class SyntheticFS2KDataset(Dataset):
    """Synthetic paired dataset mimicking FS2K photo/sketch pairs."""
    def __init__(self, n_samples: int = 16):
        self.n_samples = n_samples
        self.samples = []
        for i in range(n_samples):
            st = i % 3
            self.samples.append({
                'photo_path': f'synthetic_photo_{i}.jpg',
                'sketch_path': f'synthetic_sketch_{i}.jpg',
                'style': st,
                'image_name': f'synth_{i}'
            })

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Return photo in [-1, 1], sketch in [-1, 1], style int
        st = self.samples[idx]['style']
        photo = torch.randn(3, 128, 128).clamp(-1.0, 1.0)
        sketch = torch.randn(3, 128, 128).clamp(-1.0, 1.0)
        return {
            'photo': photo,
            'sketch': sketch,
            'style': torch.tensor(st, dtype=torch.long),
            'metadata': self.samples[idx]
        }


def test_3_end_to_end_smoke_test():
    print("\n--- TEST 3: End-to-End CPU Smoke Test (Steps 6, 7, 8, 9 Simulation) ---")
    device = torch.device('cpu')
    tmp_dir = tempfile.mkdtemp(prefix="task4_smoke_")
    
    try:
        local_ckpt_dir = os.path.join(tmp_dir, "checkpoints/task4")
        drive_ckpt_dir = os.path.join(tmp_dir, "drive/checkpoints/task4")
        eval_out_dir = os.path.join(tmp_dir, "artifacts/evaluation_task4")
        drive_eval_dir = os.path.join(tmp_dir, "drive/artifacts/evaluation_task4")
        synth_manifest_dir = os.path.join(tmp_dir, "configs/manifests")
        synth_data_dir = os.path.join(tmp_dir, "local_data/FS2K")

        os.makedirs(local_ckpt_dir, exist_ok=True)
        os.makedirs(drive_ckpt_dir, exist_ok=True)
        os.makedirs(eval_out_dir, exist_ok=True)
        os.makedirs(drive_eval_dir, exist_ok=True)
        os.makedirs(synth_manifest_dir, exist_ok=True)
        os.makedirs(os.path.join(synth_data_dir, "photo/photo1"), exist_ok=True)
        os.makedirs(os.path.join(synth_data_dir, "sketch/sketch1"), exist_ok=True)

        # Create dummy test manifest and images
        manifest_items = []
        for i in range(12):
            st = i % 3
            p_rel = f"photo/photo1/img_{i}.jpg"
            s_rel = f"sketch/sketch1/img_{i}.jpg"
            p_full = os.path.join(synth_data_dir, p_rel)
            s_full = os.path.join(synth_data_dir, s_rel)
            
            # Save dummy 128x128 images
            Image.fromarray((np.random.rand(128, 128, 3) * 255).astype(np.uint8)).save(p_full)
            Image.fromarray((np.random.rand(128, 128, 3) * 255).astype(np.uint8)).save(s_full)
            
            manifest_items.append({
                'photo_path': p_rel,
                'sketch_path': s_rel,
                'style': st,
                'image_name': f"img_{i}",
                'split': 'test'
            })

        test_manifest_file = os.path.join(synth_manifest_dir, "fs2k_test_manifest.json")
        with open(test_manifest_file, 'w', encoding='utf-8') as f:
            json.dump(manifest_items, f)

        # 1. Simulate Step 5: Save Optuna best_params JSON
        best_params = {
            'lr_g': 0.0004,
            'lr_d': 0.00008,
            'batch_size': 4,
            'lambda_l1': 100.0,
            'base_channels_g': 32,
            'emb_dim': 16,
            'dropout_rate': 0.2
        }
        local_optuna_file = os.path.join(local_ckpt_dir, "optuna_best_params.json")
        drive_optuna_file = os.path.join(drive_ckpt_dir, "optuna_best_params.json")
        with open(local_optuna_file, 'w') as f:
            json.dump(best_params, f)
        with open(drive_optuna_file, 'w') as f:
            json.dump(best_params, f)
        print("  [Step 5 Sim] Saved optuna_best_params.json to local and Drive.")

        # 2. Simulate Step 6: Full Training with drive_save_dir
        train_ds = SyntheticFS2KDataset(n_samples=8)
        val_ds = SyntheticFS2KDataset(n_samples=4)
        train_ldr = DataLoader(train_ds, batch_size=4, shuffle=True)
        val_ldr = DataLoader(val_ds, batch_size=4, shuffle=False)

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
            train_loader=train_ldr,
            val_loader=val_ldr,
            device=device,
            save_dir=local_ckpt_dir,
            drive_save_dir=drive_ckpt_dir,
            epochs=2,
            lr_g=best_params['lr_g'],
            lr_d=best_params['lr_d'],
            lambda_l1=best_params['lambda_l1'],
            patience=2
        )

        assert os.path.exists(os.path.join(local_ckpt_dir, "best_cgan_generator.pth")), "Local checkpoint not saved!"
        assert os.path.exists(os.path.join(drive_ckpt_dir, "best_cgan_generator.pth")), "Drive checkpoint not saved!"
        print(f"  [Step 6 Sim] Training complete. Checkpoints verified in both local ({os.path.getsize(os.path.join(local_ckpt_dir, 'best_cgan_generator.pth'))/1024/1024:.2f} MB) and Drive.")

        # 3. Simulate Step 7: Load from Drive checkpoint & Run Benchmark
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
            manifest_path=synth_manifest_dir,
            fs2k_root=synth_data_dir,
            device="cpu",
            output_dir=eval_out_dir,
            num_visualizations=4
        )
        assert 'summary' in bench_res
        assert os.path.exists(os.path.join(eval_out_dir, "cgan_benchmark_results.json"))
        assert os.path.exists(os.path.join(eval_out_dir, "figures/test_synthesis_comparisons.png"))
        print(f"  [Step 7 Sim] Benchmark executed successfully. Generated report and test figures.")

        # 4. Simulate Step 9: Export to ONNX & Parity check
        onnx_file = os.path.join(local_ckpt_dir, "cgan_generator.onnx")
        export_cgan_generator_to_onnx(rebuilt_g, onnx_file, opset_version=18)
        parity = verify_cgan_onnx_numerical_equivalence(rebuilt_g, onnx_file, atol=1e-4)
        assert parity['equivalent'] is True, "ONNX parity failed!"
        print(f"  [Step 9 Sim] ONNX export and numerical parity passed (Size: {parity['onnx_size_mb']:.2f} MB).")

        print("\n>>> ALL SMOKE TESTS PASSED END-TO-END! <<<")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == '__main__':
    test_1_json_validation_all_notebooks()
    test_2_static_compile_task4_notebook()
    test_3_end_to_end_smoke_test()
