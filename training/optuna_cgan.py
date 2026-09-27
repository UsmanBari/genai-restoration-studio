"""
Optuna Hyperparameter Optimization Study for Style-Conditioned cGAN (Task 4).
Searches:
  - lr_g: Generator learning rate
  - lr_d: Discriminator learning rate
  - lambda_l1: L1 paired reconstruction loss weight
  - base_channels_g: Generator base channel count
  - emb_dim: Style categorical embedding dimension
  - dropout_rate: Bottleneck dropout rate
"""

from typing import Dict, Any, Optional
import os
import torch
from torch.utils.data import DataLoader

try:
    import optuna
    from optuna.pruners import MedianPruner
    HAS_OPTUNA = True
except ImportError:
    optuna = None
    MedianPruner = None
    HAS_OPTUNA = False

from models.cgan import StyleConditionedUNetGenerator, ConditionalPatchGANDiscriminator
from training.losses_cgan import ConditionalGANLoss
from training.trainer_cgan import train_one_epoch_cgan, evaluate_cgan


def objective_cgan(
    trial: optuna.Trial,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs_per_trial: int = 5
) -> float:
    """
    Optuna objective function for cGAN. Evaluates validation PSNR across trial epochs.
    """
    # 1. Sample hyperparameters (covering all 7 spec-mandated parameters)
    lr_g = trial.suggest_float('lr_g', 1e-4, 5e-4, log=True)
    lr_d = trial.suggest_float('lr_d', 1e-4, 5e-4, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    lambda_l1 = trial.suggest_float('lambda_l1', 50.0, 150.0, step=10.0)
    base_channels_g = trial.suggest_categorical('base_channels_g', [32, 64])
    base_channels_d = 64
    emb_dim = trial.suggest_categorical('emb_dim', [16, 32, 64])
    dropout_rate = trial.suggest_categorical('dropout_rate', [0.0, 0.2, 0.5])

    # Dynamic trial DataLoader for sampled batch_size
    trial_train_loader = DataLoader(
        train_loader.dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )


    # 2. Instantiate Models and Loss
    net_g = StyleConditionedUNetGenerator(
        in_channels=3,
        out_channels=3,
        num_styles=3,
        emb_dim=emb_dim,
        emb_channels=16,
        base_channels=base_channels_g,
        dropout_rate=dropout_rate
    ).to(device)

    net_d = ConditionalPatchGANDiscriminator(
        in_channels=3,
        num_styles=3,
        emb_dim=emb_dim,
        emb_channels=16,
        base_channels=base_channels_d
    ).to(device)

    criterion = ConditionalGANLoss(lambda_l1=lambda_l1, real_label=0.9, fake_label=0.0)
    opt_g = torch.optim.Adam(net_g.parameters(), lr=lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(net_d.parameters(), lr=lr_d, betas=(0.5, 0.999))

    best_psnr = -float('inf')

    for epoch in range(1, epochs_per_trial + 1):
        train_one_epoch_cgan(
            net_g, net_d, trial_train_loader, opt_g, opt_d, criterion, device,
            d_update_freq=2
        )
        val_m = evaluate_cgan(net_g, val_loader, device)


        current_psnr = val_m['psnr']
        best_psnr = max(best_psnr, current_psnr)

        trial.report(current_psnr, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_psnr


def run_optuna_cgan_study(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    n_trials: int = 15,
    epochs_per_trial: int = 5,
    study_name: str = "cgan_face_to_sketch_study"
) -> optuna.Study:
    """Executes a 15-trial Optuna study for cGAN hyperparameters."""
    if not HAS_OPTUNA:
        raise ImportError("Optuna is not installed.")

    pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=1)
    study = optuna.create_study(direction="maximize", pruner=pruner, study_name=study_name)

    print(f"\n[OPTUNA] Starting {n_trials}-trial study for Task 4 cGAN ({epochs_per_trial} epochs/trial)...")
    study.optimize(
        lambda trial: objective_cgan(trial, train_loader, val_loader, device, epochs_per_trial),
        n_trials=n_trials
    )

    print("\n" + "=" * 60)
    print(f"[OPTUNA STUDY COMPLETE] Best Val PSNR: {study.best_value:.2f} dB")
    print("Best Hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  - {k}: {v}")
    print("=" * 60)

    return study
