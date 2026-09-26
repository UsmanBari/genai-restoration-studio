"""
Optuna Hyperparameter Optimization for Soft Mixture-of-Experts (Task 3).

Optimizes:
  - joint fine-tuning learning rate (lr_joint)
  - softmax routing temperature (tau)
  - classification loss weight (lambda_class)
  - load balancing penalty weight (lambda_balance)
  - routing entropy regularizer weight (lambda_entropy)
  - reconstruction loss weighting (alpha)
"""

from typing import Dict, Any, Optional
import os
import torch
import optuna
from optuna.pruners import MedianPruner
from torch.utils.data import DataLoader

from models.moe import SoftMoERestorationNetwork
from training.losses import SoftMoECompositeLoss
from training.trainer_moe import train_one_epoch_moe, evaluate_moe


def objective_moe(
    trial: optuna.Trial,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    warmstart_paths: Dict[str, Optional[str]],
    epochs_per_trial: int = 4
) -> float:
    """
    Optuna objective function for Soft MoE. Evaluates validation PSNR.
    """
    # 1. Sample hyperparameters
    lr_joint = trial.suggest_float('lr_joint', 5e-5, 5e-4, log=True)
    temperature = trial.suggest_float('temperature', 0.5, 2.0, step=0.1)
    lambda_class = trial.suggest_float('lambda_class', 0.1, 0.4, step=0.05)
    lambda_balance = trial.suggest_float('lambda_balance', 0.02, 0.20, step=0.02)
    lambda_entropy = trial.suggest_float('lambda_entropy', 0.002, 0.04, log=True)
    alpha = trial.suggest_float('alpha', 0.75, 0.95, step=0.05)

    # 2. Instantiate Soft MoE and load warm-start weights
    model = SoftMoERestorationNetwork(temperature=temperature).to(device)
    model.load_warmstart_weights(
        classifier_path=warmstart_paths.get('classifier'),
        sp_path=warmstart_paths.get('sp'),
        blur_path=warmstart_paths.get('blur'),
        occlusion_path=warmstart_paths.get('occlusion'),
        device=str(device)
    )

    criterion = SoftMoECompositeLoss(
        lambda_recon=0.8,
        lambda_class=lambda_class,
        lambda_balance=lambda_balance,
        lambda_entropy=lambda_entropy,
        alpha=alpha
    ).to(device)

    # Joint optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_joint, weight_decay=1e-4)

    # 3. Fast training loop with pruning
    best_psnr = -float('inf')
    for epoch in range(1, epochs_per_trial + 1):
        train_m = train_one_epoch_moe(model, train_loader, criterion, optimizer, device)
        val_m, _ = evaluate_moe(model, val_loader, criterion, device)
        current_psnr = val_m['psnr']
        best_psnr = max(best_psnr, current_psnr)

        trial.report(current_psnr, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_psnr


def run_optuna_moe_study(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    warmstart_paths: Dict[str, Optional[str]],
    n_trials: int = 15,
    epochs_per_trial: int = 4,
    study_name: str = "soft_moe_hparam_search"
) -> optuna.Study:
    """
    Executes a 15-trial Optuna study for Soft MoE hyperparameters.
    """
    pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=1)
    study = optuna.create_study(direction="maximize", pruner=pruner, study_name=study_name)

    print(f"\n[OPTUNA] Starting {n_trials}-trial study for Soft MoE ({epochs_per_trial} epochs/trial)...")
    study.optimize(
        lambda trial: objective_moe(
            trial, train_loader, val_loader, device, warmstart_paths, epochs_per_trial
        ),
        n_trials=n_trials
    )

    print("\n" + "=" * 60)
    print(f"[OPTUNA STUDY COMPLETE] Best Val PSNR: {study.best_value:.2f} dB")
    print("Best Hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  - {k}: {v}")
    print("=" * 60)

    return study
