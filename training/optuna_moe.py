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
    epochs_per_trial: int = 5
) -> float:
    """
    Optuna objective function for Soft MoE. Evaluates validation PSNR.
    Loss weights are strictly fixed per assignment spec:
      lambda_recon=0.8, lambda_class=0.2, lambda_balance=0.1, lambda_entropy=0.01, alpha=0.90
    Searches:
      - lr_joint in [5e-5, 5e-4]
      - lr_warmup in [1e-4, 1e-3]
      - temperature in [0.5, 2.0]
      - warmup_epochs in [1, 2, 3]
      - weight_decay in [1e-5, 1e-3]
    """
    # 1. Sample genuine training hyperparameters
    lr_joint = trial.suggest_float('lr_joint', 5e-5, 5e-4, log=True)
    lr_warmup = trial.suggest_float('lr_warmup', 1e-4, 1e-3, log=True)
    temperature = trial.suggest_float('temperature', 0.5, 2.0, step=0.1)
    warmup_epochs = trial.suggest_int('warmup_epochs', 1, 3)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)

    # Fixed loss formulation per assignment specification
    lambda_recon = 0.8
    lambda_class = 0.2
    lambda_balance = 0.1
    lambda_entropy = 0.01
    alpha = 0.90

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
        lambda_recon=lambda_recon,
        lambda_class=lambda_class,
        lambda_balance=lambda_balance,
        lambda_entropy=lambda_entropy,
        alpha=alpha
    ).to(device)

    # 3. Phase 1: Warm-up (Gate only)
    if warmup_epochs > 0:
        model.freeze_experts()
        model.unfreeze_gate()
        opt_warmup = torch.optim.AdamW(model.gate.parameters(), lr=lr_warmup, weight_decay=weight_decay)
        for _ in range(warmup_epochs):
            train_one_epoch_moe(model, train_loader, criterion, opt_warmup, device)

    # 4. Phase 2: Joint fine-tuning
    model.unfreeze_experts()
    model.unfreeze_gate()
    opt_joint = torch.optim.AdamW([
        {'params': model.gate.parameters(), 'lr': lr_joint},
        {'params': model.specialist_sp.parameters(), 'lr': lr_joint * 0.5},
        {'params': model.specialist_blur.parameters(), 'lr': lr_joint * 0.5},
        {'params': model.specialist_occlusion.parameters(), 'lr': lr_joint * 0.5},
    ], weight_decay=weight_decay)

    joint_eval_epochs = max(epochs_per_trial - warmup_epochs, 2)
    best_psnr = -float('inf')

    for step in range(1, joint_eval_epochs + 1):
        train_m = train_one_epoch_moe(model, train_loader, criterion, opt_joint, device)
        val_m, _ = evaluate_moe(model, val_loader, criterion, device)
        current_psnr = val_m['psnr']
        best_psnr = max(best_psnr, current_psnr)

        trial.report(current_psnr, step)
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
