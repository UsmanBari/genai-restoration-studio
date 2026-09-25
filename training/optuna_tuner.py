"""
Optuna Hyperparameter Tuning Module for Universal Autoencoder (Task 1).
Tunes:
  - learning_rate (loguniform 1e-4 to 1e-2)
  - batch_size (16, 32, 64)
  - bottleneck_dim (128, 256, 512)
  - base_channels (32, 48, 64)
  - dropout_rate (0.0, 0.1, 0.2)
  - alpha (0.5 to 0.95)
Integrates Optuna MedianPruner and logs every trial to MLflow SQLite database.
"""

from typing import Dict, Any, Optional
import os
import optuna
from optuna.pruners import MedianPruner
import mlflow

try:
    import torch
    from torch.utils.data import DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    DataLoader = None

from models.autoencoders import UniversalAutoencoder
from data.oxford_pet import get_oxford_dataloaders, OxfordPetDataset
from training.losses import RestorationLoss
from training.trainer_universal import train_one_epoch, evaluate


def objective_universal(
    trial: optuna.Trial,
    manifest_dir: str,
    images_dir: str,
    trial_epochs: int = 4,
    device: str = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"
) -> float:
    """Optuna objective function for a single trial."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required to execute Optuna tuning trials.")

    # 1. Sample hyperparameters
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    base_channels = trial.suggest_categorical("base_channels", [32, 48, 64])
    bottleneck_dim = trial.suggest_categorical("bottleneck_dim", [128, 256, 512])
    dropout_rate = trial.suggest_categorical("dropout_rate", [0.0, 0.1, 0.2])
    alpha = trial.suggest_float("alpha", 0.50, 0.95, step=0.05)

    # 2. Build DataLoaders using the unified get_oxford_dataloaders
    pin_mem = device.startswith('cuda')
    train_loader, val_loader, _ = get_oxford_dataloaders(
        manifest_dir=manifest_dir,
        images_dir=images_dir,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=pin_mem
    )

    # 3. Build Model & Optimizer
    model = UniversalAutoencoder(
        in_channels=3,
        out_channels=3,
        base_channels=base_channels,
        bottleneck_dim=bottleneck_dim,
        dropout_rate=dropout_rate
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = RestorationLoss(alpha=alpha)
    scaler = torch.cuda.amp.GradScaler() if device.startswith('cuda') else None

    # 4. Short Training Loop with Pruning
    best_val_loss = float('inf')

    # Log trial as nested run in MLflow if active
    with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
        mlflow.log_params(trial.params)

        for epoch in range(1, trial_epochs + 1):
            train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device)
            val_metrics = evaluate(model, val_loader, criterion, device)

            val_loss = val_metrics['loss']
            if val_loss < best_val_loss:
                best_val_loss = val_loss

            mlflow.log_metric("trial_train_loss", train_metrics['loss'], step=epoch)
            mlflow.log_metric("trial_val_loss", val_loss, step=epoch)
            mlflow.log_metric("trial_val_psnr", val_metrics['psnr'], step=epoch)

            # Report to Optuna for pruning
            trial.report(val_loss, epoch)
            if trial.should_prune():
                mlflow.set_tag("pruned", "true")
                raise optuna.exceptions.TrialPruned()

        mlflow.log_metric("final_val_loss", best_val_loss)
        mlflow.set_tag("pruned", "false")

    return best_val_loss


def run_optuna_study(
    manifest_dir: str,
    images_dir: str,
    n_trials: int = 15,
    trial_epochs: int = 4,
    study_name: str = "task1_universal_tuning",
    device: str = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"
) -> optuna.Study:
    """
    Executes Optuna study with MedianPruner and returns best trial configuration.
    """
    pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=1)
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        pruner=pruner
    )

    print(f"Starting Optuna Hyperparameter Search ({n_trials} trials, {trial_epochs} epochs each)...")
    study.optimize(
        lambda trial: objective_universal(
            trial,
            manifest_dir=manifest_dir,
            images_dir=images_dir,
            trial_epochs=trial_epochs,
            device=device
        ),
        n_trials=n_trials
    )

    print("\n=== Optuna Study Completed ===")
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Best Trial #{study.best_trial.number}:")
    print(f"  Best Val Loss: {study.best_value:.4f}")
    print("  Best Parameters:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    return study
