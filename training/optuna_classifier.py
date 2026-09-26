"""
Optuna Hyperparameter Tuning for Corruption Classifier (Task 2).
Tunes:
  - lr (loguniform 1e-4 to 5e-3)
  - batch_size (16, 32, 64)
  - base_channels (16, 24, 32, 48)
  - dropout_rate (0.1, 0.2, 0.3, 0.5)

Evaluates on validation accuracy / macro F1 with MedianPruner and MLflow tracking.
"""

from typing import Dict, Any, Optional
import os
import optuna
from optuna.pruners import MedianPruner
import mlflow

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    DataLoader = None

from models.classifiers import CorruptionClassifier
from data.oxford_pet import get_oxford_dataloaders
from training.trainer_classifier import train_one_epoch_classifier, evaluate_classifier


def objective_classifier(
    trial: optuna.Trial,
    manifest_dir: str,
    images_dir: str,
    trial_epochs: int = 3,
    device: str = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"
) -> float:
    """Optuna objective function for corruption classifier tuning."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for classifier Optuna tuning.")

    # 1. Sample hyperparameters
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    base_channels = trial.suggest_categorical("base_channels", [16, 24, 32, 48])
    dropout_rate = trial.suggest_categorical("dropout_rate", [0.1, 0.2, 0.3, 0.5])

    # 2. Build balanced runtime corruption DataLoaders
    pin_mem = device.startswith('cuda')
    train_loader, val_loader, _ = get_oxford_dataloaders(
        manifest_dir=manifest_dir,
        images_dir=images_dir,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=pin_mem
    )

    # 3. Instantiate model & optimizer
    model = CorruptionClassifier(
        in_channels=3,
        num_classes=4,
        base_channels=base_channels,
        dropout_rate=dropout_rate
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler() if device.startswith('cuda') else None

    best_val_acc = 0.0

    with mlflow.start_run(run_name=f"classifier_trial_{trial.number}", nested=True):
        mlflow.log_params(trial.params)

        for epoch in range(1, trial_epochs + 1):
            train_metrics = train_one_epoch_classifier(model, train_loader, optimizer, criterion, scaler, device)
            val_metrics = evaluate_classifier(model, val_loader, criterion, device)

            val_acc = val_metrics['accuracy']
            if val_acc > best_val_acc:
                best_val_acc = val_acc

            mlflow.log_metric("trial_train_loss", train_metrics['loss'], step=epoch)
            mlflow.log_metric("trial_train_acc", train_metrics['accuracy'], step=epoch)
            mlflow.log_metric("trial_val_loss", val_metrics['loss'], step=epoch)
            mlflow.log_metric("trial_val_acc", val_metrics['accuracy'], step=epoch)
            mlflow.log_metric("trial_val_macro_f1", val_metrics['macro_f1'], step=epoch)

            # Report to Optuna for MedianPruner evaluation
            trial.report(val_acc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return best_val_acc


def run_classifier_optuna_study(
    manifest_dir: str,
    images_dir: str,
    n_trials: int = 15,
    trial_epochs: int = 3,
    study_name: str = "task2_classifier_search",
    mlflow_tracking_uri: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes a 15-trial Optuna study to optimize classifier hyperparameters.
    Returns best trial hyperparameters and validation accuracy.
    """
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)

    mlflow.set_experiment(study_name)

    pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=1)
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        pruner=pruner
    )

    def _obj_wrapper(t):
        return objective_classifier(t, manifest_dir, images_dir, trial_epochs=trial_epochs)

    with mlflow.start_run(run_name="optuna_classifier_study"):
        study.optimize(_obj_wrapper, n_trials=n_trials)

        best_trial = study.best_trial
        mlflow.log_params(best_trial.params)
        mlflow.log_metric("best_val_accuracy", best_trial.value)

    print("\n" + "=" * 50)
    print("=== Optuna Classifier Study Completed ===")
    print(f"Best Trial #{best_trial.number}")
    print(f"Validation Accuracy: {best_trial.value:.2f}%")
    print("Best Hyperparameters:")
    for k, v in best_trial.params.items():
        print(f"  {k}: {v}")
    print("=" * 50 + "\n")

    return {
        'best_params': best_trial.params,
        'best_val_accuracy': best_trial.value,
        'study': study
    }
