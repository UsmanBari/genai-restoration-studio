"""
Training and evaluation routines for Corruption Classifier (Task 2).
Features:
  - 4-way CrossEntropyLoss training with mixed precision
  - Comprehensive metrics: Top-1 accuracy, per-class recall, precision, macro F1, and confusion matrix
  - MLflow logging of epoch metrics, best checkpointing, and confusion matrix artifacts
"""

import os
import time
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import mlflow

from data.corruptions import CORRUPTION_NAMES


def train_one_epoch_classifier(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: str
) -> Dict[str, float]:
    """Runs one training epoch for the 4-way corruption classifier."""
    model.train()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    num_batches = len(dataloader)

    for batch in dataloader:
        corrupted = batch['corrupted'].to(device)
        labels = batch['label'].to(device)
        bs = corrupted.size(0)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None and device.startswith('cuda'):
            with torch.cuda.amp.autocast():
                logits = model(corrupted)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(corrupted)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * bs
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total_samples += bs

    return {
        'loss': total_loss / max(1, total_samples),
        'accuracy': (correct / max(1, total_samples)) * 100.0
    }


def evaluate_classifier(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str
) -> Dict[str, Any]:
    """
    Evaluates classifier on validation/test set.
    Returns loss, overall accuracy, per-class recall, precision, macro F1, and 4x4 confusion matrix.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    num_classes = 4
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for batch in dataloader:
            corrupted = batch['corrupted'].to(device)
            labels = batch['label'].to(device)
            bs = corrupted.size(0)

            logits = model(corrupted)
            loss = criterion(logits, labels)

            total_loss += loss.item() * bs
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total_samples += bs

            preds_np = preds.cpu().numpy()
            labels_np = labels.cpu().numpy()
            for p, y in zip(preds_np, labels_np):
                if 0 <= y < num_classes and 0 <= p < num_classes:
                    conf_matrix[y, p] += 1

    overall_acc = (correct / max(1, total_samples)) * 100.0
    mean_loss = total_loss / max(1, total_samples)

    # Per-class metrics
    per_class_recalls = {}
    per_class_precisions = {}
    f1_scores = []
    for c in range(num_classes):
        c_name = CORRUPTION_NAMES.get(c, f"class_{c}")
        tp = conf_matrix[c, c]
        fn = conf_matrix[c, :].sum() - tp
        fp = conf_matrix[:, c].sum() - tp
        
        recall = tp / max(1, tp + fn)
        precision = tp / max(1, tp + fp)
        f1 = (2 * precision * recall) / max(1e-8, precision + recall)

        per_class_recalls[c_name] = recall * 100.0
        per_class_precisions[c_name] = precision * 100.0
        f1_scores.append(f1)

    macro_f1 = float(np.mean(f1_scores))

    return {
        'loss': mean_loss,
        'accuracy': overall_acc,
        'macro_f1': macro_f1,
        'confusion_matrix': conf_matrix.tolist(),
        'per_class_recall': per_class_recalls,
        'per_class_precision': per_class_precisions
    }


def train_classifier_full(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 15,
    lr: float = 1e-3,
    checkpoint_dir: str = "checkpoints/classifier",
    experiment_name: str = "Task2_Classifier_Training",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, Any]:
    """
    Full training loop for corruption classifier with MLflow logging, LR scheduler,
    and automatic best-model checkpoint saving.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler() if device.startswith('cuda') else None

    mlflow.set_experiment(experiment_name)
    best_val_acc = 0.0
    best_checkpoint_path = os.path.join(checkpoint_dir, "best_classifier.pth")
    history = []

    print(f"=== Starting Classifier Training ({epochs} epochs, device: {device}) ===")

    with mlflow.start_run(run_name="classifier_run"):
        mlflow.log_params({
            "epochs": epochs,
            "lr": lr,
            "base_channels": getattr(model, "base_channels", 32),
            "dropout_rate": getattr(model, "dropout_rate", 0.2),
            "device": device
        })

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_metrics = train_one_epoch_classifier(model, train_loader, optimizer, criterion, scaler, device)
            val_metrics = evaluate_classifier(model, val_loader, criterion, device)
            scheduler.step()
            elapsed = time.time() - t0

            # Log metrics
            mlflow.log_metric("train_loss", train_metrics['loss'], step=epoch)
            mlflow.log_metric("train_acc", train_metrics['accuracy'], step=epoch)
            mlflow.log_metric("val_loss", val_metrics['loss'], step=epoch)
            mlflow.log_metric("val_acc", val_metrics['accuracy'], step=epoch)
            mlflow.log_metric("val_macro_f1", val_metrics['macro_f1'], step=epoch)
            mlflow.log_metric("lr", optimizer.param_groups[0]['lr'], step=epoch)

            # Checkpoint on best validation accuracy
            is_best = val_metrics['accuracy'] > best_val_acc
            if is_best:
                best_val_acc = val_metrics['accuracy']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_accuracy': best_val_acc,
                    'val_macro_f1': val_metrics['macro_f1'],
                    'confusion_matrix': val_metrics['confusion_matrix'],
                    'base_channels': getattr(model, "base_channels", 32),
                    'dropout_rate': getattr(model, "dropout_rate", 0.2)
                }, best_checkpoint_path)

            status = f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.1f}% | Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.1f}% F1: {val_metrics['macro_f1']:.3f}"
            if is_best:
                status += f" -> [BEST SAVED ({best_val_acc:.1f}%)]"
            print(status)

            history.append({
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'train_acc': train_metrics['accuracy'],
                'val_loss': val_metrics['loss'],
                'val_acc': val_metrics['accuracy'],
                'val_macro_f1': val_metrics['macro_f1'],
                'is_best': is_best
            })

    print(f"\nClassifier training complete. Best Validation Accuracy: {best_val_acc:.2f}%")
    return {
        'best_val_accuracy': best_val_acc,
        'best_checkpoint_path': best_checkpoint_path,
        'history': history
    }
