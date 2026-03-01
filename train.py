"""
Training module for Caltech-101 image classification.

Implements the training loop with validation, early stopping,
backbone freezing/unfreezing, and history tracking.
"""

import time
from typing import Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

import config
from utils import (
    get_logger,
    save_checkpoint,
    EarlyStopping,
    plot_training_history,
)
from model import (
    freeze_backbone,
    unfreeze_backbone,
    get_optimizer,
    get_scheduler,
    count_parameters,
)


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Train model for one epoch.

    Returns:
        Tuple of (average_loss, accuracy_percentage).
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training", leave=False, ncols=100)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{100.0 * correct / total:.2f}%",
        )

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Validate model on given dataloader.

    Returns:
        Tuple of (average_loss, accuracy_percentage).
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Validating", leave=False, ncols=100)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    num_epochs: int = config.NUM_EPOCHS,
) -> dict:
    """
    Full training pipeline with:
      - Backbone freezing for initial epochs
      - Backbone unfreezing with differential LR
      - Early stopping
      - Best model checkpointing
      - Training history tracking

    Args:
        model: ResNet18 model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        num_epochs: Maximum number of training epochs.

    Returns:
        Training history dictionary.
    """
    logger = get_logger()

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Phase 1: Freeze backbone, train head only
    logger.info("=" * 60)
    logger.info("PHASE 1: Training classification head (backbone frozen)")
    logger.info("=" * 60)
    freeze_backbone(model)

    params = count_parameters(model)
    logger.info(f"Trainable params: {params['trainable_parameters']:,} / {params['total_parameters']:,} ({params['trainable_pct']}%)")

    optimizer = get_optimizer(model, lr=config.LEARNING_RATE)
    scheduler = get_scheduler(optimizer)
    early_stopping = EarlyStopping()

    # History tracking
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    best_val_acc = 0.0
    total_start = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        # Phase 2: Unfreeze backbone after initial epochs
        if epoch == config.FREEZE_BACKBONE_EPOCHS + 1:
            logger.info("=" * 60)
            logger.info("PHASE 2: Fine-tuning entire network (backbone unfrozen)")
            logger.info("=" * 60)
            unfreeze_backbone(model)

            # Recreate optimizer with differential LR
            optimizer = get_optimizer(model, lr=config.LEARNING_RATE * config.UNFREEZE_LR_FACTOR)
            scheduler = get_scheduler(optimizer)

            params = count_parameters(model)
            logger.info(f"Trainable params: {params['trainable_parameters']:,} / {params['total_parameters']:,} ({params['trainable_pct']}%)")

        # Training
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE
        )

        # Validation
        val_loss, val_acc = validate(model, val_loader, criterion, config.DEVICE)

        # Learning rate step
        current_lr = optimizer.param_groups[-1]["lr"]
        scheduler.step()

        # Update history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        epoch_time = time.time() - epoch_start

        logger.info(
            f"Epoch [{epoch:02d}/{num_epochs}] "
            f"| Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% "
            f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% "
            f"| LR: {current_lr:.2e} | Time: {epoch_time:.1f}s"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_loss, val_acc)
            logger.info(f"★ New best model! Val Acc: {val_acc:.2f}%")

        # Early stopping check
        if early_stopping(val_loss):
            logger.info(
                f"Early stopping triggered at epoch {epoch} "
                f"(patience={config.EARLY_STOPPING_PATIENCE})"
            )
            break

    total_time = time.time() - total_start
    logger.info("=" * 60)
    logger.info(f"Training complete in {total_time / 60:.1f} minutes")
    logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")
    logger.info("=" * 60)

    # Plot training history
    plot_training_history(history)

    history["best_val_acc"] = best_val_acc
    history["total_training_time_sec"] = total_time

    return history
