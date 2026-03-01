"""
Model module for Caltech-101 image classification.

Builds a ResNet18 model with pretrained ImageNet weights, modified for
the Caltech-101 classification task with optional layer freezing.
"""

import torch
import torch.nn as nn
from torchvision import models

import config
from utils import get_logger


def build_resnet18(num_classes: int, pretrained: bool = config.PRETRAINED) -> nn.Module:
    """
    Build a ResNet18 model for image classification.

    Architecture modifications from standard ResNet18:
      1. Load pretrained ImageNet weights for transfer learning
      2. Add dropout layer before final FC for regularization
      3. Replace final FC layer to match number of target classes

    Args:
        num_classes: Number of output classes.
        pretrained: Whether to use ImageNet pretrained weights.

    Returns:
        Modified ResNet18 model.
    """
    logger = get_logger()

    # Load pretrained ResNet18
    if pretrained:
        logger.info("Loading ResNet18 with pretrained ImageNet weights")
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
    else:
        logger.info("Loading ResNet18 without pretrained weights (random init)")
        model = models.resnet18(weights=None)

    # Get the number of features in the final FC layer
    in_features = model.fc.in_features

    # Replace final FC layer with dropout + new FC
    model.fc = nn.Sequential(
        nn.Dropout(p=config.DROPOUT_RATE),
        nn.Linear(in_features, num_classes),
    )

    logger.info(
        f"Model built: ResNet18 → {num_classes} classes "
        f"(in_features={in_features}, dropout={config.DROPOUT_RATE})"
    )

    # Move to device
    model = model.to(config.DEVICE)
    logger.info(f"Model moved to device: {config.DEVICE}")

    return model


def freeze_backbone(model: nn.Module) -> None:
    """
    Freeze all layers except the final FC head.

    This is used during the initial training phase to only update the
    classification head while keeping pretrained features intact.
    """
    logger = get_logger()
    frozen_count = 0

    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False
            frozen_count += 1

    logger.info(f"Backbone frozen: {frozen_count} parameter groups frozen")


def unfreeze_backbone(model: nn.Module) -> None:
    """
    Unfreeze all layers for full fine-tuning.

    Called after initial head-only training to allow end-to-end updates.
    """
    logger = get_logger()
    unfrozen_count = 0

    for param in model.parameters():
        param.requires_grad = True
        unfrozen_count += 1

    logger.info(f"Backbone unfrozen: {unfrozen_count} parameter groups trainable")


def get_optimizer(model: nn.Module, lr: float = config.LEARNING_RATE) -> torch.optim.Optimizer:
    """
    Create Adam optimizer with weight decay.

    Uses differential learning rates when backbone is unfrozen:
      - Backbone params: lower learning rate
      - FC head params: standard learning rate
    """
    # Separate backbone and head parameters
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "fc" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    if backbone_params:
        # Differential LR: backbone gets lower LR
        param_groups = [
            {"params": backbone_params, "lr": lr * config.UNFREEZE_LR_FACTOR},
            {"params": head_params, "lr": lr},
        ]
    else:
        param_groups = [{"params": head_params, "lr": lr}]

    optimizer = torch.optim.Adam(param_groups, weight_decay=config.WEIGHT_DECAY)
    return optimizer


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = config.SCHEDULER_TYPE,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Create learning rate scheduler."""
    if scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config.STEP_SIZE, gamma=config.GAMMA
        )
    elif scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    get_logger().info(f"Scheduler: {scheduler_type}")
    return scheduler


def count_parameters(model: nn.Module) -> dict:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": total - trainable,
        "trainable_pct": round(100 * trainable / total, 2) if total > 0 else 0,
    }
