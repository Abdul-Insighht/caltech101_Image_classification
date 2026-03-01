"""
Utility functions for the Caltech-101 classification pipeline.

Includes reproducibility, logging, checkpointing, and visualization helpers.
"""

import os
import json
import random
import logging
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

import config


def set_seed(seed: int = config.RANDOM_SEED) -> None:
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_logging(log_file: str = config.LOG_FILE) -> logging.Logger:
    """Configure logging to both file and console."""
    logger = logging.getLogger("caltech101")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def get_logger() -> logging.Logger:
    """Get the configured logger instance."""
    return logging.getLogger("caltech101")


# =========================================================================
# Checkpointing
# =========================================================================

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    val_acc: float,
    filepath: str = None,
) -> str:
    """Save model checkpoint with training state."""
    if filepath is None:
        filepath = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "val_acc": val_acc,
        "timestamp": datetime.now().isoformat(),
    }
    torch.save(checkpoint, filepath)
    get_logger().info(f"Checkpoint saved: {filepath} (epoch {epoch}, val_acc={val_acc:.4f})")
    return filepath


def load_checkpoint(
    model: torch.nn.Module,
    filepath: str = None,
    optimizer: torch.optim.Optimizer = None,
) -> dict:
    """Load model checkpoint and optionally restore optimizer state."""
    if filepath is None:
        filepath = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No checkpoint found at {filepath}")

    checkpoint = torch.load(filepath, map_location=config.DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    get_logger().info(
        f"Checkpoint loaded: {filepath} (epoch {checkpoint['epoch']}, "
        f"val_acc={checkpoint['val_acc']:.4f})"
    )
    return checkpoint


# =========================================================================
# Visualization
# =========================================================================

def plot_training_history(history: dict, save_path: str = None) -> None:
    """Plot training and validation loss/accuracy curves."""
    if save_path is None:
        save_path = os.path.join(config.PLOTS_DIR, "training_history.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    axes[0].plot(history["train_loss"], label="Train Loss", linewidth=2, color="#2196F3")
    axes[0].plot(history["val_loss"], label="Val Loss", linewidth=2, color="#F44336")
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    axes[1].plot(history["train_acc"], label="Train Accuracy", linewidth=2, color="#4CAF50")
    axes[1].plot(history["val_acc"], label="Val Accuracy", linewidth=2, color="#FF9800")
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Accuracy (%)", fontsize=12)
    axes[1].set_title("Training & Validation Accuracy", fontsize=14, fontweight="bold")
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    get_logger().info(f"Training history plot saved: {save_path}")


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    save_path: str = None,
    top_n: int = 20,
) -> None:
    """Plot confusion matrix heatmap (top-N classes for readability)."""
    if save_path is None:
        save_path = os.path.join(config.PLOTS_DIR, "confusion_matrix.png")

    # If too many classes, show only top-N by sample count
    if len(class_names) > top_n:
        class_totals = cm.sum(axis=1)
        top_indices = np.argsort(class_totals)[-top_n:]
        cm = cm[np.ix_(top_indices, top_indices)]
        class_names = [class_names[i] for i in top_indices]
        title_suffix = f" (Top {top_n} classes)"
    else:
        title_suffix = ""

    fig, ax = plt.subplots(figsize=(max(12, len(class_names) * 0.5), max(10, len(class_names) * 0.4)))
    sns.heatmap(
        cm,
        annot=True if len(class_names) <= 25 else False,
        fmt="d" if len(class_names) <= 25 else "",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix{title_suffix}", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    get_logger().info(f"Confusion matrix saved: {save_path}")


def plot_class_distribution(
    class_counts: dict,
    save_path: str = None,
    title: str = "Class Distribution",
) -> None:
    """Plot bar chart of class sample counts."""
    if save_path is None:
        save_path = os.path.join(config.PLOTS_DIR, "class_distribution.png")

    sorted_items = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    names = [item[0] for item in sorted_items]
    counts = [item[1] for item in sorted_items]

    fig, ax = plt.subplots(figsize=(max(16, len(names) * 0.2), 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
    ax.bar(range(len(names)), counts, color=colors, edgecolor="none")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=6)
    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Number of Samples", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    get_logger().info(f"Class distribution plot saved: {save_path}")


def plot_sample_predictions(
    images: list,
    true_labels: list,
    pred_labels: list,
    class_names: list,
    save_path: str = None,
    n_samples: int = 16,
) -> None:
    """Plot grid of sample images with true vs predicted labels."""
    if save_path is None:
        save_path = os.path.join(config.PLOTS_DIR, "sample_predictions.png")

    n = min(n_samples, len(images))
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.5 * rows))
    axes = axes.flatten() if n > 1 else [axes]

    mean = np.array(config.IMAGENET_MEAN)
    std = np.array(config.IMAGENET_STD)

    for i in range(n):
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        img = std * img + mean  # Denormalize
        img = np.clip(img, 0, 1)

        true_name = class_names[true_labels[i]]
        pred_name = class_names[pred_labels[i]]
        correct = true_labels[i] == pred_labels[i]

        axes[i].imshow(img)
        axes[i].set_title(
            f"True: {true_name}\nPred: {pred_name}",
            fontsize=8,
            color="green" if correct else "red",
            fontweight="bold",
        )
        axes[i].axis("off")

    # Hide empty subplots
    for i in range(n, len(axes)):
        axes[i].axis("off")

    plt.suptitle("Sample Predictions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    get_logger().info(f"Sample predictions plot saved: {save_path}")


def save_metrics(metrics: dict, save_path: str = None) -> None:
    """Save evaluation metrics to JSON file."""
    if save_path is None:
        save_path = os.path.join(config.METRICS_DIR, "evaluation_metrics.json")

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable = json.loads(json.dumps(metrics, default=convert))

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    get_logger().info(f"Metrics saved: {save_path}")


class EarlyStopping:
    """Early stopping to halt training when validation loss stops improving."""

    def __init__(
        self,
        patience: int = config.EARLY_STOPPING_PATIENCE,
        min_delta: float = config.EARLY_STOPPING_MIN_DELTA,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop
