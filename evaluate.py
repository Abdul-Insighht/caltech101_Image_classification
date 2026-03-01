"""
Evaluation module for Caltech-101 image classification.

Computes comprehensive classification metrics, generates reports,
and produces visualization outputs.
"""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    top_k_accuracy_score,
)

import config
from utils import (
    get_logger,
    load_checkpoint,
    plot_confusion_matrix,
    plot_sample_predictions,
    save_metrics,
)


@torch.no_grad()
def get_predictions(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[List, List, List, List]:
    """
    Run inference on entire dataloader and collect predictions.

    Returns:
        Tuple of (all_labels, all_preds, all_probs, sample_images).
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    sample_images = []

    pbar = tqdm(dataloader, desc="Evaluating", leave=False, ncols=100)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

        # Collect sample images for visualization
        if len(sample_images) < 16:
            remaining = 16 - len(sample_images)
            sample_images.extend(images[:remaining].cpu())

    return all_labels, all_preds, all_probs, sample_images


def evaluate_model(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    class_names: list,
    load_best: bool = True,
) -> dict:
    """
    Comprehensive model evaluation on test set.

    Computes:
      - Overall accuracy
      - Precision, recall, F1 (macro & weighted)
      - Top-5 accuracy
      - Per-class classification report
      - Confusion matrix

    Generates:
      - Confusion matrix heatmap
      - Sample predictions visualization
      - Metrics JSON file

    Args:
        model: Trained ResNet18 model.
        test_loader: Test DataLoader.
        class_names: List of class name strings.
        load_best: Whether to load the best checkpoint before evaluation.

    Returns:
        Dictionary of all evaluation metrics.
    """
    logger = get_logger()

    # Load best checkpoint
    if load_best:
        try:
            checkpoint = load_checkpoint(model)
            logger.info(
                f"Loaded best model from epoch {checkpoint['epoch']} "
                f"(val_acc={checkpoint['val_acc']:.2f}%)"
            )
        except FileNotFoundError:
            logger.warning("No checkpoint found. Evaluating current model state.")

    model = model.to(config.DEVICE)

    # Get predictions
    logger.info("Running inference on test set...")
    all_labels, all_preds, all_probs, sample_images = get_predictions(
        model, test_loader, config.DEVICE
    )

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # =====================================================================
    # Compute Metrics
    # =====================================================================
    logger.info("=" * 60)
    logger.info("TEST SET EVALUATION RESULTS")
    logger.info("=" * 60)

    # Overall accuracy
    accuracy = accuracy_score(all_labels, all_preds) * 100

    # Precision, Recall, F1
    precision_macro = precision_score(all_labels, all_preds, average="macro", zero_division=0) * 100
    recall_macro = recall_score(all_labels, all_preds, average="macro", zero_division=0) * 100
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0) * 100

    precision_weighted = precision_score(all_labels, all_preds, average="weighted", zero_division=0) * 100
    recall_weighted = recall_score(all_labels, all_preds, average="weighted", zero_division=0) * 100
    f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0) * 100

    # Top-5 accuracy
    try:
        top5_acc = top_k_accuracy_score(all_labels, all_probs, k=5, labels=range(len(class_names))) * 100
    except Exception:
        top5_acc = 0.0

    metrics = {
        "test_accuracy": round(accuracy, 2),
        "top5_accuracy": round(top5_acc, 2),
        "precision_macro": round(precision_macro, 2),
        "recall_macro": round(recall_macro, 2),
        "f1_macro": round(f1_macro, 2),
        "precision_weighted": round(precision_weighted, 2),
        "recall_weighted": round(recall_weighted, 2),
        "f1_weighted": round(f1_weighted, 2),
        "total_test_samples": len(all_labels),
        "num_classes": len(class_names),
    }

    # Log metrics
    logger.info(f"Test Accuracy       : {accuracy:.2f}%")
    logger.info(f"Top-5 Accuracy      : {top5_acc:.2f}%")
    logger.info(f"Precision (macro)   : {precision_macro:.2f}%")
    logger.info(f"Recall (macro)      : {recall_macro:.2f}%")
    logger.info(f"F1-Score (macro)    : {f1_macro:.2f}%")
    logger.info(f"Precision (weighted): {precision_weighted:.2f}%")
    logger.info(f"Recall (weighted)   : {recall_weighted:.2f}%")
    logger.info(f"F1-Score (weighted) : {f1_weighted:.2f}%")
    logger.info("=" * 60)

    # Per-class classification report
    report_str = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        zero_division=0,
    )
    logger.info("Per-Class Classification Report:\n" + report_str)
    metrics["classification_report"] = report_str

    # =====================================================================
    # Confusion Matrix
    # =====================================================================
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names)

    # =====================================================================
    # Sample Predictions Visualization
    # =====================================================================
    if sample_images:
        n = min(16, len(sample_images))
        plot_sample_predictions(
            images=sample_images[:n],
            true_labels=all_labels[:n].tolist(),
            pred_labels=all_preds[:n].tolist(),
            class_names=class_names,
        )

    # =====================================================================
    # Per-class accuracy analysis
    # =====================================================================
    per_class_acc = {}
    for idx, name in enumerate(class_names):
        mask = all_labels == idx
        if mask.sum() > 0:
            class_acc = (all_preds[mask] == all_labels[mask]).mean() * 100
            per_class_acc[name] = round(float(class_acc), 2)

    # Best and worst performing classes
    sorted_classes = sorted(per_class_acc.items(), key=lambda x: x[1])
    worst_5 = sorted_classes[:5]
    best_5 = sorted_classes[-5:]

    logger.info("Top 5 Best Performing Classes:")
    for name, acc in reversed(best_5):
        logger.info(f"  {name}: {acc:.2f}%")

    logger.info("Top 5 Worst Performing Classes:")
    for name, acc in worst_5:
        logger.info(f"  {name}: {acc:.2f}%")

    metrics["per_class_accuracy"] = per_class_acc
    metrics["best_5_classes"] = {n: a for n, a in best_5}
    metrics["worst_5_classes"] = {n: a for n, a in worst_5}

    # Save metrics to JSON
    save_metrics(metrics)

    logger.info("=" * 60)
    logger.info("Evaluation complete. All metrics and plots saved.")
    logger.info("=" * 60)

    return metrics
