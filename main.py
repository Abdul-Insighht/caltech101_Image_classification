"""
Main entry point for the Caltech-101 Image Classification Pipeline.

Usage:
    python main.py --mode full       # Run entire pipeline (analyze → train → evaluate)
    python main.py --mode analyze    # Dataset analysis only
    python main.py --mode train      # Train model only
    python main.py --mode evaluate   # Evaluate saved model only
    python main.py --mode train --epochs 10  # Override number of epochs
"""

import argparse
import os
import sys
import json
import time

import config
from utils import set_seed, setup_logging, get_logger, save_metrics, plot_class_distribution
from dataset import (
    Caltech101Dataset,
    get_train_transforms,
    get_eval_transforms,
    stratified_split,
    create_dataloaders,
    analyze_dataset,
)
from model import build_resnet18, count_parameters
from train import train_model
from evaluate import evaluate_model


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Caltech-101 Image Classification with ResNet18",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode full          Run complete pipeline
  python main.py --mode analyze       Analyze dataset only
  python main.py --mode train         Train model
  python main.py --mode evaluate      Evaluate best checkpoint
  python main.py --mode train --epochs 10 --batch-size 64
        """,
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["analyze", "train", "evaluate", "full"],
        help="Pipeline mode (default: full)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--data-dir", type=str, default=None, help="Override dataset directory")
    parser.add_argument("--no-pretrained", action="store_true", help="Train from scratch (no ImageNet weights)")

    return parser.parse_args()


def apply_overrides(args):
    """Apply CLI argument overrides to config."""
    if args.epochs is not None:
        config.NUM_EPOCHS = args.epochs
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    if args.lr is not None:
        config.LEARNING_RATE = args.lr
    if args.data_dir is not None:
        config.DATA_DIR = args.data_dir
    if args.no_pretrained:
        config.PRETRAINED = False


def run_analyze(dataset):
    """Run dataset analysis phase."""
    logger = get_logger()
    logger.info("=" * 60)
    logger.info("PHASE: DATASET ANALYSIS")
    logger.info("=" * 60)

    stats = analyze_dataset(dataset)

    # Plot class distribution
    plot_class_distribution(
        stats["class_counts"],
        title=f"Caltech-101 Class Distribution ({stats['num_classes']} classes, {stats['total_samples']} samples)",
    )

    # Save analysis results
    analysis_path = os.path.join(config.METRICS_DIR, "dataset_analysis.json")
    save_metrics(stats, analysis_path)

    return stats


def run_train(dataset):
    """Run training phase."""
    logger = get_logger()
    logger.info("=" * 60)
    logger.info("PHASE: MODEL TRAINING")
    logger.info("=" * 60)

    # Create data splits
    logger.info("Creating stratified train/val/test splits...")
    train_subset, val_subset, test_subset = stratified_split(dataset)
    logger.info(
        f"Split sizes — Train: {len(train_subset)}, Val: {len(val_subset)}, "
        f"Test: {len(test_subset)}"
    )

    # Create dataloaders with transforms
    train_loader, val_loader, test_loader = create_dataloaders(
        train_subset, val_subset, test_subset,
        train_transform=get_train_transforms(),
        eval_transform=get_eval_transforms(),
    )

    # Build model
    model = build_resnet18(num_classes=dataset.num_classes, pretrained=config.PRETRAINED)
    params = count_parameters(model)
    logger.info(f"Model parameters: {json.dumps(params, indent=2)}")

    # Train
    history = train_model(model, train_loader, val_loader, num_epochs=config.NUM_EPOCHS)

    # Save training history
    history_path = os.path.join(config.METRICS_DIR, "training_history.json")
    history_save = {k: v for k, v in history.items() if k not in ["train_loss", "train_acc", "val_loss", "val_acc", "lr"]}
    history_save.update({
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
        "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
        "final_train_acc": history["train_acc"][-1] if history["train_acc"] else None,
        "final_val_acc": history["val_acc"][-1] if history["val_acc"] else None,
        "epochs_trained": len(history["train_loss"]),
    })
    save_metrics(history_save, history_path)

    return model, dataset, test_loader


def run_evaluate(model, dataset, test_loader=None):
    """Run evaluation phase."""
    logger = get_logger()
    logger.info("=" * 60)
    logger.info("PHASE: MODEL EVALUATION")
    logger.info("=" * 60)

    # If no test_loader, create one
    if test_loader is None:
        _, _, test_subset = stratified_split(dataset)
        _, _, test_loader = create_dataloaders(
            *stratified_split(dataset),
            train_transform=get_train_transforms(),
            eval_transform=get_eval_transforms(),
        )

    # If model not trained (evaluate-only mode), build and load checkpoint
    if model is None:
        model = build_resnet18(num_classes=dataset.num_classes, pretrained=False)

    metrics = evaluate_model(model, test_loader, dataset.class_names, load_best=True)

    return metrics


def print_summary(stats, metrics, total_time):
    """Print final technical summary."""
    logger = get_logger()

    logger.info("\n" + "=" * 60)
    logger.info("TECHNICAL SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Dataset          : Caltech-101")
    logger.info(f"Classes          : {stats.get('num_classes', 'N/A')}")
    logger.info(f"Total samples    : {stats.get('total_samples', 'N/A')}")
    logger.info(f"Model            : ResNet18 (pretrained={'yes' if config.PRETRAINED else 'no'})")
    logger.info(f"Image size       : {config.IMAGE_SIZE}x{config.IMAGE_SIZE}")
    logger.info(f"Batch size       : {config.BATCH_SIZE}")
    logger.info(f"Max epochs       : {config.NUM_EPOCHS}")
    logger.info(f"Learning rate    : {config.LEARNING_RATE}")
    logger.info(f"Optimizer        : Adam (weight_decay={config.WEIGHT_DECAY})")
    logger.info(f"Scheduler        : {config.SCHEDULER_TYPE}")
    logger.info(f"Dropout          : {config.DROPOUT_RATE}")
    logger.info(f"Device           : {config.DEVICE}")
    if metrics:
        logger.info(f"Test Accuracy    : {metrics.get('test_accuracy', 'N/A')}%")
        logger.info(f"Top-5 Accuracy   : {metrics.get('top5_accuracy', 'N/A')}%")
        logger.info(f"F1 (macro)       : {metrics.get('f1_macro', 'N/A')}%")
        logger.info(f"F1 (weighted)    : {metrics.get('f1_weighted', 'N/A')}%")
    logger.info(f"Total time       : {total_time / 60:.1f} minutes")
    logger.info("=" * 60)

    # Save summary to file
    summary_path = os.path.join(config.OUTPUT_DIR, "technical_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("CALTECH-101 IMAGE CLASSIFICATION - TECHNICAL SUMMARY\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Date             : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset          : Caltech-101\n")
        f.write(f"Classes          : {stats.get('num_classes', 'N/A')}\n")
        f.write(f"Total samples    : {stats.get('total_samples', 'N/A')}\n")
        f.write(f"Model            : ResNet18 (pretrained={'yes' if config.PRETRAINED else 'no'})\n")
        f.write(f"Image size       : {config.IMAGE_SIZE}x{config.IMAGE_SIZE}\n")
        f.write(f"Batch size       : {config.BATCH_SIZE}\n")
        f.write(f"Max epochs       : {config.NUM_EPOCHS}\n")
        f.write(f"Learning rate    : {config.LEARNING_RATE}\n")
        f.write(f"Dropout          : {config.DROPOUT_RATE}\n")
        f.write(f"Device           : {config.DEVICE}\n\n")
        f.write("TRAINING STRATEGY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Phase 1: Head-only training ({config.FREEZE_BACKBONE_EPOCHS} epochs, backbone frozen)\n")
        f.write(f"Phase 2: Full fine-tuning (remaining epochs, differential LR)\n")
        f.write(f"Early stopping   : patience={config.EARLY_STOPPING_PATIENCE}\n")
        f.write(f"Data augmentation: RandomResizedCrop, HorizontalFlip, Rotation, ColorJitter\n\n")
        if metrics:
            f.write("EVALUATION RESULTS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Test Accuracy    : {metrics.get('test_accuracy', 'N/A')}%\n")
            f.write(f"Top-5 Accuracy   : {metrics.get('top5_accuracy', 'N/A')}%\n")
            f.write(f"Precision (macro): {metrics.get('precision_macro', 'N/A')}%\n")
            f.write(f"Recall (macro)   : {metrics.get('recall_macro', 'N/A')}%\n")
            f.write(f"F1 (macro)       : {metrics.get('f1_macro', 'N/A')}%\n")
            f.write(f"F1 (weighted)    : {metrics.get('f1_weighted', 'N/A')}%\n\n")
        f.write(f"Total time       : {total_time / 60:.1f} minutes\n")

    logger.info(f"Technical summary saved: {summary_path}")


def main():
    """Main pipeline orchestrator."""
    args = parse_args()
    apply_overrides(args)

    # Setup
    set_seed()
    logger = setup_logging()
    logger.info("Caltech-101 Image Classification with ResNet18")
    logger.info(f"Mode: {args.mode} | Device: {config.DEVICE}")
    logger.info(f"Data directory: {config.DATA_DIR}")

    total_start = time.time()

    # Load dataset (no transform — transforms are applied per-split in DataLoaders)
    logger.info("Loading dataset...")
    dataset = Caltech101Dataset(data_dir=config.DATA_DIR, transform=None)
    logger.info(f"Dataset loaded: {len(dataset)} samples, {dataset.num_classes} classes")

    stats = {}
    metrics = {}
    model = None
    test_loader = None

    if args.mode in ("analyze", "full"):
        stats = run_analyze(dataset)

    if args.mode in ("train", "full"):
        model, dataset, test_loader = run_train(dataset)

    if args.mode in ("evaluate", "full"):
        if args.mode == "evaluate":
            # Need to recreate splits for evaluate-only mode
            _, _, test_subset = stratified_split(dataset)
            _, _, test_loader = create_dataloaders(
                *stratified_split(dataset),
                train_transform=get_train_transforms(),
                eval_transform=get_eval_transforms(),
            )
        metrics = run_evaluate(model, dataset, test_loader)

    total_time = time.time() - total_start

    if args.mode == "full":
        print_summary(stats, metrics, total_time)

    logger.info(f"Pipeline complete. Total time: {total_time / 60:.1f} minutes")
    logger.info(f"Outputs saved to: {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
