"""
Dataset module for Caltech-101 image classification.

Handles dataset analysis, stratified splitting, preprocessing,
data augmentation, and DataLoader creation.
"""

import os
import collections
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split

import config
from utils import get_logger, set_seed


# =========================================================================
# Custom Dataset
# =========================================================================

class Caltech101Dataset(Dataset):
    """
    Custom PyTorch Dataset for Caltech-101.

    Expects the standard folder structure:
        data_dir/
            class_name_1/
                image_0001.jpg
                image_0002.jpg
                ...
            class_name_2/
                ...
    """

    def __init__(
        self,
        data_dir: str,
        transform: transforms.Compose = None,
        exclude_background: bool = config.EXCLUDE_BACKGROUND,
        min_samples: int = config.MIN_SAMPLES_PER_CLASS,
    ):
        self.data_dir = data_dir
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []   # (image_path, label_index)
        self.class_names: List[str] = []
        self.class_to_idx: Dict[str, int] = {}

        self._load_dataset(exclude_background, min_samples)

    def _load_dataset(self, exclude_background: bool, min_samples: int) -> None:
        """Scan directory and build (path, label) list."""
        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(
                f"Dataset directory not found: {self.data_dir}\n"
                f"Please download the Caltech-101 dataset and extract it to:\n"
                f"  {config.DATA_DIR}"
            )

        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}

        # Collect class directories
        all_classes = sorted([
            d for d in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, d))
        ])

        # Filter out background class if requested
        if exclude_background:
            all_classes = [c for c in all_classes if c.upper() != "BACKGROUND_GOOGLE"]

        # Build dataset
        idx = 0
        for class_name in all_classes:
            class_dir = os.path.join(self.data_dir, class_name)
            images = [
                f for f in os.listdir(class_dir)
                if os.path.splitext(f)[1].lower() in valid_extensions
            ]

            if len(images) < min_samples:
                continue

            self.class_names.append(class_name)
            self.class_to_idx[class_name] = idx

            for img_name in images:
                img_path = os.path.join(class_dir, img_name)
                self.samples.append((img_path, idx))

            idx += 1

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        img_path, label = self.samples[index]

        # Load image and convert to RGB
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback: return a blank image if loading fails
            get_logger().warning(f"Failed to load {img_path}: {e}")
            image = Image.new("RGB", (config.IMAGE_SIZE, config.IMAGE_SIZE), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, label

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


# =========================================================================
# Transforms
# =========================================================================

def get_train_transforms() -> transforms.Compose:
    """Training transforms with data augmentation."""
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE + 32, config.IMAGE_SIZE + 32)),
        transforms.RandomResizedCrop(config.IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])


def get_eval_transforms() -> transforms.Compose:
    """Validation/test transforms (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])


# =========================================================================
# Dataset Splitting
# =========================================================================

def stratified_split(
    dataset: Caltech101Dataset,
) -> Tuple[Subset, Subset, Subset]:
    """
    Perform stratified train/val/test split preserving class distribution.

    Returns:
        Tuple of (train_subset, val_subset, test_subset)
    """
    set_seed()

    labels = [label for _, label in dataset.samples]
    indices = list(range(len(dataset)))

    # First split: train vs (val + test)
    train_indices, temp_indices, train_labels, temp_labels = train_test_split(
        indices,
        labels,
        test_size=(config.VAL_RATIO + config.TEST_RATIO),
        random_state=config.RANDOM_SEED,
        stratify=labels,
    )

    # Second split: val vs test
    relative_test_ratio = config.TEST_RATIO / (config.VAL_RATIO + config.TEST_RATIO)
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=relative_test_ratio,
        random_state=config.RANDOM_SEED,
        stratify=temp_labels,
    )

    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
        Subset(dataset, test_indices),
    )


# =========================================================================
# Transformed Subset (module-level for Windows multiprocessing pickling)
# =========================================================================

class TransformedSubset(Dataset):
    """Wrapper to apply a specific transform to a Subset."""

    def __init__(self, subset: Subset, transform: transforms.Compose):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):
        img_path, label = self.subset.dataset.samples[self.subset.indices[index]]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (config.IMAGE_SIZE, config.IMAGE_SIZE), (0, 0, 0))

        if self.transform:
            image = self.transform(image)
        return image, label


# =========================================================================
# DataLoader Creation
# =========================================================================

def create_dataloaders(
    train_subset: Subset,
    val_subset: Subset,
    test_subset: Subset,
    train_transform: transforms.Compose,
    eval_transform: transforms.Compose,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders with appropriate transforms for each split.

    Since Subsets share the parent dataset's transform, we use wrapper
    datasets to apply different transforms per split.
    """
    train_ds = TransformedSubset(train_subset, train_transform)
    val_ds = TransformedSubset(val_subset, eval_transform)
    test_ds = TransformedSubset(test_subset, eval_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    return train_loader, val_loader, test_loader


# =========================================================================
# Dataset Analysis
# =========================================================================

def analyze_dataset(dataset: Caltech101Dataset) -> dict:
    """
    Analyze dataset to produce distribution statistics and quality report.

    Returns:
        Dictionary of analysis results.
    """
    logger = get_logger()

    labels = [label for _, label in dataset.samples]
    counter = collections.Counter(labels)

    class_counts = {
        dataset.class_names[idx]: count
        for idx, count in sorted(counter.items())
    }

    counts = list(counter.values())
    stats = {
        "total_samples": len(dataset),
        "num_classes": dataset.num_classes,
        "class_counts": class_counts,
        "min_samples_per_class": min(counts),
        "max_samples_per_class": max(counts),
        "mean_samples_per_class": round(np.mean(counts), 1),
        "median_samples_per_class": round(np.median(counts), 1),
        "std_samples_per_class": round(np.std(counts), 1),
    }

    # Identify imbalanced classes
    threshold_low = stats["mean_samples_per_class"] * 0.3
    threshold_high = stats["mean_samples_per_class"] * 3.0
    underrepresented = [n for n, c in class_counts.items() if c < threshold_low]
    overrepresented = [n for n, c in class_counts.items() if c > threshold_high]
    stats["underrepresented_classes"] = underrepresented
    stats["overrepresented_classes"] = overrepresented

    # Log summary
    logger.info("=" * 60)
    logger.info("DATASET ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"Total samples     : {stats['total_samples']}")
    logger.info(f"Number of classes  : {stats['num_classes']}")
    logger.info(f"Min samples/class  : {stats['min_samples_per_class']}")
    logger.info(f"Max samples/class  : {stats['max_samples_per_class']}")
    logger.info(f"Mean samples/class : {stats['mean_samples_per_class']}")
    logger.info(f"Median             : {stats['median_samples_per_class']}")
    logger.info(f"Std dev            : {stats['std_samples_per_class']}")
    if underrepresented:
        logger.info(f"Underrepresented   : {underrepresented}")
    if overrepresented:
        logger.info(f"Overrepresented    : {overrepresented}")
    logger.info("=" * 60)

    return stats
