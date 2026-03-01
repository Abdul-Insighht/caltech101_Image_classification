"""
Configuration module for Caltech-101 Image Classification with ResNet18.

Centralizes all hyperparameters, paths, and settings for reproducibility.
"""

import os
import torch

# =============================================================================
# Paths
# =============================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "Caltech101_Dataset", "caltech-101")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")

# Create output directories
for d in [OUTPUT_DIR, CHECKPOINT_DIR, PLOTS_DIR, METRICS_DIR]:
    os.makedirs(d, exist_ok=True)

# =============================================================================
# Dataset Settings
# =============================================================================
IMAGE_SIZE = 224                 # ResNet18 standard input size
NUM_WORKERS = 0                  # Set to 0 on Windows to avoid multiprocessing issues
PIN_MEMORY = True                # Pin memory for faster GPU transfer
EXCLUDE_BACKGROUND = True        # Exclude 'BACKGROUND_Google' class
MIN_SAMPLES_PER_CLASS = 20       # Minimum samples required per class

# =============================================================================
# Data Split Ratios
# =============================================================================
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# =============================================================================
# Training Hyperparameters
# =============================================================================
BATCH_SIZE = 32
NUM_EPOCHS = 25
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9                   # For SGD (if used)
STEP_SIZE = 7                    # StepLR: decay every N epochs
GAMMA = 0.1                      # StepLR: decay factor
SCHEDULER_TYPE = "step"          # Options: "step", "cosine"

# =============================================================================
# Fine-Tuning Strategy
# =============================================================================
FREEZE_BACKBONE_EPOCHS = 5       # Freeze backbone for first N epochs
UNFREEZE_LR_FACTOR = 0.1        # LR multiplier for backbone after unfreezing

# =============================================================================
# Early Stopping
# =============================================================================
EARLY_STOPPING_PATIENCE = 7     # Stop after N epochs without improvement
EARLY_STOPPING_MIN_DELTA = 1e-4 # Minimum improvement threshold

# =============================================================================
# Model Settings
# =============================================================================
MODEL_NAME = "resnet18"
PRETRAINED = True                # Use ImageNet pretrained weights
DROPOUT_RATE = 0.5               # Dropout before final FC layer

# =============================================================================
# Normalization (ImageNet stats)
# =============================================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# =============================================================================
# Device Configuration
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# Reproducibility
# =============================================================================
RANDOM_SEED = 42

# =============================================================================
# Logging
# =============================================================================
LOG_FILE = os.path.join(OUTPUT_DIR, "training.log")
VERBOSE = True
