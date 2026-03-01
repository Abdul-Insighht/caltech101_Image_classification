# Caltech-101 Image Classification with ResNet18

A complete deep learning pipeline for classifying images from the [Caltech-101](https://www.kaggle.com/datasets/imbikramsaha/caltech-101) dataset using a fine-tuned ResNet18 model built in PyTorch.

## Project Structure

```
caltech101_classification/
├── config.py          # Centralized configuration & hyperparameters
├── dataset.py         # Dataset loading, splitting, augmentation
├── model.py           # ResNet18 model architecture & fine-tuning
├── train.py           # Training loop with validation & early stopping
├── evaluate.py        # Evaluation metrics & visualization
├── utils.py           # Utilities (logging, plotting, checkpointing)
├── main.py            # Main entry point (CLI)
├── requirements.txt   # Python dependencies
├── README.md          # This file
├── data/              # Dataset directory (Caltech-101)
│   └── caltech-101/
│       └── 101_ObjectCategories/
│           ├── accordion/
│           ├── airplanes/
│           └── ...
└── outputs/           # Generated outputs
    ├── checkpoints/   # Model checkpoints
    ├── plots/         # Training curves, confusion matrix, etc.
    └── metrics/       # JSON evaluation metrics
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

Download the Caltech-101 dataset from Kaggle:
- URL: https://www.kaggle.com/datasets/imbikramsaha/caltech-101
- Extract the `101_ObjectCategories` folder into `data/caltech-101/`

Alternatively, using the Kaggle CLI:
```bash
kaggle datasets download -d imbikramsaha/caltech-101
# Extract to data/caltech-101/
```

## Usage

### Run Full Pipeline (Recommended)

```bash
python main.py --mode full
```

### Individual Modes

```bash
# Analyze dataset only
python main.py --mode analyze

# Train model
python main.py --mode train

# Evaluate saved model
python main.py --mode evaluate
```

### Override Hyperparameters

```bash
python main.py --mode full --epochs 30 --batch-size 64 --lr 0.0005
```

## Model Architecture

- **Base Model**: ResNet18 with ImageNet pretrained weights
- **Modifications**:
  - Dropout (0.5) before final fully connected layer
  - FC layer resized for 101 classes (excluding background)
- **Training Strategy**:
  - **Phase 1** (Epochs 1–5): Backbone frozen, train classification head only
  - **Phase 2** (Epochs 6–25): Full fine-tuning with differential learning rates
- **Data Augmentation**: RandomResizedCrop, HorizontalFlip, Rotation(15°), ColorJitter
- **Optimizer**: Adam with weight decay (1e-4)
- **Scheduler**: StepLR (decay every 7 epochs by 0.1)
- **Early Stopping**: Patience of 7 epochs

## Evaluation Metrics

- Test Accuracy & Top-5 Accuracy
- Precision, Recall, F1-Score (macro & weighted)
- Per-class classification report
- Confusion matrix heatmap
- Sample predictions visualization

## Output Files

After training, the following files are generated in `outputs/`:

| File | Description |
|------|-------------|
| `checkpoints/best_model.pth` | Best model checkpoint |
| `plots/training_history.png` | Loss & accuracy curves |
| `plots/confusion_matrix.png` | Confusion matrix heatmap |
| `plots/sample_predictions.png` | Sample prediction grid |
| `plots/class_distribution.png` | Dataset class distribution |
| `metrics/evaluation_metrics.json` | All evaluation metrics |
| `metrics/dataset_analysis.json` | Dataset statistics |
| `technical_summary.txt` | Human-readable summary |
| `training.log` | Full training log |
"# Image-Classification-through-caltech-101" 
"# Image-Classification-through-caltech-101" 
