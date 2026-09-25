# Kaggle Dashcam Crash Detection Pipeline

PyTorch training and inference pipeline for binary crash detection from dashcam video clips.

## Evaluation Metric
**ROC-AUC** (Area Under the Receiver Operating Characteristic Curve)

## Dataset Structure
- **Train clips**: 3,000 clips (1,500 crash, 1,500 non-crash)
- **Test clips**: 1,750 clips
- **Frame structure**: Each clip contains 30 JPEG frames (384×224 pixels)
- **Group ID**: Paired crash/non-crash clips from the same scene (split using GroupKFold)

## Architecture & Strategy
- **Model**: R2Plus1D-18 (ResNet (2+1)D) pretrained on Kinetics-400
- **Cross-Validation**: 5-Fold GroupKFold on `group_id` to prevent scene leakage
- **Sampling**: 16 frames uniformly sampled during training, all 30 frames during validation/inference
- **Augmentation**: RandomResizedCrop, RandomHorizontalFlip, ColorJitter (temporally consistent)
- **Loss**: Binary Cross-Entropy with Label Smoothing (0.05)
- **Optimizer**: AdamW with differential learning rates (1e-4 backbone, 5e-4 head)
- **Scheduler**: CosineAnnealingLR with 3-epoch linear warmup
- **Mixed Precision**: PyTorch AMP fp16 for RTX 4050 16GB VRAM efficiency
- **Test-Time Augmentation (TTA)**: Original + horizontal flip averaging across 5 folds

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Training (5-Fold Cross-Validation)

```bash
python src/train.py
```

This will:
- Create 5-Fold GroupKFold splits
- Train R2Plus1D-18 on each fold with AMP fp16
- Save best checkpoints to `checkpoints/fold_{0-4}_best.pth`
- Save Out-Of-Fold (OOF) predictions to `outputs/oof_predictions.csv`
- Report validation AUC per fold and overall OOF ROC-AUC

### 2. Inference & Submission Generation

```bash
python src/predict.py
```

This will:
- Load all 5 fold checkpoints
- Run inference with Test-Time Augmentation (TTA) on the 1,750 test clips
- Average predictions across all 5 folds
- Save `outputs/submission.csv` in Kaggle-ready format
- Validate prediction constraints (no NaNs, values in [0, 1])

## Project Structure

```
D:\omnirouter\
├── crash_competition_data\        # Local dataset directory
│   ├── train\                     # 3,000 clip folders
│   ├── test\                      # 1,750 clip folders
│   ├── train_labels.csv           # clip_id, label, group_id
│   ├── sample_submission.csv      # clip_id, label
│   └── clip_spec.yaml             # Frame specs
├── src\
│   ├── config.py                  # Paths, hyperparameters, video specs
│   ├── dataset.py                 # CrashVideoDataset, temporal transforms
│   ├── models\
│   │   ├── __init__.py
│   │   └── r2plus1d.py            # R2Plus1D-18 architecture with custom head
│   ├── utils.py                   # GroupKFold, checkpointing, scheduler, seed
│   ├── train.py                   # 5-fold CV training loop with AMP
│   └── predict.py                 # 5-fold ensemble inference with TTA
├── checkpoints\                   # Model weights (best per fold)
├── outputs\                       # oof_predictions.csv, submission.csv
├── requirements.txt
└── README.md
```
