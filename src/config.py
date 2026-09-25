"""Configuration for crash detection pipeline."""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # Paths
    DATA_DIR: Path = Path("D:/omnirouter/crash_competition_data")
    TRAIN_DIR: Path = DATA_DIR / "train"
    TEST_DIR: Path = DATA_DIR / "test"
    LABELS_PATH: Path = DATA_DIR / "train_labels.csv"
    SUBMISSION_PATH: Path = DATA_DIR / "sample_submission.csv"
    CHECKPOINT_DIR: Path = Path("D:/omnirouter/checkpoints")
    OUTPUT_DIR: Path = Path("D:/omnirouter/outputs")

    # Video parameters
    num_frames_train: int = 16  # Sample fewer frames during training for speed
    num_frames_val: int = 30    # Use all frames during validation/inference
    height: int = 224           # Square input for R2Plus1D (Kinetics pretraining)
    width: int = 224

    # Training hyperparameters
    batch_size: int = 8
    accum_steps: int = 2        # Effective batch size = 8 * 2 = 16
    lr_backbone: float = 1e-4   # Lower LR for pretrained backbone
    lr_head: float = 5e-4       # Higher LR for classifier head
    weight_decay: float = 1e-2
    epochs: int = 25
    patience: int = 7           # Early stopping patience
    n_folds: int = 5
    seed: int = 42

    # Data augmentation
    label_smoothing: float = 0.05
    crop_scale_min: float = 0.85
    crop_scale_max: float = 1.0
    color_jitter_brightness: float = 0.2
    color_jitter_contrast: float = 0.2
    color_jitter_saturation: float = 0.2
    color_jitter_hue: float = 0.05

    # Training settings
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    grad_clip_max_norm: float = 1.0

    # Model settings
    dropout_1: float = 0.3
    dropout_2: float = 0.15
    hidden_dim: int = 128

    # ImageNet normalization (used by Kinetics pretraining)
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)


config = Config()
