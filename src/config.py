"""Configuration for crash detection pipeline."""

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass
class Config:

    DATA_DIR: Path = Path(
        os.environ.get(
            "CRASH_DATA_DIR",
            "D:/omnirouter/crash_competition_data"
        )
    )

    TRAIN_DIR: Path = DATA_DIR / "train"
    TEST_DIR: Path = DATA_DIR / "test"
    LABELS_PATH: Path = DATA_DIR / "train_labels.csv"
    SUBMISSION_PATH: Path = DATA_DIR / "sample_submission.csv"

    CHECKPOINT_DIR: Path = Path(
        os.environ.get(
            "CRASH_CHECKPOINT_DIR",
            "checkpoints"
        )
    )

    OUTPUT_DIR: Path = Path(
        os.environ.get(
            "CRASH_OUTPUT_DIR",
            "outputs"
        )
    )


    num_frames_train: int = 16
    num_frames_val: int = 30

    height: int = 224
    width: int = 224

    batch_size: int = 8
    accum_steps: int = 2

    lr_backbone: float = 1e-4
    lr_head: float = 5e-4

    weight_decay: float = 1e-2

    epochs: int = 25
    patience: int = 7

    n_folds: int = 5
    seed: int = 42


    label_smoothing: float = 0.05

    crop_scale_min: float = 0.85
    crop_scale_max: float = 1.0

    color_jitter_brightness: float = 0.2
    color_jitter_contrast: float = 0.2
    color_jitter_saturation: float = 0.2
    color_jitter_hue: float = 0.05


    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True

    grad_clip_max_norm: float = 1.0


    dropout_1: float = 0.3
    dropout_2: float = 0.15

    hidden_dim: int = 128


    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)


config = Config()