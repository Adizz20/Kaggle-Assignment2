"""Dataset for crash detection from video clips."""
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import torchvision.transforms.v2 as transforms


class CrashVideoDataset(Dataset):
    """
    Dataset for loading video clips as sequences of frames.

    Each clip is a folder containing 30 JPEG frames (frame_000.jpg to frame_029.jpg).
    """
    def __init__(self, clip_ids, labels, root_dir, num_frames, is_train=False, config=None):
        """
        Args:
            clip_ids: List of clip IDs
            labels: List of labels (or None for test set)
            root_dir: Root directory containing clip folders
            num_frames: Number of frames to sample from each clip
            is_train: Whether this is training set (enables augmentation and temporal jitter)
            config: Config object with augmentation parameters
        """
        self.clip_ids = clip_ids
        self.labels = labels
        self.root_dir = Path(root_dir)
        self.num_frames = num_frames
        self.is_train = is_train
        self.config = config

        # Define transforms
        if is_train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    config.height,
                    scale=(config.crop_scale_min, config.crop_scale_max),
                    antialias=True
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=config.color_jitter_brightness,
                    contrast=config.color_jitter_contrast,
                    saturation=config.color_jitter_saturation,
                    hue=config.color_jitter_hue
                ),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(mean=config.mean, std=config.std),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256, antialias=True),
                transforms.CenterCrop(config.height),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(mean=config.mean, std=config.std),
            ])

    def __len__(self):
        return len(self.clip_ids)

    def _sample_frame_indices(self, total_frames=30):
        """
        Sample frame indices uniformly from the clip.

        Args:
            total_frames: Total number of frames in the clip (always 30)

        Returns:
            List of frame indices to load
        """
        # Uniformly sample indices
        indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)

        # Add random temporal jitter during training (±1 frame)
        if self.is_train:
            jitter = np.random.randint(-1, 2, size=len(indices))
            indices = np.clip(indices + jitter, 0, total_frames - 1)

        return indices

    def _load_frames(self, clip_path, frame_indices):
        """
        Load frames from disk.

        Args:
            clip_path: Path to clip folder
            frame_indices: List of frame indices to load

        Returns:
            Tensor of shape (T, C, H, W) with loaded frames
        """
        frames = []
        for idx in frame_indices:
            frame_path = clip_path / f"frame_{idx:03d}.jpg"
            frame = Image.open(frame_path).convert('RGB')
            frames.append(frame)

        return frames

    def __getitem__(self, idx):
        clip_id = self.clip_ids[idx]
        clip_path = self.root_dir / clip_id

        # Sample frame indices
        frame_indices = self._sample_frame_indices()

        # Load frames
        frames = self._load_frames(clip_path, frame_indices)

        # Apply transforms (same transform to all frames for temporal consistency)
        if self.transform:
            # Stack frames into a batch and apply transforms
            frames = torch.stack([transforms.ToImage()(frame) for frame in frames])
            frames = self.transform(frames)

        # Rearrange from (T, C, H, W) to (C, T, H, W) for R2Plus1D
        frames = frames.permute(1, 0, 2, 3)

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return frames, label
        else:
            return frames


def get_train_transforms(config):
    """Get training transforms."""
    return transforms.Compose([
        transforms.RandomResizedCrop(
            config.height,
            scale=(config.crop_scale_min, config.crop_scale_max),
            antialias=True
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=config.color_jitter_brightness,
            contrast=config.color_jitter_contrast,
            saturation=config.color_jitter_saturation,
            hue=config.color_jitter_hue
        ),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=config.mean, std=config.std),
    ])


def get_val_transforms(config):
    """Get validation/test transforms."""
    return transforms.Compose([
        transforms.Resize(256, antialias=True),
        transforms.CenterCrop(config.height),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=config.mean, std=config.std),
    ])
