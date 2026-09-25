"""Utility functions for training and evaluation."""
import random
import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from pathlib import Path


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_group_kfold_splits(df, n_splits=5, seed=42):
    """
    Create GroupKFold splits to prevent scene leakage.

    Args:
        df: DataFrame with 'group_id' column
        n_splits: Number of folds
        seed: Random seed

    Returns:
        DataFrame with added 'fold' column
    """
    df = df.copy()
    df['fold'] = -1

    gkf = GroupKFold(n_splits=n_splits)
    for fold, (_, val_idx) in enumerate(gkf.split(df, groups=df['group_id'])):
        df.loc[val_idx, 'fold'] = fold

    return df


def save_checkpoint(model, optimizer, scheduler, epoch, best_auc, fold, checkpoint_dir):
    """Save model checkpoint."""
    checkpoint_path = Path(checkpoint_dir) / f"fold_{fold}_best.pth"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_auc': best_auc,
        'fold': fold,
    }, checkpoint_path)
    return checkpoint_path


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint.get('epoch', 0), checkpoint.get('best_auc', 0.0)


class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class WarmupCosineScheduler:
    """Cosine annealing with linear warmup."""
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_epoch = 0

    def step(self):
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            scale = (self.current_epoch + 1) / self.warmup_epochs
            lrs = [base_lr * scale for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lrs = [
                self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
                for base_lr in self.base_lrs
            ]

        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group['lr'] = lr

        self.current_epoch += 1

    def state_dict(self):
        return {'current_epoch': self.current_epoch}

    def load_state_dict(self, state_dict):
        self.current_epoch = state_dict['current_epoch']
