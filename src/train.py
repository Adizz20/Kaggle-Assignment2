"""Training script for crash detection model."""
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# Add src to path
sys.path.append(str(Path(__file__).parent))

from project.src.config import config
from project.src.dataset import CrashVideoDataset
from project.src.models import CrashR2Plus1D
from project.src.utils import (
    set_seed, get_group_kfold_splits, save_checkpoint,
    AverageMeter, WarmupCosineScheduler
)


class LabelSmoothingBCEWithLogitsLoss(nn.Module):
    """BCE loss with label smoothing."""
    def __init__(self, smoothing=0.0):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        if self.smoothing > 0:
            targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(logits, targets)


def train_epoch(model, loader, criterion, optimizer, scaler, device, config):
    """Train for one epoch."""
    model.train()
    loss_meter = AverageMeter()

    pbar = tqdm(loader, desc='Training')
    optimizer.zero_grad()

    for batch_idx, (frames, labels) in enumerate(pbar):
        frames = frames.to(device)
        labels = labels.to(device)

        # Forward pass with AMP
        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits = model(frames)
            loss = criterion(logits, labels)
            loss = loss / config.accum_steps

        # Backward pass
        scaler.scale(loss).backward()

        # Update weights every accum_steps
        if (batch_idx + 1) % config.accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_max_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        loss_meter.update(loss.item() * config.accum_steps, frames.size(0))
        pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})

    return loss_meter.avg


@torch.no_grad()
def validate(model, loader, device):
    """Validate and return predictions and labels."""
    model.eval()
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc='Validating')
    for frames, labels in pbar:
        frames = frames.to(device)

        # Forward pass
        logits = model(frames)
        probs = torch.sigmoid(logits)

        all_preds.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    auc = roc_auc_score(all_labels, all_preds)
    return auc, all_preds, all_labels


def train_fold(fold, train_df, val_df, config, device):
    """Train a single fold."""
    print(f"\n{'='*60}")
    print(f"Training Fold {fold}")
    print(f"{'='*60}")
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")

    # Create datasets
    train_dataset = CrashVideoDataset(
        clip_ids=train_df['clip_id'].values,
        labels=train_df['label'].values,
        root_dir=config.TRAIN_DIR,
        num_frames=config.num_frames_train,
        is_train=True,
        config=config
    )

    val_dataset = CrashVideoDataset(
        clip_ids=val_df['clip_id'].values,
        labels=val_df['label'].values,
        root_dir=config.TRAIN_DIR,
        num_frames=config.num_frames_val,
        is_train=False,
        config=config
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers
    )

    # Create model
    model = CrashR2Plus1D(config).to(device)

    # Optimizer with differential learning rates
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if 'backbone.fc' in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': config.lr_backbone},
        {'params': head_params, 'lr': config.lr_head}
    ], weight_decay=config.weight_decay)

    # Scheduler with warmup
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=3,
        total_epochs=config.epochs
    )

    # Loss and scaler
    criterion = LabelSmoothingBCEWithLogitsLoss(smoothing=config.label_smoothing)
    scaler = torch.amp.GradScaler('cuda')

    # Training loop
    best_auc = 0.0
    patience_counter = 0

    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch + 1}/{config.epochs}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.2e} (backbone), {optimizer.param_groups[1]['lr']:.2e} (head)")

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device, config)

        # Validate
        val_auc, val_preds, val_labels = validate(model, val_loader, device)

        print(f"Train Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f}")

        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            checkpoint_path = save_checkpoint(
                model, optimizer, scheduler, epoch, best_auc, fold, config.CHECKPOINT_DIR
            )
            print(f"[SAVED] Checkpoint to {checkpoint_path} (AUC: {best_auc:.4f})")
        else:
            patience_counter += 1
            print(f"[INFO] No improvement ({patience_counter}/{config.patience})")

        # Early stopping
        if patience_counter >= config.patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break

        # Step scheduler
        scheduler.step()

    print(f"\nFold {fold} Best AUC: {best_auc:.4f}")

    # Return validation predictions for OOF
    return val_df['clip_id'].values, val_preds, val_labels, best_auc


def main():
    """Main training function."""
    # Setup
    set_seed(config.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directories
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading training labels...")
    df = pd.read_csv(config.LABELS_PATH)
    print(f"Total clips: {len(df)}")
    print(f"Crash: {(df['label'] == 1).sum()}, Non-crash: {(df['label'] == 0).sum()}")
    print(f"Unique groups: {df['group_id'].nunique()}")

    # Create folds
    print("\nCreating GroupKFold splits...")
    df = get_group_kfold_splits(df, n_splits=config.n_folds, seed=config.seed)

    # Track OOF predictions
    oof_predictions = []

    # Train each fold
    fold_aucs = []
    for fold in range(config.n_folds):
        train_df = df[df['fold'] != fold].reset_index(drop=True)
        val_df = df[df['fold'] == fold].reset_index(drop=True)

        clip_ids, preds, labels, best_auc = train_fold(fold, train_df, val_df, config, device)
        fold_aucs.append(best_auc)

        # Store OOF predictions
        for clip_id, pred, label in zip(clip_ids, preds, labels):
            oof_predictions.append({
                'clip_id': clip_id,
                'fold': fold,
                'prediction': pred,
                'label': label
            })

    # Save OOF predictions
    oof_df = pd.DataFrame(oof_predictions)
    oof_path = config.OUTPUT_DIR / 'oof_predictions.csv'
    oof_df.to_csv(oof_path, index=False)
    print(f"\n[SAVED] OOF predictions to {oof_path}")

    # Calculate overall OOF AUC
    overall_auc = roc_auc_score(oof_df['label'], oof_df['prediction'])

    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    for fold, auc in enumerate(fold_aucs):
        print(f"Fold {fold}: {auc:.4f}")
    print(f"\nMean CV AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
    print(f"Overall OOF AUC: {overall_auc:.4f}")
    print("="*60)


if __name__ == '__main__':
    main()
