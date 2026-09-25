"""Inference script for crash detection model."""
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent))

from project.src.config import config
from project.src.dataset import CrashVideoDataset
from project.src.models import CrashR2Plus1D
from project.src.utils import set_seed, load_checkpoint


@torch.no_grad()
def predict_with_tta(model, loader, device):
    """
    Run inference with Test-Time Augmentation.

    TTA strategy: original + horizontal flip, then average predictions.
    """
    model.eval()
    all_preds = []

    # Original predictions
    print("Running inference (original)...")
    for frames in tqdm(loader):
        frames = frames.to(device)
        logits = model(frames)
        probs = torch.sigmoid(logits)
        all_preds.append(probs.cpu().numpy())

    preds_original = np.concatenate(all_preds)

    # Horizontal flip predictions
    print("Running inference (horizontal flip)...")
    all_preds = []
    for frames in tqdm(loader):
        frames = frames.to(device)
        # Flip horizontally (flip width dimension)
        frames_flipped = torch.flip(frames, dims=[4])  # (B, C, T, H, W) -> flip W
        logits = model(frames_flipped)
        probs = torch.sigmoid(logits)
        all_preds.append(probs.cpu().numpy())

    preds_flipped = np.concatenate(all_preds)

    # Average predictions
    preds_tta = (preds_original + preds_flipped) / 2.0

    return preds_tta


def predict_fold(fold, test_dataset, config, device):
    """Run inference for a single fold checkpoint."""
    print(f"\n{'='*60}")
    print(f"Predicting with Fold {fold} checkpoint")
    print(f"{'='*60}")

    # Load model
    checkpoint_path = config.CHECKPOINT_DIR / f"fold_{fold}_best.pth"
    if not checkpoint_path.exists():
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        return None

    model = CrashR2Plus1D(config).to(device)
    epoch, best_auc = load_checkpoint(checkpoint_path, model)
    print(f"Loaded checkpoint from epoch {epoch} (Best AUC: {best_auc:.4f})")

    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers
    )

    # Predict with TTA
    predictions = predict_with_tta(model, test_loader, device)

    return predictions


def main():
    """Main inference function."""
    # Setup
    set_seed(config.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load test data
    print("\nLoading test data...")
    submission_df = pd.read_csv(config.SUBMISSION_PATH)
    test_clip_ids = submission_df['clip_id'].values
    print(f"Test clips: {len(test_clip_ids)}")

    # Create test dataset
    test_dataset = CrashVideoDataset(
        clip_ids=test_clip_ids,
        labels=None,
        root_dir=config.TEST_DIR,
        num_frames=config.num_frames_val,  # Use all 30 frames
        is_train=False,
        config=config
    )

    # Predict with each fold
    all_fold_preds = []
    for fold in range(config.n_folds):
        fold_preds = predict_fold(fold, test_dataset, config, device)
        if fold_preds is not None:
            all_fold_preds.append(fold_preds)

    if len(all_fold_preds) == 0:
        print("\n[ERROR] No predictions generated. Please train the model first.")
        return

    # Average predictions across folds
    print(f"\nAveraging predictions from {len(all_fold_preds)} folds...")
    final_predictions = np.mean(all_fold_preds, axis=0)

    # Create submission file
    submission_df['label'] = final_predictions

    # Validate submission
    print("\nValidating submission...")
    assert len(submission_df) == 1750, f"Expected 1750 rows, got {len(submission_df)}"
    assert (submission_df['label'] >= 0).all() and (submission_df['label'] <= 1).all(), \
        "Predictions must be in [0, 1]"
    assert not submission_df['label'].isna().any(), "Submission contains NaN values"

    # Save submission
    submission_path = config.OUTPUT_DIR / 'submission.csv'
    submission_df.to_csv(submission_path, index=False)

    print(f"\n[SUCCESS] Submission saved to {submission_path}")
    print(f"\nSubmission statistics:")
    print(f"  Mean: {submission_df['label'].mean():.4f}")
    print(f"  Std:  {submission_df['label'].std():.4f}")
    print(f"  Min:  {submission_df['label'].min():.4f}")
    print(f"  Max:  {submission_df['label'].max():.4f}")
    print(f"\nTop 5 most confident crash predictions:")
    print(submission_df.nlargest(5, 'label')[['clip_id', 'label']])
    print(f"\nTop 5 most confident non-crash predictions:")
    print(submission_df.nsmallest(5, 'label')[['clip_id', 'label']])


if __name__ == '__main__':
    main()
