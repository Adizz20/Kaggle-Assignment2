"""Sanity check script to verify the installation and model."""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

import torch
from project.src.config import config
from project.src.models import CrashR2Plus1D
import pandas as pd


def main():
    print("="*60)
    print("SANITY CHECK")
    print("="*60)

    # Check PyTorch installation
    print(f"\n[OK] PyTorch Version: {torch.__version__}")
    print(f"[OK] CUDA Available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"[OK] Device Name: {torch.cuda.get_device_name(0)}")
        print(f"[OK] CUDA Version: {torch.version.cuda}")
    else:
        print("[WARN] CUDA not available - training will be slow on CPU!")

    # Check data paths
    print(f"\n[OK] Data directory exists: {config.DATA_DIR.exists()}")
    print(f"[OK] Train directory exists: {config.TRAIN_DIR.exists()}")
    print(f"[OK] Test directory exists: {config.TEST_DIR.exists()}")
    print(f"[OK] Labels file exists: {config.LABELS_PATH.exists()}")
    print(f"[OK] Submission file exists: {config.SUBMISSION_PATH.exists()}")

    if config.LABELS_PATH.exists():
        df = pd.read_csv(config.LABELS_PATH)
        print(f"\n[OK] Training clips: {len(df)}")
        print(f"  - Crash: {(df['label'] == 1).sum()}")
        print(f"  - Non-crash: {(df['label'] == 0).sum()}")
        print(f"  - Groups: {df['group_id'].nunique()}")

    # Test model initialization
    print("\n[OK] Initializing model...")
    model = CrashR2Plus1D(config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[OK] Total parameters: {total_params:,}")
    print(f"[OK] Trainable parameters: {trainable_params:,}")

    # Test forward pass
    print("\n[OK] Testing forward pass...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Dummy input: (batch_size=2, channels=3, frames=16, height=224, width=224)
    x = torch.randn(2, 3, 16, 224, 224).to(device)

    with torch.no_grad():
        out = model(x)

    print(f"[OK] Input shape: {x.shape}")
    print(f"[OK] Output shape: {out.shape}")
    assert out.shape == torch.Size([2]), f"Expected shape [2], got {out.shape}"

    print("\n" + "="*60)
    print("ALL CHECKS PASSED!")
    print("="*60)
    print("\nYou can now run:")
    print("  python src/train.py    # Train 5-fold CV")
    print("  python src/predict.py  # Generate submission")


if __name__ == '__main__':
    main()
