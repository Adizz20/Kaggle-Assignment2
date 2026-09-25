"""R2Plus1D model for crash detection."""
import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights


class CrashR2Plus1D(nn.Module):
    """
    R2Plus1D-18 model for binary crash classification.

    Uses pretrained weights from Kinetics-400 and replaces the classification head.
    """
    def __init__(self, config):
        super().__init__()

        # Load pretrained R2Plus1D-18
        self.backbone = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)

        # Get the input dimension of the original fc layer
        in_features = self.backbone.fc.in_features  # 512 for R2Plus1D-18

        # Replace fc layer with custom head
        self.backbone.fc = nn.Sequential(
            nn.Dropout(config.dropout_1),
            nn.Linear(in_features, config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout_2),
            nn.Linear(config.hidden_dim, 1)
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, T, H, W)
               B = batch size
               C = 3 (RGB channels)
               T = number of frames
               H, W = height, width (224, 224)

        Returns:
            Logits of shape (B,) for binary classification
        """
        logits = self.backbone(x)  # (B, 1)
        return logits.squeeze(1)   # (B,)
