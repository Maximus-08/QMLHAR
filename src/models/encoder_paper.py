"""
Paper-compliant 4-layer 1D CNN encoder for Human Activity Recognition.

Architecture matches the specifications of the paper:
  Layer 1: Conv1d(9, 32, k=8) -> BN -> ReLU -> MaxPool(2) -> Dropout(0.35)
  Layer 2: Conv1d(32, 64, k=8) -> BN -> ReLU -> MaxPool(2)
  Layer 3: Conv1d(64, 128, k=8) -> BN -> ReLU -> MaxPool(2)
  Layer 4: Conv1d(128, 256, k=8) -> BN -> ReLU -> AdaptiveMaxPool1d(1)

Input:  (batch, in_channels, T) — raw inertial signals
Output: (batch, 256) — feature vector, directly compatible with 8-qubit AmplitudeEmbedding
"""

import torch
import torch.nn as nn


class HAREncoderPaper(nn.Module):
    """
    4-layer fully convolutional encoder for raw inertial sensor signals.
    Uses Max Pooling at the end of every block to match paper specifications.
    """

    def __init__(self, in_channels=9, feature_dim=256, dropout_p=0.35):
        """
        Args:
            in_channels: Number of input sensor channels (default 9)
            feature_dim: Output feature dimension (default 256 = 2^8)
            dropout_p: Dropout probability after the first convolutional block
        """
        super(HAREncoderPaper, self).__init__()

        self.feature_dim = feature_dim

        # Layer 1: Conv(in_channels -> 32) + BN + ReLU + MaxPool + Dropout
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=dropout_p),
        )

        # Layer 2: Conv(32 -> 64) + BN + ReLU + MaxPool
        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )

        # Layer 3: Conv(64 -> 128) + BN + ReLU + MaxPool
        self.block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )

        # Layer 4: Conv(128 -> 256) + BN + ReLU + Global Average Pool
        self.block4 = nn.Sequential(
            nn.Conv1d(128, feature_dim, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(1),  # Global max pooling -> (batch, 256, 1)
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, in_channels, T)
        Returns:
            Tensor of shape (batch, 256)
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.squeeze(-1)  # (batch, 256)
        return x


# Quick verification
if __name__ == "__main__":
    model = HAREncoderPaper()
    dummy = torch.randn(8, 9, 128)
    out = model(dummy)
    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters:   {sum(p.numel() for p in model.parameters()):,}")

    # Verify shapes at each block
    x = dummy
    for i, block in enumerate([model.block1, model.block2, model.block3, model.block4]):
        x = block(x)
        print(f"After block {i + 1}: {x.shape}")
