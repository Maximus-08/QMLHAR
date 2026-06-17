"""
Standalone 4-layer 1D CNN encoder for Human Activity Recognition.

Architecture matches the QCLHAR paper (Ren et al., 2024):
  Layer 1: Conv1d(9, 32, k=8) -> BN -> ReLU -> MaxPool(2) -> Dropout(0.35)
  Layer 2: Conv1d(32, 64, k=8) -> BN -> ReLU -> MaxPool(2)
  Layer 3: Conv1d(64, 128, k=8) -> BN -> ReLU -> MaxPool(2)
  Layer 4: Conv1d(128, 256, k=8) -> BN -> ReLU -> AdaptiveAvgPool(1)

Input:  (batch, 9, 128) — 9-channel raw inertial signals, 128 timesteps
Output: (batch, 256) — feature vector, directly compatible with 8-qubit AmplitudeEmbedding
"""

import torch
import torch.nn as nn


class HAREncoder(nn.Module):
    """
    4-layer fully convolutional encoder for raw inertial sensor signals.

    Produces a 256-dimensional feature vector that can be:
    - Fed into a quantum projection head (AmplitudeEmbedding, 8 qubits)
    - Fed into a linear classifier for supervised fine-tuning
    """

    def __init__(self, in_channels=9, feature_dim=256, dropout_p=0.35):
        """
        Args:
            in_channels: Number of input sensor channels (default 9 for UCI-HAR)
            feature_dim: Output feature dimension (default 256 = 2^8 for amplitude encoding)
            dropout_p: Dropout probability after the first convolutional block
        """
        super(HAREncoder, self).__init__()

        self.feature_dim = feature_dim

        # Layer 1: Conv(9 -> 32) + BN + ReLU + MaxPool + Dropout
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

        # Layer 4: Conv(128 -> 256) + BN + ReLU + AdaptiveAvgPool
        self.block4 = nn.Sequential(
            nn.Conv1d(128, feature_dim, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),  # Global average pooling -> (batch, 256, 1)
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, 9, 128)
        Returns:
            Tensor of shape (batch, 256)
        """
        x = self.block1(x)  # (batch, 32, 64)
        x = self.block2(x)  # (batch, 64, 32)
        x = self.block3(x)  # (batch, 128, 16)
        x = self.block4(x)  # (batch, 256, 1)
        x = x.squeeze(-1)  # (batch, 256)
        return x


# Quick verification
if __name__ == "__main__":
    model = HAREncoder()
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
