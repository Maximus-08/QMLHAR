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

    def __init__(self, in_channels=9, feature_dim=256, dropout_p=0.35, pooling="avg"):
        """
        Args:
            in_channels: Number of input sensor channels (default 9 for UCI-HAR)
            feature_dim: Output feature dimension (default 256 = 2^8 for amplitude encoding)
            dropout_p: Dropout probability after the first convolutional block
            pooling: Pooling layer to use in block4 ("avg" or "max")
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

        # Layer 4: Conv(128 -> 256) + BN + ReLU + Adaptive Pooling
        if pooling == "avg":
            pool_layer = nn.AdaptiveAvgPool1d(1)
        elif pooling == "max":
            pool_layer = nn.AdaptiveMaxPool1d(1)
        else:
            raise ValueError(f"Unknown pooling type: {pooling}")

        self.block4 = nn.Sequential(
            nn.Conv1d(128, feature_dim, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            pool_layer,
        )

    def forward_unpooled(self, x):
        """
        Extract spatial-temporal features before the final global pooling layer.

        Args:
            x: Tensor of shape (batch, in_channels, T)
        Returns:
            Tensor of shape (batch, feature_dim, L_seq)
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        # Apply layers in block4 except the final pooling layer
        for layer in list(self.block4)[:-1]:
            x = layer(x)
        return x

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, in_channels, T)
        Returns:
            Tensor of shape (batch, 256)
        """
        x = self.block1(x)  # (batch, 32, 64)
        x = self.block2(x)  # (batch, 64, 32)
        x = self.block3(x)  # (batch, 128, 16)
        x = self.block4(x)  # (batch, 256, 1)
        x = x.squeeze(-1)  # (batch, 256)
        return x


class HAREncoderPaper(HAREncoder):
    """
    Paper-compliant 4-layer 1D CNN encoder for Human Activity Recognition.
    Inherits from HAREncoder but is pre-configured with Max Pooling at the end
    of block4 to match the paper specifications.
    """

    def __init__(self, in_channels=9, feature_dim=256, dropout_p=0.35):
        super(HAREncoderPaper, self).__init__(
            in_channels=in_channels,
            feature_dim=feature_dim,
            dropout_p=dropout_p,
            pooling="max",
        )


# Quick verification
if __name__ == "__main__":
    for name, model_cls in [
        ("HAREncoder (AvgPool)", HAREncoder),
        ("HAREncoderPaper (MaxPool)", HAREncoderPaper),
    ]:
        print(f"\n--- Testing {name} ---")
        model = model_cls()
        dummy = torch.randn(8, 9, 128)
        out = model(dummy)
        unpooled = model.forward_unpooled(dummy)
        print(f"Input shape:      {dummy.shape}")
        print(f"Output shape:     {out.shape}")
        print(f"Unpooled shape:   {unpooled.shape}")
        print(f"Parameters:       {sum(p.numel() for p in model.parameters()):,}")
