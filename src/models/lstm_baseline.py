"""
Classical LSTM baseline model for Human Activity Recognition (HAR).

Processes input tensors of shape (batch, in_channels, sequence_length),
transposes them to (batch, sequence_length, in_channels) for PyTorch's LSTM,
and projects the final time step output to the number of classes.
"""

import torch
import torch.nn as nn
from src.models.qklstm import QKLSTM


class LSTMClassifier(nn.Module):
    """
    Standard QK-LSTM Classifier baseline for raw inertial sensor signals.
    """

    def __init__(
        self,
        in_channels,
        hidden_dim=128,
        num_layers=2,
        num_classes=6,
        dropout_p=0.5,
        bidirectional=False,
        n_ref=20,
        block_size=2,
    ):
        """
        Args:
            in_channels: Number of input sensor channels.
            hidden_dim: Number of features in the LSTM hidden state.
            num_layers: Number of recurrent layers.
            num_classes: Number of target activity classes.
            dropout_p: Dropout probability applied between LSTM layers.
            bidirectional: If True, becomes a bidirectional LSTM.
            n_ref: Number of reference vectors for QK-LSTM.
            block_size: Size of local blocks for BPS kernel.
        """
        super(LSTMClassifier, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # QK-LSTM Layer
        self.lstm = QKLSTM(
            input_size=in_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            n_ref=n_ref,
            block_size=block_size,
        )

        # Classification Head
        classifier_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout_p), nn.Linear(classifier_input_dim, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, in_channels, sequence_length)
        Returns:
            Tensor of shape (batch, num_classes)
        """
        # Transpose input: (batch, in_channels, sequence_length) -> (batch, sequence_length, in_channels)
        x = x.transpose(1, 2)

        # Forward pass through LSTM
        # lstm_out shape: (batch, sequence_length, hidden_dim * num_directions)
        lstm_out, _ = self.lstm(x)

        # Extract the last time step's output
        # out shape: (batch, hidden_dim * num_directions)
        out = lstm_out[:, -1, :]

        # Project to classes
        logits = self.fc(out)
        return logits


# Quick verification
if __name__ == "__main__":
    model = LSTMClassifier(in_channels=9, hidden_dim=128, num_layers=2, num_classes=6)
    dummy = torch.randn(8, 9, 128)
    out = model(dummy)
    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters:   {sum(p.numel() for p in model.parameters()):,}")
