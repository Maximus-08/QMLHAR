import torch
import torch.nn as nn
from src.models.qklstm import QKLSTM


class MPSQCLLSTMClassifier(nn.Module):
    """
    Hybrid model combining a pre-trained MPSQCL 1D CNN Encoder (HAREncoderPaper)
    with a QK-LSTM classifier head.
    """

    def __init__(
        self,
        encoder,
        lstm_hidden_dim=128,
        lstm_layers=1,
        num_classes=6,
        bidirectional=False,
        dropout_p=0.5,
        use_pool=False,
        normalize_features=True,
        n_ref=20,
        block_size=2,
        use_classical_lstm=False,
    ):
        """
        Args:
            encoder: Instance of HAREncoderPaper.
            lstm_hidden_dim: Number of features in the LSTM hidden state.
            lstm_layers: Number of recurrent layers.
            num_classes: Number of target activity classes.
            bidirectional: If True, becomes a bidirectional LSTM.
            dropout_p: Dropout probability applied between LSTM layers and classification head.
            use_pool: If True, feeds the global max-pooled representation (sequence length = 1)
                      into the LSTM. If False, feeds the sequence of features before pooling.
            normalize_features: If True, L2-normalizes the feature representations.
            n_ref: Number of reference vectors for QK-LSTM.
            block_size: Size of local blocks for BPS kernel.
            use_classical_lstm: If True, uses standard PyTorch nn.LSTM instead of QKLSTM.
        """
        super().__init__()
        self.encoder = encoder
        self.use_pool = use_pool
        self.normalize_features = normalize_features

        # The feature dimension from HAREncoderPaper is always 256
        if use_classical_lstm:
            self.lstm = nn.LSTM(
                input_size=256,
                hidden_size=lstm_hidden_dim,
                num_layers=lstm_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout_p if lstm_layers > 1 else 0.0,
            )
        else:
            self.lstm = QKLSTM(
                input_size=256,
                hidden_size=lstm_hidden_dim,
                num_layers=lstm_layers,
                batch_first=True,
                bidirectional=bidirectional,
                n_ref=n_ref,
                block_size=block_size,
            )

        classifier_input_dim = lstm_hidden_dim * 2 if bidirectional else lstm_hidden_dim
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout_p), nn.Linear(classifier_input_dim, num_classes)
        )


    def forward(self, x):
        if self.use_pool:
            # Get global pooled feature vector. Shape: (batch, 256)
            features = self.encoder(x)
            if self.normalize_features:
                features = nn.functional.normalize(features, p=2, dim=1)
            # Add sequence dimension of 1 -> (batch, 1, 256)
            features = features.unsqueeze(1)
        else:
            # Run encoder blocks except the final global pooling layer in block4
            x = self.encoder.forward_unpooled(x)

            # Shape of x: (batch, 256, L_seq)
            features = x.transpose(1, 2)  # (batch, L_seq, 256)
            if self.normalize_features:
                features = nn.functional.normalize(features, p=2, dim=2)

        # Forward pass through LSTM
        # lstm_out shape: (batch, sequence_length, hidden_dim * num_directions)
        lstm_out, _ = self.lstm(features)

        # Take the output of the last sequence step
        out = lstm_out[:, -1, :]

        # Final classification
        logits = self.fc(out)
        return logits


# Quick verification
if __name__ == "__main__":
    import os
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from src.models.encoder import HAREncoderPaper

    encoder = HAREncoderPaper(in_channels=9, feature_dim=256)

    model_pool = MPSQCLLSTMClassifier(encoder, use_pool=True, num_classes=6)
    model_seq = MPSQCLLSTMClassifier(encoder, use_pool=False, num_classes=6)

    dummy = torch.randn(8, 9, 128)

    out_pool = model_pool(dummy)
    out_seq = model_seq(dummy)

    print(f"Input shape: {dummy.shape}")
    print(f"Pool mode output shape: {out_pool.shape} | Expected: (8, 6)")
    print(f"Seq mode output shape: {out_seq.shape} | Expected: (8, 6)")
