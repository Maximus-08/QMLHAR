"""
Quantum Projection Head for contrastive learning.

Uses AmplitudeEmbedding to encode 256-dimensional feature vectors into 8-qubit
quantum states, followed by StronglyEntanglingLayers for variational processing,
and PauliZ measurements for the output projection.

This head is used only during Phase 1 (contrastive pre-training).
During Phase 2 (fine-tuning), it is replaced by a linear classifier.
"""

import torch
import torch.nn as nn
import pennylane as qml


class QuantumProjectionHead(nn.Module):
    """
    Amplitude-encoded VQC projection head for contrastive learning.

    Maps 256-dim classical features -> 8-qubit quantum state -> 8-dim projection.
    Output is L2-normalized to lie on the unit hypersphere for cosine similarity.
    """

    def __init__(
        self, input_dim=256, num_qubits=8, q_layers=3, device_type="default.qubit"
    ):
        """
        Args:
            input_dim: Dimension of input features (must equal 2^num_qubits)
            num_qubits: Number of qubits (default 8, so 2^8 = 256 features)
            q_layers: Number of StronglyEntanglingLayers
            device_type: PennyLane device backend
        """
        super(QuantumProjectionHead, self).__init__()

        assert input_dim == 2**num_qubits, (
            f"input_dim ({input_dim}) must equal 2^num_qubits ({2**num_qubits})"
        )

        self.input_dim = input_dim
        self.num_qubits = num_qubits
        self.q_layers = q_layers

        # PennyLane device
        try:
            self.dev = qml.device(device_type, wires=num_qubits)
        except Exception:
            print(
                f"Warning: {device_type} not available, falling back to default.qubit"
            )
            self.dev = qml.device("default.qubit", wires=num_qubits)

        # Define the QNode
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def qnode(inputs, weights):
            """
            Quantum circuit:
              1. AmplitudeEmbedding: encode 256-dim normalized vector into 8 qubits
              2. StronglyEntanglingLayers: variational processing
              3. PauliZ measurements on each qubit
            """
            qml.AmplitudeEmbedding(inputs, wires=range(num_qubits), normalize=True)
            qml.StronglyEntanglingLayers(weights, wires=range(num_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

        # Weight shapes for TorchLayer
        weight_shapes = {"weights": (q_layers, num_qubits, 3)}

        # Wrap QNode as a differentiable PyTorch layer
        self.q_layer = qml.qnn.TorchLayer(qnode, weight_shapes)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, 256) — encoder output features
        Returns:
            Tensor of shape (batch, num_qubits) — L2-normalized quantum projections
        """
        # L2-normalize inputs for AmplitudeEmbedding (requires unit norm)
        x_norm = torch.nn.functional.normalize(x, p=2, dim=1)

        # Pass through quantum circuit
        z = self.q_layer(x_norm)  # (batch, num_qubits)

        # L2-normalize output for cosine similarity in NT-Xent
        z = torch.nn.functional.normalize(z, p=2, dim=1)

        return z


# Quick verification
if __name__ == "__main__":
    head = QuantumProjectionHead(input_dim=256, num_qubits=8, q_layers=3)
    dummy = torch.randn(4, 256)
    out = head(dummy)
    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output norms: {torch.norm(out, dim=1)}")  # Should be ~1.0
    print(f"Quantum parameters: {sum(p.numel() for p in head.parameters()):,}")
