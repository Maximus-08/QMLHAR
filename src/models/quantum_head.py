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
import numpy as np
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


class QuantumProjectionHeadPaper(nn.Module):
    """
    Amplitude-encoded VQC projection head matching the paper specification.

    Maps 256-dim features -> 8-qubit quantum state -> 8-dim projection.
    Uses depth D = 1 by default (8 learnable parameters).
    Allows NISQ noise simulation (random Rx before each gate).
    """

    def __init__(
        self, input_dim=256, num_qubits=8, q_layers=1, device_type="default.qubit"
    ):
        """
        Args:
            input_dim: Dimension of input features (must equal 2^num_qubits)
            num_qubits: Number of qubits (default 8, so 2^8 = 256 features)
            q_layers: Depth of the PQC (default 1)
            device_type: PennyLane device backend
        """
        super(QuantumProjectionHeadPaper, self).__init__()

        assert input_dim == 2**num_qubits, (
            f"input_dim ({input_dim}) must equal 2^num_qubits ({2**num_qubits})"
        )

        self.input_dim = input_dim
        self.num_qubits = num_qubits
        self.q_layers = q_layers

        # Noise settings
        self.noise_prob = 0.0
        self.noise_std = 0.1

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
              2. Custom PQC: Ry rotation gates + CNOT gates between adjacent qubits
              3. PauliZ measurements on each qubit
            """
            qml.AmplitudeEmbedding(inputs, wires=range(num_qubits), normalize=True)

            # Helper to apply NISQ noise before gate operations
            def apply_noise(wires):
                if self.noise_prob > 0:
                    for w in wires:
                        if np.random.random() < self.noise_prob:
                            angle = np.random.normal(0, self.noise_std)
                            qml.RX(angle, wires=w)

            # Variational layers
            for d in range(q_layers):
                # Ry rotations
                for i in range(num_qubits):
                    apply_noise([i])
                    qml.RY(weights[d, i], wires=i)

                # CNOT gates between adjacent qubits
                for i in range(num_qubits - 1):
                    apply_noise([i, i + 1])
                    qml.CNOT(wires=[i, i + 1])

            return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

        # Weight shapes: (depth, num_qubits) -> exactly num_qubits parameters per layer
        weight_shapes = {"weights": (q_layers, num_qubits)}

        # Wrap QNode as a differentiable PyTorch layer
        self.q_layer = qml.qnn.TorchLayer(qnode, weight_shapes)

    def set_noise(self, noise_prob=0.0, noise_std=0.1):
        """Configure noise parameters for NISQ simulations."""
        self.noise_prob = noise_prob
        self.noise_std = noise_std

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
    for name, head_cls, layers in [
        ("QuantumProjectionHead (StronglyEntangling)", QuantumProjectionHead, 3),
        ("QuantumProjectionHeadPaper (Custom VQC)", QuantumProjectionHeadPaper, 1),
    ]:
        print(f"\n--- Testing {name} ---")
        head = head_cls(input_dim=256, num_qubits=8, q_layers=layers)
        dummy = torch.randn(4, 256)
        out = head(dummy)
        print(f"Input shape:  {dummy.shape}")
        print(f"Output shape: {out.shape}")
        print(f"Output norms: {torch.norm(out, dim=1)}")  # Should be ~1.0
        print(f"Quantum parameters: {sum(p.numel() for p in head.parameters()):,}")

    # Test noise simulation on paper head
    print("\n--- Testing Noise Simulation on Paper Head ---")
    paper_head = QuantumProjectionHeadPaper(input_dim=256, num_qubits=8, q_layers=1)
    paper_head.set_noise(noise_prob=0.7, noise_std=0.2)
    out_noise = paper_head(dummy)
    print(f"Noise output shape: {out_noise.shape}")
