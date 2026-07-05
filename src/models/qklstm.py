"""
Quantum Kernel Long Short-Term Memory (QK-LSTM) cell and layer with a Block-Product State (BPS) kernel.
Designed to replace classical LSTM cells with a parameter-efficient quantum-kernel alternative.
Highly optimized PyTorch-native implementation utilizing statevector algebra, reference state calculations,
and trainable reference vectors/scaling parameters for optimal distribution alignment and training stability.
"""

import torch
import torch.nn as nn
import numpy as np


def block_forward(x, w):
    """
    Simulates a 2-qubit BPS feature map on the initial state |00>.
    Optimized version representing the state vector explicitly as 4 elements,
    avoiding dimensions stacks, einsums, and memory allocations.
    
    Args:
        x: Input tensor of shape (..., num_blocks, 2)
        w: Weight tensor of shape (..., num_blocks, 2)
    Returns:
        Statevector tensor of shape (..., num_blocks, 4)
    """
    # Start with |00> state. After applying Hadamard to both qubits:
    # |++> = 0.5 * (|00> + |01> + |10> + |11>)
    shape = list(x.shape[:-1])
    p0 = torch.full(shape, 0.5, device=x.device, dtype=x.dtype)
    p1 = torch.full(shape, 0.5, device=x.device, dtype=x.dtype)
    p2 = torch.full(shape, 0.5, device=x.device, dtype=x.dtype)
    p3 = torch.full(shape, 0.5, device=x.device, dtype=x.dtype)
    
    # Broadcast w to match the batch dimensions of x
    w_dims = w.dim()
    x_dims = x.dim()
    if x_dims > w_dims:
        for _ in range(x_dims - w_dims):
            w = w.unsqueeze(0)
            
    # Apply RY(x_0) to qubit 0 (least significant bit)
    c0 = torch.cos(x[..., 0] / 2.0)
    s0 = torch.sin(x[..., 0] / 2.0)
    
    p0_rot0 = c0 * p0 - s0 * p1
    p1_rot0 = s0 * p0 + c0 * p1
    p2_rot0 = c0 * p2 - s0 * p3
    p3_rot0 = s0 * p2 + c0 * p3
    
    # Apply RY(x_1) to qubit 1
    c1 = torch.cos(x[..., 1] / 2.0)
    s1 = torch.sin(x[..., 1] / 2.0)
    
    p0_rot1 = c1 * p0_rot0 - s1 * p2_rot0
    p2_rot1 = s1 * p0_rot0 + c1 * p2_rot0
    p1_rot1 = c1 * p1_rot0 - s1 * p3_rot0
    p3_rot1 = s1 * p1_rot0 + c1 * p3_rot0
    
    # Apply CNOT (control=0, target=1): swaps p1 and p3
    p0_cnot = p0_rot1
    p2_cnot = p2_rot1
    p1_cnot = p3_rot1
    p3_cnot = p1_rot1
    
    # Apply RY(w_0) to qubit 0
    cw0 = torch.cos(w[..., 0] / 2.0)
    sw0 = torch.sin(w[..., 0] / 2.0)
    
    p0_wrot0 = cw0 * p0_cnot - sw0 * p1_cnot
    p1_wrot0 = sw0 * p0_cnot + cw0 * p1_cnot
    p2_wrot0 = cw0 * p2_cnot - sw0 * p3_cnot
    p3_wrot0 = sw0 * p2_cnot + cw0 * p3_cnot
    
    # Apply RY(w_1) to qubit 1
    cw1 = torch.cos(w[..., 1] / 2.0)
    sw1 = torch.sin(w[..., 1] / 2.0)
    
    p0_wrot1 = cw1 * p0_wrot0 - sw1 * p2_wrot0
    p2_wrot1 = sw1 * p0_wrot0 + cw1 * p2_wrot0
    p1_wrot1 = cw1 * p1_wrot0 - sw1 * p3_wrot0
    p3_wrot1 = sw1 * p1_wrot0 + cw1 * p3_wrot0
    
    # Apply CNOT (control=1, target=0): swaps p2 and p3
    p0_out = p0_wrot1
    p1_out = p1_wrot1
    p2_out = p3_wrot1
    p3_out = p2_wrot1
    
    # Return statevector of shape (..., num_blocks, 4)
    return torch.stack([p0_out, p1_out, p2_out, p3_out], dim=-1)


class QKLSTMCell(nn.Module):
    """
    A single cell of the Quantum Kernel LSTM (QK-LSTM) using the BPS kernel.
    """
    def __init__(self, input_size, hidden_size, n_ref=20, block_size=2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_ref = n_ref
        self.block_size = block_size
        self.v_dim = hidden_size + input_size
        
        # Calculate number of blocks (each block is 2-qubit BPS)
        self.num_blocks = int(np.ceil(self.v_dim / block_size))
        
        # Trainable coefficient vectors beta for each gate: shape (n_ref, hidden_size)
        # Initialized with small random values to start with stable gates
        self.beta_f = nn.Parameter(torch.randn(n_ref, hidden_size) * (1.0 / np.sqrt(n_ref)))
        self.beta_i = nn.Parameter(torch.randn(n_ref, hidden_size) * (1.0 / np.sqrt(n_ref)))
        self.beta_c = nn.Parameter(torch.randn(n_ref, hidden_size) * (1.0 / np.sqrt(n_ref)))
        self.beta_o = nn.Parameter(torch.randn(n_ref, hidden_size) * (1.0 / np.sqrt(n_ref)))
        
        # Trainable scaling parameter for inputs and reference vectors to learn optimal kernel bandwidth
        self.gamma = nn.Parameter(torch.tensor(3.0))
        
        # Trainable reference/support vectors: shape (n_ref, v_dim)
        # Initialized to match the scale of input features/hidden states
        self.ref_vectors = nn.Parameter(torch.randn(n_ref, self.v_dim) * 0.1)
        
        # BPS variational weights: shape (num_blocks, block_size)
        # Registered as non-trainable buffers
        bps_w_init = torch.rand(self.num_blocks, block_size) * (2 * np.pi)
        self.register_buffer('bps_weights', bps_w_init)
        
    def compute_bps_kernel(self, x):
        """
        Computes the BPS kernel evaluations kappa(x, v_j) for all reference vectors v_j.
        
        Args:
            x: Input tensor of shape (batch_size, v_dim)
        Returns:
            Kernel matrix of shape (batch_size, n_ref)
        """
        batch_size = x.shape[0]
        y = self.ref_vectors
        
        # Pad inputs if necessary
        pad_size = self.num_blocks * self.block_size - self.v_dim
        if pad_size > 0:
            x_padded = torch.cat([x, torch.zeros(batch_size, pad_size, device=x.device, dtype=x.dtype)], dim=-1)
            y_padded = torch.cat([y, torch.zeros(self.n_ref, pad_size, device=y.device, dtype=y.dtype)], dim=-1)
        else:
            x_padded = x
            y_padded = y
            
        # Reshape to (..., num_blocks, block_size)
        x_blocks = x_padded.view(batch_size, self.num_blocks, self.block_size)
        y_blocks = y_padded.view(self.n_ref, self.num_blocks, self.block_size)
        
        # Apply learnable scaling parameter gamma to control kernel resolution/bandwidth
        x_scaled = x_blocks * self.gamma
        y_scaled = y_blocks * self.gamma
        
        # Compute BPS block wavefunctions (statevectors)
        psi_X = block_forward(x_scaled, self.bps_weights) # (batch_size, num_blocks, 4)
        psi_Y = block_forward(y_scaled, self.bps_weights) # (n_ref, num_blocks, 4)
        
        # Pairwise dot product for each block
        overlaps = torch.einsum('b g d, r g d -> b r g', psi_X, psi_Y)
        fidelities = overlaps ** 2 # (batch_size, n_ref, num_blocks)
        
        # Compute the arithmetic mean of fidelities across blocks.
        # This acts as a stable and valid kernel summation, completely avoiding underflow.
        kernel_matrix = torch.mean(fidelities, dim=-1) # (batch_size, n_ref)
        
        return kernel_matrix

    def forward(self, x, states):
        """
        Forward pass for a single time step of the QK-LSTM Cell.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            states: Tuple of (h_prev, c_prev), each of shape (batch_size, hidden_size)
        Returns:
            h_next: Hidden state of shape (batch_size, hidden_size)
            states_next: Tuple of (h_next, c_next)
        """
        h, c = states
        
        # v_t = [h_{t-1}; x_t]
        v_t = torch.cat([h, x], dim=-1) # (batch_size, hidden_size + input_size)
        
        # Compute kernel evaluations
        kappa = self.compute_bps_kernel(v_t) # (batch_size, n_ref)
        
        # Compute gate activations using beta coefficients
        f_t = torch.sigmoid(torch.matmul(kappa, self.beta_f))
        i_t = torch.sigmoid(torch.matmul(kappa, self.beta_i))
        c_tilde = torch.tanh(torch.matmul(kappa, self.beta_c))
        
        c_next = f_t * c + i_t * c_tilde
        
        o_t = torch.sigmoid(torch.matmul(kappa, self.beta_o))
        h_next = o_t * torch.tanh(c_next)
        
        return h_next, (h_next, c_next)


class QKLSTM(nn.Module):
    """
    A multi-layer, optionally bidirectional Quantum Kernel LSTM (QK-LSTM) layer.
    Direct drop-in replacement for nn.LSTM.
    """
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=False, n_ref=20, block_size=2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        
        # Build layers list
        self.layers = nn.ModuleList()
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else (hidden_size * 2 if bidirectional else hidden_size)
            
            # Forward Cell
            self.layers.append(QKLSTMCell(layer_input_size, hidden_size, n_ref=n_ref, block_size=block_size))
            
            # Backward Cell
            if bidirectional:
                self.layers.append(QKLSTMCell(layer_input_size, hidden_size, n_ref=n_ref, block_size=block_size))

    def forward(self, x, init_states=None):
        """
        Args:
            x: Tensor of shape (batch, seq_len, input_size) if batch_first=True, else (seq_len, batch, input_size)
            init_states: Optional tuple of (h_0, c_0)
        Returns:
            output: Tensor of shape (batch, seq_len, hidden_size * num_directions)
            (h_n, c_n): Final states
        """
        # Ensure sequence dimension is first
        if self.batch_first:
            x = x.transpose(0, 1) # (seq_len, batch, input_size)
            
        seq_len, batch_size, _ = x.shape
        num_directions = 2 if self.bidirectional else 1
        
        if init_states is None:
            h_n = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
            c_n = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
        else:
            h_n, c_n = init_states
            
        current_input = x
        next_h_n = []
        next_c_n = []
        
        for layer_idx in range(self.num_layers):
            # Forward direction
            cell_fw = self.layers[layer_idx * num_directions]
            state_fw = (h_n[layer_idx * num_directions], c_n[layer_idx * num_directions])
            
            fw_outputs = []
            for t in range(seq_len):
                h_t, state_fw = cell_fw(current_input[t], state_fw)
                fw_outputs.append(h_t)
            fw_outputs = torch.stack(fw_outputs, dim=0) # (seq_len, batch, hidden_size)
            
            next_h_n.append(state_fw[0])
            next_c_n.append(state_fw[1])
            
            if self.bidirectional:
                # Backward direction
                cell_bw = self.layers[layer_idx * num_directions + 1]
                state_bw = (h_n[layer_idx * num_directions + 1], c_n[layer_idx * num_directions + 1])
                
                bw_outputs = []
                for t in reversed(range(seq_len)):
                    h_t, state_bw = cell_bw(current_input[t], state_bw)
                    bw_outputs.insert(0, h_t)
                bw_outputs = torch.stack(bw_outputs, dim=0) # (seq_len, batch, hidden_size)
                
                next_h_n.append(state_bw[0])
                next_c_n.append(state_bw[1])
                
                current_input = torch.cat([fw_outputs, bw_outputs], dim=-1)
            else:
                current_input = fw_outputs
                
        output = current_input
        if self.batch_first:
            output = output.transpose(0, 1) # (batch, seq_len, hidden_size * num_directions)
            
        h_n = torch.stack(next_h_n, dim=0)
        c_n = torch.stack(next_c_n, dim=0)
        
        return output, (h_n, c_n)


if __name__ == "__main__":
    # Small test script to verify implementation correctness
    print("Testing QKLSTM Cell and Layer...")
    cell = QKLSTMCell(input_size=10, hidden_size=20, n_ref=15)
    dummy_x = torch.randn(8, 10)
    dummy_h = torch.zeros(8, 20)
    dummy_c = torch.zeros(8, 20)
    
    h_next, (h_next, c_next) = cell(dummy_x, (dummy_h, dummy_c))
    print(f"Cell forward pass successful!")
    print(f"h_next shape: {h_next.shape} | expected: (8, 20)")
    print(f"c_next shape: {c_next.shape} | expected: (8, 20)")
    
    # Test multi-layer bidirectional QKLSTM
    model = QKLSTM(input_size=10, hidden_size=20, num_layers=2, bidirectional=True, n_ref=15)
    dummy_seq = torch.randn(8, 30, 10) # (batch, seq_len, input_size)
    out, (hn, cn) = model(dummy_seq)
    print("QKLSTM Layer forward pass successful!")
    print(f"out shape: {out.shape} | expected: (8, 30, 40)")
    print(f"hn shape:  {hn.shape} | expected: (4, 8, 20)")
    print(f"cn shape:  {cn.shape} | expected: (4, 8, 20)")
    
    # Verify backward gradient flow
    loss = out.sum()
    loss.backward()
    print("Backward pass successful! Parameter gradients populated:")
    for name, p in model.named_parameters():
        if p.grad is not None:
            print(f"  {name}: grad norm = {p.grad.norm().item():.4f}")
        else:
            print(f"  WARNING: {name} has no gradient!")
