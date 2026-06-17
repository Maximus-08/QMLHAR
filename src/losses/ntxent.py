"""
NT-Xent (Normalized Temperature-scaled Cross Entropy) loss for contrastive learning.

SimCLR-style contrastive loss that pushes representations of positive pairs
(two augmented views of the same sample) together and negative pairs apart
on the unit hypersphere.

Reference: Chen et al., "A Simple Framework for Contrastive Learning" (SimCLR), 2020
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    NT-Xent loss for contrastive learning.

    Given a batch of N samples, each producing 2 augmented views (z_i, z_j),
    the loss treats (z_i, z_j) as a positive pair and all other 2(N-1) views
    as negatives.

    Loss = -log( exp(sim(z_i, z_j) / τ) / Σ_{k≠i} exp(sim(z_i, z_k) / τ) )
    averaged over all 2N views.
    """

    def __init__(self, temperature=0.1):
        """
        Args:
            temperature: Temperature scaling parameter τ. Lower values make
                         the distribution sharper (more contrastive).
        """
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        Compute the NT-Xent loss.

        Args:
            z_i: Tensor of shape (N, D) — projections of first augmented views
            z_j: Tensor of shape (N, D) — projections of second augmented views
        Returns:
            Scalar loss value
        """
        N = z_i.size(0)
        device = z_i.device

        # Concatenate to get 2N representations
        z = torch.cat([z_i, z_j], dim=0)  # (2N, D)

        # Compute cosine similarity matrix: (2N, 2N)
        # Inputs should already be L2-normalized, but normalize again for safety
        z = F.normalize(z, p=2, dim=1)
        sim_matrix = torch.mm(z, z.t()) / self.temperature  # (2N, 2N)

        # Create mask to exclude self-similarity (diagonal)
        mask_self = torch.eye(2 * N, dtype=torch.bool, device=device)
        sim_matrix = sim_matrix.masked_fill(mask_self, -float("inf"))

        # For each sample i in [0, 2N), its positive pair is at index:
        #   i -> i + N  (if i < N)
        #   i -> i - N  (if i >= N)
        # This creates the positive pair indices
        pos_indices = torch.cat(
            [
                torch.arange(N, 2 * N, device=device),  # First half -> second half
                torch.arange(0, N, device=device),  # Second half -> first half
            ]
        )  # (2N,)

        # Extract positive similarities
        pos_sim = sim_matrix[torch.arange(2 * N, device=device), pos_indices]  # (2N,)

        # NT-Xent: -log(exp(pos) / sum(exp(all except self)))
        # = -pos + log(sum(exp(all except self)))
        # Using logsumexp for numerical stability
        loss = -pos_sim + torch.logsumexp(sim_matrix, dim=1)

        return loss.mean()


# Quick verification
if __name__ == "__main__":
    torch.manual_seed(42)

    # Simulate batch of 8 samples, 8-dim projections
    z_i = F.normalize(torch.randn(8, 8), dim=1)
    z_j = F.normalize(torch.randn(8, 8), dim=1)

    loss_fn = NTXentLoss(temperature=0.1)
    loss = loss_fn(z_i, z_j)
    print(f"z_i shape: {z_i.shape}")
    print(f"z_j shape: {z_j.shape}")
    print(f"NT-Xent loss: {loss.item():.4f}")

    # Verify: if z_i == z_j (perfect alignment), loss should be low
    z_j_same = z_i.clone()
    loss_same = loss_fn(z_i, z_j_same)
    print(f"Loss with identical pairs: {loss_same.item():.4f} (should be lower)")

    assert loss_same < loss, "Loss should be lower when pairs are identical"
    print("Verification passed!")
