"""
Multi-Positive Sample Quantum Contrastive Loss (MPSQCL Loss) for contrastive learning.

Pushes representations of multiple positive views (different augmentations of the same sample)
together and negative views apart on the unit hypersphere.

Reference: Ren et al., "Multi-Positive Sample Quantum Contrastive Learning" (MPSQCL), 2024
Equation (1):
  L_i = 1 / |P(i)| * sum_{n in P(i)} -log ( exp(sim(z_i, z_n) / τ) / sum_{k=1}^{M*N} 1[k not in P(i)] exp(sim(z_i, z_k) / τ) )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MPSQCLLoss(nn.Module):
    """
    Multi-Positive Sample Quantum Contrastive Loss (MPSQCL Loss).

    Given a batch of N samples, each producing M augmented views:
    - We have M * N total representations.
    - For a given view i, the set of positive views P(i) consists of the other
      M - 1 views of the same original sample.
    - The denominator sums over all views k that do NOT belong to the same original
      sample, i.e., different samples (negatives), and optionally includes the anchor itself.
    """

    def __init__(self, temperature=0.1, exclude_anchor_from_denominator=False):
        """
        Args:
            temperature: Temperature scaling parameter τ.
            exclude_anchor_from_denominator: If True, the anchor itself (k = i) is excluded
                                            from the denominator. If False (default, matching the paper),
                                            1[k not in P(i)] allows the anchor itself to be in the denominator
                                            since the anchor is not in P(i).
        """
        super(MPSQCLLoss, self).__init__()
        self.temperature = temperature
        self.exclude_anchor_from_denominator = exclude_anchor_from_denominator

    def forward(self, views):
        """
        Compute the Multi-Positive Sample Contrastive Loss.

        Args:
            views: List of M tensors, each of shape (N, D) or a single stacked tensor of shape (M, N, D)
                  or a concatenated tensor of shape (M * N, D).
        Returns:
            Scalar loss value
        """
        # If views is a list of tensors, concatenate them
        if isinstance(views, list):
            M = len(views)
            N = views[0].size(0)
            z = torch.cat(views, dim=0)  # (M * N, D)
        elif isinstance(views, torch.Tensor):
            if views.dim() == 3:
                # Shape (M, N, D)
                M, N, D = views.shape
                z = views.view(M * N, D)
            else:
                # Assume concatenated shape (M * N, D). We need to know M or N to identify samples.
                # In this case, we expect views to be passed as list/tuple or 3D tensor to avoid ambiguity.
                raise ValueError("If views is a Tensor, it must be of shape (M, N, D).")
        else:
            raise TypeError(
                "views must be a list of Tensors or a 3D Tensor of shape (M, N, D)."
            )

        device = z.device

        # L2 normalize the representations along the feature dimension
        z = F.normalize(z, p=2, dim=1)

        # Compute cosine similarity matrix: shape (M*N, M*N)
        sim_matrix = torch.mm(z, z.t()) / self.temperature

        # Helper indices to track sample identities
        indices = torch.arange(M * N, device=device)
        sample_ids = (
            indices % N
        )  # If indices are [0..N-1, N..2N-1, ...], then i % N gets sample index

        # Row sample IDs and col sample IDs to create masks
        sample_ids_row = sample_ids.unsqueeze(1)
        sample_ids_col = sample_ids.unsqueeze(0)

        # same_sample[i, j] is True if view i and view j belong to the same original sample
        same_sample = sample_ids_row == sample_ids_col

        # Exclude self-similarity (diagonal) for positive pairs
        # pos_mask[i, j] is True if i != j and they belong to the same sample
        pos_mask = same_sample & (~torch.eye(M * N, dtype=torch.bool, device=device))

        # Denominator mask 1[k not in P(i)]
        # P(i) is the set of positive samples for view i (same sample, excluding i).
        # Therefore, "k not in P(i)" means:
        #   - either k belongs to a different original sample
        #   - or k is the anchor view i itself.
        if self.exclude_anchor_from_denominator:
            # Only sum over different samples (negatives)
            denom_mask = ~same_sample
        else:
            # Sum over different samples AND the anchor itself (paper Eq 1)
            denom_mask = ~pos_mask

        # For numerical stability: log(exp(pos) / sum(exp(denom))) = pos - log(sum(exp(denom)))
        # Mask out values not in denominator by setting them to -inf (so exp is 0)
        sim_matrix_denom = sim_matrix.clone()
        sim_matrix_denom[~denom_mask] = -float("inf")

        # logsumexp along rows: shape (M * N, 1)
        log_sum_exp_denom = torch.logsumexp(sim_matrix_denom, dim=1, keepdim=True)

        # Compute the loss for all pairs
        loss_matrix = -sim_matrix + log_sum_exp_denom  # (M*N, M*N)

        # Filter to only keep positive pair losses
        loss_matrix = loss_matrix * pos_mask.float()

        # Average over all positive targets per view (each view has M - 1 positive views)
        pos_count = pos_mask.sum(dim=1)  # (M*N,)
        pos_count = torch.clamp(pos_count, min=1.0)  # Avoid division by zero

        row_loss = loss_matrix.sum(dim=1) / pos_count

        return row_loss.mean()


# Self-verification block
if __name__ == "__main__":
    import torch

    print("Verifying MPSQCLLoss implementation...")
    # 3 views, batch size of 4, feature dim of 8
    M, N, D = 3, 4, 8
    torch.manual_seed(42)

    views = [F.normalize(torch.randn(N, D), dim=1) for _ in range(M)]

    loss_fn_paper = MPSQCLLoss(temperature=0.1, exclude_anchor_from_denominator=False)
    loss_fn_std = MPSQCLLoss(temperature=0.1, exclude_anchor_from_denominator=True)

    loss_paper = loss_fn_paper(views)
    loss_std = loss_fn_std(views)

    print(f"Loss (paper formulation): {loss_paper.item():.4f}")
    print(f"Loss (exclude anchor):    {loss_std.item():.4f}")

    # Check that identical views yield lower loss
    views_identical = [views[0].clone() for _ in range(M)]
    loss_identical = loss_fn_paper(views_identical)
    print(f"Loss with identical views: {loss_identical.item():.4f} (should be lower)")

    assert loss_identical < loss_paper, "Identical views should result in lower loss!"
    print("Success! MPSQCLLoss works as expected.")
