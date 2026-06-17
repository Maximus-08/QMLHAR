"""
Paper-compliant time-series augmentation functions for contrastive learning.

Restricted only to the two strategies used in the paper:
  - Resampling: upsampling + downsampling to simulate different sampling frequencies
  - Negated: vertical flip (mirroring) of the signal
"""
import torch
import numpy as np


def negate(x):
    """
    Vertically flip (mirror) the signal values: x -> -x.
    Simulates sensor polarity inversion.
    
    Args:
        x: Tensor of shape (C, T)
    Returns:
        Augmented tensor of shape (C, T)
    """
    return -x


def resample(x, ratio_range=(0.5, 2.0)):
    """
    Resample the time-series by randomly changing the temporal resolution
    via linear interpolation, then resize back to the original length.
    
    Args:
        x: Tensor of shape (C, T)
        ratio_range: Tuple of (min_ratio, max_ratio) for resampling
    Returns:
        Augmented tensor of shape (C, T)
    """
    C, T = x.shape
    ratio = np.random.uniform(*ratio_range)
    new_len = max(2, int(T * ratio))
    
    # Interpolate: unsqueeze to (1, C, T) for F.interpolate
    x_unsq = x.unsqueeze(0)  # (1, C, T)
    resampled = torch.nn.functional.interpolate(
        x_unsq, size=new_len, mode='linear', align_corners=False
    )
    # Resize back to original length
    result = torch.nn.functional.interpolate(
        resampled, size=T, mode='linear', align_corners=False
    )
    return result.squeeze(0)  # (C, T)


# Only these two are used in the paper
AUGMENTATIONS = [resample, negate]


class ContrastiveViewGeneratorPaper:
    """
    Generates M augmented views of a single sample for contrastive learning
    using ONLY resampling and negation.
    """
    
    def __init__(self, n_views=2):
        """
        Args:
            n_views: Number of views to generate (default 2 for SimCLR, M for MPSQCL)
        """
        self.n_views = n_views
        self.augmentations = AUGMENTATIONS
    
    def __call__(self, x):
        """
        Args:
            x: Tensor of shape (C, T)
        Returns:
            Tuple of n_views augmented tensors, each of shape (C, T)
        """
        views = []
        for _ in range(self.n_views):
            while True:
                v = x.clone()
                # Apply resample with 50% probability
                if np.random.random() < 0.5:
                    v = resample(v)
                # Apply negate with 50% probability
                if np.random.random() < 0.5:
                    v = negate(v)
                
                # Ensure it is not identical to any already generated views
                is_duplicate = False
                for existing_v in views:
                    if torch.equal(v, existing_v):
                        is_duplicate = True
                        break
                if not is_duplicate:
                    views.append(v)
                    break
        
        if self.n_views == 2:
            return views[0], views[1]
        return tuple(views)


# Quick verification
if __name__ == "__main__":
    x = torch.randn(9, 128)
    print(f"Original shape: {x.shape}")
    
    for name, fn in [("negate", negate), ("resample", resample)]:
        out = fn(x)
        print(f"{name}: shape={out.shape}, mean_diff={torch.abs(out - x).mean():.4f}")
    
    gen = ContrastiveViewGeneratorPaper(n_views=4)
    views = gen(x)
    print(f"\nContrastiveViewGeneratorPaper (4 views): {[v.shape for v in views]}")
