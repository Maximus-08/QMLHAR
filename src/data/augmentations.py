"""
Time-series augmentation functions for contrastive learning on HAR sensor data.

Each augmentation takes a tensor of shape (C, T) where C=9 channels and T=128 timesteps,
and returns an augmented view of the same shape.

Reference: QCLHAR (Ren et al., 2024) — Section 3.2 Data Augmentation
"""
import torch
import numpy as np


def jitter(x, sigma=0.05):
    """
    Add zero-mean Gaussian noise to the signal.
    
    Args:
        x: Tensor of shape (C, T)
        sigma: Standard deviation of the noise
    Returns:
        Augmented tensor of shape (C, T)
    """
    noise = torch.randn_like(x) * sigma
    return x + noise


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


def permute(x, n_segments=5):
    """
    Slice the time-series into n_segments and shuffle their order.
    Disrupts temporal ordering while preserving local feature statistics.
    
    Args:
        x: Tensor of shape (C, T)
        n_segments: Number of segments to split into
    Returns:
        Augmented tensor of shape (C, T)
    """
    C, T = x.shape
    segment_len = T // n_segments
    
    # Split into segments
    segments = []
    for i in range(n_segments):
        start = i * segment_len
        end = start + segment_len if i < n_segments - 1 else T
        segments.append(x[:, start:end])
    
    # Shuffle segment order
    perm = torch.randperm(len(segments))
    shuffled = [segments[p] for p in perm]
    
    return torch.cat(shuffled, dim=1)


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


def rotate(x):
    """
    Apply a random 3D rotation matrix to each 3-axis sensor group.
    Each group is rotated by the same random rotation matrix.
    
    Args:
        x: Tensor of shape (C, T)
    Returns:
        Augmented tensor of shape (C, T)
    """
    C, T = x.shape
    # Generate a random rotation matrix via QR decomposition
    random_matrix = torch.randn(3, 3, dtype=x.dtype)
    q, r = torch.linalg.qr(random_matrix)
    # Ensure proper rotation (det = +1)
    d = torch.diag(r)
    ph = torch.sign(d)
    q = q * ph.unsqueeze(0)
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    
    result = x.clone()
    # Apply same rotation to each 3-axis group dynamically based on actual number of channels C
    for start in range(0, C - C % 3, 3):
        group = x[start:start+3, :]  # (3, T)
        rotated = q @ group  # (3, 3) @ (3, T) = (3, T)
        result[start:start+3, :] = rotated
    
    return result


def scale(x, sigma=0.1):
    """
    Multiply each channel by a random scaling factor ~ N(1, sigma).
    
    Args:
        x: Tensor of shape (C, T)
        sigma: Standard deviation for scaling factors
    Returns:
        Augmented tensor of shape (C, T)
    """
    C, T = x.shape
    factors = torch.normal(mean=1.0, std=sigma, size=(C, 1))
    return x * factors


def temporal_flip(x):
    """
    Reverse the time-series along the time dimension (T).
    
    Args:
        x: Tensor of shape (C, T)
    Returns:
        Augmented tensor of shape (C, T)
    """
    return torch.flip(x, dims=[1])


# Registry of all augmentations
AUGMENTATIONS = [jitter, negate, permute, resample, rotate, scale, temporal_flip]


def random_augment(x):
    """
    Randomly select and apply one augmentation from the registry.
    
    Args:
        x: Tensor of shape (C, T)
    Returns:
        Augmented tensor of shape (C, T)
    """
    aug_fn = AUGMENTATIONS[np.random.randint(len(AUGMENTATIONS))]
    return aug_fn(x)


class ContrastiveViewGenerator:
    """
    Generates two augmented views of a single sample for contrastive learning.
    
    Usage:
        generator = ContrastiveViewGenerator()
        view_1, view_2 = generator(x)
    
    When used as the `transform` for UCIHARRawDataset, the dataset's __getitem__
    will return (view_1, view_2) instead of the original sample.
    """
    
    def __init__(self, augmentations=None, n_views=2):
        """
        Args:
            augmentations: List of augmentation functions. Defaults to all.
            n_views: Number of views to generate (default 2 for SimCLR-style)
        """
        self.augmentations = augmentations or AUGMENTATIONS
        self.n_views = n_views
    
    def __call__(self, x):
        """
        Args:
            x: Tensor of shape (C, T)
        Returns:
            Tuple of n_views augmented tensors, each of shape (C, T)
        """
        views = []
        for _ in range(self.n_views):
            aug_fn = self.augmentations[np.random.randint(len(self.augmentations))]
            views.append(aug_fn(x.clone()))
        
        if self.n_views == 2:
            return views[0], views[1]
        return tuple(views)


# Quick verification
if __name__ == "__main__":
    x = torch.randn(9, 128)
    print(f"Original shape: {x.shape}")
    
    for name, fn in [("jitter", jitter), ("negate", negate), ("permute", permute),
                     ("resample", resample), ("rotate", rotate), ("scale", scale),
                     ("temporal_flip", temporal_flip)]:
        out = fn(x)
        print(f"{name}: shape={out.shape}, mean_diff={torch.abs(out - x).mean():.4f}")
    
    gen = ContrastiveViewGenerator()
    v1, v2 = gen(x)
    print(f"\nContrastiveViewGenerator: v1={v1.shape}, v2={v2.shape}")
    print(f"Views are different: {not torch.equal(v1, v2)}")
