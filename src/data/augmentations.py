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
        x_unsq, size=new_len, mode="linear", align_corners=False
    )
    # Resize back to original length
    result = torch.nn.functional.interpolate(
        resampled, size=T, mode="linear", align_corners=False
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
        group = x[start : start + 3, :]  # (3, T)
        rotated = q @ group  # (3, 3) @ (3, T) = (3, T)
        result[start : start + 3, :] = rotated

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


def time_warp(x, sigma=0.2, knot=4):
    """
    Time warp the signal using cubic spline interpolation.

    Args:
        x: Tensor of shape (C, T)
    Returns:
        Augmented tensor of shape (C, T)
    """
    from scipy.interpolate import CubicSpline

    C, T = x.shape
    x_np = x.cpu().numpy().T[np.newaxis, :, :]  # (1, T, C)

    orig_steps = np.arange(T)
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(1, knot + 2, C))
    warp_steps = (np.ones((C, 1)) * (np.linspace(0, T - 1, num=knot + 2))).T

    ret = np.zeros_like(x_np)
    for dim in range(C):
        t_warp = CubicSpline(
            warp_steps[:, dim], warp_steps[:, dim] * random_warps[0, :, dim]
        )(orig_steps)
        scale = (T - 1) / t_warp[-1] if t_warp[-1] != 0 else 1.0
        ret[0, :, dim] = np.interp(
            orig_steps, np.clip(scale * t_warp, 0, T - 1), x_np[0, :, dim]
        ).T

    y_np = ret[0].T  # (C, T)
    return torch.tensor(y_np, dtype=x.dtype, device=x.device)


def window_warp(x, window_ratio=0.1, scales=[0.5, 2.0]):
    """
    Warp a random window of the signal.

    Args:
        x: Tensor of shape (C, T)
    Returns:
        Augmented tensor of shape (C, T)
    """
    C, T = x.shape
    x_np = x.cpu().numpy().T[np.newaxis, :, :]  # (1, T, C)

    warp_scale = np.random.choice(scales)
    warp_size = np.ceil(window_ratio * T).astype(int)
    window_steps = np.arange(warp_size)

    window_start = np.random.randint(low=1, high=T - warp_size - 1)
    window_end = window_start + warp_size

    ret = np.zeros_like(x_np)
    for dim in range(C):
        start_seg = x_np[0, :window_start, dim]
        window_seg = np.interp(
            np.linspace(0, warp_size - 1, num=int(warp_size * warp_scale)),
            window_steps,
            x_np[0, window_start:window_end, dim],
        )
        end_seg = x_np[0, window_end:, dim]
        warped = np.concatenate((start_seg, window_seg, end_seg))
        ret[0, :, dim] = np.interp(
            np.arange(T), np.linspace(0, T - 1, num=warped.size), warped
        ).T

    y_np = ret[0].T  # (C, T)
    return torch.tensor(y_np, dtype=x.dtype, device=x.device)


def channel_shuffle(x):
    """
    Randomly shuffle/permute the order of the channels.

    Args:
        x: Tensor of shape (C, T)
    Returns:
        Augmented tensor of shape (C, T)
    """
    C, T = x.shape
    perm = torch.randperm(C)
    return x[perm, :]


def perm_jit(x, n_segments=5, sigma=0.05):
    """
    Slice the time-series into n_segments, shuffle them, and add zero-mean Gaussian noise.

    Args:
        x: Tensor of shape (C, T)
    Returns:
        Augmented tensor of shape (C, T)
    """
    return jitter(permute(x, n_segments=n_segments), sigma=sigma)


noise = jitter


# Registry of all augmentations
AUGMENTATIONS = [
    jitter,
    negate,
    permute,
    resample,
    rotate,
    scale,
    temporal_flip,
    time_warp,
    window_warp,
    channel_shuffle,
    perm_jit,
]


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


# Dataset-specific optimal augmentation sets from Table III of the paper:
OPTIMAL_AUGMENTATIONS = {
    "ucihar": [resample, perm_jit, noise, temporal_flip, rotate],
    "hhar": [resample, perm_jit, negate, noise, temporal_flip, rotate],
    "motionsense": [resample, perm_jit, noise, rotate, permute],
    "uschad": [time_warp, resample, perm_jit, noise],
}


class ContrastiveViewGeneratorPaper:
    """
    Generates M augmented views of a single sample for contrastive pre-training.
    Supports dataset-specific optimal augmentation combinations from Table III.
    """

    def __init__(self, n_views=2, dataset_name=None):
        """
        Args:
            n_views: Number of views to generate. Overridden if dataset_name matches.
            dataset_name: Dataset name to load optimal paper combinations.
        """
        self.dataset_name = dataset_name

        if dataset_name and dataset_name.lower() in OPTIMAL_AUGMENTATIONS:
            self.augmentations = OPTIMAL_AUGMENTATIONS[dataset_name.lower()]
            self.n_views = len(self.augmentations)
            print(
                f"Initialized ContrastiveViewGeneratorPaper for '{dataset_name}' with {self.n_views} optimal paper-compliant augmentations."
            )
        else:
            self.n_views = n_views
            self.augmentations = [resample, negate]
            print(
                f"Initialized ContrastiveViewGeneratorPaper with default {self.n_views} views (resample, negate)."
            )

    def __call__(self, x):
        """
        Args:
            x: Tensor of shape (C, T)
        Returns:
            Tuple of n_views augmented tensors, each of shape (C, T)
        """
        views = []

        # If dataset-specific optimal set is defined, apply each function exactly once
        if self.dataset_name and self.dataset_name.lower() in OPTIMAL_AUGMENTATIONS:
            for aug_fn in self.augmentations:
                views.append(aug_fn(x.clone()))
        else:
            # Default probabilistic view generator using only resample and negate
            for _ in range(self.n_views):
                while True:
                    v = x.clone()
                    if np.random.random() < 0.5:
                        v = resample(v)
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

    for name, fn in [
        ("jitter", jitter),
        ("negate", negate),
        ("permute", permute),
        ("resample", resample),
        ("rotate", rotate),
        ("scale", scale),
        ("temporal_flip", temporal_flip),
    ]:
        out = fn(x)
        print(f"{name}: shape={out.shape}, mean_diff={torch.abs(out - x).mean():.4f}")

    gen = ContrastiveViewGenerator()
    v1, v2 = gen(x)
    print(f"\nContrastiveViewGenerator: v1={v1.shape}, v2={v2.shape}")
    print(f"Views are different: {not torch.equal(v1, v2)}")
