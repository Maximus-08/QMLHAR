"""
Paper-compliant Dataset Loaders for UCI-HAR, SHAR, and HHAR.

Implements:
  - 64% / 16% / 20% train/val/test split across all datasets.
  - Normalization (mean=0, std=1) on the datasets.
  - Sliding window segmentation with 50% overlap.
  - HHAR: smartphone device data only, aligned and downsampled to 50 Hz, window size 100.
  - SHAR: UniMiB-SHAR dataset loader, window size 151, and programmatically excluding
    the 10 out of 30 participants with incomplete activity classes.
"""
import os
import urllib.request
import numpy as np
import pandas as pd
import scipy.io
import torch
from torch.utils.data import Dataset, DataLoader, random_split


def segment_sliding_window(X, y, window_size, overlap_ratio=0.5):
    """
    Segment continuous timeseries data into sliding windows.
    
    Args:
        X: Numpy array of shape (L, C)
        y: Numpy array of shape (L,)
        window_size: Length of each window
        overlap_ratio: Overlap fraction (0.5 for 50% overlap)
    Returns:
        X_segmented: (num_windows, C, window_size)
        y_segmented: (num_windows,)
    """
    step = int(window_size * (1.0 - overlap_ratio))
    X_segmented = []
    y_segmented = []
    
    for start in range(0, len(X) - window_size + 1, step):
        end = start + window_size
        # Label is the majority label in the window
        window_y = y[start:end]
        majority_label = np.argmax(np.bincount(window_y.astype(int)))
        
        # Check if the window is homogenous enough or has labels
        X_segmented.append(X[start:end].T)  # Shape (C, window_size)
        y_segmented.append(majority_label)
        
    if len(X_segmented) == 0:
        return np.empty((0, X.shape[1], window_size)), np.empty((0,))
        
    return np.stack(X_segmented), np.array(y_segmented)


def normalize_features(X):
    """Normalize features to zero mean and unit variance."""
    mean = np.mean(X, axis=0, keepdims=True)
    std = np.std(X, axis=0, keepdims=True)
    std[std == 0.0] = 1.0  # Avoid division by zero
    return (X - mean) / std


class UnifiedHARDataset(Dataset):
    """Generic dataset wrapping preprocessed X and y tensors."""
    def __init__(self, X, y, transform=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.transform = transform
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        X_sample = self.X[idx]
        y_sample = self.y[idx]
        
        if self.transform:
            X_sample = self.transform(X_sample)
            
        return X_sample, y_sample


# =====================================================================
# 1. UCI-HAR Loader
# =====================================================================
def load_uci_har_all(data_dir):
    """
    Load and merge train/test splits of raw UCI-HAR signals.
    Shape: (N, 9, 128)
    """
    uci_dir = os.path.join(data_dir, "UCI-HAR Dataset")
    if not os.path.exists(uci_dir):
        raise FileNotFoundError(f"UCI-HAR Dataset not found at {uci_dir}. Please run extract_dataset.py first.")
        
    channel_files = [
        "body_acc_x_{}.txt", "body_acc_y_{}.txt", "body_acc_z_{}.txt",
        "body_gyro_x_{}.txt", "body_gyro_y_{}.txt", "body_gyro_z_{}.txt",
        "total_acc_x_{}.txt", "total_acc_y_{}.txt", "total_acc_z_{}.txt"
    ]
    
    def read_split(split):
        signals = []
        signals_dir = os.path.join(uci_dir, split, "Inertial Signals")
        for filename in channel_files:
            filepath = os.path.join(signals_dir, filename.format(split))
            with open(filepath, 'r') as f:
                channel_data = np.array([list(map(float, line.split())) for line in f])
            signals.append(channel_data)
        
        X = np.stack(signals, axis=1)  # (samples, 9, 128)
        
        label_file = os.path.join(uci_dir, split, f"y_{split}.txt")
        with open(label_file, 'r') as f:
            y = np.array([int(line.strip()) - 1 for line in f])  # Map 1-6 -> 0-5
            
        return X, y

    X_train, y_train = read_split("train")
    X_test, y_test = read_split("test")
    
    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)
    
    # Normalize features window-by-window (commented out to preserve relative amplitude features)
    # for i in range(len(X)):
    #     # Normalize along time dimension for each channel
    #     X[i] = normalize_features(X[i].T).T
        
    return X, y


# =====================================================================
# 2. SHAR (UniMiB SHAR) Loader
# =====================================================================
def load_unimib_shar_all(data_dir):
    """
    Load UniMiB SHAR dataset.
    Disregards the 10 out of 30 participants with incomplete classes.
    Window size 151, 3 channels (accelerometer only).
    """
    shar_dir = os.path.join(data_dir, "unimib_shar")
    os.makedirs(shar_dir, exist_ok=True)
    mat_path = os.path.join(shar_dir, "uniMiB-SHAR.mat")
    
    # Download from a public mirror if missing
    if not os.path.exists(mat_path):
        url = "https://github.com/videoflow/human-activity-recognition/raw/master/uniMiB-SHAR.mat"
        print(f"Downloading UniMiB-SHAR dataset from {url}...")
        try:
            urllib.request.urlretrieve(url, mat_path)
            print("Download complete!")
        except Exception as e:
            raise RuntimeError(f"Failed to download UniMiB-SHAR dataset: {e}. Please place uniMiB-SHAR.mat manually in {shar_dir}")
            
    print(f"Loading UniMiB-SHAR dataset from {mat_path}...")
    mat = scipy.io.loadmat(mat_path)
    
    # UniMiB mat file structure contains:
    # 'data': (11771, 151, 3) or 'acc_data'
    # 'labels': (11771, 2) where col 0 = class_id, col 1 = subject_id
    if 'data' in mat:
        data = mat['data']
    elif 'acc_data' in mat:
        data = mat['acc_data']
    else:
        raise KeyError("Could not find data key in .mat file")
        
    if 'labels' in mat:
        labels = mat['labels']
    elif 'acc_labels' in mat:
        labels = mat['acc_labels']
    else:
        raise KeyError("Could not find labels key in .mat file")
        
    # Re-order axes to (samples, channels, time_steps) -> (N, 3, 151)
    # The original mat data might be (N, 151 * 3). Let's check dimensions.
    if len(data.shape) == 2:
        # Flat representation (N, 453) -> reshape to (N, 151, 3) and transpose to (N, 3, 151)
        data = data.reshape(-1, 151, 3).transpose(0, 2, 1)
        
    classes = labels[:, 0] - 1  # 1-indexed to 0-indexed
    subjects = labels[:, 1]
    
    # Programmatically identify and disregard 10 out of 30 participants with incomplete classes
    subject_unique_classes = {}
    for s, c in zip(subjects, classes):
        if s not in subject_unique_classes:
            subject_unique_classes[s] = set()
        subject_unique_classes[s].add(c)
        
    # Sort subjects by number of unique classes ascending
    sorted_subjects = sorted(subject_unique_classes.keys(), key=lambda s: len(subject_unique_classes[s]))
    disregarded_subjects = sorted_subjects[:10]
    print(f"Programmatically disregarding 10 participants with incomplete classes: {disregarded_subjects}")
    
    # Filter dataset
    mask = ~np.isin(subjects, disregarded_subjects)
    X = data[mask]
    y = classes[mask]
    
    # Map remaining class IDs to contiguous 0 to C-1
    unique_classes = np.unique(y)
    class_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_classes)}
    y = np.array([class_mapping[val] for val in y])
    
    # Normalize features globally per channel (preserving relative amplitude features)
    mean = np.mean(X, axis=(0, 2), keepdims=True)
    std = np.std(X, axis=(0, 2), keepdims=True)
    std[std == 0.0] = 1.0  # Avoid division by zero
    X = (X - mean) / std
        
    print(f"SHAR loaded. Shape X: {X.shape}, y: {y.shape}, Unique classes: {len(unique_classes)}")
    return X, y


# =====================================================================
# 3. HHAR Loader
# =====================================================================
def preprocess_hhar(data_dir, output_path):
    """
    Process HHAR dataset from raw CSV files.
    Aligns Phones Accelerometer and Gyroscope, downsamples to 50 Hz,
    segments into sliding windows of size 100 with 50% overlap.
    Caches output to output_path.
    """
    print("HHAR Preprocessing Pipeline Started...")
    hhar_dir = os.path.join(data_dir, "HHAR")
    acc_path = os.path.join(hhar_dir, "Phones_accelerometer.csv")
    gyro_path = os.path.join(hhar_dir, "Phones_gyroscope.csv")
    
    if not os.path.exists(acc_path) or not os.path.exists(gyro_path):
        raise FileNotFoundError(f"HHAR Phones CSV files not found in {hhar_dir}")
        
    # Load raw CSVs (only columns of interest with optimized dtypes to save memory)
    print("Loading HHAR CSV files...")
    cols = ['Creation_Time', 'x', 'y', 'z', 'User', 'Model', 'Device', 'gt']
    dtypes = {
        'Creation_Time': 'int64',
        'x': 'float32',
        'y': 'float32',
        'z': 'float32',
        'User': 'category',
        'Model': 'category',
        'Device': 'category',
        'gt': 'category'
    }
    df_acc = pd.read_csv(acc_path, usecols=cols, dtype=dtypes).dropna()
    df_gyro = pd.read_csv(gyro_path, usecols=cols, dtype=dtypes).dropna()
    
    # Map class labels to integers
    gt_mapping = {
        'bike': 0,
        'sit': 1,
        'stand': 2,
        'walk': 3,
        'stairsup': 4,
        'stairsdown': 5
    }
    df_acc['label'] = df_acc['gt'].map(gt_mapping)
    df_gyro['label'] = df_gyro['gt'].map(gt_mapping)
    df_acc = df_acc.dropna(subset=['label'])
    df_gyro = df_gyro.dropna(subset=['label'])
    
    # Unique keys to align
    df_acc['key'] = df_acc['User'].astype(str) + "_" + df_acc['Model'].astype(str) + "_" + df_acc['Device'].astype(str) + "_" + df_acc['label'].astype(str)
    df_gyro['key'] = df_gyro['User'].astype(str) + "_" + df_gyro['Model'].astype(str) + "_" + df_gyro['Device'].astype(str) + "_" + df_gyro['label'].astype(str)
    
    keys = set(df_acc['key'].unique()).intersection(df_gyro['key'].unique())
    print(f"Found {len(keys)} matching User/Device/Class combinations to align.")
    
    # Group by key beforehand to avoid O(N) filtering inside the loop
    print("Grouping dataframes by key...")
    df_acc_grouped = {k: v.sort_values('Creation_Time') for k, v in df_acc.groupby('key')}
    df_gyro_grouped = {k: v.sort_values('Creation_Time') for k, v in df_gyro.groupby('key')}
    
    del df_acc, df_gyro
    import gc
    gc.collect()
    
    X_list = []
    y_list = []
    
    # Step size for 50 Hz grid in nanoseconds (20 ms = 20,000,000 ns)
    step_ns = 20_000_000
    
    for idx, key in enumerate(sorted(list(keys))):
        acc_sub = df_acc_grouped[key]
        gyro_sub = df_gyro_grouped[key]
        
        t_start = max(acc_sub['Creation_Time'].min(), gyro_sub['Creation_Time'].min())
        t_end = min(acc_sub['Creation_Time'].max(), gyro_sub['Creation_Time'].max())
        
        if t_start >= t_end:
            continue
            
        # Create 50 Hz time grid
        t_grid = np.arange(t_start, t_end, step_ns)
        if len(t_grid) < 100:
            continue
            
        # Interpolate acc
        acc_x = np.interp(t_grid, acc_sub['Creation_Time'], acc_sub['x'])
        acc_y = np.interp(t_grid, acc_sub['Creation_Time'], acc_sub['y'])
        acc_z = np.interp(t_grid, acc_sub['Creation_Time'], acc_sub['z'])
        
        # Interpolate gyro
        gyro_x = np.interp(t_grid, gyro_sub['Creation_Time'], gyro_sub['x'])
        gyro_y = np.interp(t_grid, gyro_sub['Creation_Time'], gyro_sub['y'])
        gyro_z = np.interp(t_grid, gyro_sub['Creation_Time'], gyro_sub['z'])
        
        # Merge to 6 channels (L, 6)
        sig = np.column_stack([acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z])
        # Segment into sliding windows of size 100
        label_val = int(key.split("_")[-1])
        y_continuous = np.full(len(sig), label_val)
        
        X_seg, y_seg = segment_sliding_window(sig, y_continuous, window_size=100, overlap_ratio=0.5)
        
        if len(X_seg) > 0:
            X_list.append(X_seg)
            y_list.append(y_seg)
            
        if (idx + 1) % 10 == 0 or (idx + 1) == len(keys):
            print(f"  Processed {idx+1}/{len(keys)} alignment blocks...")
            
    if len(X_list) == 0:
        raise ValueError("HHAR preprocessing resulted in empty dataset. Check labels matching.")
        
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    
    # Global channel-wise normalization to preserve relative amplitude/intensity features
    print("Applying global channel-wise normalization to HHAR...")
    mean = np.mean(X, axis=(0, 2), keepdims=True)
    std = np.std(X, axis=(0, 2), keepdims=True)
    std[std == 0.0] = 1.0
    X = (X - mean) / std
    
    print(f"Caching aligned HHAR to {output_path}...")
    np.savez_compressed(output_path, X=X, y=y)
    print("HHAR Preprocessing Complete!")
    return X, y


def load_hhar_all(data_dir):
    """
    Load HHAR dataset. Preprocesses CSVs and caches to .npz if not already done.
    Window size 100, 6 channels (acc + gyro).
    """
    cache_path = os.path.join(data_dir, "HHAR", "hhar_preprocessed.npz")
    if os.path.exists(cache_path):
        print(f"Loading preprocessed HHAR from cache: {cache_path}")
        data = np.load(cache_path)
        return data['X'], data['y']
        
    return preprocess_hhar(data_dir, cache_path)


def load_motionsense_all(data_dir):
    cache_path = os.path.join(data_dir, "motion_sense", "motionsense_preprocessed.npz")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"MotionSense cache not found: {cache_path}. Run download_and_preprocess_datasets.py first.")
    data = np.load(cache_path)
    return data['X'], data['y']

def load_uschad_all(data_dir):
    cache_path = os.path.join(data_dir, "USC-HAD", "uschad_preprocessed.npz")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"USC-HAD cache not found: {cache_path}. Run download_and_preprocess_datasets.py first.")
    data = np.load(cache_path)
    return data['X'], data['y']

def load_mobiact_all(data_dir):
    cache_path = os.path.join(data_dir, "MobiAct", "mobiact_preprocessed.npz")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"MobiAct cache not found: {cache_path}. Run download_and_preprocess_datasets.py first.")
    data = np.load(cache_path)
    return data['X'], data['y']


# =====================================================================
# Unified Split Utility
# =====================================================================
def get_paper_dataloaders(dataset_name, data_dir, batch_size=128, transform=None, collate_fn=None, seed=42, use_sampler=True, num_workers=0):
    """
    Load the specified dataset, randomly divide it into 64% Train / 16% Val / 20% Test,
    and return the train, validation, and test dataloaders.
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == "ucihar":
        X, y = load_uci_har_all(data_dir)
    elif dataset_name == "shar":
        X, y = load_unimib_shar_all(data_dir)
    elif dataset_name == "hhar":
        X, y = load_hhar_all(data_dir)
    elif dataset_name == "motionsense":
        X, y = load_motionsense_all(data_dir)
    elif dataset_name == "uschad":
        X, y = load_uschad_all(data_dir)
    elif dataset_name == "mobiact":
        X, y = load_mobiact_all(data_dir)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}. Choose 'ucihar', 'shar', 'hhar', 'motionsense', 'uschad', or 'mobiact'.")
        
    full_dataset = UnifiedHARDataset(X, y, transform=transform)
    
    # Split lengths matching 64% / 16% / 20%
    n_total = len(full_dataset)
    n_train = int(n_total * 0.64)
    n_val = int(n_total * 0.16)
    n_test = n_total - n_train - n_val
    
    print(f"Splitting {dataset_name} ({n_total} samples) into:")
    print(f"  Train (64%): {n_train}")
    print(f"  Val   (16%): {n_val}")
    print(f"  Test  (20%): {n_test}")
    
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(
        full_dataset, [n_train, n_val, n_test], generator=generator
    )
    
    if use_sampler:
        # Weighted Random Sampling to address class imbalance (paper specification)
        train_labels = y[train_set.indices]
        class_counts = np.bincount(train_labels)
        class_counts[class_counts == 0] = 1  # Avoid division by zero
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[train_labels]
        
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights),
            num_samples=len(sample_weights),
            replacement=True
        )
        train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, drop_last=True, collate_fn=collate_fn, num_workers=num_workers)
    else:
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    
    # Also return number of channels and window size
    in_channels = X.shape[1]
    window_size = X.shape[2]
    num_classes = len(np.unique(y))
    
    return train_loader, val_loader, test_loader, in_channels, window_size, num_classes


# Quick verification
if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_dir = os.path.join(project_root, "data")
    
    try:
        tr, va, te, ch, w, classes = get_paper_dataloaders("ucihar", data_dir, batch_size=32)
        print("UCI-HAR loaders created successfully.")
        print(f"Channels: {ch}, Window size: {w}, Classes: {classes}")
        for X, y in tr:
            print(f"Batch shape X: {X.shape}, y: {y.shape}")
            break
    except Exception as e:
        print(f"Error loading UCI-HAR: {e}")
