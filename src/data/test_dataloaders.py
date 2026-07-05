import os
import sys

# Adjust python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from src.data.har_datasets_paper import get_paper_dataloaders  # noqa: E402

data_dir = os.path.join(project_root, "data")
datasets = ["ucihar", "shar", "hhar", "motionsense", "uschad", "mobiact"]

print("=== VERIFYING ALL 6 HAR DATASETS ===")
for name in datasets:
    print("-" * 50)
    print(f"Loading {name.upper()}...")
    try:
        train_loader, val_loader, test_loader, in_channels, window_size, num_classes = (
            get_paper_dataloaders(
                dataset_name=name, data_dir=data_dir, batch_size=128, seed=42
            )
        )

        print(f"Successfully loaded {name.upper()}!")
        print(f"  Channels: {in_channels}")
        print(f"  Window Size: {window_size}")
        print(f"  Num Classes: {num_classes}")
        print(
            f"  Train batches: {len(train_loader)} (total samples: {len(train_loader.dataset)})"
        )
        print(
            f"  Val batches:   {len(val_loader)} (total samples: {len(val_loader.dataset)})"
        )
        print(
            f"  Test batches:  {len(test_loader)} (total samples: {len(test_loader.dataset)})"
        )

        # Take a batch
        for X, y in train_loader:
            print(f"  Batch X shape: {X.shape}")
            print(f"  Batch y shape: {y.shape}")
            break
    except Exception as e:
        print(f"  Error loading {name}: {e}")

print("-" * 50)
print("Verification complete!")
