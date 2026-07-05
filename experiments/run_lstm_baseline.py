"""
Classical LSTM baseline training and evaluation pipeline for Human Activity Recognition.

Loads any of the 6 benchmark datasets, trains an LSTM classifier,
selects the best model based on validation performance, and evaluates on the test set.

Usage:
  python experiments/run_lstm_baseline.py --dataset ucihar --epochs 50 --batch_size 128
"""

import os
import sys
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np

# Adjust python path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.har_datasets_paper import get_paper_dataloaders
from src.models.lstm_baseline import LSTMClassifier


def train_eval_lstm(args):
    print("=" * 60)
    print(f"Classical LSTM Baseline — Dataset: {args.dataset.upper()}")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1. Load dataset splits (64/16/20)
    print(f"\nLoading dataset '{args.dataset}'...")
    train_loader, val_loader, test_loader, in_channels, window_size, num_classes = (
        get_paper_dataloaders(
            dataset_name=args.dataset,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            transform=None,
            seed=args.seed,
            use_sampler=True,  # Match QCL/MPSQCL class balancing sampler
        )
    )

    print(
        f"Input channels: {in_channels} | Window size: {window_size} | Num classes: {num_classes}"
    )

    # 2. Initialize Model
    model = LSTMClassifier(
        in_channels=in_channels,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=num_classes,
        dropout_p=args.dropout,
        bidirectional=args.bidirectional,
        n_ref=args.n_ref,
        block_size=args.block_size,
    ).to(device)

    print("\nModel architecture:")
    print(model)
    print(f"Total model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 3. Loss & Optimizer & Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Output file paths
    dataset_suffix = args.dataset.lower()
    best_model_path = os.path.join(
        os.path.dirname(args.output_file), f"lstm_best_model_{dataset_suffix}.pt"
    )
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    # 4. Training Loop
    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
    }

    best_val_f1 = 0.0
    best_epoch = 0
    total_start_time = time.time()

    print(f"\nStarting training for {args.epochs} epochs...")

    for epoch in range(args.epochs):
        epoch_start_time = time.time()

        # --- Train ---
        model.train()
        running_train_loss = 0.0
        all_train_preds, all_train_targets = [], []

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            logits = model(inputs)
            loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(logits, 1)
            all_train_preds.extend(preds.cpu().numpy())
            all_train_targets.extend(targets.cpu().numpy())

        train_loss = running_train_loss / len(train_loader.dataset)
        train_acc = accuracy_score(all_train_targets, all_train_preds)
        train_f1 = f1_score(all_train_targets, all_train_preds, average="macro")

        # --- Validate ---
        model.eval()
        running_val_loss = 0.0
        all_val_preds, all_val_targets = [], []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits = model(inputs)
                loss = criterion(logits, targets)

                running_val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(logits, 1)
                all_val_preds.extend(preds.cpu().numpy())
                all_val_targets.extend(targets.cpu().numpy())

        val_loss = running_val_loss / len(val_loader.dataset)
        val_acc = accuracy_score(all_val_targets, all_val_preds)
        val_f1 = f1_score(all_val_targets, all_val_preds, average="macro")

        scheduler.step()
        epoch_time = time.time() - epoch_start_time

        # Save history
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_f1"].append(train_f1)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc * 100:.2f}%, F1: {train_f1:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc * 100:.2f}%, F1: {val_f1:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Track best model based on validation Macro F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_f1": val_f1,
                    "val_acc": val_acc,
                    "args": vars(args),
                },
                best_model_path,
            )
            print(f"  --> Saved new best model checkpoint to {best_model_path}")

    total_training_time = time.time() - total_start_time
    print(
        f"\nTraining completed in {total_training_time:.1f}s ({total_training_time / 60:.2f} minutes)"
    )

    # 5. Evaluate Best Model on Test Set
    print(
        f"\nLoading best model checkpoint from epoch {best_epoch} for final test evaluation..."
    )
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_loss = 0.0
    all_test_preds, all_test_targets = [], []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)

            test_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(logits, 1)
            all_test_preds.extend(preds.cpu().numpy())
            all_test_targets.extend(targets.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    test_acc = accuracy_score(all_test_targets, all_test_preds)
    test_f1 = f1_score(all_test_targets, all_test_preds, average="macro")
    test_cm = confusion_matrix(all_test_targets, all_test_preds)

    print("=" * 60)
    print("Test Evaluation Complete!")
    print(f"Test Loss:            {test_loss:.4f}")
    print(f"Test Accuracy:        {test_acc * 100:.2f}%")
    print(f"Test Macro F1-score:  {test_f1:.4f}")
    print("Confusion Matrix:")
    print(test_cm)
    print("=" * 60)

    # Save results to a log file
    results_path = args.output_file.replace(".txt", f"_{dataset_suffix}.txt")
    with open(results_path, "w") as f:
        f.write("=== Classical LSTM Baseline Results ===\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Learning Rate: {args.lr}\n")
        f.write(f"Hidden Dim: {args.hidden_dim}\n")
        f.write(f"Num Layers: {args.num_layers}\n")
        f.write(f"Bidirectional: {args.bidirectional}\n")
        f.write(f"Dropout: {args.dropout}\n")
        f.write(f"Total Training Time: {total_training_time:.2f}s\n\n")
        f.write(f"Best Val Epoch: {best_epoch}\n")
        f.write(f"Best Val Macro F1: {best_val_f1:.4f}\n")
        f.write(f"Final Test Accuracy: {test_acc * 100:.2f}%\n")
        f.write(f"Final Test F1 (Macro): {test_f1:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(test_cm))
        f.write("\n\nEpoch History:\n")
        f.write("Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Val F1\n")
        f.write("-" * 70 + "\n")
        for i in range(len(history["epoch"])):
            f.write(
                f"{history['epoch'][i]:<6}| "
                f"{history['train_loss'][i]:<11.4f}| "
                f"{history['train_acc'][i] * 100:<10.2f}| "
                f"{history['val_loss'][i]:<9.4f}| "
                f"{history['val_acc'][i] * 100:<8.2f}| "
                f"{history['val_f1'][i]:.4f}\n"
            )

    print(f"\nSaved results to: {results_path}")


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_data_dir = os.path.join(project_root, "data")
    default_output = os.path.join(project_root, "results", "lstm_results.txt")

    parser = argparse.ArgumentParser(description="Classical LSTM Baseline for HAR")
    parser.add_argument(
        "--dataset",
        type=str,
        default="ucihar",
        choices=["ucihar", "shar", "hhar", "motionsense", "uschad", "mobiact"],
        help="Dataset name",
    )
    parser.add_argument("--data_dir", type=str, default=default_data_dir)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument(
        "--bidirectional", action="store_true", help="Use bidirectional LSTM"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_ref", type=int, default=20, help="Number of reference vectors for QK-LSTM")
    parser.add_argument("--block_size", type=int, default=2, help="Block size for BPS kernel")
    parser.add_argument("--output_file", type=str, default=default_output)

    args = parser.parse_args()

    train_eval_lstm(args)


if __name__ == "__main__":
    main()
