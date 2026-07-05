"""
Training and evaluation pipeline for pre-trained MPSQCL Encoder + LSTM Classifier.

Loads any of the 6 benchmark datasets, initializes the hybrid MPSQCL-LSTM model,
loads a pre-trained MPSQCL encoder checkpoint, and trains the LSTM head.

Usage:
  python experiments/run_mpsqcl_lstm.py --dataset ucihar --epochs 50 --batch_size 128
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
from src.models.encoder import HAREncoder, HAREncoderPaper
from src.models.mpsqcl_lstm import MPSQCLLSTMClassifier


def train_eval_mpsqcl_lstm(args):
    print("=" * 60)
    print(f"MPSQCL Encoder + LSTM Classifier — Dataset: {args.dataset.upper()}")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1. Load dataset splits
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

    # 2. Initialize Encoder
    if args.use_paper_encoder:
        print("\nInitializing paper-compliant encoder (HAREncoderPaper)...")
        encoder = HAREncoderPaper(in_channels=in_channels, feature_dim=256).to(device)
    else:
        print("\nInitializing standard encoder (HAREncoder)...")
        encoder = HAREncoder(in_channels=in_channels, feature_dim=256).to(device)

    # Load pre-trained weights if checkpoint is provided
    dataset_suffix = args.dataset.lower()
    checkpoint_path = args.checkpoint

    if checkpoint_path and checkpoint_path.lower() == "none":
        checkpoint_path = None
        print("Explicitly training from scratch: no encoder checkpoint will be loaded.")
    elif not checkpoint_path or checkpoint_path.lower() == "default":
        # Construct default checkpoint path depending on encoder type
        if args.use_paper_encoder:
            checkpoint_name = f"mpsqcl_paper_encoder_best_{dataset_suffix}.pt"
        else:
            checkpoint_name = f"mpsqcl_encoder_pretrained_{dataset_suffix}.pt"
        checkpoint_path = os.path.join(args.results_dir, checkpoint_name)

    # Fallback to epoch checkpoints if default path does not exist
    if checkpoint_path and not os.path.exists(checkpoint_path):
        if not args.use_paper_encoder:
            fallback_names = [
                f"mpsqcl_checkpoint_epoch150_{dataset_suffix}.pt",
                f"mpsqcl_checkpoint_epoch120_{dataset_suffix}.pt",
                f"mpsqcl_checkpoint_epoch100_{dataset_suffix}.pt",
            ]
            for name in fallback_names:
                p = os.path.join(args.results_dir, name)
                if os.path.exists(p):
                    print(f"Default pretrained weights not found. Falling back to: {p}")
                    checkpoint_path = p
                    break

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading pre-trained encoder weights from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        if isinstance(ckpt, dict) and "encoder" in ckpt:
            encoder.load_state_dict(ckpt["encoder"])
        else:
            encoder.load_state_dict(ckpt)
        print("Pre-trained encoder weights loaded successfully!")
    elif checkpoint_path:
        print(
            f"WARNING: Pre-trained encoder checkpoint NOT found at {checkpoint_path}."
        )
        print("Training from scratch or with randomly initialized encoder.")
    else:
        # User explicitly specified "none", so checkpoint_path is None
        print("Training from scratch: no encoder checkpoint loaded.")

    # Freeze or keep unfrozen
    if args.freeze_encoder:
        print("Freezing encoder parameters (requires_grad = False).")
        for param in encoder.parameters():
            param.requires_grad = False
        encoder.eval()
    else:
        print("Fine-tuning encoder parameters (requires_grad = True).")
        encoder.train()

    # 3. Initialize Hybrid Model
    model = MPSQCLLSTMClassifier(
        encoder=encoder,
        lstm_hidden_dim=args.hidden_dim,
        lstm_layers=args.num_layers,
        num_classes=num_classes,
        bidirectional=args.bidirectional,
        dropout_p=args.dropout,
        use_pool=args.use_pool,
        normalize_features=args.normalize_features,
        n_ref=args.n_ref,
        block_size=args.block_size,
        use_classical_lstm=args.classical_lstm,
    ).to(device)

    print("\nModel architecture:")
    print(model)
    print(f"Total model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"Trainable parameters:   {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    # 4. Loss & Optimizer & Scheduler
    criterion = nn.CrossEntropyLoss()

    # Configure optimizer with optional differential learning rates
    if not args.freeze_encoder and args.encoder_lr_factor != 1.0:
        print(
            f"Using differential learning rates: Encoder LR = {args.lr * args.encoder_lr_factor:.2e}, LSTM/Classifier LR = {args.lr:.2e}"
        )
        encoder_params = list(model.encoder.parameters())
        other_params = list(model.lstm.parameters()) + list(model.fc.parameters())

        optimizer = optim.Adam(
            [
                {"params": encoder_params, "lr": args.lr * args.encoder_lr_factor},
                {"params": other_params, "lr": args.lr},
            ],
            weight_decay=1e-5,
        )
    else:
        # Optimize only the parameters that require grad
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.Adam(trainable_params, lr=args.lr, weight_decay=1e-5)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Output file paths
    checkpoint_suffix = ""
    if args.checkpoint and "depth2" in args.checkpoint:
        checkpoint_suffix = "_depth2"
    elif args.checkpoint and "depth1" in args.checkpoint:
        checkpoint_suffix = "_depth1"
    
    best_model_path = os.path.join(
        args.results_dir, f"mpsqcl_lstm_best_model{checkpoint_suffix}_{dataset_suffix}.pt"
    )
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    # 5. Training Loop
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

    print(f"\nStarting training for {args.epochs} epochs (use_pool={args.use_pool})...")

    for epoch in range(args.epochs):
        epoch_start_time = time.time()

        # --- Train ---
        if args.freeze_encoder:
            encoder.eval()  # Make sure encoder stays in eval mode
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

        # Track best model based on validation Macro F1 (aligning with the paper compliant metric)
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

    # 6. Evaluate Best Model on Test Set
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
    suffix = ""
    if args.checkpoint and "depth2" in args.checkpoint:
        suffix = "_depth2"
    elif args.checkpoint and "depth1" in args.checkpoint:
        suffix = "_depth1"
        
    results_path = os.path.join(
        args.results_dir, f"mpsqcl_lstm_results{suffix}_{dataset_suffix}.txt"
    )
    with open(results_path, "w") as f:
        f.write("=== MPSQCL Encoder + LSTM Classifier Results ===\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Pre-trained checkpoint loaded: {checkpoint_path}\n")
        f.write(f"Freeze Encoder: {args.freeze_encoder}\n")
        f.write(f"Use Pool Mode: {args.use_pool}\n")
        f.write(f"Normalize Features: {args.normalize_features}\n")
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


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_data_dir = os.path.join(project_root, "data")
    default_results_dir = os.path.join(project_root, "results")

    parser = argparse.ArgumentParser(
        description="MPSQCL Encoder + LSTM Classifier Training"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ucihar",
        choices=["ucihar", "shar", "hhar", "motionsense", "uschad", "mobiact", "opportunity", "opportunity_gestures"],
        help="Dataset name",
    )
    parser.add_argument("--data_dir", type=str, default=default_data_dir)
    parser.add_argument("--results_dir", type=str, default=default_results_dir)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="default",
        help="Path to pre-trained encoder checkpoint (use 'none' to train from scratch)",
    )
    parser.add_argument(
        "--use_paper_encoder",
        type=str2bool,
        default=False,
        help="Use paper-compliant HAREncoderPaper instead of standard HAREncoder",
    )
    parser.add_argument(
        "--freeze_encoder",
        type=str2bool,
        default=True,
        help="Freeze the encoder weights",
    )
    parser.add_argument(
        "--use_pool",
        action="store_true",
        help="Use global-pooled features (otherwise sequence features)",
    )
    parser.add_argument(
        "--normalize_features",
        type=str2bool,
        default=True,
        help="L2 normalize features before LSTM",
    )
    parser.add_argument(
        "--encoder_lr_factor",
        type=float,
        default=0.1,
        help="Learning rate factor for the encoder relative to head (default: 0.1)",
    )

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
    parser.add_argument(
        "--classical_lstm",
        action="store_true",
        help="Use standard PyTorch nn.LSTM instead of QKLSTM",
    )

    args = parser.parse_args()

    train_eval_mpsqcl_lstm(args)


if __name__ == "__main__":
    main()
