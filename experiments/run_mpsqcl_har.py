"""
Multi-Positive Sample Quantum Contrastive Learning for Human Activity Recognition (MPSQCL HAR).

Two-phase training pipeline:
  Phase 1 (pretrain):  Self-supervised contrastive pre-training using M augmented views
                       of unlabeled sensor data. Encoder + Quantum Projection Head are
                       trained with MPSQCLLoss (no labels used).
  Phase 2 (finetune):  Supervised fine-tuning. Load pre-trained encoder, replace quantum
                       head with a linear classifier, and train on labeled data.

Reference: MPSQCL (Ren et al., 2024)

Usage:
  # Phase 1: Pre-train
  python experiments/run_mpsqcl_har.py --phase pretrain --epochs 10 --n_views 4 --subset_fraction 0.5

  # Phase 2: Fine-tune
  python experiments/run_mpsqcl_har.py --phase finetune --epochs 20 --checkpoint results/mpsqcl_encoder_pretrained.pt
"""

import os
import sys
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np

# Adjust python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.har_datasets_paper import get_paper_dataloaders
from src.data.augmentations import ContrastiveViewGenerator
from src.models.encoder import HAREncoder
from src.models.quantum_head import QuantumProjectionHead, QuantumProjectionHeadPaper
from src.losses.mpsqcl_loss import MPSQCLLoss


def mps_contrastive_collate_fn(batch):
    """
    Custom collate function for MPSQCL pre-training.

    When using ContrastiveViewGenerator as transform with n_views > 2,
    each item in the batch is:
        ((view_1, view_2, ..., view_M), label)

    We stack the views for each view index separately.
    """
    n_views = len(batch[0][0])
    views_list = [[] for _ in range(n_views)]
    labels = []

    for views, label in batch:
        for idx in range(n_views):
            views_list[idx].append(views[idx])
        labels.append(label)

    stacked_views = [torch.stack(v) for v in views_list]
    return tuple(stacked_views), torch.stack(labels)


# ============================================================
# Phase 1: Multi-Positive Contrastive Pre-Training
# ============================================================
def pretrain(args):
    """Self-supervised contrastive pre-training of encoder + quantum head using MPSQCL."""

    print("=" * 60)
    print(f"MPSQCL HAR — Phase 1: Contrastive Pre-Training ({args.n_views} views)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1. Data with contrastive augmentation transform
    print(f"\nLoading dataset '{args.dataset}' with {args.n_views} augmented views...")
    transform = ContrastiveViewGenerator(n_views=args.n_views)

    train_loader, _, _, in_channels, _, _ = get_paper_dataloaders(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        transform=transform,
        collate_fn=mps_contrastive_collate_fn,
        seed=args.seed,
        use_sampler=args.use_sampler,
        num_workers=args.num_workers,
    )

    # Subset if requested
    if args.subset_fraction < 1.0:
        n = len(train_loader.dataset)
        indices = np.random.choice(n, int(n * args.subset_fraction), replace=False)
        train_loader = DataLoader(
            Subset(train_loader.dataset, indices),
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=mps_contrastive_collate_fn,
            num_workers=0,
        )
        print(
            f"Using {len(indices)} samples ({args.subset_fraction * 100:.0f}% subset)"
        )

    # 2. Models
    print("\nInitializing encoder and quantum projection head...")
    encoder = HAREncoder(in_channels=in_channels, feature_dim=2**args.num_qubits).to(
        device
    )

    if args.use_paper_head:
        print("Using paper-compliant quantum head (Ry + CNOT ring)")
        quantum_head = QuantumProjectionHeadPaper(
            input_dim=2**args.num_qubits,
            num_qubits=args.num_qubits,
            q_layers=args.q_layers,
            device_type=args.device_type,
        ).to(device)
    else:
        print("Using standard StronglyEntanglingLayers quantum head")
        quantum_head = QuantumProjectionHead(
            input_dim=2**args.num_qubits,
            num_qubits=args.num_qubits,
            q_layers=args.q_layers,
            device_type=args.device_type,
        ).to(device)

    print(f"Encoder parameters:      {sum(p.numel() for p in encoder.parameters()):,}")
    print(
        f"Quantum head parameters: {sum(p.numel() for p in quantum_head.parameters()):,}"
    )

    # 3. Loss and Optimizer
    criterion = MPSQCLLoss(
        temperature=args.temperature,
        exclude_anchor_from_denominator=args.exclude_anchor,
    )

    # Combine parameters from both models
    all_params = list(encoder.parameters()) + list(quantum_head.parameters())
    optimizer = optim.Adam(all_params, lr=args.lr, weight_decay=1e-5)

    # 4. Resume from checkpoint if provided
    start_epoch = 0
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"\nResuming from checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        encoder.load_state_dict(ckpt["encoder"])
        quantum_head.load_state_dict(ckpt["quantum_head"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)
        # Reset learning rates to args.lr so CosineAnnealingLR initializes base_lrs correctly
        for g in optimizer.param_groups:
            g["lr"] = args.lr
        print(f"Resumed from epoch {start_epoch}")

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        last_epoch=start_epoch - 1 if start_epoch > 0 else -1,
    )

    # 5. Training loop
    print(f"\nStarting multi-positive pre-training for {args.epochs} epochs...")
    print(
        f"Temperature: {args.temperature}, LR: {args.lr}, Batch size: {args.batch_size}"
    )

    history = {"epoch": [], "loss": [], "lr": [], "epoch_time": []}
    total_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        encoder.train()
        quantum_head.train()

        epoch_loss = 0.0
        n_batches = 0
        epoch_start = time.time()

        for batch_idx, (batch_views, _labels) in enumerate(train_loader):
            # batch_views: Tuple of M tensors of shape (batch_size, 9, 128)
            views = [v.to(device) for v in batch_views]

            # Forward pass: extract features and map to VQC projections
            z_list = []
            for v in views:
                h = encoder(v)  # (batch, 256)
                z = quantum_head(h)  # (batch, 8)
                z_list.append(z)

            # Compute multi-positive sample loss
            loss = criterion(z_list)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(train_loader):
                elapsed = time.time() - epoch_start
                print(
                    f"  Epoch {epoch + 1}/{args.epochs} | "
                    f"Batch {batch_idx + 1}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Time: {elapsed:.1f}s"
                )

        scheduler.step()

        avg_loss = epoch_loss / n_batches
        epoch_time = time.time() - epoch_start
        current_lr = scheduler.get_last_lr()[0]

        history["epoch"].append(epoch + 1)
        history["loss"].append(avg_loss)
        history["lr"].append(current_lr)
        history["epoch_time"].append(epoch_time)

        print(
            f"\n  [Epoch {epoch + 1}] Avg Loss: {avg_loss:.4f} | "
            f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s\n"
        )

        # Save checkpoint every N epochs
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            ckpt_prefix = f"mpsqcl_checkpoint_depth{args.q_layers}" if args.q_layers != 3 else "mpsqcl_checkpoint"
            ckpt_path = os.path.join(
                os.path.dirname(args.output_file),
                f"{ckpt_prefix}_epoch{epoch + 1}_{args.dataset}.pt",
            )
            torch.save(
                {
                    "epoch": epoch + 1,
                    "encoder": encoder.state_dict(),
                    "quantum_head": quantum_head.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "loss": avg_loss,
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"  Saved checkpoint: {ckpt_path}")

    total_time = time.time() - total_start
    print(
        f"\nPre-training completed in {total_time:.1f}s ({total_time / 3600:.2f} hours)"
    )

    # Save final encoder weights (for fine-tuning)
    encoder_prefix = f"mpsqcl_encoder_pretrained_depth{args.q_layers}" if args.q_layers != 3 else "mpsqcl_encoder_pretrained"
    encoder_path = os.path.join(
        os.path.dirname(args.output_file),
        f"{encoder_prefix}_{args.dataset}.pt",
    )
    torch.save(encoder.state_dict(), encoder_path)
    print(f"Saved pre-trained encoder: {encoder_path}")

    # Save training history
    results_path = args.output_file.replace(".txt", f"_{args.dataset}_pretrain.txt")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        f.write("=== MPSQCL HAR Pre-Training Results ===\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Learning Rate: {args.lr}\n")
        f.write(f"Views (M): {args.n_views}\n")
        f.write(f"Temperature: {args.temperature}\n")
        f.write(f"Exclude Anchor in Denom: {args.exclude_anchor}\n")
        f.write(f"Num Qubits: {args.num_qubits}\n")
        f.write(f"Q Layers: {args.q_layers}\n")
        f.write(f"Subset Fraction: {args.subset_fraction}\n")
        f.write(f"Total Time: {total_time:.2f}s\n\n")
        f.write("Epoch | Loss     | LR         | Time (s)\n")
        f.write("-" * 50 + "\n")
        for i in range(len(history["epoch"])):
            f.write(
                f"{history['epoch'][i]:<6}| "
                f"{history['loss'][i]:<9.4f}| "
                f"{history['lr'][i]:<11.6f}| "
                f"{history['epoch_time'][i]:.1f}\n"
            )

    print(f"Saved pre-training results: {results_path}")
    return encoder_path


# ============================================================
# Phase 2: Supervised Fine-Tuning
# ============================================================
def finetune(args):
    """Supervised fine-tuning with pre-trained encoder + linear classifier."""

    print("=" * 60)
    print("MPSQCL HAR — Phase 2: Supervised Fine-Tuning")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1. Data (no augmentation for fine-tuning)
    print(f"\nLoading dataset '{args.dataset}'...")
    train_loader, _, test_loader, in_channels, _, num_classes = get_paper_dataloaders(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        transform=None,
        collate_fn=None,
        seed=args.seed,
        use_sampler=args.use_sampler,
        num_workers=args.num_workers,
    )

    if args.subset_fraction < 1.0:
        n = len(train_loader.dataset)
        indices = np.random.choice(n, int(n * args.subset_fraction), replace=False)
        train_loader = DataLoader(
            Subset(train_loader.dataset, indices),
            batch_size=args.batch_size,
            shuffle=True,
        )
        print(f"Using {len(indices)} train samples ({args.subset_fraction * 100:.0f}%)")

    # 2. Model: encoder + linear classifier
    feature_dim = 2**args.num_qubits
    encoder = HAREncoder(in_channels=in_channels, feature_dim=feature_dim).to(device)
    classifier = nn.Linear(feature_dim, num_classes).to(device)

    # Load pre-trained encoder
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"\nLoading pre-trained encoder from: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)

        # Handle both full checkpoint and encoder-only saves
        if isinstance(ckpt, dict) and "encoder" in ckpt:
            encoder.load_state_dict(ckpt["encoder"])
        else:
            encoder.load_state_dict(ckpt)
        print("Pre-trained encoder loaded successfully!")
    else:
        print(
            "\nWARNING: No checkpoint provided. Training encoder from scratch (no pre-training)."
        )

    print(f"Encoder parameters:    {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"Classifier parameters: {sum(p.numel() for p in classifier.parameters()):,}")

    # 3. Optimizer — lower LR for encoder (fine-tuning), higher for classifier
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        [
            {
                "params": encoder.parameters(),
                "lr": args.lr * 0.1,
            },  # Fine-tune encoder slowly
            {
                "params": classifier.parameters(),
                "lr": args.lr,
            },  # Train classifier faster
        ],
        weight_decay=1e-5,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 4. Training loop
    print(f"\nStarting fine-tuning for {args.epochs} epochs...")

    history = {
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "test_loss": [],
        "test_acc": [],
        "test_f1": [],
    }

    best_test_acc = 0.0
    best_epoch = 0
    total_start = time.time()

    for epoch in range(args.epochs):
        # --- Train ---
        encoder.train()
        classifier.train()

        running_loss = 0.0
        all_preds, all_targets = [], []
        epoch_start = time.time()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            features = encoder(inputs)  # (batch, 256)
            logits = classifier(features)  # (batch, 6)
            loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(logits, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(train_loader):
                print(
                    f"  Epoch {epoch + 1}/{args.epochs} | "
                    f"Batch {batch_idx + 1}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f}"
                )

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = accuracy_score(all_targets, all_preds)
        train_f1 = f1_score(all_targets, all_preds, average="macro")

        # --- Evaluate ---
        encoder.eval()
        classifier.eval()

        test_loss = 0.0
        test_preds, test_targets = [], []

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                features = encoder(inputs)
                logits = classifier(features)
                loss = criterion(logits, targets)

                test_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(logits, 1)
                test_preds.extend(preds.cpu().numpy())
                test_targets.extend(targets.cpu().numpy())

        test_loss /= len(test_loader.dataset)
        test_acc = accuracy_score(test_targets, test_preds)
        test_f1 = f1_score(test_targets, test_preds, average="macro")
        test_cm = confusion_matrix(test_targets, test_preds)

        scheduler.step()
        epoch_time = time.time() - epoch_start

        # Track history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_f1"].append(train_f1)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["test_f1"].append(test_f1)

        # Track best
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_test_f1 = test_f1
            best_epoch = epoch + 1
            best_cm = test_cm
            # Save best model
            best_path = os.path.join(
                os.path.dirname(args.output_file),
                f"mpsqcl_best_model_{args.dataset}.pt",
            )
            torch.save(
                {
                    "encoder": encoder.state_dict(),
                    "classifier": classifier.state_dict(),
                    "test_acc": test_acc,
                    "test_f1": test_f1,
                    "epoch": epoch + 1,
                },
                best_path,
            )

        print(
            f"\n  [Epoch {epoch + 1}] "
            f"Train: loss={train_loss:.4f}, acc={train_acc * 100:.2f}%, f1={train_f1:.4f} | "
            f"Test: loss={test_loss:.4f}, acc={test_acc * 100:.2f}%, f1={test_f1:.4f} | "
            f"Time: {epoch_time:.1f}s\n"
        )

    total_time = time.time() - total_start

    # Final summary
    print("=" * 60)
    print("MPSQCL Fine-Tuning Complete!")
    print(f"Total time: {total_time:.1f}s ({total_time / 3600:.2f} hours)")
    print(f"Best test accuracy: {best_test_acc * 100:.2f}% (Epoch {best_epoch})")
    print(f"Best test F1:       {best_test_f1:.4f}")
    print(f"Final test accuracy: {test_acc * 100:.2f}%")
    print(f"Final test F1:       {test_f1:.4f}")
    print("\nConfusion Matrix:")
    print(best_cm)
    print("=" * 60)

    # Save results
    results_path = args.output_file.replace(".txt", f"_{args.dataset}_finetune.txt")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        f.write("=== MPSQCL HAR Fine-Tuning Results ===\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Pre-trained checkpoint: {args.checkpoint}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Learning Rate: {args.lr}\n")
        f.write(f"Num Qubits: {args.num_qubits}\n")
        f.write(f"Subset Fraction: {args.subset_fraction}\n")
        f.write(f"Total Time: {total_time:.2f}s\n\n")
        f.write(
            f"Best Test Accuracy: {best_test_acc * 100:.2f}% (Epoch {best_epoch})\n"
        )
        f.write(f"Best Test F1 (Macro): {best_test_f1:.4f}\n")
        f.write(f"Final Test Accuracy: {test_acc * 100:.2f}%\n")
        f.write(f"Final Test F1 (Macro): {test_f1:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(best_cm))
        f.write("\n\nEpoch History:\n")
        f.write("Epoch | Train Loss | Train Acc | Test Loss | Test Acc | Test F1\n")
        f.write("-" * 70 + "\n")
        for i in range(len(history["train_loss"])):
            f.write(
                f"{i + 1:<6}| {history['train_loss'][i]:<11.4f}| "
                f"{history['train_acc'][i] * 100:<10.2f}| "
                f"{history['test_loss'][i]:<10.4f}| "
                f"{history['test_acc'][i] * 100:<9.2f}| "
                f"{history['test_f1'][i]:.4f}\n"
            )

    print(f"\nSaved fine-tuning results: {results_path}")


# ============================================================
# Main CLI
# ============================================================
def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_data_dir = os.path.join(project_root, "data")
    default_output = os.path.join(project_root, "results", "mpsqcl_results.txt")

    parser = argparse.ArgumentParser(
        description="MPSQCL HAR: Multi-Positive Sample Quantum Contrastive Learning for Human Activity Recognition"
    )

    # Phase
    parser.add_argument(
        "--phase",
        type=str,
        required=True,
        choices=["pretrain", "finetune"],
        help="Training phase: 'pretrain' (Phase 1) or 'finetune' (Phase 2)",
    )

    # Data
    parser.add_argument(
        "--dataset",
        type=str,
        default="ucihar",
        choices=["ucihar", "shar", "hhar", "motionsense", "uschad", "mobiact", "opportunity", "opportunity_gestures"],
        help="Dataset name",
    )
    parser.add_argument("--data_dir", type=str, default=default_data_dir)
    parser.add_argument(
        "--subset_fraction",
        type=float,
        default=1.0,
        help="Fraction of training data to use (0.0-1.0)",
    )

    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use_sampler",
        action="store_true",
        help="Use WeightedRandomSampler to address class imbalance",
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="Number of dataloader workers"
    )

    # Contrastive / Multi-Positive (Phase 1)
    parser.add_argument(
        "--n_views",
        type=int,
        default=4,
        help="Number of augmented views per sample (default 4)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="MPSQCL loss temperature parameter",
    )
    parser.add_argument(
        "--exclude_anchor",
        action="store_true",
        help="Exclude the anchor itself from the loss denominator",
    )

    # Quantum
    parser.add_argument(
        "--num_qubits",
        type=int,
        default=8,
        help="Number of qubits (feature_dim = 2^num_qubits)",
    )
    parser.add_argument(
        "--q_layers", type=int, default=3, help="Number of VQC layers"
    )
    parser.add_argument(
        "--use_paper_head",
        action="store_true",
        help="Use paper-compliant quantum head (Ry + CNOT ring) instead of StronglyEntanglingLayers",
    )
    parser.add_argument(
        "--device_type",
        type=str,
        default="default.qubit",
        help="PennyLane device (default.qubit for backprop diff)",
    )

    # Checkpoints
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint (resume pretrain or load encoder for finetune)",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=5,
        help="Save checkpoint every N epochs during pre-training",
    )

    # Output
    parser.add_argument("--output_file", type=str, default=default_output)
    parser.add_argument("--num_classes", type=int, default=6)

    args = parser.parse_args()

    print(f"\nMPSQCL HAR — Phase: {args.phase.upper()}")
    print(f"Dataset: {args.dataset.upper()}")
    if args.phase == "pretrain":
        print(f"Augmented Views (M): {args.n_views}")
        print(f"Exclude Anchor: {args.exclude_anchor}")
    print(f"Qubits: {args.num_qubits} | Feature dim: {2**args.num_qubits}")
    print(f"Device: {args.device_type}\n")

    if args.phase == "pretrain":
        pretrain(args)
    elif args.phase == "finetune":
        finetune(args)


if __name__ == "__main__":
    main()
