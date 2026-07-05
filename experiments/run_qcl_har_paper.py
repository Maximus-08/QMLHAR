"""
Paper-compliant QCL HAR training pipeline (Phase 1: Pre-training, Phase 2: Fine-tuning).

Matches:
  - 120 epochs, batch size 128 (64 for MotionSense if needed)
  - Adam optimizer with Cosine Annealing decay
  - Stage 1 LR: 3e-3 (1e-2 for SHAR)
  - Stage 2 LR: 1e-1 with frozen encoder (feature extraction only)
  - Validation-based model selection: saves encoder with lowest validation loss during pre-training
  - Pre-training augmentation: resampling + negation only
  - 64% / 16% / 20% splits
  - NISQ noise simulation support
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

# Adjust python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.har_datasets_paper import get_paper_dataloaders
from src.data.augmentations import ContrastiveViewGeneratorPaper
from src.models.encoder import HAREncoderPaper
from src.models.quantum_head import QuantumProjectionHeadPaper
from src.losses.ntxent import NTXentLoss


def contrastive_collate_fn(batch):
    """Stack augmented views for contrastive learning."""
    views_1, views_2, labels = [], [], []
    for (v1, v2), label in batch:
        views_1.append(v1)
        views_2.append(v2)
        labels.append(label)
    return (torch.stack(views_1), torch.stack(views_2)), torch.stack(labels)


# ============================================================
# Phase 1: Pre-Training with Validation Loop
# ============================================================
def pretrain(args):
    print("=" * 60)
    print("QCL HAR Paper Version — Phase 1: Pre-Training")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    best_encoder_path = os.path.join(
        os.path.dirname(args.output_file), f"qcl_paper_encoder_best_{args.dataset}.pt"
    )

    # Seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load dataset splits (64/16/20)
    print(
        f"\nLoading dataset '{args.dataset}' with resampling and negate augmentations..."
    )
    transform = ContrastiveViewGeneratorPaper(n_views=2)

    # Pre-training uses validation data loader as well
    train_loader, val_loader, _, in_channels, _, _ = get_paper_dataloaders(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        transform=transform,
        collate_fn=contrastive_collate_fn,
        seed=args.seed,
    )

    # Initialize Models
    print("\nInitializing paper-compliant encoder and custom VQC projection head...")
    encoder = HAREncoderPaper(in_channels=in_channels, feature_dim=256).to(device)
    quantum_head = QuantumProjectionHeadPaper(
        input_dim=256,
        num_qubits=8,
        q_layers=args.q_layers,
        device_type=args.device_type,
    ).to(device)

    # Set noise parameters
    quantum_head.set_noise(noise_prob=args.noise_prob, noise_std=args.noise_std)
    if args.noise_prob > 0:
        print(f"NISQ Noise Enabled: prob={args.noise_prob}, std={args.noise_std}")

    print(f"Encoder parameters:      {sum(p.numel() for p in encoder.parameters()):,}")
    print(
        f"Quantum head parameters: {sum(p.numel() for p in quantum_head.parameters()):,}"
    )

    criterion = NTXentLoss(temperature=args.temperature)

    # Optimizer & LR Schedule
    # Pre-training LR is 3e-3 (1e-2 for SHAR)
    lr = args.lr if args.dataset.lower() != "shar" else 1e-2
    print(f"Using pre-training learning rate: {lr}")

    all_params = list(encoder.parameters()) + list(quantum_head.parameters())
    optimizer = optim.Adam(all_params, lr=lr, weight_decay=1e-5)

    # Resume
    start_epoch = 0
    best_val_loss = float("inf")
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"\nResuming from checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        encoder.load_state_dict(ckpt["encoder"])
        quantum_head.load_state_dict(ckpt["quantum_head"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)
        best_val_loss = ckpt.get("best_val_loss", ckpt.get("val_loss", float("inf")))
        print(
            f"Resumed from epoch {start_epoch} with best_val_loss {best_val_loss:.4f}"
        )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        last_epoch=start_epoch - 1 if start_epoch > 0 else -1,
    )

    # Training Loop
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "lr": [],
        "epoch_time": [],
    }
    total_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # --- Train ---
        encoder.train()
        quantum_head.train()
        train_loss = 0.0
        n_train_batches = 0

        for batch_idx, ((views_1, views_2), _) in enumerate(train_loader):
            views_1 = views_1.to(device)
            views_2 = views_2.to(device)

            # Forward
            h_i = encoder(views_1)
            h_j = encoder(views_2)
            z_i = quantum_head(h_i)
            z_j = quantum_head(h_j)

            loss = criterion(z_i, z_j)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_train_batches += 1

        avg_train_loss = train_loss / n_train_batches

        # --- Validation ---
        encoder.eval()
        quantum_head.eval()
        val_loss = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for (views_1, views_2), _ in val_loader:
                views_1 = views_1.to(device)
                views_2 = views_2.to(device)

                h_i = encoder(views_1)
                h_j = encoder(views_2)
                z_i = quantum_head(h_i)
                z_j = quantum_head(h_j)

                loss = criterion(z_i, z_j)
                val_loss += loss.item()
                n_val_batches += 1

        avg_val_loss = val_loss / n_val_batches
        scheduler.step()

        epoch_time = time.time() - epoch_start
        current_lr = scheduler.get_last_lr()[0]

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["lr"].append(current_lr)
        history["epoch_time"].append(epoch_time)

        print(
            f"Epoch {epoch + 1}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.6f} | Time: {epoch_time:.1f}s"
        )

        # Model Selection: Save lowest validation loss checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_encoder_path = os.path.join(
                os.path.dirname(args.output_file),
                f"qcl_paper_encoder_best_{args.dataset}.pt",
            )
            os.makedirs(os.path.dirname(best_encoder_path), exist_ok=True)
            torch.save(encoder.state_dict(), best_encoder_path)
            print(
                f"  --> New best validation loss! Saved encoder state to {best_encoder_path}"
            )

        # Periodic checkpoint
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            ckpt_path = os.path.join(
                os.path.dirname(args.output_file),
                f"qcl_paper_checkpoint_epoch{epoch + 1}_{args.dataset}.pt",
            )
            torch.save(
                {
                    "epoch": epoch + 1,
                    "encoder": encoder.state_dict(),
                    "quantum_head": quantum_head.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_loss": avg_val_loss,
                    "best_val_loss": best_val_loss,
                    "args": vars(args),
                },
                ckpt_path,
            )

    total_time = time.time() - total_start
    print(
        f"\nPre-training completed in {total_time:.1f}s ({total_time / 3600:.2f} hours)"
    )

    # Save training history
    results_path = args.output_file.replace(".txt", f"_{args.dataset}_pretrain.txt")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        f.write("=== QCL HAR Paper Pre-Training Results ===\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Best Val Loss: {best_val_loss:.4f}\n")
        f.write(f"Total Time: {total_time:.2f}s\n\n")
        f.write("Epoch | Train Loss | Val Loss  | LR         | Time (s)\n")
        f.write("-" * 60 + "\n")
        for i in range(len(history["epoch"])):
            f.write(
                f"{history['epoch'][i]:<6}| "
                f"{history['train_loss'][i]:<11.4f}| "
                f"{history['val_loss'][i]:<10.4f}| "
                f"{history['lr'][i]:<11.6f}| "
                f"{history['epoch_time'][i]:.1f}\n"
            )

    print(f"Saved pre-training logs: {results_path}")
    return best_encoder_path


# ============================================================
# Phase 2: Supervised Fine-Tuning
# ============================================================
def finetune(args):
    print("=" * 60)
    print("QCL HAR Paper Version — Phase 2: Supervised Fine-Tuning")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load dataset splits (No augmentations for downstream evaluation)
    train_loader, _, test_loader, in_channels, _, num_classes = get_paper_dataloaders(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        transform=None,
        seed=args.seed,
    )

    # Model: encoder + linear classifier
    encoder = HAREncoderPaper(in_channels=in_channels, feature_dim=256).to(device)
    classifier = nn.Linear(256, num_classes).to(device)

    # Load best pre-trained weights
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"\nLoading best pre-trained encoder from: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        if isinstance(ckpt, dict) and "encoder" in ckpt:
            encoder.load_state_dict(ckpt["encoder"])
        else:
            encoder.load_state_dict(ckpt)
        print("Best pre-trained encoder loaded successfully!")
    else:
        print("\nWARNING: No checkpoint loaded. Training from scratch.")

    # Freeze the encoder parameters completely (matches paper: "freeze the encoder, using it solely for feature extraction")
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()

    print(
        f"Encoder parameters (FROZEN): {sum(p.numel() for p in encoder.parameters()):,}"
    )
    print(
        f"Classifier parameters:      {sum(p.numel() for p in classifier.parameters()):,}"
    )

    criterion = nn.CrossEntropyLoss()
    # Fine-tuning LR stage 2 is exactly 1e-1 (Adam, T_max=epochs)
    optimizer = optim.Adam(
        classifier.parameters(), lr=args.finetune_lr, weight_decay=1e-5
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "test_loss": [],
        "test_acc": [],
        "test_f1": [],
    }

    best_test_f1 = 0.0
    best_test_acc = 0.0
    best_epoch = 0
    best_cm = None
    total_start = time.time()

    for epoch in range(args.epochs):
        # --- Train (Classifier only) ---
        classifier.train()
        running_loss = 0.0
        all_preds, all_targets = [], []
        epoch_start = time.time()

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            with torch.no_grad():
                features = encoder(inputs)  # Frozen features
                features = torch.nn.functional.normalize(features, p=2, dim=1)

            logits = classifier(features)
            loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(logits, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = accuracy_score(all_targets, all_preds)
        train_f1 = f1_score(all_targets, all_preds, average="macro")

        # --- Evaluate ---
        classifier.eval()
        test_loss = 0.0
        test_preds, test_targets = [], []

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                features = encoder(inputs)
                features = torch.nn.functional.normalize(features, p=2, dim=1)
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

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_f1"].append(train_f1)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["test_f1"].append(test_f1)

        # Track best macro F1-score as primary evaluation metric
        if test_f1 > best_test_f1:
            best_test_f1 = test_f1
            best_test_acc = test_acc
            best_epoch = epoch + 1
            best_cm = test_cm
            best_model_path = os.path.join(
                os.path.dirname(args.output_file),
                f"qcl_paper_best_model_{args.dataset}.pt",
            )
            torch.save(
                {
                    "classifier": classifier.state_dict(),
                    "test_acc": test_acc,
                    "test_f1": test_f1,
                    "epoch": epoch + 1,
                },
                best_model_path,
            )

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc * 100:.2f}%, F1: {train_f1:.4f} | "
            f"Test Loss: {test_loss:.4f}, Acc: {test_acc * 100:.2f}%, F1: {test_f1:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

    total_time = time.time() - total_start
    print("=" * 60)
    print("QCL Fine-Tuning Complete!")
    print(f"Best test F1-score: {best_test_f1 * 100:.2f}% (Epoch {best_epoch})")
    print(f"Best test Accuracy:  {best_test_acc * 100:.2f}%")
    print("Confusion Matrix:")
    print(best_cm)
    print("=" * 60)

    # Save results
    results_path = args.output_file.replace(".txt", f"_{args.dataset}_finetune.txt")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        f.write("=== QCL HAR Paper Fine-Tuning Results ===\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Pre-trained checkpoint: {args.checkpoint}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Fine-tuning LR: {args.finetune_lr}\n")
        f.write(f"Total Time: {total_time:.2f}s\n\n")
        f.write(
            f"Best Test F1 (Macro): {best_test_f1 * 100:.2f}% (Epoch {best_epoch})\n"
        )
        f.write(f"Best Test Accuracy:  {best_test_acc * 100:.2f}%\n\n")
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

    print(f"Saved fine-tuning logs: {results_path}")


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_data_dir = os.path.join(project_root, "data")
    default_output = os.path.join(project_root, "results", "qcl_paper_results.txt")

    parser = argparse.ArgumentParser(description="QCL HAR Paper Training Pipelines")
    parser.add_argument(
        "--phase", type=str, required=True, choices=["pretrain", "finetune"]
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ucihar",
        choices=["ucihar", "shar", "hhar", "motionsense", "uschad", "mobiact"],
    )
    parser.add_argument("--data_dir", type=str, default=default_data_dir)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--lr", type=float, default=3e-3, help="Stage 1 Pre-training learning rate"
    )
    parser.add_argument(
        "--finetune_lr",
        type=float,
        default=1e-1,
        help="Stage 2 Fine-tuning learning rate",
    )
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--q_layers", type=int, default=1)
    parser.add_argument("--device_type", type=str, default="default.qubit")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--noise_prob",
        type=float,
        default=0.0,
        help="NISQ noise probability (e.g. 0.1, 0.7)",
    )
    parser.add_argument(
        "--noise_std", type=float, default=0.1, help="NISQ noise rotation std dev"
    )
    parser.add_argument("--output_file", type=str, default=default_output)

    args = parser.parse_args()

    # Adjust batch size for HHAR/SHAR/UCI-HAR if needed, or keep 128 as default
    print(
        f"\nQCL HAR (Paper Version) — Dataset: {args.dataset.upper()} | Phase: {args.phase.upper()}"
    )

    if args.phase == "pretrain":
        pretrain(args)
    elif args.phase == "finetune":
        finetune(args)


if __name__ == "__main__":
    main()
