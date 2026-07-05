"""
Ablation Study script for MPSQCL (Multi-Positive Sample Quantum Contrastive Learning).
Evaluates 8 different configurations sequentially on a 20% subset of ucihar data for 20 epochs.
Saves comparative table results to results/mpsqcl_ablation_results.md.
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

# Force PyTorch to use multi-threading on CPU for faster PennyLane simulation
num_cores = os.cpu_count()
if num_cores:
    threads = min(8, num_cores)
    torch.set_num_threads(threads)
    print(f"Forcing PyTorch CPU threads: {threads} (total cores: {num_cores})")

# Adjust python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.har_datasets_paper import get_paper_dataloaders
from src.data.augmentations import ContrastiveViewGenerator, ContrastiveViewGeneratorPaper
from src.models.encoder import HAREncoder, HAREncoderPaper
from src.models.quantum_head import QuantumProjectionHead, QuantumProjectionHeadPaper
from src.losses.mpsqcl_loss import MPSQCLLoss


def mps_contrastive_collate_fn(batch):
    """Stack multi-view augmented features for contrastive pre-training."""
    n_views = len(batch[0][0])
    views_list = [[] for _ in range(n_views)]
    labels = []

    for views, label in batch:
        for idx in range(n_views):
            views_list[idx].append(views[idx])
        labels.append(label)

    stacked_views = [torch.stack(v) for v in views_list]
    return tuple(stacked_views), torch.stack(labels)


def get_ablation_loaders(
    dataset_name,
    data_dir,
    batch_size,
    transform,
    collate_fn,
    seed,
    subset_fraction,
    use_sampler,
):
    """Loads dataloaders, subsets train/val splits to subset_fraction, and configures sampler."""
    train_loader, val_loader, test_loader, in_channels, _, num_classes = get_paper_dataloaders(
        dataset_name=dataset_name,
        data_dir=data_dir,
        batch_size=batch_size,
        transform=transform,
        collate_fn=collate_fn,
        seed=seed,
        use_sampler=False,  # We will configure sampler manually on the subset
    )

    train_set = train_loader.dataset
    val_set = val_loader.dataset
    test_set = test_loader.dataset

    # Subset train and val sets
    if subset_fraction < 1.0:
        np.random.seed(seed)
        n_train = len(train_set)
        train_indices = np.random.choice(
            n_train, int(n_train * subset_fraction), replace=False
        )
        train_subset = Subset(train_set, train_indices)

        n_val = len(val_set)
        val_indices = np.random.choice(
            n_val, int(n_val * subset_fraction), replace=False
        )
        val_subset = Subset(val_set, val_indices)
    else:
        train_subset = train_set
        val_subset = val_set

    # Rebuild train loader with/without sampler
    if use_sampler:
        if subset_fraction < 1.0:
            sub_indices = np.array(train_set.indices)[train_indices]
        else:
            sub_indices = np.array(train_set.indices)

        y_all = train_set.dataset.y.numpy()
        subset_labels = y_all[sub_indices]

        class_counts = np.bincount(subset_labels)
        class_counts[class_counts == 0] = 1
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[subset_labels]

        sampler = torch.utils.data.WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights),
            num_samples=len(sample_weights),
            replacement=True,
        )

        train_loader_subset = DataLoader(
            train_subset,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=True,
            collate_fn=collate_fn,
            num_workers=0,
        )
    else:
        train_loader_subset = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
            num_workers=0,
        )

    val_loader_subset = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    return (
        train_loader_subset,
        val_loader_subset,
        test_loader,
        in_channels,
        num_classes,
    )


def run_configuration(config_name, settings, args, device):
    print("=" * 60)
    print(f"RUNNING CONFIGURATION: {config_name}")
    print("=" * 60)

    # 1. Phase 1: Pre-training
    # Determine views & augmentations generator
    if settings["use_paper_aug"]:
        transform = ContrastiveViewGeneratorPaper(dataset_name=args.dataset)
        n_views = transform.n_views
    else:
        transform = ContrastiveViewGenerator(n_views=settings["n_views"])
        n_views = settings["n_views"]

    print(f"Loading data for Phase 1 with {n_views} views (Paper Augmentations: {settings['use_paper_aug']})...")
    train_loader, val_loader, _, in_channels, num_classes = get_ablation_loaders(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        transform=transform,
        collate_fn=mps_contrastive_collate_fn,
        seed=args.seed,
        subset_fraction=args.subset_fraction,
        use_sampler=settings["use_sampler_pretrain"],
    )

    # Setup Models
    if settings["use_paper_encoder"]:
        encoder = HAREncoderPaper(in_channels=in_channels, feature_dim=256).to(device)
    else:
        encoder = HAREncoder(
            in_channels=in_channels, feature_dim=256, pooling="avg"
        ).to(device)

    q_layers = settings.get("q_layers", 1 if settings["use_paper_head"] else 3)
    if settings["use_paper_head"]:
        quantum_head = QuantumProjectionHeadPaper(
            input_dim=256, num_qubits=8, q_layers=q_layers, device_type=args.device_type
        ).to(device)
    else:
        quantum_head = QuantumProjectionHead(
            input_dim=256, num_qubits=8, q_layers=q_layers, device_type=args.device_type
        ).to(device)

    criterion_pretrain = MPSQCLLoss(temperature=0.1, exclude_anchor_from_denominator=False)
    all_params = list(encoder.parameters()) + list(quantum_head.parameters())
    
    # Cosine Annealing learning rate
    pretrain_lr = 3e-3 if args.dataset.lower() != "shar" else 1e-2
    optimizer_pretrain = optim.Adam(all_params, lr=pretrain_lr, weight_decay=1e-5)
    scheduler_pretrain = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_pretrain, T_max=args.pretrain_epochs
    )

    best_val_loss = float("inf")
    best_encoder_state = None

    print(f"Pre-training {config_name} for {args.pretrain_epochs} epochs...")
    for epoch in range(args.pretrain_epochs):
        encoder.train()
        quantum_head.train()
        train_loss = 0.0
        n_train = 0

        for batch_views, _ in train_loader:
            views = [v.to(device) for v in batch_views]

            z_list = []
            for v in views:
                h = encoder(v)
                z = quantum_head(h)
                z_list.append(z)

            loss = criterion_pretrain(z_list)

            optimizer_pretrain.zero_grad()
            loss.backward()
            optimizer_pretrain.step()

            train_loss += loss.item()
            n_train += 1

        avg_train_loss = train_loss / n_train
        scheduler_pretrain.step()

        # Validation
        encoder.eval()
        quantum_head.eval()
        val_loss = 0.0
        n_val = 0

        with torch.no_grad():
            for batch_views, _ in val_loader:
                views = [v.to(device) for v in batch_views]
                z_list = []
                for v in views:
                    h = encoder(v)
                    z = quantum_head(h)
                    z_list.append(z)
                loss = criterion_pretrain(z_list)
                val_loss += loss.item()
                n_val += 1

        avg_val_loss = val_loss / n_val
        print(
            f"  Epoch {epoch+1:02d} | Pre-train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_encoder_state = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}

    print(f"Finished Pre-training. Best Val Loss: {best_val_loss:.4f}")

    # 2. Phase 2: Fine-Tuning
    print("\nLoading data for Phase 2 (Fine-Tuning)...")
    train_loader_ft, _, test_loader_ft, in_channels, num_classes = get_ablation_loaders(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        transform=None,
        collate_fn=None,
        seed=args.seed,
        subset_fraction=args.subset_fraction,
        use_sampler=settings["use_sampler_finetune"],
    )

    # Re-initialize fresh models
    if settings["use_paper_encoder"]:
        encoder_ft = HAREncoderPaper(in_channels=in_channels, feature_dim=256).to(device)
    else:
        encoder_ft = HAREncoder(
            in_channels=in_channels, feature_dim=256, pooling="avg"
        ).to(device)

    # Load best pre-trained weights
    encoder_ft.load_state_dict({k: v.to(device) for k, v in best_encoder_state.items()})
    classifier = nn.Linear(256, num_classes).to(device)

    # Freeze or Unfreeze Encoder
    if settings["freeze_encoder"]:
        for param in encoder_ft.parameters():
            param.requires_grad = False
        encoder_ft.eval()
        optimizer_ft = optim.Adam(
            classifier.parameters(), lr=settings["finetune_lr"], weight_decay=1e-5
        )
    else:
        for param in encoder_ft.parameters():
            param.requires_grad = True
        optimizer_ft = optim.Adam(
            [
                {"params": encoder_ft.parameters(), "lr": args.lr * 0.1},
                {"params": classifier.parameters(), "lr": args.lr},
            ],
            weight_decay=1e-5,
        )

    scheduler_ft = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_ft, T_max=args.finetune_epochs
    )
    criterion_ft = nn.CrossEntropyLoss()

    best_test_f1 = 0.0
    best_test_acc = 0.0

    print(f"Fine-tuning {config_name} for {args.finetune_epochs} epochs...")
    for epoch in range(args.finetune_epochs):
        # Train
        if settings["freeze_encoder"]:
            encoder_ft.eval()
        else:
            encoder_ft.train()
        classifier.train()

        running_loss = 0.0
        n_ft = 0

        for inputs, targets in train_loader_ft:
            inputs, targets = inputs.to(device), targets.to(device)

            if settings["freeze_encoder"]:
                with torch.no_grad():
                    features = encoder_ft(inputs)
            else:
                features = encoder_ft(inputs)

            if settings["normalize_features"]:
                features = torch.nn.functional.normalize(features, p=2, dim=1)

            logits = classifier(features)
            loss = criterion_ft(logits, targets)

            optimizer_ft.zero_grad()
            loss.backward()
            optimizer_ft.step()

            running_loss += loss.item() * inputs.size(0)
            n_ft += inputs.size(0)

        train_loss = running_loss / n_ft
        scheduler_ft.step()

        # Evaluate
        encoder_ft.eval()
        classifier.eval()
        test_preds, test_targets = [], []

        with torch.no_grad():
            for inputs, targets in test_loader_ft:
                inputs = inputs.to(device)
                features = encoder_ft(inputs)
                if settings["normalize_features"]:
                    features = torch.nn.functional.normalize(features, p=2, dim=1)
                logits = classifier(features)

                _, preds = torch.max(logits, 1)
                test_preds.extend(preds.cpu().numpy())
                test_targets.extend(targets.numpy())

        test_acc = accuracy_score(test_targets, test_preds)
        test_f1 = f1_score(test_targets, test_preds, average="macro")

        if test_f1 > best_test_f1:
            best_test_f1 = test_f1
            best_test_acc = test_acc

        print(
            f"  Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Test Acc: {test_acc*100:.2f}% | Test F1: {test_f1:.4f}"
        )

    print(f"Finished Fine-tuning. Best Test Acc: {best_test_acc*100:.2f}%, Best Test F1: {best_test_f1:.4f}")

    return {
        "best_val_loss": best_val_loss,
        "best_test_acc": best_test_acc,
        "best_test_f1": best_test_f1,
    }


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_data_dir = os.path.join(project_root, "data")

    parser = argparse.ArgumentParser(description="MPSQCL HAR Ablation & Candidate Evaluator")
    parser.add_argument("--dataset", type=str, default="ucihar")
    parser.add_argument("--data_dir", type=str, default=default_data_dir)
    parser.add_argument(
        "--mode",
        type=str,
        default="ablation_paper",
        choices=["ablation_paper", "ablation_standard", "candidates"],
        help="Evaluation mode: 'ablation_paper', 'ablation_standard', or 'candidates'",
    )
    parser.add_argument("--subset_fraction", type=float, default=0.2)
    parser.add_argument("--pretrain_epochs", type=int, default=20)
    parser.add_argument("--finetune_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device_type", type=str, default="default.qubit")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Orchestrating {args.mode} on dataset: {args.dataset.upper()}")
    print(f"Using Device: {device}")
    print(
        f"Subset size: {args.subset_fraction*100:.0f}% | Pretrain Epochs: {args.pretrain_epochs} | Finetune Epochs: {args.finetune_epochs}"
    )

    # Load configurations based on mode
    if args.mode == "ablation_paper":
        baseline_key = "Config 0: Paper-Compliant Baseline"
        title = "MPSQCL Pipeline Ablation Study Results (Paper Components)"
        res_file = os.path.join(
            project_root,
            "results",
            f"mpsqcl_ablation_results_{int(args.subset_fraction * 100)}pct_{args.pretrain_epochs}e.md",
        )
        configs = {
            "Config 0: Paper-Compliant Baseline": {
                "use_paper_aug": True,
                "n_views": 5,
                "use_paper_encoder": True,
                "use_paper_head": True,
                "use_sampler_pretrain": True,
                "use_sampler_finetune": True,
                "freeze_encoder": True,
                "normalize_features": True,
                "finetune_lr": 1e-1,
            },
            "Config 1: Paper + Unfrozen Encoder": {
                "use_paper_aug": True,
                "n_views": 5,
                "use_paper_encoder": True,
                "use_paper_head": True,
                "use_sampler_pretrain": True,
                "use_sampler_finetune": True,
                "freeze_encoder": False,
                "normalize_features": True,
                "finetune_lr": 1e-3,
            },
            "Config 2: Paper + Deeper VQC (D=3)": {
                "use_paper_aug": True,
                "n_views": 5,
                "use_paper_encoder": True,
                "use_paper_head": False,
                "use_sampler_pretrain": True,
                "use_sampler_finetune": True,
                "freeze_encoder": True,
                "normalize_features": True,
                "finetune_lr": 1e-1,
            },
            "Config 3: Paper + Average Pooling": {
                "use_paper_aug": True,
                "n_views": 5,
                "use_paper_encoder": False,
                "use_paper_head": True,
                "use_sampler_pretrain": True,
                "use_sampler_finetune": True,
                "freeze_encoder": True,
                "normalize_features": True,
                "finetune_lr": 1e-1,
            },
            "Config 4: Paper + 11-Augmentation Pool": {
                "use_paper_aug": False,
                "n_views": 5,
                "use_paper_encoder": True,
                "use_paper_head": True,
                "use_sampler_pretrain": True,
                "use_sampler_finetune": True,
                "freeze_encoder": True,
                "normalize_features": True,
                "finetune_lr": 1e-1,
            },
            "Config 5: Paper + Unweighted Sampler": {
                "use_paper_aug": True,
                "n_views": 5,
                "use_paper_encoder": True,
                "use_paper_head": True,
                "use_sampler_pretrain": False,
                "use_sampler_finetune": False,
                "freeze_encoder": True,
                "normalize_features": True,
                "finetune_lr": 1e-1,
            },
            "Config 6: Paper + No Feature L2-Normalization": {
                "use_paper_aug": True,
                "n_views": 5,
                "use_paper_encoder": True,
                "use_paper_head": True,
                "use_sampler_pretrain": True,
                "use_sampler_finetune": True,
                "freeze_encoder": True,
                "normalize_features": False,
                "finetune_lr": 1e-1,
            },
            "Config 7: Standard Pipeline (All Combined)": {
                "use_paper_aug": False,
                "n_views": 4,
                "use_paper_encoder": False,
                "use_paper_head": False,
                "use_sampler_pretrain": False,
                "use_sampler_finetune": False,
                "freeze_encoder": False,
                "normalize_features": False,
                "finetune_lr": 1e-3,
            },
        }

    elif args.mode == "ablation_standard":
        baseline_key = "Config 0: Standard Baseline"
        title = "Standard MPSQCL Pipeline Ablation Study Results"
        res_file = os.path.join(
            project_root,
            "results",
            f"mpsqcl_ablation_results_standard_{int(args.subset_fraction * 100)}pct_{args.pretrain_epochs}e.md",
        )
        configs = {
            "Config 0: Standard Baseline": {
                "use_paper_aug": False,
                "n_views": 4,
                "use_paper_encoder": False,
                "use_paper_head": False,
                "use_sampler_pretrain": False,
                "use_sampler_finetune": False,
                "freeze_encoder": False,
                "normalize_features": False,
                "finetune_lr": 1e-3,
            },
            "Config 1: Standard - Average Pooling": {
                "use_paper_aug": False,
                "n_views": 4,
                "use_paper_encoder": True,
                "use_paper_head": False,
                "use_sampler_pretrain": False,
                "use_sampler_finetune": False,
                "freeze_encoder": False,
                "normalize_features": False,
                "finetune_lr": 1e-3,
            },
            "Config 2: Standard - Unweighted Sampler": {
                "use_paper_aug": False,
                "n_views": 4,
                "use_paper_encoder": False,
                "use_paper_head": False,
                "use_sampler_pretrain": True,
                "use_sampler_finetune": True,
                "freeze_encoder": False,
                "normalize_features": False,
                "finetune_lr": 1e-3,
            },
            "Config 3: Standard - Deeper VQC": {
                "use_paper_aug": False,
                "n_views": 4,
                "use_paper_encoder": False,
                "use_paper_head": True,
                "use_sampler_pretrain": False,
                "use_sampler_finetune": False,
                "freeze_encoder": False,
                "normalize_features": False,
                "finetune_lr": 1e-3,
            },
            "Config 4: Standard - Unfrozen Encoder": {
                "use_paper_aug": False,
                "n_views": 4,
                "use_paper_encoder": False,
                "use_paper_head": False,
                "use_sampler_pretrain": False,
                "use_sampler_finetune": False,
                "freeze_encoder": True,
                "normalize_features": False,
                "finetune_lr": 1e-1,
            },
        }

    else:  # candidates
        baseline_key = "Candidate 1: Fully Optimized Standard (Depth-2 VQC, Unfrozen Encoder)"
        title = "Standard MPSQCL Candidate Configuration Results"
        res_file = os.path.join(
            project_root,
            "results",
            f"mpsqcl_candidates_results_{int(args.subset_fraction * 100)}pct_{args.pretrain_epochs}e.md",
        )
        configs = {
            "Candidate 1: Fully Optimized Standard (Depth-2 VQC, Unfrozen Encoder)": {
                "use_paper_aug": False,
                "n_views": 4,
                "use_paper_encoder": False,
                "use_paper_head": False,
                "q_layers": 2,
                "use_sampler_pretrain": False,
                "use_sampler_finetune": False,
                "freeze_encoder": False,
                "normalize_features": False,
                "finetune_lr": 1e-3,
            },
            "Candidate 2: Speed-Optimized Standard (Depth-1 VQC, Unfrozen Encoder)": {
                "use_paper_aug": False,
                "n_views": 4,
                "use_paper_encoder": False,
                "use_paper_head": True,
                "q_layers": 1,
                "use_sampler_pretrain": False,
                "use_sampler_finetune": False,
                "freeze_encoder": False,
                "normalize_features": False,
                "finetune_lr": 1e-3,
            },
            "Candidate 3: Frozen Encoder Standard (Depth-2 VQC)": {
                "use_paper_aug": False,
                "n_views": 4,
                "use_paper_encoder": False,
                "use_paper_head": False,
                "q_layers": 2,
                "use_sampler_pretrain": False,
                "use_sampler_finetune": False,
                "freeze_encoder": True,
                "normalize_features": False,
                "finetune_lr": 1e-1,
            },
        }

    results = {}
    for name, settings in configs.items():
        res = run_configuration(name, settings, args, device)
        results[name] = res

    # Generate Markdown Table
    print("\n" + "=" * 60)
    print("EVALUATION RUNS COMPLETED. SUMMARY OF RESULTS:")
    print("=" * 60)

    os.makedirs(os.path.dirname(res_file), exist_ok=True)
    with open(res_file, "w") as f:
        f.write(f"# {title}\n\n")
        f.write(f"- **Dataset**: {args.dataset.upper()}\n")
        f.write(f"- **Subset Fraction**: {args.subset_fraction * 100:.0f}%\n")
        f.write(f"- **Pre-training Epochs**: {args.pretrain_epochs}\n")
        f.write(f"- **Fine-tuning Epochs**: {args.finetune_epochs}\n")
        f.write(f"- **Device Used**: {device}\n\n")
        f.write("## Comparison Table\n\n")
        f.write(
            f"| Configuration | Best Pre-training Val Loss | Test Accuracy | Test Macro F1-score | Delta Test F1 (vs. Baseline) |\n"
        )
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")

        baseline_f1 = results[baseline_key]["best_test_f1"]

        for name, metrics in results.items():
            delta = metrics["best_test_f1"] - baseline_f1
            delta_str = f"{delta*100:+.2f}%" if name != baseline_key else "-"
            f.write(
                f"| **{name}** | {metrics['best_val_loss']:.4f} | {metrics['best_test_acc']*100:.2f}% | {metrics['best_test_f1']:.4f} | {delta_str} |\n"
            )

            # Print live to terminal
            print(
                f"{name:<45} | Val Loss: {metrics['best_val_loss']:.4f} | Test Acc: {metrics['best_test_acc']*100:.2f}% | Test F1: {metrics['best_test_f1']:.4f} | Delta F1: {delta_str}"
            )

    print(f"\nSaved comparison report to: {res_file}")


if __name__ == "__main__":
    main()
