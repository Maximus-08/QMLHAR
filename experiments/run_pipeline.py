"""
Unified Pipeline Runner for QMLHAR models.
Sequentially executes pre-training, fine-tuning, or baseline models across multiple datasets.

Usage:
  # Run standard MPSQCL pipeline
  python experiments/run_pipeline.py --model mpsqcl --epochs_pretrain 150 --epochs_finetune 100

  # Run standard LSTM baseline
  python experiments/run_pipeline.py --model lstm --epochs_finetune 100

  # Run MPSQCL + LSTM classifier pipeline
  python experiments/run_pipeline.py --model mpsqcl_lstm --epochs_finetune 100
"""

import os
import sys
import argparse
import subprocess
import time
import glob
import re


def run_cmd(cmd, log_file):
    print(f"Running command: {cmd}")
    print(f"Logging to: {log_file}")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "w") as f:
        process = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        # Stream output live to console and file
        for line in process.stdout:
            f.write(line)
            f.flush()
            print(line, end="")
        process.wait()
    return process.returncode


def get_latest_checkpoint(results_dir, prefix, dataset):
    """Find the latest saved checkpoint for the dataset."""
    pattern = os.path.join(results_dir, f"{prefix}*_{dataset}.pt")
    files = glob.glob(pattern)
    if not files:
        return None
    latest_epoch = -1
    latest_file = None
    for f in files:
        filename = os.path.basename(f)
        match = re.search(rf"{prefix}(\d+)_", filename)
        if match:
            epoch = int(match.group(1))
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_file = f
    return latest_file


def run_dataset_pipeline(dataset, args, results_dir):
    print("\n" + "=" * 80)
    print(f"PROCESSING DATASET: {dataset.upper()}")
    print("=" * 80)

    # Determine batch sizes per dataset to prevent OOM
    if dataset == "motionsense":
        batch_size = 32
    elif dataset == "hhar":
        batch_size = 512 if args.model in ["mpsqcl", "mpsqcl_paper"] else 128
    else:
        batch_size = args.batch_size

    # Checkpoint prefix and final path naming
    is_depth_nonstandard = (args.q_layers != 3) or args.use_paper_head
    depth_str = f"depth{args.q_layers}"
    
    if is_depth_nonstandard:
        final_pretrained_path = os.path.join(
            results_dir, f"mpsqcl_encoder_pretrained_{depth_str}_{dataset}.pt"
        )
        checkpoint_prefix = f"mpsqcl_checkpoint_{depth_str}_epoch"
    else:
        final_pretrained_path = os.path.join(
            results_dir, f"mpsqcl_encoder_pretrained_{dataset}.pt"
        )
        checkpoint_prefix = "mpsqcl_checkpoint_epoch"

    # --- Mode 1: Classical LSTM Baseline ---
    if args.model == "lstm":
        log_file = os.path.join(results_dir, f"log_lstm_{dataset}.log")
        cmd = (
            f"{sys.executable} -u experiments/run_lstm_baseline.py "
            f"--dataset {dataset} --epochs {args.epochs_finetune} --batch_size {batch_size}"
        )
        return run_cmd(cmd, log_file) == 0

    # --- Mode 2: Variational Quantum pre-training / fine-tuning ---
    elif args.model in ["mpsqcl", "mpsqcl_paper"]:
        # 1. Pre-training (Phase 1)
        suffix = f"_{depth_str}" if is_depth_nonstandard else ""
        pre_log_file = os.path.join(results_dir, f"log_mpsqcl{suffix}_{dataset}_pretrain.log")

        if os.path.exists(final_pretrained_path):
            print(f"Pre-trained encoder weights exist at {final_pretrained_path}. Skipping pre-training.")
        else:
            latest_ckpt = get_latest_checkpoint(results_dir, checkpoint_prefix, dataset)
            
            if args.model == "mpsqcl":
                pretrain_cmd = (
                    f"{sys.executable} -u experiments/run_mpsqcl_har.py --phase pretrain --dataset {dataset} "
                    f"--epochs {args.epochs_pretrain} --save_every 30 --batch_size {batch_size} "
                    f"--q_layers {args.q_layers}"
                )
                if args.use_paper_head:
                    pretrain_cmd += " --use_paper_head"
                if is_depth_nonstandard:
                    pretrain_cmd += f" --output_file {results_dir}/mpsqcl_{depth_str}_results.txt"
            else:
                # mpsqcl_paper
                pretrain_cmd = (
                    f"{sys.executable} -u experiments/run_mpsqcl_har_paper.py --phase pretrain --dataset {dataset} "
                    f"--epochs {args.epochs_pretrain} --save_every 30 --batch_size {batch_size}"
                )

            if dataset == "hhar":
                pretrain_cmd += " --use_sampler"
            if latest_ckpt:
                print(f"Resuming pre-training from checkpoint: {latest_ckpt}")
                pretrain_cmd += f" --checkpoint {latest_ckpt}"

            print(f"Executing Pre-training on {dataset}...")
            ret = run_cmd(pretrain_cmd, pre_log_file)
            if ret != 0:
                print(f"Error during pre-training on {dataset}.")
                return False

            # Rename default outputs if we ran with a custom depth/paper head
            default_pretrained_path = os.path.join(results_dir, f"mpsqcl_encoder_pretrained_{dataset}.pt")
            if is_depth_nonstandard and os.path.exists(default_pretrained_path):
                if os.path.exists(final_pretrained_path):
                    os.remove(final_pretrained_path)
                os.rename(default_pretrained_path, final_pretrained_path)
                print(f"Renamed pre-trained weights to {final_pretrained_path}")

        # 2. Fine-tuning (Phase 2)
        fine_log_file = os.path.join(results_dir, f"log_mpsqcl{suffix}_{dataset}_finetune.log")
        results_txt_path = os.path.join(
            results_dir, 
            f"mpsqcl_results_{depth_str}_{dataset}_finetune.txt" if is_depth_nonstandard else f"mpsqcl_results_{dataset}_finetune.txt"
        )

        if os.path.exists(results_txt_path):
            print(f"Fine-tuning results exist at {results_txt_path}. Skipping fine-tuning.")
        else:
            print(f"Executing Fine-tuning on {dataset}...")
            if args.model == "mpsqcl":
                finetune_cmd = (
                    f"{sys.executable} -u experiments/run_mpsqcl_har.py --phase finetune --dataset {dataset} "
                    f"--epochs {args.epochs_finetune} --checkpoint {final_pretrained_path} --batch_size 128 "
                    f"--q_layers {args.q_layers}"
                )
                if args.use_paper_head:
                    finetune_cmd += " --use_paper_head"
            else:
                # mpsqcl_paper
                finetune_cmd = (
                    f"{sys.executable} -u experiments/run_mpsqcl_har_paper.py --phase finetune --dataset {dataset} "
                    f"--epochs {args.epochs_finetune} --checkpoint {final_pretrained_path}"
                )

            if dataset == "hhar":
                finetune_cmd += " --use_sampler"
            ret = run_cmd(finetune_cmd, fine_log_file)
            if ret != 0:
                print(f"Error during fine-tuning on {dataset}.")
                return False

        return True

    # --- Mode 3: MPSQCL Pre-trained Encoder + LSTM Classifier ---
    elif args.model == "mpsqcl_lstm":
        log_file = os.path.join(results_dir, f"log_mpsqcl_lstm_{depth_str}_{dataset}.log" if is_depth_nonstandard else f"log_mpsqcl_lstm_{dataset}.log")
        
        # Check if pre-trained checkpoint exists
        if not os.path.exists(final_pretrained_path):
            print(f"Error: Pre-trained encoder checkpoint not found at {final_pretrained_path}!")
            print("Please run pre-training first using `--model mpsqcl`.")
            return False

        # Epochs mapping for LSTM classifier head
        epochs = 80 if dataset == "hhar" else args.epochs_finetune

        cmd = (
            f"{sys.executable} -u experiments/run_mpsqcl_lstm.py --dataset {dataset} --epochs {epochs} "
            f"--batch_size {batch_size} --freeze_encoder False --encoder_lr_factor 1.0 "
            f"--use_paper_encoder False --checkpoint {final_pretrained_path} --classical_lstm"
        )
        return run_cmd(cmd, log_file) == 0

    return False


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    results_dir = os.path.join(project_root, "results")

    parser = argparse.ArgumentParser(description="QMLHAR Unified Pipeline Runner")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["mpsqcl", "lstm", "mpsqcl_lstm", "mpsqcl_paper"],
        help="Model pipeline to run",
    )
    parser.add_argument("--q_layers", type=int, default=3, help="VQC depth")
    parser.add_argument("--use_paper_head", action="store_true", help="Use custom D=1 VQC")
    parser.add_argument("--epochs_pretrain", type=int, default=150, help="Pre-training epochs")
    parser.add_argument("--epochs_finetune", type=int, default=100, help="Fine-tuning/Baseline epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Standard batch size")
    parser.add_argument(
        "--datasets",
        type=str,
        default="ucihar,shar,motionsense,uschad,mobiact,hhar",
        help="Comma-separated dataset list",
    )

    args = parser.parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    print("=" * 80)
    print(f"RUNNING UNIFIED PIPELINE FOR MODEL: {args.model.upper()}")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"VQC layers: {args.q_layers} | Paper Head: {args.use_paper_head}")
    print("=" * 80)

    start_time = time.time()
    results_summary = []

    for dataset in datasets:
        dataset_start = time.time()
        success = run_dataset_pipeline(dataset, args, results_dir)
        dataset_time = time.time() - dataset_start
        status = "SUCCESS" if success else "FAILED"
        results_summary.append((dataset, status, dataset_time))
        if not success:
            print(f"Pipeline failed on dataset: {dataset}. Stopping execution.")
            break

    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"ALL PIPELINES PROCESSED in {total_time / 3600:.2f} hours!")
    print("=" * 80)
    print(f"{'Dataset':<15} | {'Status':<10} | {'Time (s)':<10}")
    print("-" * 45)
    for ds, status, t in results_summary:
        print(f"{ds:<15} | {status:<10} | {t:<10.1f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
