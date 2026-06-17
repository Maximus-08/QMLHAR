import subprocess
import time
import os
import glob
import re


def run_cmd(cmd, log_file):
    print(f"Running command: {cmd}")
    print(f"Logging to: {log_file}")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "w") as f:
        # Run and print output live to log file
        process = subprocess.Popen(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
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
        # Match e.g., mpsqcl_checkpoint_epoch120_ucihar.pt
        match = re.search(rf"{prefix}(\d+)_", filename)
        if match:
            epoch = int(match.group(1))
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_file = f
    return latest_file


def run_dataset_pipeline(dataset, batch_size):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    results_dir = os.path.join(project_root, "results")

    print("=" * 80)
    print(f"PROCESSING DATASET FOR MPSQCL: {dataset.upper()}")
    print("=" * 80)

    # 1. Phase 1: Contrastive Pre-training
    mpsqcl_pre_log = os.path.join(results_dir, f"log_mpsqcl_{dataset}_pretrain.log")
    final_pretrained_path = os.path.join(
        results_dir, f"mpsqcl_encoder_pretrained_{dataset}.pt"
    )

    # Skip pre-training if final weights already exist
    if os.path.exists(final_pretrained_path):
        print(
            f"Pre-trained encoder weights already exist at {final_pretrained_path}. Skipping pre-training."
        )
    else:
        # Check if we can resume from a latest checkpoint
        latest_ckpt = get_latest_checkpoint(
            results_dir, "mpsqcl_checkpoint_epoch", dataset
        )
        pretrain_cmd = f".venv/bin/python -u experiments/run_mpsqcl_har.py --phase pretrain --dataset {dataset} --epochs 150 --save_every 30 --batch_size {batch_size}"
        if dataset == "hhar":
            pretrain_cmd += " --use_sampler"
        if latest_ckpt:
            print(f"Found latest pre-training checkpoint to resume: {latest_ckpt}")
            pretrain_cmd += f" --checkpoint {latest_ckpt}"

        print(f"Executing Pre-training on {dataset}...")
        ret_code = run_cmd(pretrain_cmd, mpsqcl_pre_log)
        if ret_code != 0:
            print(f"Error during pre-training on {dataset}. Return code: {ret_code}")
            return False
        print(f"Finished Pre-training on {dataset}")

    # 2. Phase 2: Supervised Fine-tuning
    mpsqcl_fine_log = os.path.join(results_dir, f"log_mpsqcl_{dataset}_finetune.log")
    finetune_res_path = os.path.join(
        results_dir, f"mpsqcl_results_{dataset}_finetune.txt"
    )

    # Skip fine-tuning if final result file already exists
    if os.path.exists(finetune_res_path):
        print(
            f"Fine-tuning results already exist at {finetune_res_path}. Skipping fine-tuning."
        )
    else:
        print(f"Executing Fine-tuning on {dataset}...")
        finetune_cmd = (
            f".venv/bin/python -u experiments/run_mpsqcl_har.py --phase finetune --dataset {dataset} "
            f"--epochs 100 --checkpoint {final_pretrained_path} --batch_size 128"
        )
        if dataset == "hhar":
            finetune_cmd += " --use_sampler"
        ret_code = run_cmd(finetune_cmd, mpsqcl_fine_log)
        if ret_code != 0:
            print(f"Error during fine-tuning on {dataset}. Return code: {ret_code}")
            return False
        print(f"Finished Fine-tuning on {dataset}")

    return True


def main():
    print("=" * 80)
    print("RUNNING MPSQCL HAR PIPELINE RUNNER FOR ALL DATASETS (150 EPOCHS)")
    print("=" * 80)

    datasets = ["ucihar", "shar", "motionsense", "uschad", "mobiact", "hhar"]

    start_time = time.time()
    for dataset in datasets:
        # Use batch size 32 for motionsense to prevent CUDA OOM, 512 for hhar to speed up pre-training, and 64 for all others
        if dataset == "motionsense":
            bs = 32
        elif dataset == "hhar":
            bs = 512
        else:
            bs = 64
        success = run_dataset_pipeline(dataset, bs)
        if not success:
            print(f"Pipeline failed on dataset: {dataset}. Stopping execution.")
            return

    total_time = time.time() - start_time
    print("=" * 80)
    print(
        f"ALL MPSQCL PIPELINES PROCESSED SUCCESSFULLY in {total_time / 3600:.2f} hours!"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
