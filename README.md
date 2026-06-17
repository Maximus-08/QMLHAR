# QMLHAR: Quantum Machine Learning for Human Activity Recognition

This repository explores the application of Quantum Machine Learning (QML) models to Human Activity Recognition (HAR) tasks across six benchmark datasets. It implements and evaluates Quantum Contrastive Learning (QCL) and Multi-Positive Sample Quantum Contrastive Learning (MPSQCL) pipelines, comparing them against classical and early-stage quantum baselines.

---

## Project Structure

- `src/`
  - `data/`
    - `har_datasets_paper.py`: Paper-compliant loaders for 6 HAR datasets (`ucihar`, `shar`, `hhar`, `motionsense`, `uschad`, `mobiact`) supporting standard and weighted class random sampling.
    - `augmentations.py`: Full augmentation suite (7 strategies: jitter, negate, permute, resample, rotate, scale, temporal flip) for standard contrastive pre-training.
    - `augmentations_paper.py`: Paper-compliant augmentations (resampling and negation only).
    - `download_and_preprocess_datasets.py`: Automated dataset downloader and preprocessor for all 6 datasets.
  - `models/`
    - `encoder.py`: 4-layer 1D CNN encoder producing 256-dim features for standard pipelines.
    - `encoder_paper.py`: Paper-compliant encoder variant for frozen-encoder linear evaluation.
    - `quantum_head.py`: Amplitude-encoded VQC projection head with `StronglyEntanglingLayers` (depth 3).
    - `quantum_head_paper.py`: Paper-compliant VQC with custom Ry rotation + CNOT ring (depth 1), includes NISQ noise simulation.
  - `losses/`
    - `ntxent.py`: NT-Xent (SimCLR-style) contrastive loss for 2-view QCL pre-training.
    - `mpsqcl_loss.py`: Multi-Positive Sample contrastive loss for M-view MPSQCL pre-training.
- `experiments/`
  - `run_qcl_har.py` / `run_qcl_har_paper.py`: Standard and paper-compliant Quantum Contrastive Learning pipelines (2-view pre-training + supervised fine-tuning).
  - `run_mpsqcl_har.py` / `run_mpsqcl_har_paper.py`: Standard and paper-compliant Multi-Positive Sample QCL pipelines (M-view pre-training + supervised fine-tuning).
  - `run_pipeline_mpsqcl.py`: Automated pipeline runner for sequentially executing MPSQCL across all 6 datasets with checkpoint resume support.
- `results/`: Saved metrics, training logs, pre-training/fine-tuning history, model checkpoints, and confusion matrices.
- `qclimplementation.md`: Detailed implementation specification for the QCL architecture.
- `mpsqclimplementation.md`: Detailed implementation specification for the MPSQCL architecture.
- `report.md`: Full results table with all model evaluations across datasets.

---

## Model Architectures

### Quantum Contrastive Learning (QCL)

A two-phase self-supervised learning framework that leverages quantum circuits for representation learning on raw inertial sensor signals.

- **Phase 1 (Pre-training):** A 1D CNN encoder and an 8-qubit Quantum Projection Head are jointly trained using data augmentations and NT-Xent contrastive loss. The encoder learns to map raw sensor windows to robust 256-dimensional feature vectors by pushing augmented views of the same sample together and different samples apart in the quantum-projected space.
- **Phase 2 (Fine-tuning):** The quantum head is discarded. The pre-trained encoder is paired with a simple linear classifier and fine-tuned on labeled data with differential learning rates.

See [qclimplementation.md](qclimplementation.md) for full architectural details.

### Multi-Positive Sample QCL (MPSQCL)

An extension of QCL that generates **M positive views** (default M=4) per sample instead of 2, using a richer augmentation set. The multi-positive contrastive loss simultaneously pulls all M-1 positive views of the same sample together, leading to more robust and diverse representations that better handle sensor heterogeneity and environmental variation.

See [mpsqclimplementation.md](mpsqclimplementation.md) for full architectural details.

### Key Design Decisions

#### Amplitude Encoding for Bottleneck-Free Representation
Amplitude encoding maps $2^N$ features onto $N$ qubits ($2^8 = 256$ features on 8 qubits), completely bypassing the dimensionality compression bottleneck that limits supervised hybrid approaches to 4–6 features. This preserves the full information density of the CNN encoder output.

#### Amplitude Encoding Sign Invariance
A negated feature vector $-\mathbf{h}$ produces the same quantum state as $\mathbf{h}$ up to a global phase of $-1$, making VQC expectation values invariant to signal polarity. This causes the paper-compliant frozen-encoder pipeline to underperform on datasets where orientation information is critical. The standard pipeline recovers by unfreezing the encoder during fine-tuning, allowing supervised gradients to restore orientation sensitivity.

#### Self-Supervised Pre-Training vs. Supervised Training
Direct supervised training through quantum circuits suffers from barren plateaus, noisy gradients, and exponential simulation overhead. Contrastive pre-training sidesteps these issues by learning representations under a self-supervised objective, then fine-tuning classically — avoiding VQC gradient computation during the labeled training phase entirely.

---

## Results & Benchmarks

The table below summarizes the performance of all evaluated models. Classical baselines and early quantum models (Hybrid QCNN, QSVM) were evaluated on UCI-HAR only. QCL and MPSQCL were evaluated across all 6 datasets.

### UCI-HAR Results

| Model | Data Input / Features | Dimensionality Constraint | Test Accuracy | Macro F1 | Training Time | Notes / Decisional Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SVM (Linear)** | Hand-crafted 561-Feat | None | **96.10%** | **0.9608** | **0.83s** | Best performing classical baseline. Linear boundaries separate hand-crafted features easily. |
| **SVM (RBF)** | Hand-crafted 561-Feat | None | 95.22% | 0.9515 | 1.58s | Standard baseline for non-linear decision boundary testing. |
| **MLP Classifier** | Hand-crafted 561-Feat | None | 95.11% | 0.9509 | 5.91s | Evaluated to represent neural network behavior on hand-crafted features. |
| **Random Forest** | Hand-crafted 561-Feat | None | 92.57% | 0.9241 | 8.26s | Ensemble tree baseline. |
| **Classical 1D CNN** | Raw Signals (9x128) | No Bottleneck (128 features) | 91.55% | 0.9160 | 15.74s | Learns end-to-end representations directly from local spatial-temporal patterns. |
| **Classical 1D CNN** | Raw Signals (9x128) | 4-Dim Bottleneck (Linear + Tanh) | 93.01% | 0.9305 | 12.53s | Demonstrates that classical gradient descent can successfully optimize through a 4-dim bottleneck. |
| **Hybrid QCNN (10e)** | Raw Signals (9x128) | 4-Dim Bottleneck (4 Qubits VQC) | 73.39% | 0.6955 | 1118.89s | Suffers from slow convergence and simulation overhead (run for only 10 epochs on a 50% subset). |
| **Hybrid QCNN (50e)** | Raw Signals (9x128) | 4-Dim Bottleneck (4 Qubits VQC) | **92.53%** | **0.9256** | **3.82 hours (13768s)** | **Completed.** Final test accuracy 92.53%, peaked at **93.21%** (Epoch 47). Surpasses classical 1D CNN without bottleneck (91.55%). |
| **Hybrid QCNN (6q, 30e)** | Raw Signals (9x128) | 6-Dim Bottleneck (6 Qubits VQC) | **93.76%** | **0.9387** | **4.39 hours (15805s)** | **Completed.** Final test accuracy 93.76%, peaked at **93.89%** (Epoch 23). Best performing QML model, outperforming all classical CNN baselines. |
| **Quantum SVM (300s)** | Hand-crafted 561-Feat | 6-Dim Bottleneck (6 Qubits + PCA) | 45.00% | 0.4228 | 87.87s | Suffers from severe overfitting due to downsampling (300 train samples) to make kernel computation feasible. |
| **Quantum SVM (1000s)**| Hand-crafted 561-Feat | 8-Dim Bottleneck (8 Qubits + PCA) | 53.00% | 0.5060 | 852.43s | Accuracy improved with more samples and qubits, but simulation time scaled quadratically to 14.2 minutes. |
| **Quantum SVM (1000s, iqp)**| Hand-crafted 561-Feat| 10-Dim Bottleneck (10 Qubits + PCA)| **59.50%** | **0.5329** | **2635.20s** | **Using entangling IQPEmbedding and C=1.0 sweep.** Significantly improves representation, but entangling gates scale runtimes. |
| **Quantum SVM (1000s, amp)**| Hand-crafted 561-Feat| 256-Dim Bottleneck (8 Qubits + PCA)| **91.50%** | **0.9146** | **9.13 hours (32886s)** | **Completed.** Amplitude Embedding maps 256 features to 8 qubits, dramatically improving accuracy (91.50%) but scaling runtime significantly. |
| **QCL HAR (Pre+Fine, 150e)** | Raw Signals (9x128) | 256-Dim Features (8 Qubits VQC) | **96.64%** | **0.9665** | **56.0 mins (3359s)** | **Completed.** 150-epoch self-supervised pre-training + 30-epoch classical fine-tuning. Best performing model overall, outperforming the best classical SVM (96.10%). |
| **QCL HAR (Pre+Fine, 50e)** | Raw Signals (9x128) | 256-Dim Features (8 Qubits VQC) | **95.83%** | **0.9578** | **18.8 mins (1130s)** | **Completed.** 50-epoch self-supervised pre-training + 30-epoch classical fine-tuning. Reaches high accuracy very quickly, retaining over 99% of the performance of the 150-epoch run. |
| **MPSQCL HAR (Pre+Fine, 50e)** | Raw Signals (9x128)| 256-Dim Features (8 Qubits VQC) | **95.08%** | **0.9497** | **46.8 mins (2808s)** | **Completed.** 50-epoch multi-positive sample (M=4 views) pre-training + 30-epoch classical fine-tuning. Successfully learns representations without disturbing standard QCL code. |

### Multi-Dataset QCL & MPSQCL Results

| Model | Dataset | Test Accuracy | Macro F1 | Training Time | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MPSQCL HAR (150e)** | UCI-HAR | **98.50%** | **0.9858** | 1.54 hours | 150-epoch pre-training (M=4) + 100-epoch fine-tuning |
| **MPSQCL HAR (150e)** | MotionSense | **99.69%** | **0.9954** | 1.94 hours | Near-perfect classification |
| **MPSQCL HAR (120e)** | MobiAct | **98.31%** | **0.9559** | 48.6 mins | Excluding falls, 9 ADL classes |
| **MPSQCL HAR (150e)** | USC-HAD | **88.34%** | **0.8503** | 3.23 hours | Hardest dataset due to 12 activity classes |
| **MPSQCL HAR (150e)** | SHAR | **83.52%** | **0.7524** | 1.26 hours | Layout scrambling bug fixed |
| **QCL HAR (120e)** | MotionSense | **95.79%** | **0.9596** | 25.8 mins | Standard QCL baseline |
| **QCL HAR (120e)** | HHAR | **93.28%** | **0.8816** | 7.66 hours | Large dataset (224k windows) |
| **QCL HAR (120e)** | MobiAct | **91.70%** | **0.8180** | 11.5 mins | Standard QCL baseline |
| **QCL HAR (120e)** | SHAR | **91.06%** | **0.8667** | 2.28 hours | Standard QCL baseline |
| **QCL HAR (120e)** | USC-HAD | **71.97%** | **0.7002** | ~25 mins | Standard QCL baseline |

### Paper Reference Comparisons

| Model | Source | Test Accuracy | Macro F1 |
| :--- | :--- | :--- | :--- |
| **MPSQCL (M=4, 50e/30e)** | Paper | **97.28%** | **0.9722** |
| **MPSCL (M=4, classical head)** | Paper | **97.50%** | **0.9745** |
| **TS-TCC (classical SOTA)** | Paper | **96.41%** | **0.9635** |
| **QSSL (VQC in encoder)** | Paper | **83.59%** | **0.8351** |

---

## Future Directions: Underresearched Quantum Architectures

While standard Hybrid QCNNs, QSVMs, and Quantum Contrastive Learning (QCL/MPSQCL) are the primary models currently explored, several underresearched quantum machine learning (QML) architectures hold significant potential for Human Activity Recognition (HAR) tasks:

### 1. Quantum Long Short-Term Memory (QLSTM) / Quantum RNNs
*   **Concept**: Time-series sensor data exhibits rich sequential and temporal correlations. QLSTMs replace the classical LSTM cell's internal gates (input, forget, cell, and output gates) with parameterized variational quantum circuits (VQCs).
*   **Rationale**: The VQC gates process temporal hidden states in the quantum state space, potentially capturing non-linear phase relationships and dynamics of raw inertial signals that classical recurrences struggle to resolve. This remains highly underresearched in self-supervised and contrastive contexts.

### 2. Vision/Sequence Quantum Transformers (QST)
*   **Concept**: Replaces the classical self-attention mechanism in sequence-to-sequence transformers with a Quantum Self-Attention (QSA) mechanism, mapping sequence tokens to quantum states and using entangling gates to calculate the attention map.
*   **Rationale**: Quantum attention could offer superior sequence alignment and representation richness for high-frequency time-series windows, though it currently faces heavy qubit scaling and gate depth limitations.

### 3. Quantum Graph Convolutional Networks (QGCN)
*   **Concept**: Under multi-sensor configurations (e.g., smartwatches, chest sensors, smart shoes), the relationship between body nodes can be represented as a spatial graph. QGCNs map the graph nodes and adjacency matrices directly into quantum states and evolve them via entangling PQCs.
*   **Rationale**: This enables processing multi-node spatial-temporal topology quantumly, which is ideal for advanced HAR tasks (e.g., USC-HAD or HHAR) but has seen almost no research in QML contrastive settings.

### 4. Quantum Generative Adversarial Networks (QGANs) for State Augmentation
*   **Concept**: A generator circuit learns to map random distributions into synthetic quantum states representing realistic activity signals, while a discriminator network distinguishes them from actual encoded signals.
*   **Rationale**: Instead of relying on manual classical augmentations (e.g., `jitter`, `negate`), a pre-trained QGAN can generate high-fidelity, physically consistent positive views of signals directly in the quantum state space, improving the stability of contrastive pre-training.

### 5. Trotterized Hamiltonian Evolution Feature Maps
*   **Concept**: Instead of static `AngleEmbedding` or deep `AmplitudeEmbedding` circuits (which require hundreds of entangling gates for 256 dimensions), data is encoded dynamically through Trotter-step Hamiltonian time evolution: $U(x) = e^{-i H(x) t}$.
*   **Rationale**: Hamiltonian feature maps could allow the circuit to naturally represent continuous physical signal dynamics while keeping gate depth manageable, mitigating simulation bottlenecks and improving gradient flow.

---

## How to Run

Ensure you have activated your virtual environment containing PyTorch, PennyLane, and Scikit-Learn.

### 1. Automatic Multi-Dataset Pipeline Runner (Recommended)
To run the full MPSQCL suite (pre-training + fine-tuning) sequentially across all datasets with checkpoint resume support:
```bash
python experiments/run_pipeline_mpsqcl.py
```

### 2. Quantum Contrastive Learning (QCL HAR)
To run pre-training and fine-tuning on a specific dataset manually (e.g. `hhar`):
```bash
# Phase 1: Self-supervised pre-training
python experiments/run_qcl_har.py --phase pretrain --dataset hhar --epochs 120 --save_every 20

# Phase 2: Supervised fine-tuning
python experiments/run_qcl_har.py --phase finetune --dataset hhar --epochs 100 --checkpoint results/qcl_checkpoint_epoch120_hhar.pt
```
To run the paper-compliant version (frozen encoder linear evaluation):
```bash
# Phase 1: Pre-training
python experiments/run_qcl_har_paper.py --phase pretrain --dataset hhar --epochs 120 --save_every 20

# Phase 2: Fine-tuning
python experiments/run_qcl_har_paper.py --phase finetune --dataset hhar --epochs 100 --checkpoint results/qcl_paper_checkpoint_epoch120_hhar.pt
```

### 3. Multi-Positive Sample QCL (MPSQCL HAR)
Train the multi-positive sample quantum contrastive pipeline manually:
```bash
# Phase 1: Self-supervised pre-training (4 views)
python experiments/run_mpsqcl_har.py --phase pretrain --dataset ucihar --epochs 150 --n_views 4 --save_every 30

# Phase 2: Supervised fine-tuning
python experiments/run_mpsqcl_har.py --phase finetune --dataset ucihar --epochs 100 --checkpoint results/mpsqcl_encoder_pretrained_ucihar.pt
```
