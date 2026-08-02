# QMLHAR: Quantum Machine Learning for Human Activity Recognition

This repository explores the application of Quantum Machine Learning (QML) models to Human Activity Recognition (HAR) tasks across six benchmark datasets. It implements and evaluates Quantum Contrastive Learning (QCL) and Multi-Positive Sample Quantum Contrastive Learning (MPSQCL) pipelines, comparing them against classical and early-stage quantum baselines.

---

## Project Structure

- `src/`
  - `data/`
    - `har_datasets_paper.py`: Paper-compliant loaders for 6 HAR datasets (`ucihar`, `shar`, `hhar`, `motionsense`, `uschad`, `mobiact`) supporting standard and weighted class random sampling.
    - `augmentations.py`: Full augmentation suite (11 strategies: jitter, negate, permute, resample, rotate, scale, temporal flip, time warp, window warp, channel shuffle, permutation-jitter) for standard contrastive pre-training.
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
  - `run_mpsqcl_lstm.py`: Hybrid pre-trained MPSQCL CNN encoder + LSTM classifier pipeline.
  - `run_lstm_baseline.py`: Standalone classical LSTM sequence model baseline.
  - `run_mpsqcl_ablation.py`: Pipeline component ablation study runner.
  - `run_mpsqcl_views_ablation.py`: Multi-view count ($M \in \{2, 3, 4, 5, 6\}$) ablation study runner.
  - `run_pipeline.py` / `run_pipeline_mpsqcl.py`: Automated pipeline runners for sequentially executing MPSQCL across all datasets.
- `results/`: Saved metrics, training logs, pre-training/fine-tuning history, model checkpoints, and ablation study reports (`mpsqcl_ablation_results.md`, `mpsqcl_views_ablation_results.md`).
- `qclimplementation.md`: Detailed implementation specification for the QCL architecture.
- `mpsqclimplementation.md`: Detailed implementation specification for the MPSQCL architecture.
- `master_results.md`: Consolidated master experimental results across all datasets, ablation studies, and baselines.
- `report.md`: Full benchmark comparison report across evaluated HAR datasets.

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
| **Classical LSTM** | Raw Signals (9x128) | None | **95.49%** | **0.9591** | **65.3s** | Processes temporal patterns sequentially, significantly outperforming the 1D CNN. |
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
| **MPSQCL + LSTM (Ours, Frozen)** | Raw Signals (9x128) | None | **96.99%** | **0.9720** | **32.8s** | **Completed.** Frozen standard encoder (`HAREncoder`) + LSTM classifier head (100e). Outperforms baseline by +1.50% Acc. |
| **MPSQCL + LSTM (Ours, Fine-tuned)** | Raw Signals (9x128) | None | **98.20%** | **0.9837** | **43.2s** | **Completed.** Pre-trained standard MPSQCL CNN encoder + LSTM jointly fine-tuned (100e) with equal LR. Outperforms baseline by +2.71% Acc. |




### Multi-Dataset QCL, MPSQCL, & LSTM Results

| Model | Dataset | Test Accuracy | Macro F1 | Training Time | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MPSQCL HAR (150e)** | UCI-HAR | **98.50%** | **0.9858** | 1.54 hours | 150-epoch pre-training (M=4) + 100-epoch fine-tuning |
| **MPSQCL HAR (150e)** | MotionSense | **99.69%** | **0.9954** | 1.94 hours | Near-perfect classification |
| **MPSQCL HAR (120e)** | MobiAct | **98.31%** | **0.9559** | 48.6 mins | Excluding falls, 9 ADL classes |
| **MPSQCL HAR (150e)** | USC-HAD | **88.34%** | **0.8503** | 3.23 hours | Hardest dataset due to 12 activity classes |
| **MPSQCL HAR (150e)** | SHAR | **83.52%** | **0.7524** | 1.26 hours | Layout scrambling bug fixed |
| **MPSQCL HAR (150e)** | HHAR | **96.22%** | **0.9279** | 4.21 hours | 150-epoch pre-training (M=4) + 100-epoch fine-tuning |
| **QCL HAR (120e)** | MotionSense | **95.79%** | **0.9596** | 25.8 mins | Standard QCL baseline |
| **QCL HAR (120e)** | HHAR | **93.28%** | **0.8816** | 7.66 hours | Large dataset (224k windows) |
| **QCL HAR (120e)** | MobiAct | **91.70%** | **0.8180** | 11.5 mins | Standard QCL baseline |
| **QCL HAR (120e)** | SHAR | **91.06%** | **0.8667** | 2.28 hours | Standard QCL baseline |
| **QCL HAR (120e)** | USC-HAD | **71.97%** | **0.7002** | ~25 mins | Standard QCL baseline |
| **Classical LSTM (100e)** | UCI-HAR | **95.49%** | **0.9591** | 56.9s | Classical LSTM baseline with class balancing sampler |
| **Classical LSTM (100e)** | SHAR | **76.09%** | **0.6992** | 50.2s | Classical LSTM baseline with class balancing sampler |
| **Classical LSTM (100e)** | MotionSense | **98.47%** | **0.9788** | 93.1s | Classical LSTM baseline with class balancing sampler |
| **Classical LSTM (100e)** | USC-HAD | **90.95%** | **0.8832** | 197.7s | Classical LSTM baseline with class balancing sampler |
| **Classical LSTM (100e)** | MobiAct | **98.70%** | **0.9632** | 36.9s | Classical LSTM baseline with class balancing sampler |
| **Classical LSTM (100e)** | HHAR | **99.32%** | **0.9865** | 16.3 mins | Classical LSTM baseline with class balancing sampler |
| **MPSQCL + LSTM (Ours, Frozen, 100e)** | UCI-HAR | **96.99%** | **0.9720** | 32.8s | Pre-trained standard MPSQCL CNN encoder + LSTM classifier (frozen) |
| **MPSQCL + LSTM (Ours, Fine-tuned, 100e)** | UCI-HAR | **98.20%** | **0.9837** | 43.2s | Pre-trained standard MPSQCL CNN encoder + LSTM classifier (unfrozen) |

### Consolidated Parameter & Performance Benchmark

The table below contrasts the classical LSTM baseline against the pre-trained hybrid MPSQCL + LSTM model across all 6 benchmark datasets. We compare sequence configurations, parameter divisions, classification metrics, and training runtimes (evaluated on a CUDA-capable system).

| Dataset | Channels | Classes | Model | Encoder Params | Head Params | Total Params | Test Accuracy | Macro F1 | Training Time |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **UCI-HAR** | 9 | 6 | Classical LSTM | - | 204,038 | 204,038 | 95.49% | 0.9591 | 56.9s |
| | | | **MPSQCL + LSTM (Ours)** | **347,808** | **330,502** | **678,310** | **98.20%** | **0.9837** | **43.2s** |
| **SHAR** | 3 | 17 | Classical LSTM | - | 202,385 | 202,385 | 76.09% | 0.6992 | 50.2s |
| | | | **MPSQCL + LSTM (Ours)** | **346,272** | **331,921** | **678,193** | **94.65%** | **0.9151** | **37.4s** |
| **HHAR** | 6 | 6 | Classical LSTM | - | 202,502 | 202,502 | **99.32%** | **0.9865** | 980.4s |
| | | | **MPSQCL + LSTM (Ours)** | **347,040** | **330,502** | **677,542** | **98.88%** | **0.9791** | **901.6s** |
| **MotionSense** | 12 | 6 | Classical LSTM | - | 205,574 | 205,574 | 98.47% | 0.9788 | 93.1s |
| | | | **MPSQCL + LSTM (Ours)** | **348,576** | **330,502** | **679,078** | **99.23%** | **0.9885** | **40.8s** |
| **USC-HAD** | 6 | 12 | Classical LSTM | - | 203,276 | 203,276 | 90.95% | 0.8832 | 197.7s |
| | | | **MPSQCL + LSTM (Ours)** | **347,040** | **331,276** | **678,316** | **93.43%** | **0.9083** | **98.7s** |
| **MobiAct** | 6 | 9 | Classical LSTM | - | 202,889 | 202,889 | 98.70% | 0.9632 | 36.9s |
| | | | **MPSQCL + LSTM (Ours)** | **347,040** | **330,889** | **677,929** | **99.62%** | **0.9889** | **28.9s** |





### Detailed Comparison: Linear Head vs. LSTM Head (MPSQCL Ours)

The table below directly contrasts the classical Linear classification head (used in standard MPSQCL fine-tuning) against our hybrid LSTM head across all 6 benchmark datasets. Both configurations use the standard unfrozen `HAREncoder` pre-trained under multi-positive quantum contrastive learning.

| Dataset | Metric | MPSQCL (Ours) + Linear Head | MPSQCL (Ours) + LSTM Head | Difference ($\Delta$) | Head Params (Linear vs. LSTM) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **UCI-HAR** | Accuracy <br> Macro F1 <br> Time | **98.35%** <br> **0.9849** <br> **37.5s** | 98.20% <br> 0.9837 <br> 43.2s | -0.15% <br> -0.12% <br> +5.7s | **1,542** <br> vs. <br> 330,502 |
| **SHAR** | Accuracy <br> Macro F1 <br> Time | 83.52% <br> 0.7524 <br> **29.3s** | **94.65%** <br> **0.9151** <br> 37.4s | **+11.13%** <br> **+16.27%** <br> +8.1s | **4,369** <br> vs. <br> 331,921 |
| **HHAR** | Accuracy <br> Macro F1 <br> Time | 92.97% <br> 0.8725 <br> **143.5s** | **98.88%** <br> **0.9791** <br> 901.6s | **+5.91%** <br> **+10.66%** <br> +758.1s | **1,542** <br> vs. <br> 330,502 |
| **MotionSense** | Accuracy <br> Macro F1 <br> Time | **99.69%** <br> **0.9954** <br> **30.1s** | 99.23% <br> 0.9885 <br> 40.8s | -0.46% <br> -0.69% <br> +10.7s | **1,542** <br> vs. <br> 330,502 |
| **USC-HAD** | Accuracy <br> Macro F1 <br> Time | 88.34% <br> 0.8503 <br> **66.1s** | **93.43%** <br> **0.9083** <br> 98.7s | **+5.09%** <br> **+5.80%** <br> +32.6s | **3,084** <br> vs. <br> 331,276 |
| **MobiAct** | Accuracy <br> Macro F1 <br> Time | 98.31% <br> 0.9559 <br> **22.8s** | **99.62%** <br> **0.9889** <br> 28.9s | **+1.31%** <br> **+3.30%** <br> +6.1s | **2,313** <br> vs. <br> 330,889 |

### Comparison to Published SOTA Papers

The hybrid MPSQCL + LSTM model is evaluated against the SOTA results from the two distinct source papers:

#### 1. Published MPSQCL SOTA (Qproj, IEEE Globecom 2024)

This paper proposed Multi-Positive Sample Quantum Contrastive Learning (MPSQCL) using a quantum projection head (**Qproj**). The table below contrasts the published Qproj results against our hybrid model:

| Dataset | Published Qproj SOTA (Ren et al., 2024) | MPSQCL + LSTM (Ours) | Improvement ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **UCI-HAR** | 94.13% | **98.20%** | **+4.07%** |
| **HHAR** | 94.83% | **98.64%** | **+3.81%** |
| **MotionSense** | 98.19% | **99.23%** | **+1.04%** |
| **USC-HAD** | 91.66% | **93.43%** | **+1.77%** |

#### 2. Published QCL SOTA (QCLHAR, Smart Health 2025)

This paper proposed standard Quantum Contrastive Learning (QCL) for HAR (**QCLHAR**). The table below contrasts the published QCLHAR results against our hybrid model:

| Dataset | Published QCLHAR SOTA (Ren et al., 2025) | MPSQCL + LSTM (Ours) | Improvement ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **UCI-HAR** | 94.13% | **98.20%** | **+4.07%** |
| **SHAR** | 86.18% | **94.65%** | **+8.47%** |
| **HHAR** | 94.83% | **98.64%** | **+3.81%** |
| **MotionSense** | 99.10% | **99.23%** | **+0.13%** |
| **USC-HAD** | 91.66% | **93.43%** | **+1.77%** |
| **MobiAct** | 99.07% | **99.62%** | **+0.55%** |


### Dataset Processing Differences from Source Papers

Upon cross-referencing the dataset configuration in our codebase against the published text files in the `papers/` directory, we identified two specific discrepancies in channel dimensions and class structures:

1. **MotionSense Channel Count (12 channels vs. 3 channels)**:
   * **Our Codebase:** We extract **12 channels** representing raw sensor readings (attitude pitch/roll/yaw, gravity x/y/z, rotation rate x/y/z, and user acceleration x/y/z).
   * **Globecom 2024 (MPSQCL) Paper:** Section IV.A.1 explicitly states: *"This paper uses the signals from the three-axis accelerometer sensor."* (i.e. **3 channels**). This higher input channel count in our codebase provides our model with attitude and gyroscopic data, which explains our superior classification baseline (98.47% vs. paper's SimCLR 97.85%).

2. **MobiAct Class Count (9 classes vs. 11 classes)**:
   * **Our Codebase:** Our preprocessing script (`preprocess_mobiact`) parses only **9 classes** (STD, WAL, JOG, JUM, STU, STN, SCH, CSI, CSO), which map to indices 0–8, excluding fall classes and car step-in/out.
   * **Smart Health 2025 (QCLHAR) Paper:** Table 2 in the paper lists **11 classes** (indices 0–10), which includes *9: car step-in* and *10: car step-out*. Both codebase and paper utilize the same **6 channels** (accelerometer + gyroscope).

3. **Other Datasets (UCI-HAR, SHAR, HHAR, USC-HAD)**:
   * **UCI-HAR:** Exactly matches (9 channels, 6 classes, window size 128, 50% overlap).
   * **SHAR (UniMiB SHAR):** Exactly matches (3 channels, 17 classes, window size 151, 10 out of 30 subjects with incomplete classes excluded).
   * **HHAR:** Exactly matches (6 channels, 6 classes, window size 100, downsampled to 50 Hz, smartphone only).
   * **USC-HAD:** Exactly matches (6 channels, 12 classes, window size 250, 100 Hz sampling).

### Paper Reference Comparisons (UCI-HAR)

| Model | Source | Test Accuracy | Macro F1 |
| :--- | :--- | :--- | :--- |
| **MPSQCL (M=4, 50e/30e)** | Paper | **97.28%** | **0.9722** |
| **MPSCL (M=4, classical head)** | Paper | **97.50%** | **0.9745** |
| **TS-TCC (classical SOTA)** | Paper | **96.41%** | **0.9635** |
| **QSSL (VQC in encoder)** | Paper | **83.59%** | **0.8351** |

---

## Ablation Studies & Architectural Trade-offs

### 1. Pipeline Component Ablation Study (UCI-HAR, 20% Subset, 20 Pre-train / 20 Fine-tune Epochs)

The table below evaluates individual component modifications against the strict paper-compliant baseline under controlled settings:

| Configuration | Pre-train Val Loss | Test Accuracy | Test Macro F1 | Delta F1 vs. Paper Baseline | Key Takeaway / Insight |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Config 0: Paper Baseline** | 3.8601 | 83.16% | 0.8292 | - | Default 2-view, frozen encoder, depth-1 Ry+CNOT VQC, max pooling. |
| **Config 1: Unfrozen Encoder** | 3.5746 | 76.52% | 0.7522 | -7.70% | Unfreezing 348k params in short runs (20e) overfits; stable on 100e+ runs. |
| **Config 2: Deeper VQC (D=3)** | 3.6691 | 86.12% | 0.8584 | +2.92% | StronglyEntangling VQC ($D=3$) increases representation mapping capacity. |
| **Config 3: Average Pooling** | 3.8858 | 84.76% | 0.8452 | +1.60% | Avg pooling preserves global temporal context better than max pooling. |
| **Config 4: 11-Augmentations** | 3.6292 | 86.07% | 0.8544 | +2.52% | Broader augmentation pool prevents encoder shortcut learning. |
| **Config 5: Unweighted Sampler**| 3.8166 | 86.66% | 0.8620 | +3.28% | Balanced datasets (UCI-HAR) perform better without oversampling. |
| **Config 6: No Feature L2-Norm** | 4.0917 | 85.88% | 0.8535 | +2.43% | Preserves magnitude/scale information for downstream classification. |
| **Config 7: Standard Pipeline** | 3.3437 | **89.67%** | **0.8911** | **+6.20%** | **Combined optimizations yield best performance overall (+6.20% F1).** |

### 2. Multi-View Count ($M$) Ablation Study (UCI-HAR, 20% Subset, 20 Pre-train / 20 Fine-tune Epochs)

Empirical evaluation of downstream classification accuracy, macro F1 score, and pre-training runtime scaling across $M \in \{2, 3, 4, 5, 6\}$ positive augmented views:

| Number of Views ($M$) | Positive Pair Density | Pre-train Val Loss | Test Accuracy | Test Macro F1 | Delta F1 vs. $M=2$ | Pre-train Time (s) | Sec / Epoch |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M = 2** | 1 pair / sample | 2.4372 | 87.19% | 0.8654 | - | 110.4s | 5.52s |
| **M = 3** | 2 pairs / sample | 2.5869 | 88.60% | 0.8823 | +1.69% | 151.5s | 7.57s |
| **M = 4** | 3 pairs / sample | 2.6198 | 88.79% | 0.8839 | +1.84% | 191.8s | 9.59s |
| **M = 5** | 4 pairs / sample | 2.8145 | 89.42% | 0.8912 | +2.58% | 258.6s | 12.93s |
| **M = 6** | 5 pairs / sample | **2.9698** | **90.20%** | **0.8996** | **+3.42%** | 294.6s | 14.73s |

#### Key Takeaways & Trade-offs:
1. **Multi-Positive Representation Boost**: Increasing $M$ from 2 (SimCLR baseline) to 6 yields a continuous **+3.42% F1 boost** (up to 90.20% accuracy), demonstrating that multi-positive contrastive learning enhances signal invariance.
2. **Runtime Scaling**: Pre-training execution time scales linearly with $M$ (from 5.52s/epoch at $M=2$ to 14.73s/epoch at $M=6$). $M=4$ remains the recommended balance point between speed and performance.

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

### 4. Classical LSTM Baseline
Train and evaluate the classical LSTM classifier baseline:
```bash
python experiments/run_lstm_baseline.py --dataset ucihar --epochs 100 --batch_size 128
```

### 5. MPSQCL Encoder + LSTM Classifier
Train and evaluate the hybrid pre-trained MPSQCL CNN encoder + LSTM classifier:
```bash
# Fine-tune the pre-trained MPSQCL encoder jointly with the LSTM (Recommended)
python experiments/run_mpsqcl_lstm.py --dataset ucihar --epochs 50 --batch_size 128 --freeze_encoder False

# Evaluate with frozen pre-trained encoder (feature extraction only)
python experiments/run_mpsqcl_lstm.py --dataset ucihar --epochs 50 --batch_size 128 --freeze_encoder True
```

### 6. Pipeline Component & View Ablation Studies
Run component ablation study:
```bash
./.venv/bin/python experiments/run_mpsqcl_ablation.py --subset_fraction 0.2 --epochs_pretrain 20 --epochs_finetune 20
```

Run number of views ($M \in \{2, 3, 4, 5, 6\}$) ablation study:
```bash
./.venv/bin/python experiments/run_mpsqcl_views_ablation.py --dataset ucihar --subset_fraction 0.2 --epochs_pretrain 20 --epochs_finetune 20 --views 2 3 4 5 6
```


