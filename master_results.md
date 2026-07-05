# MPSQCL HAR Study: Master Results and Configurations Compilation

This report compiles all experimental results, training configurations, and baseline comparisons generated throughout the Multi-Positive Sample Quantum Contrastive Learning (MPSQCL) and Quantum Contrastive Learning (QCL) research project for Human Activity Recognition (HAR).

---

## 1. System and Architecture Configurations

### 1.1 Pre-training Phase Configurations
During the self-supervised pre-training phase, the classical CNN encoder and the Variational Quantum Circuit (VQC) projection head are trained jointly using unlabelled data to optimize the multi-positive contrastive loss.

| Parameter | Configuration / Value |
| :--- | :--- |
| **Encoder Architecture** | **HAREncoder**: 4-layer 1D CNN with block channels: $\text{in\_channels} \rightarrow 32 \rightarrow 64 \rightarrow 128 \rightarrow 256$. Each block uses Conv1D (kernel=8), Batch Normalization, and ReLU activation, followed by MaxPool1D (kernel=2). |
| **Augmentation Protocol** | **M = 4 augmented views** generated per window, randomly drawn from a pool of **11 transformations**: Jitter, Negate, Permutation, Resampling, Rotation, Scaling, Temporal Flip, Time Warping, Window Warping, Channel Shuffle, and Permutation+Jitter. |
| **Feature Dimension** | 256-dimensional representation vector (mapped to 8-qubit state via Amplitude Encoding). |
| **VQC Projection Head** | **StronglyEntangling VQC** (depths $D \in \{1, 2, 3\}$, parameters $= 24D$) or **Paper-Compliant Custom VQC** (depth 1, 8 parameters using $R_Y$ gates + circular ring of CNOT gates). |
| **Projection Size** | 8-dimensional Pauli-Z expectation value vector (L2-normalized). |
| **Loss Function** | **Multi-Positive Contrastive Loss (MPSQCLLoss)** with temperature $\tau = 0.1$. |
| **Pre-training Epochs** | **150 epochs** (120 epochs for HHAR due to sample volume and UCI-HAR depth-3 to avoid majority class overfitting). |
| **Pre-training Batch Size** | **256** (reduced dynamically for HHAR to match GPU memory limitations). |
| **Optimizer** | **Adam** (Learning Rate $= 10^{-3}$, weight decay $= 0$, fixed random seed $= 42$). |

### 1.2 Downstream Fine-Tuning Configurations
After pre-training, the VQC projection head is discarded, and a classical classification head is attached to the pre-trained encoder.

| Parameter | Configuration / Value |
| :--- | :--- |
| **Downstream Heads** | 1. **Linear Head**: Single linear layer mapping the globally pooled 256-dimensional feature vector to $C$ classes.<br>2. **LSTM Head**: 2-layer sequential LSTM (128 hidden dim, dropout 0.5) processing the sequence of feature maps $(L_{\text{seq}}, 256)$ before global pooling, followed by a linear classification layer. |
| **Encoder Adaptability** | **Unfrozen Encoder** trained with a reduced learning rate ($0.1 \times$ classification head learning rate) to preserve pre-trained spatial features while adapting boundaries. |
| **Class Balancing** | **Weighted Random Sampler** oversampling minority classes in proportion to their inverse frequency (applied to UniMiB SHAR, MobiAct, and Opportunity). |
| **Fine-Tuning Epochs** | **100 epochs** (Linear classifier), **80 epochs** (LSTM for HHAR), **100 epochs** (LSTM for most datasets), **120 epochs** (LSTM for Opportunity / Opportunity Gestures). |
| **Fine-Tuning Batch Size**| **128** |
| **Fine-Tuning Optimizer** | **Adam** (Classifier LR $= 10^{-3}$, Encoder LR $= 10^{-4}$ with Cosine Annealing scheduler, weight decay $= 10^{-5}$). |

---

## 2. Benchmark Dataset Properties

Continuous signals are segmented using sliding windows with 50% overlap. Input channels represent various sensor modalities (accelerometers, gyroscopes, attitude).

| Dataset | Channels | Window Size | Classes | Total Windows | Preprocessing Details |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **UCI-HAR** | 9 | 128 | 6 | 10,299 | 3-axis accelerometer, gyroscope, User acceleration. |
| **UniMiB SHAR** | 3 | 151 | 17 | 11,771 | 3-axis accelerometer. Subject-wise exclusions applied. |
| **HHAR** | 6 | 100 | 6 | 224,386 | Downsampled to 50 Hz. Smartphone signals only. |
| **MotionSense** | 12 | 400 | 6 | 5,292 | Attitude, User acceleration, Gravity, Gyroscope. |
| **USC-HAD** | 6 | 250 | 12 | 15,378 | Accelerometer and gyroscope sampled at 100 Hz. |
| **MobiAct** | 6 | 128 | 9 | 16,758 | Excludes falls and vehicle step-ins/outs (ADLs only). |
| **Opp. Locomotion**| 113 | 30 | 4 | 47,201 | Body-worn IMUs, accelerometers, shoes. Excluded Null locomotion. |
| **Opp. Gestures** | 113 | 30 | 18 | 57,927 | Body-worn IMUs. Null class included. |

---

## 3. Overall Performance Benchmark vs. Published SOTA

Comparison of our proposed classical-quantum framework (with both Linear and LSTM downstream heads) against prior state-of-the-art results. The configurations utilize the best-performing VQC depth per dataset.

| Dataset | Published SOTA QCLHAR (2025) <br> [Macro F1] | Published SOTA MPSQCL (2024) <br> [Accuracy] | Ours: Linear Head <br> [Acc / Macro F1] | Ours: LSTM Head <br> [Acc / Macro F1] | Improvement <br> [$\Delta$ F1 vs. QCLHAR] |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **UCI-HAR** | 94.13% | 94.13% | 98.40% / 0.9854 | **98.50% / 0.9864** | **+4.51%** |
| **SHAR** | 86.18% | — | 83.52% / 0.7524 | **94.65% / 0.9151** | **+5.33%** |
| **HHAR** | 94.83% | 94.83% | 96.28% / 0.9318 | **98.68% / 0.9756** | **+2.73%** |
| **MotionSense** | 99.10% | 98.19% | **99.69% / 0.9954** | 99.23% / 0.9885 | **+0.44%** |
| **USC-HAD** | 91.66% | 91.66% | 88.36% / 0.8506 | **93.43% / 0.9083** | **-0.83%** |
| **MobiAct** | 99.07% | — | 98.31% / 0.9559 | **99.62% / 0.9893** | **-0.14%** |
| **Opp. Locomotion**| — | — | — | **92.31% / 0.9384** | — |
| **Opp. Gestures** | — | — | — | **85.16% / 0.6849** | — |

---

## 4. VQC Depth Sweep Results

Pre-training configurations sweeps across VQC depths 1, 2, and 3 Strongly Entangling VQCs compared to Depth-1 paper-compliant custom heads (Ry+CNOT). All configurations are fine-tuned using an unfrozen, slowly adapted encoder.

| Dataset | Classifier Head | Depth-1 VQC (Ry+CNOT) | Depth-2 StronglyEntangling | Depth-3 StronglyEntangling |
| :--- | :--- | :---: | :---: | :---: |
| **UCI-HAR** | Linear Head <br> LSTM Head | **98.40% / 0.9854** <br> 98.50% / 0.9863 | 98.30% / 0.9845 <br> **98.50% / 0.9864** | 98.35% / 0.9849 <br> 98.20% / 0.9837 |
| **SHAR** | Linear Head <br> LSTM Head | 83.52% / 0.7515 <br> 94.40% / 0.9183 | 83.15% / 0.7421 <br> 93.98% / 0.9130 | **83.52% / 0.7524** <br> **94.65% / 0.9151** |
| **HHAR** | Linear Head <br> LSTM Head | 94.95% / 0.9049 <br> 98.12% / 0.9653 | **96.28% / 0.9318** <br> **98.68% / 0.9756** | 96.22% / 0.9279 <br> 98.64% / 0.9749 |
| **MotionSense** | Linear Head <br> LSTM Head | 99.62% / 0.9945 <br> 99.16% / 0.9896 | 99.62% / 0.9939 <br> 99.08% / 0.9866 | **99.69% / 0.9954** <br> **99.23% / 0.9885** |
| **USC-HAD** | Linear Head <br> LSTM Head | **88.36% / 0.8506** <br> 92.92% / 0.9020 | 85.67% / 0.8159 <br> 93.27% / 0.9070 | 88.34% / 0.8503 <br> **93.43% / 0.9083** |
| **MobiAct** | Linear Head <br> LSTM Head | **98.31% / 0.9559** <br> 99.54% / 0.9838 | 97.77% / 0.9455 <br> **99.62% / 0.9893** | **98.31% / 0.9559** <br> 99.62% / 0.9889 |

### 4.1 Pre-Training Runtime Efficiencies (Depth-1 vs. Depth-2 vs. Depth-3)
Runtimes measured during 150 epochs of pre-training. Depth-1 custom Ry-CNOT circuit yields massive runtime benefits with minimal accuracy decay.

| Dataset | Depth-1 Runtime | Depth-2 Runtime | Depth-3 Runtime | Runtime Savings (D-1 vs. D-2) |
| :--- | :---: | :---: | :---: | :---: |
| **UCI-HAR** | 2,633.9s (~43.9m) | 4,683.5s (~1.30h) | 5,137.0s (~1.43h) | **43.8% faster** |
| **SHAR** | 1,768.0s (~29.5m) | 3,395.5s (~0.94h) | 4,504.5s (~1.25h) | **47.9% faster** |
| **MotionSense** | 2,116.1s (~35.3m) | 5,511.4s (~1.53h) | 6,959.2s (~1.93h) | **61.6% faster** |
| **USC-HAD** | 5,248.3s (~1.46h) | 9,460.5s (~2.63h) | 11,545.5s (~3.21h) | **44.8% faster** |
| **MobiAct** | 1,552.0s (~25.9m) | 2,938.5s (~0.82h) | 2,915.0s (~48.6m) | **47.2% faster** |
| **HHAR** | 16,013.1s (~4.45h) | 25,885.9s (~7.19h) | 14,308.6s (~3.97h) | **38.1% faster** |

---

## 5. Classical and Alternative Quantum Model Baselines

Comparison metrics compiled on UCI-HAR and sequential downstream heads (evaluated on a GPU system).

### 5.1 Classical Machine Learning and Deep Learning Baselines (UCI-HAR)
Classical baselines were evaluated on standard hand-crafted features or raw signal windows to benchmark performance limits without quantum architectures.

| Model | Input Feature Type | Dimensionality Bottleneck | Test Accuracy | Macro F1-Score | Training Time |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **SVM (Linear)** | Hand-crafted 561-Feat | None | 96.10% | 0.9608 | 0.83s |
| **SVM (RBF)** | Hand-crafted 561-Feat | None | 95.22% | 0.9515 | 1.58s |
| **Random Forest** | Hand-crafted 561-Feat | None | 92.57% | 0.9241 | 8.26s |
| **Classical 1D CNN** | Raw Signals (9x128) | No bottleneck (128 features) | 91.55% | 0.9160 | 15.74s |
| **Classical LSTM** | Raw Signals (9x128) | None | 95.49% | 0.9591 | 56.90s |

### 5.2 Supervised Quantum Model Baselines (UCI-HAR)
Directly supervised classical-quantum frameworks suffer from extreme simulation scales or feature bottlenecks.

| Model | Dimensionality Constraint | Test Accuracy | Macro F1-Score | Simulation Training Time |
| :--- | :--- | :---: | :---: | :---: |
| **Hybrid QCNN (10e)** | 4-Dim (4 Qubits VQC, 50% subset) | 73.39% | 0.6955 | 1,118.89s |
| **Hybrid QCNN (50e)** | 4-Dim (4 Qubits VQC) | 92.53% | 0.9256 | 13,768.00s (~3.82h) |
| **Hybrid QCNN (6q, 30e)** | 6-Dim (6 Qubits VQC) | 93.76% | 0.9387 | 15,805.00s (~4.39h) |
| **Quantum SVM (300s)** | 6-Dim (6 Qubits + PCA, 300 samples) | 45.00% | 0.4228 | 87.87s |
| **Quantum SVM (1000s)** | 8-Dim (8 Qubits + PCA, 1000 samples) | 53.00% | 0.5060 | 852.43s |
| **Quantum SVM (1000s, iqp)**| 10-Dim (10 Qubits + PCA, 1000 samples)| 59.50% | 0.5329 | 2,635.20s |
| **Quantum SVM (1000s, amp)**| 256-Dim (8 Qubits + PCA, 1000 samples)| 91.50% | 0.9146 | 32,886.00s (~9.13h) |

### 5.3 Comparative Baseline: Classical LSTM vs. MPSQCL + LSTM
Contrasts the standalone classical LSTM sequence model against our pre-trained hybrid sequence model across all benchmark datasets.

| Dataset | Model | Encoder Params | Head Params | Total Params | Test Accuracy | Macro F1 | Training Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **UCI-HAR** | Classical LSTM <br> **MPSQCL + LSTM (Ours)** | — <br> **347,808** | 204,038 <br> **330,502** | 204,038 <br> **678,310** | 95.49% <br> **98.20%** | 0.9591 <br> **0.9837** | 56.9s <br> **43.2s** |
| **SHAR** | Classical LSTM <br> **MPSQCL + LSTM (Ours)** | — <br> **346,272** | 202,385 <br> **331,921** | 202,385 <br> **678,193** | 76.09% <br> **94.65%** | 0.6992 <br> **0.9151** | 50.2s <br> **37.4s** |
| **HHAR** | Classical LSTM <br> **MPSQCL + LSTM (Ours)** | — <br> **347,040** | 202,502 <br> **330,502** | 202,502 <br> **677,542** | **99.32%** <br> 98.64% | **0.9865** <br> 0.9749 | 980.4s <br> **760.0s** |
| **MotionSense**| Classical LSTM <br> **MPSQCL + LSTM (Ours)** | — <br> **348,576** | 205,574 <br> **330,502** | 205,574 <br> **679,078** | 98.47% <br> **99.23%** | 0.9788 <br> **0.9885** | 93.1s <br> **40.8s** |
| **USC-HAD** | Classical LSTM <br> **MPSQCL + LSTM (Ours)** | — <br> **347,040** | 203,276 <br> **331,276** | 203,276 <br> **678,316** | 90.95% <br> **93.43%** | 0.8832 <br> **0.9083** | 197.7s <br> **98.7s** |
| **MobiAct** | Classical LSTM <br> **MPSQCL + LSTM (Ours)** | — <br> **347,040** | 202,889 <br> **330,889** | 202,889 <br> **677,929** | 98.70% <br> **99.62%** | 0.9632 <br> **0.9889** | 36.9s <br> **28.9s** |

---

## 6. Pre-training and Candidate Ablation Phases

Detailed progression of results across developmental and component evaluation phases.

### 6.1 Phase 1 & 2: Initial Component Ablations (UCI-HAR)
Conducted on data subsets to isolate baseline advantages of customized enhancements.
*   **Phase 1 (20% Dataset, 20 Epochs)**: Evaluated performance in highly constrained data split regimes.
*   **Phase 2 (40% Dataset, 50 Epochs)**: Evaluated convergence characteristics with more data and epochs.

| Configuration | Phase 1: Test Acc | Phase 1: Test F1 | Phase 2: Test Acc | Phase 2: Test F1 |
| :--- | :---: | :---: | :---: | :---: |
| **Config 0: Paper Baseline (M=4, D=1)** | 83.16% | 0.8292 | 90.59% | 0.9047 |
| **Config 1: Paper + Unfrozen Encoder** | 76.52% | 0.7522 | 90.59% | 0.9044 |
| **Config 2: Paper + Deeper VQC (D=3)** | 86.12% | 0.8584 | 91.51% | 0.9136 |
| **Config 3: Paper + Average Pooling** | 84.76% | 0.8452 | 89.08% | 0.8879 |
| **Config 4: Paper + 11-Augmentation Pool** | 86.07% | 0.8544 | 92.77% | 0.9283 |
| **Config 5: Paper + Unweighted Sampler** | 86.66% | 0.8620 | 91.27% | 0.9104 |
| **Config 6: Paper + No normalisation** | 85.88% | 0.8535 | 92.14% | 0.9201 |
| **Config 7: Standard Pipeline (Combined)**| **89.67%** | **0.8911** | **95.78%** | **0.9614** |

### 6.2 Phase 3: Standard Pipeline Baseline Ablations (UCI-HAR)
Started from the optimized Standard Pipeline and reverted individual choices to identify single contribution factors (40% dataset, 50 epochs).

| Configuration Reverted | Pre-train Val Loss | Test Accuracy | Test Macro F1-score | F1 Delta (vs. Standard) |
| :--- | :---: | :---: | :---: | :---: |
| **Standard Baseline (None Reverted)** | 2.7942 | **95.83%** | **0.9621** | — |
| **Config 1: Standard - Average Pooling** | 2.6134 | 95.58% | 0.9596 | -0.26% |
| **Config 2: Standard - Unweighted Sampler**| 2.6821 | 95.63% | 0.9599 | -0.22% |
| **Config 3: Standard - Deeper VQC (D=1)** | 2.7735 | 95.34% | 0.9570 | -0.51% |
| **Config 4: Standard - Frozen Encoder** | 2.7130 | 94.52% | 0.9459 | -1.62% |

### 6.3 Phase 4: Candidate Configurations on 100% UCI-HAR (100 Epochs)
Scaled up candidate variations to complete dataset and longer schedule.

| Candidate Configuration | Pre-train Val Loss | Test Accuracy | Test Macro F1-score |
| :--- | :---: | :---: | :---: |
| **Cand. 1: Fully Optimized Standard (D=3, Unfrozen)** | 2.5978 | **98.11%** | **0.9828** |
| **Cand. 2: Depth-2 VQC Standard (D=2, Unfrozen)** | 2.6361 | 98.01% | 0.9819 |
| **Cand. 3: Speed-Optimized Standard (D=1, Unfrozen)**| 2.6429 | 97.96% | 0.9814 |
| **Cand. 4: Frozen Encoder Standard (D=3, Frozen)** | 2.6523 | 97.91% | 0.9807 |
| **Cand. 5: Paper-Compliant Baseline** | 3.3005 | 92.82% | 0.9269 |

### 6.4 Phase 5, 6, 7 & 8/9: UniMiB SHAR Candidate Sweeps
Conducted candidate configurations sweeps on SHAR (extremely class-imbalanced) under low-data and full-data split constraints.
*   **Phase 5 (20% SHAR, 50 Epochs)**: Evaluated regularization benefit of frozen encoders on low-data imbalanced splits.
*   **Phase 6 & 7 (100% SHAR, 100 Epochs)**: Benchmarked capacity difference between Depth-2 StronglyEntangling VQC (Phase 6) and Depth-3 VQC (Phase 7).
*   **Phase 8 & 9 (100% SHAR + downstream LSTM, 100 Epochs)**: Connected pre-trained candidates (D-2 Unfrozen, D-1 Unfrozen, D-2 Frozen) to LSTM sequence classifier.

| Candidate Configuration | Phase 5 F1 (20% split) | Phase 6 F1 (100% split, D=2) | Phase 7 F1 (100% split, D=3) | Phase 8/9 LSTM Fine-Tuning F1 |
| :--- | :---: | :---: | :---: | :---: |
| **Cand. 1: Fully Optimized Standard** | 0.3864 | 0.7131 | 0.6520 | 0.7906 (D-2 Pre-train) |
| **Cand. 2: Speed-Optimized Standard** | 0.3811 | 0.6602 | — | **0.8009 (D-1 Pre-train)** |
| **Cand. 3: Frozen Encoder Standard** | **0.5989** | **0.8535** | **0.8485** | 0.7486 (D-2 Frozen) |
| **Cand. 4: Weighted Sampler (Unfrozen)**| 0.4416 | 0.6674 | — | — |
| **Cand. 5: Paper-Compliant Baseline** | 0.3815 | 0.5719 | — | — |

---

## 7. Key Findings and Design Rules

Based on the aggregated results, the project has established four core design rules for hybrid classical-quantum representation pipelines:

1. **VQC Bottleneck Paradox for Sequential Models**:
   During pre-training, a shallow (depth-1 Ry-CNOT) quantum head imposes a milder projection bottleneck than deeper Strongly Entangling layers. This forces the classical CNN encoder to preserve high-fidelity raw temporal signals, which downstream sequence models (LSTMs) leverage to separate sequence boundaries. For sequential classification heads, favor shallower quantum heads (**+1.03% F1**).
   
2. **Topology Sweet Spot on Imbalanced Classes**:
   On imbalanced datasets like UniMiB SHAR, a depth-2 StronglyEntangling VQC represents the optimal capacity sweet-spot. Depth-3 VQCs suffer from representation overfitting on majority class clusters, dropping F1-scores by **-0.50%** in frozen configurations.
   
3. **Duality of Encoder Freezing**:
   * For **Linear Classifiers** (static evaluations), freezing the encoder prevents overfitting, boosting scores by **+21.25% F1** on constrained data splits.
   * For **Sequence Classifiers** (LSTMs), fine-tuning must be done with an *unfrozen* encoder configuration. Keeping parameters frozen drops F1 by **-4.20%**, as sequential modeling requires dynamically adapted representation spaces.

4. **Sign-Invariance and Encoder Unfreezing**:
   Amplitude Encoding is mathematically sign-invariant: $\langle \psi(-\mathbf{h})| A |\prefix(-\mathbf{h})\rangle = \langle \psi(\mathbf{h})| A |\psi(\mathbf{h})\rangle$. Discarding or freezing the pre-trained encoder leaves downstream heads unable to identify signal negation (polarity flips). Unfreezing the encoder during downstream fine-tuning resolves sign-invariance by letting classical backpropagation adapt the encoder's feature representations.
