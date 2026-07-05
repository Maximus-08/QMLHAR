## Quantum Circuit (VQC / Projection Head)

- Number of qubits: **8**
- PQC depth D: **1** (tested 1–4, depth 1 was optimal)
- Learnable parameters in projection head: **8** (one Ry rotation angle θ per qubit)
- Encoding method: **Amplitude encoding** (maps 256-dim vector onto 8 qubits)
- Gates used: **Ry rotation gates + CNOT gates** between adjacent qubits
- Measurement: **Pauli-Z operator** on each qubit
- Implementation library: **PennyLane** (as a torch layer)

---

## Encoder (Classical CNN)

- Type: **Fully Convolutional Neural Network (FCN)**
- Number of conv layers: **4**
- Channel counts: **32 → 64 → 128 → 256**
- Kernel size: **8** for all layers
- Stride: **1** for all layers
- After each conv layer: **BatchNorm → ReLU → MaxPool**
- Dropout: **p = 0.35**, applied only after the first conv layer
- Output: **256-dimensional** feature vector

---

## Classifier

- Type: **Single linear layer**
- Loss: **Cross-entropy loss**

---

## Contrastive Learning (NT-Xent Loss)

- Temperature parameter τ: **0.1** (default)
- Similarity metric: **cosine similarity** using quantum inner product

---

## Training Setup

- Batch size: **128** (64 for MotionSense due to memory constraints)
- Optimizer: **Adam**
- Learning rate schedule: **cosine annealing decay**
- Learning rate stage 1 (pre-training): **3e-3** (1e-2 for SHAR)
- Learning rate stage 2 (fine-tuning): **1e-1**
- Total epochs: **120**
- Model selection: lowest validation loss during pre-training

---

## Data Augmentation

Two strategies used in the paper:
- **Resampling** — upsampling + downsampling to simulate different sampling frequencies
- **Negated** — vertical flip of the signal

---

## Data Preprocessing

- Normalization applied to all datasets
- Sliding window segmentation with **50% overlap** across all datasets
- Window sizes per dataset:
  - UCI-HAR: **128** time steps
  - SHAR: **151** time steps
  - HHAR: **100** time steps
  - MotionSense: **400** time steps
  - USC-HAD: **250** time steps
  - MobiAct: not explicitly stated
- Train/val/test split: **64% / 16% / 20%**

---

## Dataset-specific Constraints

- SHAR: **10 out of 30 participants removed** (incomplete activity classes)
- HHAR: **only phone device data used**, downsampled to **50 Hz** (to reduce quantum circuit computation time)

---

## Noise Simulation (NISQ experiments)

- Low noise: **10%** probability of random Rx rotation gate applied per qubit before each gate operation
- High noise: **70%** probability
- Rotation angles drawn from a **normal distribution**

---

## Evaluation Metrics

- Primary: **F1-score**
- Secondary: **Test Accuracy** = N_correct / N_total

---
### 2. Key Differences in Our Implementations

| Feature / Setting | Baseline QCL (`run_qcl_har.py`) | Paper-Compliant QCL (`run_qcl_har_paper.py`) | Performance / Speed Impact |
| :--- | :--- | :--- | :--- |
| **Encoder Freezing** | **Unfrozen** (encoder parameters fine-tuned slowly with `lr = 0.0001`) | **Frozen** (encoder parameters locked, only linear layer trained with `lr = 0.1`) | Unfreezing the encoder allows representation adaptation, boosting baseline accuracy to **95.83%** compared to **85.93%** for the frozen representation at 50 epochs. |
| **VQC Depth & Structure** | Depth **$D=3$** (`StronglyEntanglingLayers` with 24 parameters) | Depth **$D=1$** (Custom `RY` + CNOT ring with 8 parameters) | Deeper VQC layers increase projection head capacity, but significantly scale gradient computation time. |
| **Data Augmentation** | **11 Strategies** (Jitter, Negate, Scale, Resample, Permute, Temporal Flip, Rotate, Time Warp, Window Warp, Channel Shuffle, Permutation-Jitter) | **2 Strategies** (Resample and Negate only) | Diverse augmentations prevent the model from learning easy shortcut features (e.g. noise or specific orientations), forcing robust semantic representation learning. |
| **Class Balancing** | Standard random shuffle loader (no weights) | **Weighted Random Sampling** (oversamples minority classes) | Balances class representation inside batches, resulting in more robust self-supervised contrastive learning. |
| **Signal Normalization** | Raw signal normalization bounded to $[-1, 1]$ | Bounded to $[-1, 1]$ (window-by-window standardization commented out) | Window-by-window standardization destroys absolute amplitude features (e.g., distinguishing high-intensity walking from low-intensity sitting), which drops accuracy to **77.83%**. Commenting it out recovers accuracy to **85.93%**. |
| **Simulator Gradient Method** | Parameter-Shift (implicit default) | **Statevector Backpropagation** (`diff_method="backprop"`) | Parameter-Shift requires running the VQC $2 \times N_{\text{params}}$ times per sample ($2 \times 36 \times 64 = 4608$ simulations per batch). Backpropagation runs in a single pass, speeding up training by **100x** (3.6s vs. 22s per epoch). |
| **Dataset Partitioning** | Subject-wise train/test split (no shared subjects) | Sample-wise random split ($64\%/16\%/20\%$) | Sample-wise splitting places overlapping windows from the same subjects in both train and test sets, which is simpler to classify. |