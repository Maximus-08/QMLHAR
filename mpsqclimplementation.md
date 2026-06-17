## Quantum Circuit (VQC / Projection Head)

- Number of qubits: **8**
- PQC depth D: **3** (standard pipeline uses `StronglyEntanglingLayers`; paper-compliant uses D=1 with custom Ry + CNOT ring)
- Learnable parameters (standard): **72** (3 layers × 8 qubits × 3 rotation angles)
- Learnable parameters (paper): **8** (one Ry rotation angle θ per qubit)
- Encoding method: **Amplitude encoding** (maps 256-dim vector onto 8 qubits)
- Gates used (standard): **StronglyEntanglingLayers** (Rot + CNOT entangling)
- Gates used (paper): **Ry rotation gates + CNOT gates** between adjacent qubits
- Measurement: **Pauli-Z operator** on each qubit
- Implementation library: **PennyLane** (as a torch layer, `diff_method="backprop"`)

---

## Encoder (Classical CNN)

Identical to QCL — same 4-layer Fully Convolutional Network (FCN):

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

- Type: **Single linear layer** (256 → num_classes)
- Loss: **Cross-entropy loss**

---

## Multi-Positive Sample Contrastive Loss (MPSQCL Loss)

The core difference from QCL. Instead of NT-Xent (2 views), MPSQCL uses a multi-positive formulation:

- **M views** per sample (default M=4)
- For a batch of N samples, there are **M × N** total representations
- For each view $z_i$, the **positive set** $P(i)$ contains the other $M - 1$ views of the same original sample
- Loss equation (Ren et al., 2024, Eq. 1):

$$L_i = \frac{1}{|P(i)|} \sum_{n \in P(i)} -\log \frac{\exp(\text{sim}(z_i, z_n) / \tau)}{\sum_{k \notin P(i)} \exp(\text{sim}(z_i, z_k) / \tau)}$$

- Temperature parameter τ: **0.1** (default)
- Similarity metric: **cosine similarity**
- Denominator includes: all negative views (different samples) + the anchor itself (paper formulation)
- Optional: `exclude_anchor_from_denominator` flag (set to `False` by default to match paper)

---

## Data Augmentation

### Standard Pipeline (`augmentations.py`) — 7 strategies:
1. **Jitter** — additive Gaussian noise (σ = 0.05)
2. **Negate** — vertical flip of the signal (x → -x)
3. **Permute** — slice into 5 segments and shuffle
4. **Resample** — random temporal resampling (ratio 0.5–2.0×) via linear interpolation
5. **Rotate** — random 3D rotation matrix applied to each 3-axis sensor group
6. **Scale** — per-channel random scaling ~ N(1, 0.1)
7. **Temporal flip** — reverse the time dimension

Each view randomly selects **one** augmentation from the registry.

### Paper-Compliant Pipeline (`augmentations_paper.py`) — 2 strategies only:
1. **Resample** — same as above
2. **Negate** — same as above

Each view applies resample with 50% probability and negate with 50% probability (independently), with deduplication to ensure distinct views.

---

## Training Setup

### Phase 1: Contrastive Pre-Training
- Batch size: **64** (32 for MotionSense, 512 for HHAR)
- Optimizer: **Adam** with weight decay 1e-5
- Learning rate (standard): **1e-3**
- Learning rate (paper): **3e-3** (1e-2 for SHAR)
- Learning rate schedule: **Cosine Annealing** decay
- Total epochs: **150** (standard pipeline), **120** (paper-compliant)
- Checkpoint saving: every **30 epochs** (standard), every **10 epochs** (paper)
- Model selection (paper): lowest **validation loss** during pre-training
- Number of positive views M: **4** (default)

### Phase 2: Supervised Fine-Tuning
- Batch size: **128**
- Total epochs: **100**
- Optimizer: **Adam** with weight decay 1e-5
- Learning rate (standard): **1e-3** (encoder: 1e-4 differential, classifier: 1e-3)
- Learning rate (paper): **1e-1** (classifier only, encoder frozen)
- Encoder freezing (paper): **Frozen** — `requires_grad = False`, features L2-normalized before classifier
- Encoder freezing (standard): **Unfrozen** — fine-tuned with 10× lower LR than classifier
- Best model selection: **accuracy** (standard), **macro F1** (paper)

---

## Data Preprocessing

Same as QCL:
- Normalization applied to all datasets
- Sliding window segmentation with **50% overlap** across all datasets
- Window sizes per dataset:
  - UCI-HAR: **128** time steps
  - SHAR: **151** time steps
  - HHAR: **100** time steps
  - MotionSense: **400** time steps
  - USC-HAD: **250** time steps
  - MobiAct: **100** time steps (padded to 128 for standard pipeline)
- Train/val/test split: **64% / 16% / 20%**

---

## Dataset-specific Constraints

- SHAR: **10 out of 30 participants removed** (incomplete activity classes)
- HHAR: **only phone device data used**, downsampled to **50 Hz**
- HHAR: uses **WeightedRandomSampler** (`--use_sampler`) due to class imbalance
- MobiAct: **falls excluded**, 9 contiguous ADL classes retained

---

## Key Differences from QCL

| Feature / Setting | QCL (`run_qcl_har.py`) | MPSQCL (`run_mpsqcl_har.py`) |
| :--- | :--- | :--- |
| **Number of views** | **2** (SimCLR-style) | **M = 4** (configurable) |
| **Contrastive loss** | **NT-Xent** (1 positive pair) | **MPSQCLLoss** (M-1 positive pairs per anchor) |
| **Pre-training epochs** | **120** | **150** |
| **Collate function** | `contrastive_collate_fn` (2 views) | `mps_contrastive_collate_fn` (M views) |
| **Forward pass** | 2 encoder calls per batch | M encoder calls per batch |
| **Loss denominator** | All 2(N-1) non-self views | All views from different samples + anchor |
| **Paper reference** | Ren et al. (2024) — QCL baseline | Ren et al. (2024) — MPSQCL extension |

---

### Key Differences: Standard vs. Paper-Compliant MPSQCL

| Feature / Setting | Standard MPSQCL (`run_mpsqcl_har.py`) | Paper-Compliant MPSQCL (`run_mpsqcl_har_paper.py`) |
| :--- | :--- | :--- |
| **Encoder** | `HAREncoder` (unfrozen during fine-tuning) | `HAREncoderPaper` (frozen during fine-tuning) |
| **VQC Depth** | D=3 (`StronglyEntanglingLayers`, 72 params) | D=1 (Custom Ry + CNOT ring, 8 params) |
| **Data Augmentation** | 7 strategies (jitter, negate, permute, resample, rotate, scale, t_flip) | 2 strategies (resample + negate only) |
| **Fine-tuning LR** | 1e-3 (encoder: 1e-4, classifier: 1e-3) | 1e-1 (classifier only) |
| **Feature normalization** | None | L2-normalized before classifier |
| **Best metric** | Test accuracy | Macro F1 score |
| **NISQ noise support** | No | Yes (`--noise_prob`, `--noise_std`) |
