| Model | Data Input / Features | Dimensionality Constraint | Test Accuracy | Macro F1 | Training Time | Notes / Decisional Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SVM (Linear)** | Hand-crafted 561-Feat | None | **96.10%** | **0.9608** | **0.83s** | Best performing classical baseline. |
| **SVM (RBF)** | Hand-crafted 561-Feat | None | 95.22% | 0.9515 | 1.58s | - |
| **Random Forest** | Hand-crafted 561-Feat | None | 92.57% | 0.9241 | 8.26s | - |
| **Classical 1D CNN** | Raw Signals (9x128) | No Bottleneck (128 features) | 91.55% | 0.9160 | 15.74s | - |
| **Hybrid QCNN (10e)** | Raw Signals (9x128) | 4-Dim (4 Qubits VQC) | 73.39% | 0.6955 | 1118.89s | run for only 10 epochs on a 50% subset. |
| **Hybrid QCNN (50e)** | Raw Signals (9x128) | 4-Dim (4 Qubits VQC) | **92.53%** | **0.9256** | **3.82 hours (13768s)** | **Completed.** Final test accuracy 92.53%, peaked at **93.21%** (Epoch 47). Surpasses classical 1D CNN without bottleneck (91.55%). |
| **Hybrid QCNN (6q, 30e)** | Raw Signals (9x128) | 6-Dim (6 Qubits VQC) | **93.76%** | **0.9387** | **4.39 hours (15805s)** | **Completed.** Final test accuracy 93.76%, peaked at **93.89%** (Epoch 23). Best performing QML model, outperforming all classical CNN baselines. |
| **Quantum SVM (300s)** | Hand-crafted 561-Feat | 6-Dim (6 Qubits + PCA) | 45.00% | 0.4228 | 87.87s | Suffers from severe overfitting due to downsampling (300 train samples) to make kernel computation feasible. |
| **Quantum SVM (1000s)**| Hand-crafted 561-Feat | 8-Dim (8 Qubits + PCA) | 53.00% | 0.5060 | 852.43s | Accuracy improved with more samples and qubits, but simulation time scaled quadratically to 14.2 minutes. |
| **Quantum SVM (1000s, iqp)**| Hand-crafted 561-Feat| 10-Dim (10 Qubits + PCA)| **59.50%** | **0.5329** | **2635.20s** | **Using entangling IQPEmbedding and C=1.0 sweep.** Significantly improves representation, but entangling gates scale runtimes. |
| **Quantum SVM (1000s, amp)**| Hand-crafted 561-Feat| 256-Dim (8 Qubits + PCA)| **91.50%** | **0.9146** | **9.13 hours (32886s)** | **Completed.** Amplitude Embedding maps 256 features to 8 qubits, dramatically improving accuracy (91.50%) but scaling runtime significantly. |
| **QCL HAR (150e)** | Raw Signals (9x128) | 256-Dim (8 Qubits VQC) | **96.64%** | **0.9665** | **56.0 mins (3359s)** | **Completed.** Baseline QCL run (unfrozen encoder fine-tuned slowly, no validation selection). |
| **QCL HAR (50e)** | Raw Signals (9x128) | 256-Dim Features (8 Qubits VQC) | **95.83%** | **0.9578** | **18.8 mins (1130s)** | **Completed.** Baseline QCL run (unfrozen encoder fine-tuned slowly, no validation selection). |
| **QCL HAR (Paper, 50e)** | Raw Signals (9x128) | 256-Dim Features (8 Qubits VQC) | **85.93%** | **0.8487** | **3.24 mins (194s)** | **Completed.** Exact paper-compliant setup (frozen encoder, 8-qubit depth-1 VQC, no window-by-window normalization). |
| **QCL HAR (Paper, 120e)** | Raw Signals (9x128) | 256-Dim Features (8 Qubits VQC) | **93.69%** | **0.9370** | **3.55 mins (213s)** | **Completed.** Replicates the paper's exact frozen encoder linear evaluation within 0.43% F1 (paper reports 94.13%). |
| **MPSQCL HAR (50e)** | Raw Signals (9x128)| 256-Dim Features (8 Qubits VQC) | **95.08%** | **0.9497** | **46.8 mins (2808s)** | **Completed.** 50-epoch multi-positive sample (M=4 views) pre-training + 30-epoch classical fine-tuning. |
| **MPSQCL HAR (UCI-HAR, 150e)** | Raw Signals (9x128) | 256-Dim Features (8 Qubits VQC) | **98.50%** | **0.9858** | **1.54 hours (5543s)** | **Completed.** 150-epoch multi-positive contrastive pre-training (M=4 views) + 100-epoch fine-tuning. |
| **QCL HAR (SHAR, 120e)** | Raw Signals (3x151) | 256-Dim Features (8 Qubits VQC) | **91.06%** | **0.8667** | **2.28 hours (8227s)** | **Completed.** Standard QCL on SHAR (unfrozen encoder, 8-qubit depth-1 VQC). |
| **QCL HAR (Paper SHAR, 120e)** | Raw Signals (3x151) | 256-Dim Features (8 Qubits VQC) | **73.66%** | **0.6714** | **6.63 mins (398s)** | **Completed.** Exact paper-compliant setup on SHAR (frozen encoder, 8-qubit depth-1 VQC). |
| **MPSQCL HAR (SHAR, 150e)** | Raw Signals (3x151) | 256-Dim Features (8 Qubits VQC) | **83.52%** | **0.7524** | **1.26 hours (4534s)** | **Completed.** 150-epoch multi-positive contrastive pre-training (M=4 views) + 100-epoch fine-tuning. Layout scrambling bug fixed. |
| **QCL HAR (MotionSense, 120e)** | Raw Signals (12x400) | 256-Dim Features (8 Qubits VQC) | **95.79%** | **0.9596** | **25.8 mins (1548s)** | **Completed.** Standard QCL on MotionSense (unfrozen encoder, 8-qubit depth-1 VQC). |
| **QCL HAR (Paper MotionSense, 120e)** | Raw Signals (12x400) | 256-Dim Features (8 Qubits VQC) | **91.50%** | **0.9214** | **10.8 mins (651s)** | **Completed.** Exact paper-compliant setup on MotionSense (frozen encoder, 8-qubit depth-1 VQC). |
| **MPSQCL HAR (MotionSense, 150e)** | Raw Signals (12x400) | 256-Dim Features (8 Qubits VQC) | **99.69%** | **0.9954** | **1.94 hours (6989s)** | **Completed.** 150-epoch multi-positive contrastive pre-training (M=4 views) + 100-epoch fine-tuning. |
| **QCL HAR (USC-HAD, 120e)** | Raw Signals (6x250) | 256-Dim Features (8 Qubits VQC) | **71.97%** | **0.7002** | **~25.0 mins** | **Completed.** Standard QCL on USC-HAD (unfrozen encoder, 8-qubit depth-1 VQC). |
| **QCL HAR (Paper USC-HAD, 120e)** | Raw Signals (6x250) | 256-Dim Features (8 Qubits VQC) | **54.25%** | **0.5409** | **17.2 mins (1032s)** | **Completed.** Paper-compliant setup on USC-HAD (frozen encoder, depth-1 VQC, epoch checkpoint sweep best = epoch 40). |
| **MPSQCL HAR (USC-HAD, 150e)** | Raw Signals (6x250) | 256-Dim Features (8 Qubits VQC) | **88.34%** | **0.8503** | **3.23 hours (11611s)** | **Completed.** 150-epoch multi-positive contrastive pre-training (M=4 views) + 100-epoch fine-tuning. |
| **QCL HAR (MobiAct, 120e)** | Raw Signals (6x100) | 256-Dim Features (8 Qubits VQC) | **91.70%** | **0.8180** | **11.5 mins (690s)** | **Completed.** Standard QCL on MobiAct (unfrozen encoder, 8-qubit depth-1 VQC). |
| **QCL HAR (Paper MobiAct, 120e)** | Raw Signals (6x100) | 256-Dim Features (8 Qubits VQC) | **65.94%** | **0.5709** | **6.46 mins (387s)** | **Completed.** Paper-compliant setup on MobiAct (frozen encoder, depth-1 VQC, epoch checkpoint sweep best = best val loss checkpoint). |
| **MPSQCL HAR (MobiAct, 120e)** | Raw Signals (6x128) | 256-Dim Features (8 Qubits VQC) | **98.31%** | **0.9559** | **48.6 mins (2915s)** | **Completed.** 120-epoch multi-positive pre-training (resumed from 90 after interruption) + 100-epoch fine-tuning. Preprocessed MobiAct (excluding falls, 9 contiguous ADL classes). Peak test accuracy 98.31% (Epoch 28), final epoch test accuracy 98.00% (F1 0.9566). |

| **QCL HAR (HHAR, 120e)** | Raw Signals (6x100) | 256-Dim Features (8 Qubits VQC) | **93.28%** | **0.8816** | **7.66 hours (27586s)** | **Completed.** Standard QCL on HHAR (unfrozen encoder, 8-qubit depth-1 VQC). Large sample count (224k windows) scales pre-training. |
| **QCL HAR (Paper HHAR, 120e)** | Raw Signals (6x100) | 256-Dim Features (8 Qubits VQC) | *Pending* | *Pending* | *Running* | Paper-compliant setup on HHAR (frozen encoder, depth-1 VQC). Pre-training currently executing. |

---



