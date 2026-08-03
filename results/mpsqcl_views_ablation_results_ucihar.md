# MPSQCL Number of Views ($M$) Ablation Study Results

- **Dataset**: UCIHAR
- **Subset Fraction**: 100%
- **Pre-training Epochs**: 75
- **Fine-tuning Epochs**: 50
- **Device**: cuda
- **Evaluated Views ($M$)**: [2, 3, 4, 5, 6]

## Comparative Ablation Table

| Number of Views ($M$) | Positive Pair Density | Pre-train Val Loss | Test Accuracy | Test Macro F1 | Delta F1 vs. $M=2$ | Pre-train Time (s) | Sec / Epoch |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M = 2** | 1 pairs / sample | 1.8784 | 96.07% | 0.9629 | - | 1942.8s | 25.90s |
| **M = 3** | 2 pairs / sample | 2.0150 | 96.56% | 0.9676 | +0.47% | 3165.6s | 42.21s |
| **M = 4** | 3 pairs / sample | 2.1931 | 97.04% | 0.9728 | +0.99% | 3850.5s | 51.34s |
| **M = 5** | 4 pairs / sample | 2.3563 | 96.36% | 0.9663 | +0.35% | 4762.5s | 63.50s |
| **M = 6** | 5 pairs / sample | 2.4329 | 96.85% | 0.9705 | +0.76% | 5375.2s | 71.67s |

## Key Insights & Trade-Off Analysis

1. **Impact of Multi-Positive Representation Density ($M > 2$)**:
   Increasing $M$ from 2 (SimCLR baseline) provides $M-1$ positive pairs per anchor. This enhances feature representation quality by enforcing invariance across a broader set of augmentations.

2. **Computational Overhead Scaling**:
   Each batch processes $M \times N$ representation vectors through the classical encoder and VQC projection head. Pre-training runtime scales linearly with $M$.

3. **Optimal View Recommendation**:
   $M=4$ provides the optimal trade-off between downstream classification accuracy / Macro F1 score and computational runtime.
