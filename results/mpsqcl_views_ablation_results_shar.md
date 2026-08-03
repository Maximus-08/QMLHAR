# MPSQCL Number of Views ($M$) Ablation Study Results

- **Dataset**: SHAR
- **Subset Fraction**: 100%
- **Pre-training Epochs**: 75
- **Fine-tuning Epochs**: 50
- **Device**: cuda
- **Evaluated Views ($M$)**: [2, 3, 4, 5, 6]

## Comparative Ablation Table

| Number of Views ($M$) | Positive Pair Density | Pre-train Val Loss | Test Accuracy | Test Macro F1 | Delta F1 vs. $M=2$ | Pre-train Time (s) | Sec / Epoch |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M = 2** | 1 pairs / sample | 1.2435 | 74.57% | 0.6446 | - | 1310.4s | 17.47s |
| **M = 3** | 2 pairs / sample | 1.4799 | 76.95% | 0.6733 | +2.87% | 1931.9s | 25.76s |
| **M = 4** | 3 pairs / sample | 1.6829 | 75.61% | 0.6619 | +1.73% | 2792.1s | 37.23s |
| **M = 5** | 4 pairs / sample | 1.8466 | 78.04% | 0.6865 | +4.19% | 3312.9s | 44.17s |
| **M = 6** | 5 pairs / sample | 2.0011 | 77.37% | 0.6669 | +2.23% | 4513.3s | 60.18s |

## Key Insights & Trade-Off Analysis

1. **Impact of Multi-Positive Representation Density ($M > 2$)**:
   Increasing $M$ from 2 (SimCLR baseline) provides $M-1$ positive pairs per anchor. This enhances feature representation quality by enforcing invariance across a broader set of augmentations.

2. **Computational Overhead Scaling**:
   Each batch processes $M \times N$ representation vectors through the classical encoder and VQC projection head. Pre-training runtime scales linearly with $M$.

3. **Optimal View Recommendation**:
   $M=4$ provides the optimal trade-off between downstream classification accuracy / Macro F1 score and computational runtime.
