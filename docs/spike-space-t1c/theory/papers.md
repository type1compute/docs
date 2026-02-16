---
sidebar_position: 5
---

# Implemented Papers

SpikeSEG synthesises methods from four publications. This page summarises each paper's contribution and how it maps to the codebase.

## [1] Kheradpisheh et al. 2018 -- STDP-Based Spiking Deep CNNs

**Title:** *STDP-based spiking deep convolutional neural networks for object recognition*
**Venue:** Neural Networks, vol. 99, pp. 56--67

**Contributions used:**
- Simplified STDP rule with soft-bounded multiplicative updates ($\Delta w = a^{\pm} \cdot w(1-w)$).
- Three-layer spiking convolutional architecture with max pooling.
- Winner-Take-All lateral inhibition (global + local).
- Convergence criterion $C_l < 0.01$.
- Weight initialisation near 1 ($\mu = 0.8$).

**Codebase mapping:** `spikeseg.learning.stdp`, `spikeseg.learning.wta`, `spikeseg.core.neurons`.

---

## [2] Kirkland et al. 2020 -- SpikeSEG

**Title:** *SpikeSEG: Spiking segmentation via STDP saliency mapping*
**Venue:** 2020 IEEE IJCNN, pp. 1--8

**Contributions used:**
- Encoder-decoder architecture with tied weights.
- Transposed convolutions and max unpooling to trace classification spikes back to pixel space (saliency map).

**Codebase mapping:** `spikeseg.models.decoder`, `spikeseg.models.spikeseg`.

---

## [3] Kirkland et al. 2022 -- HULK-SMASH

**Title:** *Unsupervised spiking instance segmentation on event data using STDP features*
**Venue:** IEEE Transactions on Neural Networks and Learning Systems

**Contributions used:**
- **HULK** (Hierarchical Unravelling of Linked Kernels): decode each classification spike individually to get per-instance pixel masks.
- **ASH** (Active Spike Hashing): compress 4D spike activity into a 2D binary feature-time matrix.
- **SMASH** score: Jaccard(ASH) x IoU(BBox) for grouping instances into objects.

**Codebase mapping:** `spikeseg.algorithms.hulk`, `spikeseg.algorithms.smash`.

---

## [4] Kirkland et al. 2023 -- IGARSS Space Domain Awareness

**Title:** *Neuromorphic sensing and processing for space domain awareness*
**Venue:** IGARSS 2023

**Contributions used:**
- Layer-wise subtractive leak ($\lambda$ = 90% and 10% of $\theta$ in layers 1 and 2).
- 10x higher STDP learning rates ($a^{+} = 0.04$, $a^{-} = 0.03$) for faster convergence.
- Volume-based evaluation with informedness as the primary metric (target: 89.1%).
- Application to the EBSSA satellite dataset.

**Codebase mapping:** `EncoderConfig.from_paper("igarss2023")`, `scripts/evaluate.py --volume-based`.

---

## Additional References

| # | Citation |
|---|---------|
| [5] | W. Maass, "Networks of spiking neurons: The third generation of neural network models," *Neural Networks*, vol. 10, no. 9, pp. 1659--1671, 1997. |
| [6] | G.-Q. Bi and M.-M. Poo, "Synaptic modifications in cultured hippocampal neurons," *Journal of Neuroscience*, vol. 18, no. 24, pp. 10464--10472, 1998. |
| [7] | S. Afshar et al., "Event-based object detection and tracking for space situational awareness," *IEEE Sensors Journal*, vol. 20, no. 24, pp. 15117--15132, 2020. |
| [8] | G. Orchard et al., "Converting static image datasets to spiking neuromorphic datasets using saccades," *Frontiers in Neuroscience*, vol. 9, p. 437, 2015. |

See [Citation](../citation) for BibTeX entries.
