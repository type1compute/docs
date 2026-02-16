---
sidebar_position: 1
---

# Architecture Overview

SpikeSEG is a spiking encoder-decoder network with an instance-segmentation stage (HULK-SMASH) built on top of the encoder.

## Full Pipeline

```mermaid
flowchart TB
    subgraph Input
        EV[Event Stream]
        PP[Preprocessing<br/>DoG / Gabor / Temporal Binning]
    end
    subgraph Encoder
        C1[Conv1 5x5, 4ch, LIF 90% leak]
        P1[Pool1 2x2]
        C2[Conv2 5x5, 36ch, LIF 10% leak]
        P2[Pool2 2x2]
        C3[Conv3 7x7, Nclass ch, IF 0% leak]
    end
    subgraph Decoder
        TC3[TransConv3 7x7]
        UP2[Unpool2]
        TC2[TransConv2 5x5]
        UP1[Unpool1]
        TC1[TransConv1 5x5]
        SAL[Saliency Map]
    end
    subgraph InstanceSeg[Instance Segmentation]
        HULK[HULK Decoder]
        ASH[Active Spike Hash]
        SMASH[SMASH Grouping]
        OBJ[Detected Objects]
    end

    EV --> PP --> C1 --> P1 --> C2 --> P2 --> C3
    C3 -->|classification spikes| TC3 --> UP2 --> TC2 --> UP1 --> TC1 --> SAL
    C3 -->|per-spike| HULK --> ASH --> SMASH --> OBJ
    P1 -.->|indices| UP1
    P2 -.->|indices| UP2
```

## Layer Summary (IGARSS 2023)

| Layer | Type | Kernel | Channels | Threshold | Leak | Neuron |
|-------|------|--------|----------|-----------|------|--------|
| Conv1 | Spiking Conv2d | 5x5 | 4 | 10.0 | 9.0 (90%) | LIF subtractive |
| Pool1 | Max Pool | 2x2 | -- | -- | -- | -- |
| Conv2 | Spiking Conv2d | 5x5 | 36 | 10.0 | 1.0 (10%) | LIF subtractive |
| Pool2 | Max Pool | 2x2 | -- | -- | -- | -- |
| Conv3 | Spiking Conv2d | 7x7 | $n_\text{classes}$ | 10.0 | 0.0 | IF |

## Weight Initialization

All convolutional weights are drawn from $\mathcal{N}(0.8, 0.01)$ and clipped to $[0, 1]$. Starting near 1 accelerates STDP convergence.

## Two Output Paths

1. **Saliency map** (decoder path): all classification spikes are decoded together into a single pixel-level heat map.
2. **Instance masks** (HULK-SMASH path): each classification spike is decoded separately, producing one mask per instance, then grouped by SMASH similarity.

See [Encoder](encoder), [Decoder](decoder), [HULK-SMASH](hulk-smash).
