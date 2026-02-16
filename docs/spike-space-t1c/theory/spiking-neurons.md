---
sidebar_position: 1
---

# Spiking Neural Networks

Spiking Neural Networks (SNNs) are the **third generation** of neural network models [5]. Unlike rate-coded artificial neurons that output continuous activations, spiking neurons communicate through discrete events (*spikes*) whose precise timing carries information. This temporal coding enables efficient, event-driven processing that is well suited to neuromorphic sensor data.

## Neuron Models

SpikeSEG provides two neuron models, both implemented as stateful PyTorch modules.

### Integrate-and-Fire (IF)

The simplest spiking neuron accumulates input current until its membrane potential reaches a threshold, at which point it fires a spike and resets:

$$
V(t) = V(t-1) + I(t)
$$

$$
S(t) = \begin{cases} 1 & \text{if } V(t) \geq \theta \\ 0 & \text{otherwise} \end{cases}
$$

After spiking, the membrane is reset: $V(t) \leftarrow 0$.

**Use case:** The classification layer (Conv3) in the IGARSS 2023 configuration uses IF neurons with zero leak, so the output layer simply integrates evidence over time.

### Leaky Integrate-and-Fire (LIF)

LIF adds membrane potential decay, providing temporal filtering and preventing indefinite charge accumulation. SpikeSEG supports two leak modes:

#### Subtractive mode (IGARSS 2023 [4])

$$
V(t) = V(t-1) + I(t) - \lambda
$$

where $\lambda$ is a constant leak term. The IGARSS 2023 paper specifies layer-wise leak configuration:

> *"$\lambda$ is set to 90% and 10% of the neuron threshold in layers 1 and 2 respectively."* [4]

#### Multiplicative mode (Kheradpisheh 2018 [1])

$$
V(t) = \beta \cdot V(t-1) + I(t)
$$

where $\beta \in (0, 1)$ is a decay factor. Smaller $\beta$ yields faster forgetting.

### Symbol Reference

| Symbol | Meaning | Typical value |
|--------|---------|---------------|
| $V(t)$ | Membrane potential at time $t$ | -- |
| $I(t)$ | Input current (weighted sum of incoming spikes) | -- |
| $\theta$ | Firing threshold | 10.0 (IGARSS) or 0.1 (sparse EBSSA) |
| $\lambda$ | Subtractive leak | 90% or 10% of $\theta$ |
| $\beta$ | Multiplicative decay | 0.9 typical |
| $S(t)$ | Binary spike output | 0 or 1 |

## Spike Generation

Spike generation is a non-differentiable Heaviside step:

$$
S(t) = H\bigl(V(t) - \theta\bigr)
$$

where $H$ is the Heaviside function. In the implementation, `spike_function` and `spike_fn` in `spikeseg.core.functional` compute this element-wise on a membrane tensor.

## Why Spiking?

1. **Temporal precision** -- spike timing, not just firing rates, encodes information.
2. **Compatibility with event cameras** -- asynchronous events map naturally to spike trains.
3. **Biological plausibility** -- enables unsupervised learning rules like STDP.
4. **Sparse computation** -- only active neurons consume energy (relevant for neuromorphic hardware).

## Implementation

```python
from spikeseg.core.neurons import LIFNeuron, IFNeuron, create_neuron

# LIF with subtractive leak (IGARSS 2023, Layer 1)
neuron = LIFNeuron(threshold=10.0, leak_factor=9.0, leak_mode="subtractive")

# Forward: returns (spikes, membrane_after_reset, membrane_before_reset)
spikes, membrane, pre_reset = neuron(input_current, membrane)

# Factory helper
neuron = create_neuron("lif", threshold=10.0, leak_factor=9.0, leak_mode="subtractive")
```

See [API: Core](../api/core) for complete signatures.
