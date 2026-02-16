---
sidebar_position: 2
---

# Spike-Timing Dependent Plasticity (STDP)

STDP is a biologically observed synaptic learning rule in which the change in connection strength depends on the **relative timing** of pre- and post-synaptic spikes [6]. SpikeSEG uses the simplified STDP formulation of Kheradpisheh et al. [1].

## Learning Rule

### Long-Term Potentiation (LTP)

If the pre-synaptic neuron fires **before or at the same time** as the post-synaptic neuron, the synapse is strengthened:

$$
\Delta w = a^{+} \cdot w \cdot (1 - w) \qquad \text{if } t_{\text{pre}} \leq t_{\text{post}}
$$

### Long-Term Depression (LTD)

If the pre-synaptic neuron fires **after** the post-synaptic neuron, the synapse is weakened:

$$
\Delta w = -a^{-} \cdot w \cdot (1 - w) \qquad \text{if } t_{\text{pre}} > t_{\text{post}}
$$

### Key Properties

- The **sign** of the timing difference matters, not its magnitude: *"The exact time difference between two spikes does not affect the weight change, but only its sign is considered."* [1]
- The multiplicative term $w(1 - w)$ provides **soft bounds** that keep weights in $[0, 1]$ and naturally drive them toward binary values (0 or 1).
- A pre-synaptic neuron that **never fires** is assumed to fire later (treated as LTD).

## Learning Rate Parameters

| Source | $a^{+}$ | $a^{-}$ | Convergence speed |
|--------|----------|----------|-------------------|
| Kheradpisheh 2018 [1] | 0.004 | 0.003 | Standard |
| IGARSS 2023 [4] | 0.04 | 0.03 | 10x faster |

The IGARSS 2023 rates are 10x larger for faster convergence on the smaller EBSSA dataset.

## Convergence Criterion

Weights converge when they polarize toward 0 or 1. The convergence metric from Equation 4 of [1]:

$$
C_l = \frac{1}{n_w} \sum_{f} \sum_{i} w_{f,i} \cdot (1 - w_{f,i})
$$

- $C_l = 0$: all weights are exactly 0 or 1 (fully converged).
- $C_l = 0.25$: all weights are at 0.5 (maximum uncertainty).

**Training stops when $C_l < 0.01$.**

## Weight Initialization

Weights are drawn from $\mathcal{N}(\mu=0.8,\, \sigma=0.01)$ and clipped to $[0, 1]$. Starting near 1 accelerates STDP convergence: features that should be potentiated are already close to their final value.

## Implementation

```python
from spikeseg.learning import STDPLearner, STDPConfig

# Paper preset
config = STDPConfig.from_paper("igarss2023")
# config.lr_plus = 0.04, config.lr_minus = 0.03

learner = STDPLearner(config)

# Update weights for a WTA winner
learner.update_weights_for_winner(
    weights=layer.conv.weight,
    pre_spike_times=pre_times,
    post_spike_time=winner_time,
    winner_y=wy, winner_x=wx,
    kernel_size=5, stride=1, padding=0,
)

# Check convergence
metric = learner.get_convergence(layer.conv.weight)
if metric < 0.01:
    print("Layer converged")
```

See [API: Learning](../api/learning) for complete signatures and the additive STDP variant.
