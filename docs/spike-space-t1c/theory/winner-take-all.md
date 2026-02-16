---
sidebar_position: 3
---

# Winner-Take-All Competition

Winner-Take-All (WTA) lateral inhibition enforces competition among neurons so that different filters learn different features [1]. SpikeSEG uses two complementary mechanisms that operate simultaneously.

## Mechanisms

### Global Intra-Map Inhibition

Within each feature map, the **first neuron to fire** inhibits all other neurons in the same map for the remainder of the stimulus. This ensures each learned feature activates at most one spatial location per input.

### Local Inter-Map Inhibition

At each spatial position, a firing neuron inhibits neurons in **other feature maps** within a local radius $r$. This encourages different features to activate at different locations, increasing spatial diversity.

> *"We use a winner-take-all (WTA) mechanism to enforce competition among neurons. The first neuron that fires inhibits the others, preventing them from firing and receiving plasticity updates."* [1]

### Configuration

| Mode | Config value | Behaviour |
|------|-------------|-----------|
| Global only | `"global"` | Intra-map inhibition only |
| Local only | `"local"` | Inter-map inhibition only |
| Both | `"both"` | Paper default: global + local |

## Homeostatic Plasticity

Without regulation, a small subset of neurons may dominate (fire on every input) while others become "dead" (never fire). Adaptive thresholds restore balance:

$$
\theta_{\text{new}} = \theta_{\text{old}} + \theta^{+} \qquad \text{(after spike)}
$$

$$
\theta_{\text{new}} = \theta_{\text{old}} - \frac{\theta_{\text{old}} - \theta_{\text{rest}}}{\tau_\theta} \qquad \text{(decay toward rest)}
$$

| Parameter | Symbol | Default | Purpose |
|-----------|--------|---------|---------|
| Rest threshold | $\theta_{\text{rest}}$ | 0.1 | Baseline threshold |
| Increment | $\theta^{+}$ | 0.02 | Penalty for firing |
| Time constant | $\tau_\theta$ | 500 | Decay rate |
| Maximum | $\theta_{\max}$ | 10.0 | Cap on threshold |

Additionally, **dead neuron recovery** perturbs the weights of neurons whose firing rate falls below a threshold (`dead_threshold = 0.01`), giving them a chance to learn new features.

## Implementation

```python
from spikeseg.learning import WTAInhibition, WTAConfig, WTAMode

config = WTAConfig(
    mode=WTAMode.BOTH,
    local_radius=2,
    enable_homeostasis=True,
    target_rate=0.1,
)

wta = WTAInhibition(config=config, n_channels=36, spatial_shape=(32, 32))

# Apply WTA to a spike tensor
filtered_spikes, winner_mask = wta(spikes, membrane, pre_reset_membrane)
```

See [API: Learning](../api/learning) for `AdaptiveThreshold` and `ConvergenceTracker`.
