---
sidebar_position: 1
---

# API Reference

## Module Map

| Module | Classes / Key Functions |
|--------|------------------------|
| [`spikeseg.core`](core) | `IFNeuron`, `LIFNeuron`, `SpikingConv2d`, `SpikingPool2d`, `spike_fn`, `create_dog_filters`, `create_gabor_filters` |
| [`spikeseg.models`](models) | `SpikeSEG`, `SpikeSEGEncoder`, `SpikeSEGDecoder`, `EncoderConfig`, `EncoderOutput` |
| [`spikeseg.learning`](learning) | `STDPLearner`, `STDPConfig`, `WTAInhibition`, `AdaptiveThreshold`, `ConvergenceTracker` |
| [`spikeseg.algorithms`](algorithms) | `HULKDecoder`, `ActiveSpikeHash`, `Instance`, `Object`, `BoundingBox`, `group_instances_to_objects` |
| [`spikeseg.data`](data) | `EBSSADataset`, `NMNISTDataset`, `EventData`, `SpikeSEGPreprocessor`, `SpykeTorchPreprocessor` |

## Quick Imports

```python
# Models
from spikeseg.models import SpikeSEG

# Learning
from spikeseg.learning import STDPLearner, STDPConfig, WTAInhibition

# Instance segmentation
from spikeseg.algorithms import HULKDecoder, group_instances_to_objects

# Data
from spikeseg.data import EBSSADataset, NMNISTDataset

# Neurons and layers
from spikeseg.core.neurons import LIFNeuron, IFNeuron, create_neuron
from spikeseg.core.layers import SpikingConv2d, SpikingPool2d
from spikeseg.core.functional import create_dog_filters, create_gabor_filters

# Config
from spikeseg.config import load_config, get_model_params, get_stdp_params

# Visualization
from spikeseg.utils.visualization import plot_learned_filters, plot_saliency_map

# Logging
from spikeseg.utils.logging import TensorBoardLogger, MetricsLogger
```
