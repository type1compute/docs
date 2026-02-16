---
sidebar_position: 3
---

# Project Structure

```text
spikeseg/
├── spikeseg/                   # Python package
│   ├── __init__.py             # Version: 0.1.0
│   ├── config.py               # YAML loading, param dataclasses
│   ├── core/
│   │   ├── neurons.py          # IFNeuron, LIFNeuron, BaseNeuron
│   │   ├── layers.py           # SpikingConv2d, SpikingPool2d, SpikingUnpool2d, SpikingTransposedConv2d
│   │   └── functional.py       # spike_fn, DoG/Gabor filters, temporal encoding, WTA ops
│   ├── models/
│   │   ├── encoder.py          # SpikeSEGEncoder, EncoderConfig, EncoderOutput
│   │   ├── decoder.py          # SpikeSEGDecoder, DecoderConfig, DelayConnection
│   │   └── spikeseg.py         # SpikeSEG (full model)
│   ├── learning/
│   │   ├── stdp.py             # STDPLearner, STDPConfig, STDPStats
│   │   └── wta.py              # WTAInhibition, AdaptiveThreshold, ConvergenceTracker
│   ├── algorithms/
│   │   ├── hulk.py             # HULKDecoder, HULKResult, LayerSpikeActivity
│   │   └── smash.py            # ActiveSpikeHash, Instance, Object, BoundingBox, SMASH score
│   ├── data/
│   │   ├── datasets.py         # EBSSADataset, NMNISTDataset, SyntheticEventDataset
│   │   ├── events.py           # DoGFilter, IntensityToLatency, EventStreamProcessor
│   │   └── preprocessing.py    # SpykeTorchPreprocessor, SpikeSEGPreprocessor, GaborFilterBank
│   └── utils/
│       ├── logging.py          # TensorBoardLogger, MetricsLogger, ProgressTracker
│       └── visualization.py    # 18 plot functions (filters, saliency, rasters, etc.)
├── scripts/
│   ├── train.py                # Layer-wise STDP training
│   ├── train_cv.py             # k-fold cross-validation
│   ├── evaluate.py             # Pixel / object / volume-based evaluation
│   ├── inference.py            # Bounding-box detection + visualization
│   ├── diagnose.py             # Diagnostic plots (input, GT, spikes, overlay)
│   ├── threshold_sweep.py      # Sweep inference thresholds for optimal informedness
│   └── demo.py                 # Quick demonstration
├── configs/
│   ├── config.yaml             # Default training configuration
│   └── experiments/
│       └── satellite.yaml      # Satellite-specific overrides
├── tests/
│   ├── conftest.py             # Shared fixtures (device, neurons, models)
│   ├── test_neurons.py         # IF/LIF neuron tests
│   ├── test_stdp.py            # STDP rule and convergence tests
│   └── test_model.py           # Encoder, decoder, full model tests
├── website/                    # This documentation (Docusaurus)
├── pyproject.toml              # Package metadata, tool config
├── requirements.txt
└── README.md
```

## Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `spikeseg.core` | Neuron dynamics (IF, LIF), spiking conv/pool layers, stateless functional ops (DoG, Gabor, spike generation) |
| `spikeseg.models` | Encoder (3 spiking conv + 2 pool), decoder (tied transposed conv + unpool), and the combined SpikeSEG model |
| `spikeseg.learning` | STDP weight updates (multiplicative/additive), WTA inhibition (global/local/both), adaptive thresholds, convergence tracking |
| `spikeseg.algorithms` | HULK instance-wise decoding, ASH spike hashing, SMASH similarity scoring, instance-to-object grouping |
| `spikeseg.data` | EBSSA and N-MNIST dataset loaders, event-to-voxel/time-surface conversion, DoG/Gabor preprocessing pipelines |
| `spikeseg.utils` | TensorBoard + JSON metrics logging, 18 matplotlib plot functions, progress tracking |

## Configuration System

Configuration is managed through YAML files parsed by `spikeseg.config`:

- `load_config(path)` reads a YAML and returns a dictionary.
- `merge_configs(base, override)` deep-merges two configs (CLI overrides).
- Typed dataclasses (`STDPParams`, `ModelParams`, `DataParams`) provide validated access to parameters.

See [Configuration Guide](../guides/configuration) for all keys.
