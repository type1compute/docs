---
sidebar_position: 1
---

# T1C Ecosystem Introduction

**T1C-IR** is a neuromorphic intermediate representation designed for Type 1 Compute hardware. The ecosystem consists of four packages that work together to bridge ML frameworks with neuromorphic deployment.

## The T1C-SDK

For most users, the recommended entry point is the **T1C-SDK**:

```bash
pip install t1c-sdk
```

This installs all ecosystem packages and provides a unified CLI. See the [SDK Documentation](./sdk) for details.

## Package Overview

| Package | Purpose | Dependencies |
|---------|---------|--------------|
| **t1c-sdk** | SDK meta-package and CLI | t1c-ir, t1c-bridge, t1c-viz |
| **t1c.ir** (t1c-ir) | Core IR primitives and HDF5 serialization | numpy, h5py, rustworkx |
| **t1c.bridge** (t1c-bridge) | PyTorch export/import bridge | t1c-ir, torch, snntorch |
| **t1c.viz** (t1c-viz) | Interactive graph and spike visualization | t1c-ir, pillow |

## Design Philosophy

The architecture mirrors the ONNX ecosystem:

```
PyTorch Model → bridge.to_ir() → T1C-IR Graph → ir.write() → .t1c file
                                                                     ↓
Hardware Runtime ← ir.read() ← .t1c file ← bridge.ir_to_torch() ←
```

(This assumes `from t1c import ir, bridge`.)

This separation enables:

- **ML engineers** export models without understanding hardware details
- **Hardware teams** consume standardized IR without touching PyTorch
- **Visualization** works independently of both

## Key Features

- **22+ primitives** covering SNNs, ANNs, and hybrid architectures
- **Hybrid ANN-SNN support** with activations (ReLU, Sigmoid, GELU, Softmax), normalization (BatchNorm, LayerNorm), and dropout
- **NIR-compliant** LIF neuron dynamics for interoperability
- **RustworkX-powered** graph algorithms for 100x faster validation and cycle detection
- **HDF5 serialization** for efficient storage and portability
- **snnTorch integration** via thin wrapper functions
- **Netron-style visualization** with clickable nodes and parameter inspection
- **Spike/event visualization** for neuromorphic datasets (23-30M events/sec)

## Use Cases

T1C-IR is designed for:

- **Model deployment** to Type 1 Compute neuromorphic hardware
- **Hybrid ANN-SNN architectures** with encoder-SNN-decoder patterns
- **Model exchange** between ML and hardware teams
- **Architecture visualization** for debugging and documentation
- **Event-based data visualization** for neuromorphic datasets
- **Round-trip verification** ensuring export/import fidelity

## Project Repositories

Repositories (Git clone names) and their installed package names:

- **t1c-sdk**: SDK meta-package and CLI (installs as `t1c-sdk`)
- **t1cir**: Core IR package (installs as `t1c-ir`; use `from t1c import ir`)
- **t1ctorch**: PyTorch bridge (installs as `t1c-bridge`; use `from t1c import bridge`)
- **t1cviz**: Visualization tools (installs as `t1c-viz`; use `from t1c import viz`)

## Next Steps

- [SDK Overview](./sdk) - Unified entry point and CLI
- [Architecture Overview](./architecture) - How the packages work together
- [Installation Guide](./installation) - Set up all packages
- [Quick Start](./quick-start) - Export your first model in 5 minutes
