---
sidebar_position: 1
---

# Introduction

**TALON IR** is a neuromorphic intermediate representation designed for Type 1 Compute hardware. The ecosystem consists of seven packages that work together to bridge ML frameworks with neuromorphic deployment, hardware mapping, FPGA compilation, and real-time event streaming.

![Talon Ecosystem](/img/talon-ecosystem/talon_ecosystem.png)
*High-Level Architecture Overview*

## TALON

For most users, the recommended entry point is **TALON** — Tactical AI at Low-power On-device Nodes:

```bash
pip install t1c-talon
```

This installs all ecosystem packages and provides both the `talon` and `t1c` CLI commands. See the [TALON Documentation](./talon) for details.

## Package Overview

| Package | PyPI Name | Purpose | Key Dependencies |
|---------|-----------|---------|------------------|
| **talon.sdk** | `t1c-talon` | SDK meta-package and CLI (`talon` / `t1c`) | all packages below |
| **talon.ir** | `talon-ir` | Core IR primitives and HDF5 serialization | numpy, h5py, rustworkx |
| **talon.bridge** | `talon-bridge` | PyTorch export/import bridge | talon-ir, torch, snntorch |
| **talon.viz** | `talon-viz` | Interactive graph and spike visualization | talon-ir, pillow |
| **talon.graph** | `talon-graph` | Graph partitioning, placement, routing, resource allocation | talon-ir, rustworkx |
| **talon.backend** | `talon-backend` | Backend compilation, CPU simulation, HLS4ML FPGA backend | talon-ir, talon-graph, numpy |
| **talon.io** | `talon-io` | Event streaming, format conversion, neural encoding | numpy |

<br/>
![Talon Modules](/img/talon-ecosystem/talon_sdk_packages.png)
*Talon Modules/Packages*

## Design Philosophy

The architecture mirrors the ONNX ecosystem:

```
PyTorch Model → bridge.to_ir() → TALON IR Graph → ir.write() → .t1c file
                                                     │
                                        ┌────────────┼────────────┐
                                        ▼            ▼            ▼
                                   graph.partition  backend.compile  viz.visualize
                                   graph.route      backend.simulate
                                   graph.place      backend.profile
                                        │            │
                                        ▼            ▼
                              Hardware Mapping   HW Descriptor (JSON)
```

(This assumes `from talon import ir, bridge, graph, backend, viz`.)

This separation enables:

- **ML engineers** export models without understanding hardware details
- **Hardware teams** consume standardized IR without touching PyTorch
- **Deployment engineers** partition, map, and compile graphs to hardware targets
- **Visualization** works independently of all other packages
- **I/O engineers** stream and encode real-time event camera data

## Key Features

- **36 primitives** covering SNNs, ANNs, hybrid architectures, Ghost blocks, and detection heads
- **Hybrid ANN-SNN support** with activations (ReLU, Sigmoid, GELU, Softmax), normalization (BatchNorm, LayerNorm), and dropout
- **NIR-compliant** LIF neuron dynamics for interoperability
- **RustworkX-powered** graph algorithms for 100x faster validation and cycle detection
- **HDF5 serialization** for efficient storage and portability
- **snnTorch integration** via thin wrapper functions
- **Graph partitioning** across neuromorphic cores (greedy, edge-map, spectral algorithms)
- **Core placement and spike routing** on physical mesh topologies
- **Backend compilation** to hardware descriptors (JSON/binary) for runtime configuration
- **CPU simulation and profiling** with from-scratch latency and energy estimation
- **HLS4ML FPGA backend** for Xilinx Zynq targets with C++ header generation
- **Netron-style visualization** with clickable nodes, pattern detection, and parameter inspection
- **Spike/event visualization** for neuromorphic datasets (23-30M events/sec)
- **Event I/O** with vectorized EVT2/EVT3 decoding (>5M events/sec), AEDAT4 unpacking, UDP streaming

## Use Cases

TALON IR is designed for:

- **Model deployment** to Type 1 Compute neuromorphic hardware
- **Hybrid ANN-SNN architectures** with encoder-SNN-decoder patterns
- **FPGA prototyping** via HLS4ML config generation and Vivado TCL scripts
- **Hardware mapping** partitioning graphs across neuromorphic cores with placement and routing
- **Model exchange** between ML and hardware teams
- **Architecture visualization** for debugging and documentation
- **Event-based data processing** real-time event camera streaming and neural encoding
- **Round-trip verification** ensuring export/import fidelity
- **Energy and latency profiling** from-scratch CPU simulation with MAC-based energy estimation

## Project Repositories

Repositories (Git clone names) and their installed package names:

- **t1ctalon**: SDK meta-package and CLI (installs as `t1c-talon`; commands: `talon` and `t1c`)
- **talonir**: Core IR package (installs as `talon-ir`; use `from talon import ir`)
- **talonbridge**: PyTorch bridge (installs as `talon-bridge`; use `from talon import bridge`)
- **talonviz**: Visualization tools (installs as `talon-viz`; use `from talon import viz`)
- **talongraph**: Graph partitioning and hardware mapping (installs as `talon-graph`; use `from talon import graph`)
- **talonbackend**: Backend compilation and simulation (installs as `talon-backend`; use `from talon import backend`)
- **talonio**: Event streaming and I/O (installs as `talon-io`; use `from talon import io`)

## Next Steps

- [Quick Start](./quick-start) — Export your first model in 5 minutes
- [Installation Guide](./installation) — Set up all packages
- [Architecture Overview](./architecture) — How the packages work together
- [TALON Overview](./talon) — Unified entry point and CLI

### Tutorials

- [TALON IR Basics](./tutorial/tutorial_talonir_basics) — Primitives, graphs, serialization
- [talonviz](./tutorial/tutorial_talonviz) — Graph visualization and event processing
- [TALON SDK](./tutorial/tutorial_talon) — Analysis, profiling, linting, fingerprinting
- [Bridge](./tutorial/tutorial_bridge) — Export/import PyTorch models
- [Ghost & Detection](./tutorial/tutorial_ghost_detection) — GhostNet and detection pipeline
- [Hardware Mapping](./tutorial/tutorial_hardware_mapping) — Partition, route, place
- [Backend](./tutorial/tutorial_backend) — Simulate, profile, FPGA
- [Event I/O](./tutorial/tutorial_event_io) — Neural encoding, HDF5, throughput
- [snnTorch Integration](./tutorial/tutorial_snntorch_integration) — snnTorch export/import round-trip
- [End-to-End Pipeline](./tutorial/tutorial_end_to_end) — Full workflow: model → export → analyze → partition → simulate → profile
