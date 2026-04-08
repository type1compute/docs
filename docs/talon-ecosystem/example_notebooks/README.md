# TALON Tutorial Notebooks

Interactive Jupyter notebooks for learning the TALON ecosystem.

## Notebooks

| Notebook | Description |
|----------|-------------|
| [tutorial_talonir_basics.ipynb](./tutorial_talonir_basics.ipynb) | Building neural network graphs with TALON IR primitives (talon.ir) |
| [tutorial_talonviz.ipynb](./tutorial_talonviz.ipynb) | talon.viz: visualization for graphs and neuromorphic events |
| [tutorial_talon.ipynb](./tutorial_talon.ipynb) | TALON: analysis, profiling, and deployment |
| [tutorial_bridge.ipynb](./tutorial_bridge.ipynb) | Bridge: PyTorch ↔ TALON IR export, import, and stateful LIF execution |
| [tutorial_ghost_detection.ipynb](./tutorial_ghost_detection.ipynb) | Ghost architecture and NMS detection pipeline |
| [tutorial_hardware_mapping.ipynb](./tutorial_hardware_mapping.ipynb) | Hardware mapping: partitioning, routing, allocation, and placement |
| [tutorial_backend.ipynb](./tutorial_backend.ipynb) | Backend simulation, energy profiling, and FPGA compilation |
| [tutorial_event_io.ipynb](./tutorial_event_io.ipynb) | Event I/O: neural encoding, HDF5 storage, and throughput benchmarks |
| [tutorial_snntorch_integration.ipynb](./tutorial_snntorch_integration.ipynb) | snnTorch integration: export/import round-trip with SDK analysis |
| [tutorial_end_to_end.ipynb](./tutorial_end_to_end.ipynb) | End-to-end pipeline: model → export → analyze → partition → simulate → profile → visualize |

## Installation

### Prerequisites

You need SSH access to the GitLab repositories. Set up your SSH key:
- [GitLab SSH Documentation](https://docs.gitlab.com/ee/user/ssh.html)

### Quick Setup (Recommended)

```bash
./setup_kernel.sh
```

This will create a virtual environment, install all dependencies, and register the Jupyter kernel.

### Manual Setup

```bash
# 1. Create virtual environment
uv venv

# 2. Activate the environment
source .venv/bin/activate

# 3. Install dependencies
uv pip install -r requirements.txt

# 4. Register as Jupyter kernel
python -m ipykernel install --user --name=talon-tutorials --display-name="TALON Tutorials"
```

### Running Notebooks

```bash
# Make sure environment is activated
source .venv/bin/activate

## Selecting the Kernel in VSCode

After running the setup:

1. Open a notebook (`.ipynb` file)
2. Click the kernel name in the top-right (e.g., "Python 3.11.14")
3. Click **"Select Another Kernel..."**
4. Select **"Python Environments..."**
5. Look for **"talon-tutorials"** or the `.venv` path
6. If not visible, click the refresh icon (⟳) in the kernel selector
7. Or reload VSCode window: `Cmd+Shift+P` → "Developer: Reload Window"

The kernel location: `~/Library/Jupyter/kernels/talon-tutorials`

## Topics Covered

### TALON IR Basics (talon.ir)
- Creating primitives (Input, Output, Affine, LIF, IF, Conv2d, SConv, SDConv, SGhostConv, NMS)
- Building graphs with nodes and edges
- Serialization (HDF5 format)
- Graph validation
- Convolutional networks and skip connections

### talon.viz (talon.viz)
- Graph visualization to HTML with localhost serving
- Event processing (frames, grid, raster)
- Pattern detection (ResNet, SPP)
- Spike raster plots with axis labels

### TALON (talon.sdk)
- PyTorch to TALON IR conversion
- Graph analysis and statistics
- Hardware profiling with energy presets
- Graph comparison, linting, and validation
- Fingerprinting for reproducibility

### Bridge (talon.bridge)
- Standard and spiking-aware export (`SpikingAffine`)
- Round-trip import with `GraphExecutor`
- Stateful LIF execution across timesteps
- `CyclicGraphExecutor` for recurrent networks

### Ghost Architecture & Detection
- Spiking Ghost Convolutions (`SGhostConv`)
- `GhostBasicBlock1` (stride-2) and `GhostBasicBlock2` (no-stride)
- `SGhostEncoderLite` stem encoder
- Detection pipeline: `SDDetect` → `DFLDecode` → `Dist2BBox` → `NMS`

### Hardware Mapping (talon.graph)
- `HardwareSpec` with device presets (Zynq 7020, Zynq US+, TALON ASIC)
- Graph partitioning (greedy, spectral algorithms)
- Resource allocation and SRAM budgeting
- Core routing and placement optimization

### Backend Simulation (talon.backend)
- CPU backend: compile, simulate, profile
- Energy presets (45nm CMOS, neuromorphic INT8, Zynq US+)
- Per-layer energy breakdown and neuron utilization
- FPGA backend: validation, compilation, bitstream generation

### Event IO (talon.io)
- Neural encoding (rate, latency, delta)
- HDF5 event storage with time-windowed queries
- Throughput benchmarking

### snnTorch Integration
- Export snnTorch models to TALON IR
- Import TALON IR graphs as PyTorch executors
- Hybrid ANN-SNN workflows
- Round-trip verification

### End-to-End Pipeline
- Complete workflow: define → export → analyze → quantize → partition → simulate → profile → visualize
- Event encoding and I/O integration
- `run_pipeline()` single-call execution
