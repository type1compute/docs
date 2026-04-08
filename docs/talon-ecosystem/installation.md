---
sidebar_position: 3
---

# Installation

## Requirements

- Python 3.10+
- PyTorch 2.0+ (for talon-bridge)
- uv package manager (recommended)

## Quick Install

The easiest way to get started is with **TALON**, which installs all ecosystem packages and provides the `talon` and `t1c` CLI commands:

```bash
# Using uv (recommended)
uv add t1c-talon

# Or using pip
pip install t1c-talon

# Or using poetry
poetry add t1c-talon

# Or using conda/mamba
conda install t1c-talon

# Or using pipenv
pipenv install t1c-talon
```

Verify installation:

```bash
talon info
# or: talon info
```

## Development Install

For development, install all packages in editable mode. Repository names (talonir, talonbridge, talonviz, talongraph, talonbackend, talonio, t1ctalon) are the Git clone directory names; the installed package names are talon-ir, talon-bridge, talon-viz, talon-graph, talon-backend, talon-io, t1c-talon.

```bash
# Clone repositories
git clone <t1ctalon-repo> t1ctalon
git clone <talonir-repo> talonir
git clone <talonbridge-repo> talonbridge
git clone <talonviz-repo> talonviz
git clone <talongraph-repo> talongraph
git clone <talonbackend-repo> talonbackend
git clone <talonio-repo> talonio

# Create virtual environment (using uv recommended)
uv venv .venv
source .venv/bin/activate     # Linux/Mac
# .\.venv\Scripts\Activate.ps1  # Windows

# Or use other package managers:
# python -m venv .venv          # Standard venv
# conda create -n talon python=3.10  # Using conda
# poetry shell                  # Using poetry

# Install in dependency order (using uv recommended)
uv pip install -e ./talonir
uv pip install -e ./talonio
uv pip install -e ./talonbridge
uv pip install -e ./talonviz
uv pip install -e ./talongraph
uv pip install -e ./talonbackend
uv pip install -e ./t1ctalon

# Or use pip:
# pip install -e ./talonir
# pip install -e ./talonio
# pip install -e ./talonbridge
# pip install -e ./talonviz
# pip install -e ./talongraph
# pip install -e ./talonbackend
# pip install -e ./t1ctalon
```

## Verify Installation

### Via CLI

```bash
talon info
talon primitives
# or: talon info  (both commands are equivalent)
```

### Via Python

```python
from talon import sdk

# Show all versions
sdk.info()

# Or check individually
from talon import ir, bridge, viz, graph, backend, io

versions = sdk.get_versions()
print(f"talon version: {versions['talon']}")
print(f"talon.ir version: {versions['talon.ir']}")
print(f"talon.bridge version: {versions['talon.bridge']}")
print(f"talon.viz version: {versions['talon.viz']}")
print(f"talon.graph version: {versions['talon.graph']}")
print(f"talon.backend version: {versions['talon.backend']}")
print(f"talon.io version: {versions['talon.io']}")

# List available primitives
print(f"Primitives ({len(sdk.list_primitives())}): {sdk.list_primitives()}")
```

Expected output:

```
talon version: 0.0.1
talon.ir version: 0.0.1
talon.bridge version: 0.0.1
talon.viz version: 0.0.1
talon.graph version: 0.0.1
talon.backend version: 0.0.1
talon.io version: 0.0.1
Primitives (36): ['Affine', 'AvgPool2d', 'BatchNorm1d', 'BatchNorm2d', 'ChannelSplit', 'Concat', 'Conv1d', 'Conv2d', 'DFLDecode', 'Dist2BBox', 'Dropout', 'ELU', 'Flatten', 'GELU', 'GhostBasicBlock1', 'GhostBasicBlock2', 'HybridRegion', 'IF', 'LIF', 'LayerNorm', 'MaxPool2d', 'NMS', 'PReLU', 'ReLU', 'SConv', 'SDConv', 'SDDetect', 'SGhostConv', 'SGhostEncoderLite', 'SepConv2d', 'Sigmoid', 'Skip', 'Softmax', 'SpikingAffine', 'Tanh', 'Upsample']
```

## snnTorch Integration

If you're using snnTorch, install the fork with TALON IR support:

```bash
git clone <snntorch-fork-repo> snntorch

# Using uv (recommended)
uv pip install -e ./snntorch

# Or using pip
# pip install -e ./snntorch
```

This adds two wrapper functions (they use talon.ir and talon.bridge under the hood):
- `snntorch.export_talonir.export_to_ir()` - Exports snnTorch models
- `snntorch.import_talonir.import_from_ir()` - Imports TALON IR graphs

## Troubleshooting

### Import Error: No module named 'talon.ir'

Ensure packages are installed in the correct order. talon-ir must be installed before all other talon packages.

```bash
# Using uv (recommended)
uv pip install -e ./talonir --no-deps
uv pip install -e ./talonio --no-deps
uv pip install -e ./talonbridge --no-deps
uv pip install -e ./talonviz --no-deps
uv pip install -e ./talongraph --no-deps
uv pip install -e ./talonbackend --no-deps

# Or using pip
# pip install -e ./talonir --no-deps
# pip install -e ./talonio --no-deps
# pip install -e ./talonbridge --no-deps
# pip install -e ./talonviz --no-deps
# pip install -e ./talongraph --no-deps
# pip install -e ./talonbackend --no-deps
```

### HDF5 Error on Windows

If h5py installation fails, install from conda-forge:

```bash
conda install -c conda-forge h5py
```

### PyTorch Version Mismatch

talon-bridge requires PyTorch 2.0+. Check your version:

```python
import torch
print(torch.__version__)
```

## Development Setup

For contributing to the packages:

```bash
# Install dev dependencies (using uv recommended)
uv pip install pytest numpy torch snntorch

# Or using pip
# pip install pytest numpy torch snntorch

# Run tests
cd talonir && pytest tests/
cd ../talonbridge && pytest tests/
cd ../talonviz && pytest tests/
cd ../talongraph && pytest tests/
cd ../talonbackend && pytest tests/
cd ../talonio && pytest tests/
cd ../t1ctalon && pytest tests/
```
