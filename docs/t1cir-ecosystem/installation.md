---
sidebar_position: 3
---

# Installation

## Requirements

- Python 3.10+
- PyTorch 2.0+ (for t1c-bridge)
- uv package manager (recommended)

## Quick Install

The easiest way to get started is with the **T1C-SDK**, which installs all ecosystem packages and provides a CLI:

```bash
# Using uv (recommended)
uv add t1c-sdk

# Or using pip
pip install t1c-sdk

# Or using poetry
poetry add t1c-sdk

# Or using conda/mamba
conda install t1c-sdk

# Or using pipenv
pipenv install t1c-sdk
```

Verify installation:

```bash
t1c info
```

## Development Install

For development, install all packages in editable mode. Repository names (t1cir, t1ctorch, t1cviz, t1c-sdk) are the Git clone directory names; the installed package names are t1c-ir, t1c-bridge, t1c-viz, t1c-sdk.

```bash
# Clone repositories
git clone <t1c-sdk-repo> t1c-sdk
git clone <t1cir-repo> t1cir
git clone <t1ctorch-repo> t1ctorch
git clone <t1cviz-repo> t1cviz

# Create virtual environment (using uv recommended)
uv venv .venv
source .venv/bin/activate     # Linux/Mac
# .\.venv\Scripts\Activate.ps1  # Windows

# Or use other package managers:
# python -m venv .venv          # Standard venv
# conda create -n t1c python=3.10  # Using conda
# poetry shell                  # Using poetry

# Install in dependency order (using uv recommended)
uv pip install -e ./t1cir
uv pip install -e ./t1ctorch
uv pip install -e ./t1cviz
uv pip install -e ./t1c-sdk

# Or use pip:
# pip install -e ./t1cir
# pip install -e ./t1ctorch
# pip install -e ./t1cviz
# pip install -e ./t1c-sdk
```

## Verify Installation

### Via CLI

```bash
t1c info
t1c primitives
```

### Via Python

```python
from t1c import sdk

# Show all versions
sdk.info()

# Or check individually
from t1c import ir, bridge, viz

versions = sdk.get_versions()
print(f"t1c-sdk version: {versions['t1c-sdk']}")
print(f"t1c.ir version: {versions['t1c.ir']}")
print(f"t1c.bridge version: {versions['t1c.bridge']}")
print(f"t1c.viz version: {versions['t1c.viz']}")

# List available primitives
print(f"Primitives: {sdk.list_primitives()}")
```

Expected output:

```
t1c-sdk version: 0.0.1
t1c.ir version: 0.0.1
t1c.bridge version: 0.0.1
t1c.viz version: 0.0.1
Primitives: ['Affine', 'AvgPool2d', 'Conv2d', 'Flatten', 'LIF', 'MaxPool2d', 'SepConv2d', 'Skip', 'SpikingAffine', 'Upsample']
```

## snnTorch Integration

If you're using snnTorch, install the fork with T1C-IR support:

```bash
git clone <snntorch-fork-repo> snntorch

# Using uv (recommended)
uv pip install -e ./snntorch

# Or using pip
# pip install -e ./snntorch
```

This adds two wrapper functions (they use t1c.ir and t1c.bridge under the hood):
- `snntorch.export_t1cir.export_to_ir()` - Exports snnTorch models
- `snntorch.import_t1cir.import_from_ir()` - Imports T1C-IR graphs

## Troubleshooting

### Import Error: No module named 't1c.ir'

Ensure packages are installed in the correct order. t1c-ir must be installed before t1c-bridge.

```bash
# Using uv (recommended)
uv pip install -e ./t1cir --no-deps
uv pip install -e ./t1ctorch --no-deps
uv pip install -e ./t1cviz --no-deps

# Or using pip
# pip install -e ./t1cir --no-deps
# pip install -e ./t1ctorch --no-deps
# pip install -e ./t1cviz --no-deps
```

### HDF5 Error on Windows

If h5py installation fails, install from conda-forge:

```bash
conda install -c conda-forge h5py
```

### PyTorch Version Mismatch

t1c-bridge requires PyTorch 2.0+. Check your version:

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
cd t1cir && pytest tests/
cd ../t1ctorch && pytest tests/
cd ../t1cviz && pytest tests/
```
