# T1C Tutorial Notebooks

Interactive Jupyter notebooks for learning the T1C ecosystem.

## Notebooks

| Notebook | Description |
|----------|-------------|
| [tutorial_t1cir_basics.ipynb](./tutorial_t1cir_basics.ipynb) | Building neural network graphs with T1C-IR primitives (t1c.ir) |
| [tutorial_t1cviz.ipynb](./tutorial_t1cviz.ipynb) | T1C-Viz: visualization for graphs and neuromorphic events |
| [tutorial_t1c_sdk.ipynb](./tutorial_t1c_sdk.ipynb) | T1C-SDK: analysis, profiling, and deployment |

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
python -m ipykernel install --user --name=t1cir-tutorials --display-name="T1C Tutorials"
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
5. Look for **"t1cir-tutorials"** or the `.venv` path
6. If not visible, click the refresh icon (⟳) in the kernel selector
7. Or reload VSCode window: `Cmd+Shift+P` → "Developer: Reload Window"

The kernel location: `~/Library/Jupyter/kernels/t1cir-tutorials`

## Topics Covered

### T1C-IR Basics (t1c.ir)
- Creating primitives (Input, Output, Affine, LIF, Conv2d)
- Building graphs with nodes and edges
- Serialization (HDF5 format)
- Graph validation
- Convolutional networks
- Skip connections

### T1C-Viz (t1c.viz)
- Graph visualization to HTML
- Event processing (frames, grid, raster)
- Pattern detection (ResNet, SPP)
- Performance benchmarks

### T1C-SDK (t1c.sdk)
- PyTorch to T1C-IR conversion
- Graph analysis and statistics
- Hardware profiling
- Graph comparison
- Linting and validation
- Fingerprinting for reproducibility
