---
sidebar_position: 2
---

# Architecture

## Overview

The T1C-IR ecosystem follows a strict separation of concerns, similar to how ONNX separates IR definition from framework bindings and runtimes.

## Package Dependencies

```
              ┌─────────────┐
              │  t1c.ir     │  ← Core IR (no torch dependency)
              │  (numpy)    │
              └──────┬──────┘
                     │
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
 ┌─────────────────┐   ┌─────────────────┐
 │   t1c.bridge    │   │    t1c.viz      │
 │ (torch, t1c.ir) │   │   (t1c.ir)      │
 └─────────────────┘   └─────────────────┘
```

Key constraints:
- **t1c.ir** has no knowledge of PyTorch or visualization
- **t1c.bridge** may import t1c.ir, never the reverse
- **t1c.viz** may import t1c.ir, does not import t1c.bridge

## t1c.ir: Core IR

The foundation layer defines:

### Primitives

T1C-IR provides 10 primitives for hardware-mapped SNN operations:

| Primitive | Description | Parameters |
|-----------|-------------|------------|
| Affine | Linear transform (y = Wx + b) | weight, bias |
| SpikingAffine | Hardware-optimized affine | weight, bias, spike_mode, weight_bits |
| Conv2d | 2D convolution | weight, bias, stride, padding |
| SepConv2d | Depthwise separable conv | depthwise_weight, pointwise_weight |
| MaxPool2d | 2D max pooling | kernel_size, stride, padding |
| AvgPool2d | 2D average pooling | kernel_size, stride, padding |
| Upsample | 2D upsampling | scale_factor, size, mode |
| Flatten | Reshape to 1D | start_dim, end_dim |
| LIF | Leaky integrate-and-fire neuron | tau, r, v_leak, v_threshold |
| Skip | Residual/skip connection | skip_type |

Note: Input and Output are graph boundary markers, not primitives.

### Graph Structure

```python
from t1c import ir
import numpy as np

graph = ir.Graph(
    nodes={
        'input': ir.Input(np.array([784])),
        'fc1': ir.Affine(weight=W, bias=b),
        'lif1': ir.LIF(tau=tau, r=r, v_leak=v_leak, v_threshold=thr),
        'output': ir.Output(np.array([128]))
    },
    edges=[('input', 'fc1'), ('fc1', 'lif1'), ('lif1', 'output')]
)
```

### Serialization

HDF5 format with `.t1c` extension:
- Stores all node parameters as datasets
- Preserves graph topology in edges group
- Includes version metadata for compatibility

## t1c.bridge: PyTorch Bridge

Handles bidirectional conversion between PyTorch and T1C-IR.

### Export Flow

```python
from t1c import bridge

# Model → Graph
graph = bridge.to_ir(model, sample_input)

# What happens internally:
# 1. Register forward hooks to capture shapes
# 2. Run forward pass with sample input
# 3. Convert each nn.Module to T1C-IR node
# 4. Build edge list from module execution order
```

### Import Flow

```python
# Graph → Executable Module
executor = bridge.ir_to_torch(graph, return_state=True)
output, state = executor(input_tensor, state)

# What happens internally:
# 1. Convert each T1C-IR node to nn.Module
# 2. Build GraphExecutor with topological ordering
# 3. Route tensors through edges during forward pass
```

### LIF Neuron Dynamics

The `LIFModule` implements NIR-compliant dynamics:

```
tau * dv/dt = (v_leak - v) + r * I
```

Discretized as:
```
mem = beta * mem + (1 - beta) * (v_leak + r * x)
spike when mem >= v_threshold
reset to 0 on spike
```

Where `beta = 1 - 1/tau`. This matches snnTorch's Leaky neuron when `r = tau` and `v_leak = 0`.

## t1c.viz: Visualization

Generates self-contained HTML visualizations using D3.js and Dagre.

### Features

- **Automatic layout**: DAG rendering with Dagre algorithm
- **Interactive**: Pan, zoom, node selection
- **Detail panel**: Shows shapes, parameters with dtypes, configuration
- **Standalone**: No server required, works offline

### Output

```python
from t1c import viz

viz.visualize(graph)  # Opens in browser or Jupyter
viz.export_html(graph, "model.html")  # Saves to file
```

## Data Flow Example

Complete workflow from snnTorch model to visualization:

```python
import snntorch as snn
import torch.nn as nn
from t1c import ir, bridge, viz

# 1. Define model
class SNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.lif1 = snn.Leaky(beta=0.9)
        self.fc2 = nn.Linear(128, 10)
        self.lif2 = snn.Leaky(beta=0.9)
    
    def forward(self, x):
        # ... forward pass
        pass

model = SNN()

# 2. Export to T1C-IR
sample = torch.randn(1, 784)
graph = bridge.to_ir(model, sample)

# 3. Serialize
ir.write('model.t1c', graph)

# 4. Visualize
viz.visualize(graph)

# 5. Import back (verification)
executor = bridge.ir_to_torch('model.t1c', return_state=True)
```

## Design Decisions

### Why HDF5?

- Efficient storage of large weight matrices
- Self-describing format with metadata
- Wide language support (Python, C++, MATLAB)
- Compression support for deployment

### Why Separate Packages?

- **Dependency isolation**: t1c.ir works without PyTorch
- **Independent versioning**: IR can evolve separately from bindings
- **Clear ownership**: Different teams maintain different packages
- **Testing**: Each package has focused test suites

