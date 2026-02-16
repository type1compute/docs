---
sidebar_position: 6
---

# Export Guide

Convert PyTorch and snnTorch models to T1C-IR graphs using t1c.bridge.

## Basic Export

```python
import torch
import torch.nn as nn
from t1c import bridge

model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# Sample input with batch dimension
sample = torch.randn(1, 784)

# Export
graph = bridge.to_ir(model, sample)
```

## snnTorch Export

```python
import snntorch as snn

class SNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.lif1 = snn.Leaky(beta=0.9)
        self.fc2 = nn.Linear(128, 10)
        self.lif2 = snn.Leaky(beta=0.9)
    
    def forward(self, x, mem1=None, mem2=None):
        if mem1 is None:
            mem1 = self.lif1.init_leaky()
        if mem2 is None:
            mem2 = self.lif2.init_leaky()
        
        cur1 = self.fc1(x)
        spk1, mem1 = self.lif1(cur1, mem1)
        cur2 = self.fc2(spk1)
        spk2, mem2 = self.lif2(cur2, mem2)
        
        return spk2, mem1, mem2

model = SNN()
sample = torch.randn(1, 784)
graph = bridge.to_ir(model, sample)
```

### Using snnTorch Wrapper

If using the snnTorch fork with T1C-IR support (wrappers use t1c.ir and t1c.bridge under the hood):

```python
from snntorch.export_t1cir import export_to_ir

graph = export_to_ir(model, sample)
```

This is a thin wrapper around `bridge.to_ir()`.

## SpikingAffine Export

For hardware-optimized quantization, use the `spiking` flag:

```python
graph = bridge.to_ir(
    model, 
    sample,
    spiking=True,           # Use SpikingAffine instead of Affine
    spike_mode='binary',    # 'binary', 'graded', 'rate'
    weight_bits=8,          # Weight quantization
    accumulator_bits=16     # MAC accumulator precision
)
```

This converts `nn.Linear` layers to `SpikingAffine` with hardware hints.

## Custom Module Mapping

Override default conversion for specific module types:

```python
from t1c import ir, bridge
import numpy as np

def my_linear_converter(module: nn.Linear) -> ir.Node:
    """Custom converter that always uses SpikingAffine."""
    return ir.SpikingAffine(
        weight=module.weight.detach().numpy(),
        bias=module.bias.detach().numpy() if module.bias is not None 
             else np.zeros(module.out_features, np.float32),
        spike_mode='graded',
        weight_bits=4,
        accumulator_bits=8
    )

custom_map = {
    nn.Linear: my_linear_converter
}

graph = bridge.to_ir(model, sample, custom_map=custom_map)
```

## T1CExporter Class

For advanced control, use the exporter class directly:

```python
from t1c.bridge import T1CExporter

exporter = T1CExporter(
    spiking=True,
    spike_mode='binary',
    weight_bits=8,
    accumulator_bits=16
)

graph = exporter.export(model, sample, custom_map=None)
```

## GraphExecutor Round-Trip

When you import a T1C-IR graph using `bridge.ir_to_torch()`, you get a `GraphExecutor` which preserves the graph's structure. You can then export this back to T1C-IR, and **all topology is preserved** - including multi-branch skip connections.

### Why This Matters

Sequential models export with inferred edges. But imported graphs may have complex topology:

- **Residual connections** (element-wise add)
- **Concatenate connections** (channel concat)
- **Multi-branch paths** (SPP, SPPF, RepConv patterns)

The exporter detects `GraphExecutor` instances and uses their stored edge list instead of inferring edges sequentially.

### Round-Trip Example

```python
from t1c import ir, bridge

# 1. Load graph with residual connections
graph = ir.read("resnet_block.t1c")
print(f"Original: {len(graph.nodes)} nodes, {len(graph.edges)} edges")

# 2. Import to PyTorch for training
executor = bridge.ir_to_torch(graph, return_state=True)

# 3. Train/fine-tune the model
optimizer = torch.optim.Adam(executor.parameters(), lr=1e-4)
for epoch in range(10):
    for x, y in dataloader:
        output, state = executor(x, state)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# 4. Export back to T1C-IR - topology preserved!
sample = torch.randn(1, 784)
trained_graph = bridge.to_ir(executor, sample)

# Verify preservation
assert set(graph.edges) == set(trained_graph.edges)
assert graph.nodes['skip'].skip_type == trained_graph.nodes['skip'].skip_type
print("Topology preserved!")
```

### What's Preserved

| Element | Sequential Export | GraphExecutor Export |
|---------|-------------------|----------------------|
| Node count | Yes | Yes |
| Edge list | Inferred (sequential) | **Exact match** |
| Multi-branch topology | No | **Yes** |
| skip_type (residual/concatenate) | Always passthrough | **Preserved** |
| Weights and biases | Yes | Yes |
| LIF parameters | Yes | Yes |

### T1C-Bridge Module Conversion

For round-trip support, the exporter handles t1c.bridge modules:

| T1C-Bridge Module | T1C-IR Node | Preserved |
|-------------------|------------|-----------|
| `LIFModule` | `LIF` | tau, r, v_leak, v_threshold |
| `SkipModule` | `Skip` | skip_type |
| `SepConv2dModule` | `SepConv2d` | All weights, stride, padding, dilation |

This allows imported models to be re-exported exactly.

## How Export Works

### 1. Shape Capture

The exporter registers forward hooks on all modules:

```python
def hook(module, input, output):
    input_shapes[module] = input[0].shape[1:]   # Exclude batch
    output_shapes[module] = output.shape[1:]    # Exclude batch
```

### 2. Forward Pass

A forward pass with the sample input captures all shapes:

```python
with torch.no_grad():
    model(sample)
```

### 3. Module Conversion

Each module is converted to its T1C-IR equivalent:

| PyTorch | T1C-IR | Notes |
|---------|--------|-------|
| nn.Linear | Affine (or SpikingAffine) | `spiking=True` for SpikingAffine |
| nn.Conv2d | Conv2d | All params preserved |
| nn.Flatten | Flatten | start_dim adjusted |
| nn.MaxPool2d | MaxPool2d | All params preserved |
| nn.AvgPool2d | AvgPool2d | All params preserved |
| nn.Upsample | Upsample | scale_factor, size, mode, align_corners |
| nn.Identity | Skip | passthrough type |
| snn.Leaky | LIF | beta → tau conversion |
| LIFModule | LIF | Round-trip support |
| SkipModule | Skip | Preserves skip_type |

### 4. Graph Assembly

Edges are created based on module execution order:

```python
edges = [
    ('input', 'fc1'),
    ('fc1', 'lif1'),
    ('lif1', 'fc2'),
    ('fc2', 'lif2'),
    ('lif2', 'output')
]
```

## Supported Modules

### Fully Supported

- `nn.Linear` → Affine / SpikingAffine
- `nn.Conv2d` → Conv2d
- `nn.Flatten` → Flatten
- `nn.MaxPool2d` → MaxPool2d
- `nn.AvgPool2d` → AvgPool2d
- `nn.Upsample` → Upsample (spatial upsampling for FPN necks)
- `nn.Identity` → Skip (passthrough)
- `snn.Leaky` → LIF
- `LIFModule` → LIF (round-trip)
- `SkipModule` → Skip (preserves skip_type: residual/concatenate/passthrough)

### Partially Supported

- `nn.Sequential` → Nodes chained automatically
- `nn.ReLU` → Skipped (handled by LIF in SNNs)
- `nn.Dropout` → Skipped (training-only)

### Not Supported

- `nn.BatchNorm*` → Fold into preceding conv/linear
- `nn.LSTM/GRU` → Use spiking equivalents
- Custom modules → Provide custom_map converter

## Troubleshooting

### Empty Graph

If the exported graph has no nodes, check:

1. Model has named modules (not just functions)
2. Sample input has correct shape with batch dimension
3. Forward pass completes without errors

```python
# Debug: print module list
for name, module in model.named_modules():
    print(f"{name}: {type(module)}")
```

### Shape Mismatch

If shapes don't match expected values:

```python
# Check captured shapes
print(f"Input shapes: {exporter._input_shapes}")
print(f"Output shapes: {exporter._output_shapes}")
```

### snnTorch Version

Ensure snnTorch version is compatible:

```python
import snntorch
print(snntorch.__version__)  # Should be 0.9.x
```
