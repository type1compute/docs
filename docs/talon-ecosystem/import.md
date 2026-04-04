---
sidebar_position: 7
---

# Import Guide

Convert TALON IR graphs back to executable PyTorch modules using talon.bridge.

## Basic Import

```python
from talon import ir, bridge

# Load from file
graph = ir.read('model.t1c')

# Convert to PyTorch module
executor = bridge.ir_to_torch(graph)

# Run inference
output = executor(input_tensor)
```

## Stateful Execution

For models with LIF neurons, enable state tracking:

```python
executor = bridge.ir_to_torch(graph, return_state=True)

# Initialize state
state = {}

# Run timesteps
for t in range(num_timesteps):
    output, state = executor(input_tensor, state)
    print(f"t={t}: output={output.sum().item():.2f}")
```

The state dictionary maps node names to their internal states:

```python
print(state.keys())  # dict_keys(['lif1', 'lif2'])
```

## Using snnTorch Wrapper

If using the snnTorch fork (wrappers use talon.ir and talon.bridge under the hood):

```python
from snntorch.import_t1cir import import_from_ir

executor = import_from_ir('model.t1c', return_state=True)
```

## GraphExecutor

The imported module is a `GraphExecutor` instance:

```python
from talon.bridge import GraphExecutor

executor = bridge.ir_to_torch(graph, return_state=True)
print(type(executor))  # <class 'talon.bridge.executor.GraphExecutor'>

# Access submodules
for name, module in executor.named_children():
    print(f"{name}: {type(module).__name__}")
```

Output:

```
input: Identity
fc1: Linear
lif1: LIFModule
fc2: Linear
lif2: LIFModule
output: Identity
```

## LIFModule

The LIF neuron is imported as `LIFModule`:

```python
from talon.bridge import LIFModule

# Access parameters
lif = executor.lif1
print(f"tau: {lif.tau}")
print(f"beta: {lif.beta}")  # Computed: 1 - 1/tau
print(f"threshold: {lif.v_threshold}")
```

### Forward Signature

```python
def forward(self, x: torch.Tensor, mem: Optional[torch.Tensor] = None) -> tuple:
    """
    Args:
        x: Input current
        mem: Previous membrane potential (optional)
    
    Returns:
        (spikes, membrane_potential)
    """
```

### Dynamics

Implements NIR-compliant discretized LIF:

```python
# Update membrane
mem = beta * mem + (1 - beta) * (v_leak + r * x)

# Generate spikes
spk = (mem >= v_threshold).float()

# Reset
mem = mem * (1 - spk)

return spk, mem
```

## Custom Node Mapping

Override default import for specific node types:

```python
def my_lif_converter(node: ir.LIF) -> nn.Module:
    """Custom LIF that uses snnTorch Leaky instead."""
    import snntorch as snn
    beta = 1.0 - 1.0 / node.tau[0]
    return snn.Leaky(beta=beta)

executor = bridge.ir_to_torch(
    graph, 
    node_map=my_lif_converter,
    return_state=True
)
```

## Low-Level Import

For more control, use the module-level functions:

```python
from talon.bridge import from_ir, GraphExecutor

# Get dict of modules
modules = from_ir(graph)
print(modules.keys())  # dict_keys(['input', 'fc1', 'lif1', ...])

# Build executor manually
executor = GraphExecutor(
    modules=modules,
    edges=graph.edges,
    return_state=True
)
```

## Node-by-Node Conversion

Convert individual nodes:

```python
from talon.bridge import node_to_module

# Convert single node
lif_node = graph.nodes['lif1']
lif_module = node_to_module(lif_node)
print(type(lif_module))  # <class 'talon.bridge.from_ir.LIFModule'>
```

## Supported Nodes

| TALON IR | PyTorch | Notes |
|--------|---------|-------|
| Input | nn.Identity | Graph marker |
| Output | nn.Identity | Graph marker |
| Affine | nn.Linear | |
| SpikingAffine | nn.Linear | Quantization hints in metadata |
| Conv1d | nn.Conv1d | 1D temporal convolution |
| Conv2d | nn.Conv2d | |
| Flatten | nn.Flatten | |
| MaxPool2d | nn.MaxPool2d | |
| AvgPool2d | nn.AvgPool2d | |
| Upsample | nn.Upsample | Spatial upsampling (FPN necks) |
| SepConv2d | SepConv2dModule | Preserves identity for round-trip |
| Skip | SkipModule | Preserves skip_type |
| LIF | LIFModule | Stateful neuron |
| ReLU | nn.ReLU | Supports negative_slope (LeakyReLU) |
| Sigmoid | nn.Sigmoid | |
| Tanh | nn.Tanh | |
| Softmax | nn.Softmax | Preserves dim |
| GELU | nn.GELU | Preserves approximate flag |
| ELU | nn.ELU | Preserves alpha |
| PReLU | nn.PReLU | Preserves learned weight |
| BatchNorm1d | nn.BatchNorm1d | Preserves running stats |
| BatchNorm2d | nn.BatchNorm2d | Preserves running stats |
| LayerNorm | nn.LayerNorm | |
| Dropout | nn.Dropout | Preserves p, no-op in eval |
| HybridRegion | nn.Identity | ANN/SNN boundary marker |
| ChannelSplit | ChannelSplitModule | Multi-output channel split |
| Concat | ConcatModule | Channel concatenation |
| SGhostConv | SGhostConvModule | Primary + cheap depthwise concat |
| SGhostEncoderLite | SGhostEncoderLiteModule | Ghost stem encoder |
| GhostBasicBlock1 | GhostBasicBlock1Module | CSP-ELAN backbone (stride-2) |
| GhostBasicBlock2 | GhostBasicBlock2Module | CSP-ELAN FPN head |
| SDDetect | SDDetectModule | Per-scale detection head |
| DFLDecode | DFLDecodeModule | DFL box decoder |
| Dist2BBox | Dist2BBoxModule | Distance to bounding box |
| NMS | NMSModule | Non-Maximum Suppression |
| SConv | nn.Conv2d | Spiking standard conv |
| SDConv | nn.Conv2d | Spiking depthwise conv |
| IF | IFModule | Integrate-and-fire (no leak) |

## SepConv2dModule

Depthwise separable convolution preserving TALON IR identity for round-trip:

```python
from talon.bridge import SepConv2dModule

# Structure: depthwise (groups=in_ch) -> pointwise (1x1)
sepconv = executor.sepconv1
print(f"Depthwise: {sepconv.depthwise}")
print(f"Pointwise: {sepconv.pointwise}")
print(f"Stride: {sepconv.stride}")
```

The module stores all configuration needed for export back to TALON IR:

- `depthwise`: nn.Conv2d with groups=in_channels
- `pointwise`: nn.Conv2d with 1x1 kernel
- `stride`, `padding`, `dilation`: Applied to depthwise conv

## SkipModule

The Skip primitive is imported as `SkipModule`, which preserves the `skip_type` for proper merge behavior:

```python
from talon.bridge import SkipModule

skip = SkipModule(skip_type='residual')
print(skip.skip_type)  # 'residual'
```

### Skip Types and Merge Behavior

When multiple edges feed into a Skip node, the `GraphExecutor` performs the merge based on `skip_type`:

| skip_type | Merge Operation | Use Case |
|-----------|-----------------|----------|
| `passthrough` | Identity (single input) | SPPF identity branch |
| `residual` | Element-wise add | ResNet blocks |
| `concatenate` | Channel concat (dim=1) | SPP, DenseNet |

Example of residual merge:

```python
# Graph: input -> fc -> skip, input -> skip (residual connection)
modules = {
    'input': nn.Identity(),
    'fc': nn.Linear(10, 10),
    'skip': SkipModule('residual'),  # Adds inputs together
    'output': nn.Identity(),
}
edges = [
    ('input', 'fc'),
    ('fc', 'skip'),      # main path
    ('input', 'skip'),   # residual path (added to main)
    ('skip', 'output'),
]

executor = GraphExecutor(modules, edges)
# Output: fc(x) + x
```

Example of concatenate merge:

```python
modules = {
    'input': nn.Identity(),
    'pool5': nn.MaxPool2d(5, stride=1, padding=2),
    'pool9': nn.MaxPool2d(9, stride=1, padding=4),
    'concat': SkipModule('concatenate'),  # Concatenates channels
    'output': nn.Identity(),
}
edges = [
    ('input', 'pool5'),
    ('input', 'pool9'),
    ('pool5', 'concat'),   # 64 channels
    ('pool9', 'concat'),   # 64 channels
    ('concat', 'output'),  # 128 channels (concatenated)
]
```

## Round-Trip Verification

Verify export/import produces identical results:

```python
import torch

# Original model
model = SNN()
model.eval()

# Export
sample = torch.randn(1, 784)
graph = bridge.to_ir(model, sample)

# Import
executor = bridge.ir_to_torch(graph, return_state=True)
executor.eval()

# Compare outputs
x = torch.randn(1, 784)
state = {}

with torch.no_grad():
    # Original
    mem1 = model.lif1.init_leaky()
    mem2 = model.lif2.init_leaky()
    out_orig, _, _ = model(x, mem1, mem2)
    
    # Imported
    out_import, state = executor(x, state)

# Should be very close
print(f"Max diff: {(out_orig - out_import).abs().max().item()}")
assert torch.allclose(out_orig, out_import, atol=1e-5)
```

## Troubleshooting

### State Key Errors

If state dict has wrong keys:

```python
# Check expected keys
print([name for name, m in executor.named_children() 
       if isinstance(m, LIFModule)])
```

### Shape Mismatch

If input shapes don't match:

```python
# Check graph input shape
input_node = graph.nodes['input']
expected_shape = input_node.input_type['input']
print(f"Expected: {expected_shape}")
```

### Missing Node Type

If a node type isn't supported:

```python
from talon.bridge import node_to_module

# Returns None for unsupported types
result = node_to_module(unknown_node)
if result is None:
    print(f"Unsupported: {type(unknown_node)}")
```

## Tutorials

- [Tutorial: Bridge](./tutorial/tutorial_bridge) — Step-by-step import with stateful LIF and CyclicGraphExecutor
- [Tutorial: Hardware Mapping](./tutorial/tutorial_hardware_mapping) — Partition, allocate, route, and place imported graphs
- [Tutorial: snnTorch Integration](./tutorial/tutorial_snntorch_integration) — snnTorch import/export round-trip
- [Tutorial: End-to-End Pipeline](./tutorial/tutorial_end_to_end) — Full import through simulation pipeline
