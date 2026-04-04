---
sidebar_position: 4
---

# Quick Start

Export your first snnTorch model to TALON IR in under 5 minutes.

:::tip SDK Alternative
You can also use the unified SDK: `from talon import sdk` which includes all packages.
:::

## Step 1: Define a Model

```python
import torch
import torch.nn as nn
import snntorch as snn

class SimpleSNN(nn.Module):
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

model = SimpleSNN()
```

## Step 2: Export to TALON IR

```python
from talon import bridge, ir

# Sample input (batch_size=1, features=784)
sample = torch.randn(1, 784)

# Export
graph = bridge.to_ir(model, sample)

print(f"Nodes: {list(graph.nodes.keys())}")
print(f"Edges: {graph.edges}")
```

Output:

```
Nodes: ['input', 'fc1', 'lif1', 'fc2', 'lif2', 'output']
Edges: [('input', 'fc1'), ('fc1', 'lif1'), ('lif1', 'fc2'), ('fc2', 'lif2'), ('lif2', 'output')]
```

## Step 3: Save to File

```python
# Save as HDF5
ir.write('model.t1c', graph)

# Verify
print(f"Saved. Version: {ir.read_version('model.t1c')}")
```

## Step 4: Visualize

```python
from talon import viz

# Display in browser or Jupyter
viz.visualize(graph, title="SimpleSNN")

# Or export to HTML file
path = viz.export_html(graph, "model.html")
print(f"Visualization: {path}")
```

## Step 5: Import and Verify

```python
# Load graph
loaded = ir.read('model.t1c')

# Convert to executable PyTorch module
executor = bridge.ir_to_torch(loaded, return_state=True)

# Run inference
x = torch.randn(1, 784)
state = {}

for t in range(10):  # 10 timesteps
    output, state = executor(x, state)
    print(f"t={t}: spikes={output.sum().item():.0f}")
```

## Complete Example

```python
import torch
import torch.nn as nn
import snntorch as snn
from talon import ir, bridge, viz

# 1. Model
class SimpleSNN(nn.Module):
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

model = SimpleSNN()

# 2. Export
sample = torch.randn(1, 784)
graph = bridge.to_ir(model, sample)

# 3. Save
ir.write('model.t1c', graph)

# 4. Visualize
viz.visualize(graph)

# 5. Import and run
executor = bridge.ir_to_torch('model.t1c', return_state=True)
output, state = executor(torch.randn(1, 784), {})
print(f"Output shape: {output.shape}")
```

## Tutorials

Step-by-step guides with inline code and output:

- [Tutorial: TALON IR Basics](./tutorial/tutorial_t1cir_basics) — Primitives, graphs, serialization, ghost & detection
- [Tutorial: T1CViz](./tutorial/tutorial_t1cviz) — Graph visualization and event processing
- [Tutorial: TALON SDK](./tutorial/tutorial_talon) — Analysis, profiling, linting, fingerprinting
- [Tutorial: Bridge](./tutorial/tutorial_bridge) — Export/import, stateful LIF, CyclicGraphExecutor
- [Tutorial: Ghost & Detection](./tutorial/tutorial_ghost_detection) — GhostNet primitives and detection pipeline
- [Tutorial: Hardware Mapping](./tutorial/tutorial_hardware_mapping) — Partition, allocate, route, place
- [Tutorial: Backend](./tutorial/tutorial_backend) — Simulate, profile, energy presets, FPGA
- [Tutorial: Event I/O](./tutorial/tutorial_event_io) — Neural encoding, HDF5, throughput
- [Tutorial: snnTorch Integration](./tutorial/tutorial_snntorch_integration) — snnTorch export/import round-trip with SDK analysis
- [Tutorial: End-to-End Pipeline](./tutorial/tutorial_end_to_end) — Full workflow from model definition through hardware simulation

For local notebook execution, see the [example_notebooks](./example_notebooks/) directory.

## What's Next

- [Primitives Reference](./primitives) — All 36 TALON IR primitives
- [Export Guide](./export) — Advanced export options
- [Import Guide](./import) — GraphExecutor details
- [Visualization](./visualization) — Customizing visualizations

