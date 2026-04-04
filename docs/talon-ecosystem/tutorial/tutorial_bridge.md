---
sidebar_position: 14
---

# Tutorial: Bridge — Export & Import

The talon.bridge (`talon.bridge`) provides bidirectional conversion between PyTorch models and TALON IR graphs. This tutorial covers standard export, spiking-aware export, stateful LIF execution, and cyclic (recurrent) networks.

## What You'll Learn

1. Export a PyTorch model to TALON IR
2. Import a TALON IR graph as a PyTorch executor
3. Round-trip verification (numerical equivalence)
4. Spiking-aware export with `SpikingAffine`
5. Stateful LIF execution across timesteps
6. CyclicGraphExecutor for recurrent networks

## Prerequisites

```bash
pip install t1c-talon
```

---

## 1. Standard Export

`bridge.to_ir()` traces a PyTorch model with a sample input and converts each module into the corresponding TALON IR primitive.

```python
import torch
import torch.nn as nn
from talon import bridge, ir

class SimpleSNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

model = SimpleSNN()
sample = torch.randn(1, 784)

graph = bridge.to_ir(model, sample)

print(f"Exported TALON IR graph:")
print(f"  Nodes: {len(graph.nodes)}")
print(f"  Edges: {len(graph.edges)}")
for name, node in graph.nodes.items():
    print(f"  {name}: {type(node).__name__}")
```

**Output:**

```
Exported TALON IR graph:
  Nodes: 5
  Edges: 4
  input: Input
  fc1: Affine
  relu: ReLU
  fc2: Affine
  output: Output
```

---

## 2. Round-Trip Verification

Import the TALON IR graph back as a PyTorch `GraphExecutor` and verify numerical equivalence with the original model.

```python
executor = bridge.ir_to_torch(graph)

with torch.no_grad():
    orig = model(sample)
    exec_out = executor(sample)
    diff = (orig - exec_out).abs().max().item()

print(f"Max difference: {diff:.2e}")
print(f"Equivalent: {diff < 1e-5}")
```

**Output:**

```
Max difference: 0.00e+00
Equivalent: True
```

---

## 3. Spiking-Aware Export

When `spiking=True`, the exporter converts `nn.Linear` layers into `SpikingAffine` primitives, which are optimized for hardware-quantized spike-based computation.

```python
graph_spiking = bridge.to_ir(
    model, sample,
    spiking=True,
    weight_bits=8,
    accumulator_bits=16,
)

for name, node in graph_spiking.nodes.items():
    print(f"  {name}: {type(node).__name__}")
```

**Output:**

```
  input: Input
  fc1: SpikingAffine
  relu: ReLU
  fc2: SpikingAffine
  output: Output
```

:::tip Precision Hints
The `weight_bits` and `accumulator_bits` parameters are recorded as metadata on each primitive. They inform downstream tools (backend profiler, FPGA compiler) about the intended quantization level.
:::

---

## 4. Stateful LIF Execution

For spiking neural networks, LIF neurons maintain membrane potential state across timesteps. Use `return_state=True` to get the state dict back from the executor.

```python
import numpy as np

nodes = {
    "input": ir.Input(np.array([128])),
    "fc": ir.Affine(
        weight=np.random.randn(64, 128).astype(np.float32) * 0.05,
        bias=np.zeros(64, dtype=np.float32),
    ),
    "lif": ir.LIF(
        tau=np.ones(64, dtype=np.float32) * 5.0,
        r=np.ones(64, dtype=np.float32),
        v_leak=np.zeros(64, dtype=np.float32),
        v_threshold=np.ones(64, dtype=np.float32) * 0.5,
    ),
    "output": ir.Output(np.array([64])),
}
edges = [("input", "fc"), ("fc", "lif"), ("lif", "output")]
lif_graph = ir.Graph(nodes=nodes, edges=edges)

executor = bridge.ir_to_torch(lif_graph, return_state=True)

x = torch.randn(1, 128) * 0.3
state = {}
for t in range(5):
    out, state = executor(x, state)
    spike_count = (out > 0).sum().item()
    print(f"  t={t}: spikes={spike_count}/64, state keys={list(state.keys())}")
```

**Output:**

```
  t=0: spikes=0/64, state keys=['lif']
  t=1: spikes=0/64, state keys=['lif']
  t=2: spikes=0/64, state keys=['lif']
  t=3: spikes=0/64, state keys=['lif']
  t=4: spikes=0/64, state keys=['lif']
```

The `state` dictionary holds the membrane potential for each LIF node. Passing it back on the next call allows the membrane to accumulate across timesteps.

---

## 5. CyclicGraphExecutor

For recurrent (cyclic) SNN architectures with feedback connections, use `CyclicGraphExecutor` directly. It accepts pre-built PyTorch modules and an edge list, running the graph for a specified number of internal timesteps per call.

```python
modules = {
    "fc1": nn.Linear(64, 32),
    "lif1": bridge.LIFModule(
        tau=np.ones(32, dtype=np.float32) * 10.0,
        r=np.ones(32, dtype=np.float32),
        v_leak=np.zeros(32, dtype=np.float32),
        v_threshold=np.ones(32, dtype=np.float32),
    ),
    "fc_fb": nn.Linear(32, 64),
}

edges = [
    ("input", "fc1"),
    ("fc1", "lif1"),
    ("lif1", "output"),
    ("lif1", "fc_fb"),
    ("fc_fb", "fc1"),   # feedback connection
]

exec_cyclic = bridge.CyclicGraphExecutor(
    modules=modules,
    edges=edges,
    timesteps=3,
    output_nodes=["lif1"],
    return_state=True,
)

x = torch.randn(1, 64) * 0.1
outputs, state = exec_cyclic(x, {})

print(f"Timestep outputs: {len(outputs)} tensors")
for t, out in enumerate(outputs):
    print(f"  t={t}: shape={out.shape}, spike count={(out > 0).sum().item()}")
print(f"State keys: {list(state.keys())}")
```

**Output:**

```
Timestep outputs: 3 tensors
  t=0: shape=torch.Size([1, 32]), spike count=0
  t=1: shape=torch.Size([1, 32]), spike count=0
  t=2: shape=torch.Size([1, 32]), spike count=0
State keys: ['lif1']
```

The `CyclicGraphExecutor` returns a list of output tensors — one per internal timestep — so you can observe the temporal dynamics of the recurrent network.

---

## 6. Serialization Round-Trip

Save an exported graph, load it back, and verify the executor produces identical results.

```python
ir.write("models/snn_bridge_demo.t1c", graph)
loaded = ir.read("models/snn_bridge_demo.t1c")

exec_loaded = bridge.ir_to_torch(loaded)
with torch.no_grad():
    out_loaded = exec_loaded(sample)
    diff = (exec_out - out_loaded).abs().max().item()

print(f"Serialization round-trip: max diff = {diff:.2e}")
```

**Output:**

```
Serialization round-trip: max diff = 0.00e+00
```

---

## API Reference

| Function / Class | Purpose |
|------------------|---------|
| `bridge.to_ir(model, sample)` | Export PyTorch model to TALON IR graph |
| `bridge.ir_to_torch(graph)` | Import TALON IR graph as `GraphExecutor` |
| `bridge.ir_to_torch(graph, return_state=True)` | Stateful executor returning `(output, state)` |
| `bridge.CyclicGraphExecutor(modules, edges, ...)` | Executor for recurrent/cyclic graphs |
| `bridge.node_to_module(node)` | Convert single IR node to `nn.Module` |
| `bridge.nodes_to_modules(graph)` | Convert all IR nodes to modules dict |

---

## Multi-Output Routing: ChannelSplit and Concat

`ChannelSplit` splits a tensor into two or more channel groups and routes each group to a different downstream branch. The `GraphExecutor` treats these differently from stateful LIF nodes:

- **Stateful nodes** (LIF, IF) return `(spike, membrane_state, ...)` — the executor caches the spike and carries the rest as state across timesteps.
- **Multi-output routing nodes** (`ChannelSplit`) return a tuple of tensors, one per outgoing branch. The executor caches the full tuple and indexes into it per outgoing edge.

This distinction is marked by `_is_multi_output = True` on `ChannelSplitModule` and respected by both `GraphExecutor` and `CyclicGraphExecutor`.

```python
import numpy as np, torch
from talon import ir, bridge

nodes = {
    "input":  ir.Input(np.array([32])),
    "split":  ir.ChannelSplit(split_sections=(16, 16)),
    "branch0": ir.Affine(weight=np.eye(16, dtype=np.float32), bias=np.zeros(16, dtype=np.float32)),
    "branch1": ir.Affine(weight=np.eye(16, dtype=np.float32), bias=np.zeros(16, dtype=np.float32)),
    "concat": ir.Concat(dim=1),
    "output": ir.Output(np.array([32])),
}
edges = [
    ("input",   "split"),
    ("split",   "branch0"),
    ("split",   "branch1"),
    ("branch0", "concat"),
    ("branch1", "concat"),
    ("concat",  "output"),
]
graph = ir.Graph(nodes=nodes, edges=edges)
executor = bridge.ir_to_torch(graph)

x = torch.randn(1, 32)
with torch.no_grad():
    y = executor(x)
print(y.shape)   # torch.Size([1, 32])
```

The `Concat` module respects its `dim` parameter (default `1` = channel dimension), which is propagated correctly by the executor.

## What's Next

- [Tutorial: Ghost & Detection](./tutorial_ghost_detection) — GhostNet architecture and detection pipeline
- [Tutorial: Hardware Mapping](./tutorial_hardware_mapping) — Partition, route, place on hardware
- [Export Guide](../export) — Full export API reference
- [Import Guide](../import) — Full import API reference
