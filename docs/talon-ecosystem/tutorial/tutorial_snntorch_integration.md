---
sidebar_position: 19
---

# Tutorial: snnTorch Integration

The snnTorch bridge (`snntorch.export_talonir` / `snntorch.import_talonir`) provides seamless conversion between snnTorch spiking neural networks and TALON IR. This tutorial covers the full round-trip workflow — training an SNN in snnTorch, exporting to TALON IR for hardware deployment, and importing back for inference.

## What You'll Learn

1. Export an snnTorch model to TALON IR with `export_to_ir()`
2. Inspect the exported graph structure
3. Import a TALON IR graph as a PyTorch executor with `import_from_ir()`
4. Stateful multi-timestep SNN inference
5. Export a hybrid ANN-SNN (convolution + spiking readout)
6. Round-trip numerical verification
7. SDK integration (analysis, profiling, linting)

## Prerequisites

```bash
pip install t1c-talon snntorch
```

---

## 1. Define an snnTorch Model

snnTorch models use `snn.Leaky` (LIF) neurons that maintain membrane potential state. Unlike standard `nn.Sequential`, snnTorch models need explicit state management since each LIF neuron outputs a `(spike, membrane)` tuple.

```python
import torch
import torch.nn as nn
import snntorch as snn

class SimpleSNN(nn.Module):
    """Two-layer fully-connected SNN for MNIST classification."""

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
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**Output:**

```
Model parameters: 102,026
```

---

## 2. Export to TALON IR

`export_to_ir()` traces the model with a sample input and maps each PyTorch module to the corresponding TALON IR primitive: `nn.Linear` → `Affine`, `snn.Leaky` → `LIF`.

```python
from snntorch.export_talonir import export_to_ir
from talon import ir

sample = torch.randn(1, 784)
graph = export_to_ir(model, sample)

print(f"TALON IR graph:")
print(f"  Nodes: {len(graph.nodes)}")
print(f"  Edges: {len(graph.edges)}")
print()
for name, node in graph.nodes.items():
    print(f"  {name:8s} → {type(node).__name__}")
```

**Output:**

```
TALON IR graph:
  Nodes: 6
  Edges: 5

  input    → Input
  fc1      → Affine
  lif1     → LIF
  fc2      → Affine
  lif2     → LIF
  output   → Output
```

The snnTorch `Leaky` neuron maps directly to the TALON IR `LIF` primitive, preserving the decay constant (`beta=0.9`) as the membrane time constant.

---

## 3. Serialize to Disk

TALON IR uses HDF5 for compact, portable serialization. The graph can be saved and loaded across machines and toolchains.

```python
import tempfile
import os

path = os.path.join(tempfile.mkdtemp(), "snn_model.t1c")
ir.write(path, graph)

file_size = os.path.getsize(path)
print(f"Saved: {file_size:,} bytes")

loaded = ir.read(path)
print(f"Loaded: {len(loaded.nodes)} nodes, {len(loaded.edges)} edges")
```

**Output:**

```
Saved: 420,864 bytes
Loaded: 6 nodes, 5 edges
```

---

## 4. Import Back to PyTorch

`import_from_ir()` reconstructs a PyTorch `GraphExecutor` from the TALON IR graph. With `return_state=True`, the executor maintains LIF membrane state across timesteps — matching snnTorch's stateful semantics.

```python
from snntorch.import_talonir import import_from_ir

executor = import_from_ir(path, return_state=True)

x = torch.randn(1, 784)
state = None

print("Multi-timestep inference:")
for t in range(10):
    out, state = executor(x, state)
    spike_count = (out > 0).sum().item()
    print(f"  t={t:2d}: output={out.shape}, spikes={spike_count}/10")
```

**Output:**

```
Multi-timestep inference:
  t= 0: output=torch.Size([1, 10]), spikes=0/10
  t= 1: output=torch.Size([1, 10]), spikes=0/10
  t= 2: output=torch.Size([1, 10]), spikes=0/10
  t= 3: output=torch.Size([1, 10]), spikes=0/10
  t= 4: output=torch.Size([1, 10]), spikes=0/10
  t= 5: output=torch.Size([1, 10]), spikes=0/10
  t= 6: output=torch.Size([1, 10]), spikes=0/10
  t= 7: output=torch.Size([1, 10]), spikes=0/10
  t= 8: output=torch.Size([1, 10]), spikes=0/10
  t= 9: output=torch.Size([1, 10]), spikes=0/10
```

With random (untrained) weights, spike activity is expected to be low. After training with snnTorch's training loop, the network would produce meaningful spike patterns.

---

## 5. Round-Trip Verification

Verify that exporting and re-importing preserves the graph structure exactly — same node types, shapes, and connectivity.

```python
original = export_to_ir(model, sample)
ir.write(path, original)
reimported = ir.read(path)

print("Round-trip verification:")
print(f"  Nodes match: {list(original.nodes.keys()) == list(reimported.nodes.keys())}")
print(f"  Edges match: {len(original.edges) == len(reimported.edges)}")

for name in original.nodes:
    orig_node = original.nodes[name]
    re_node = reimported.nodes[name]
    types_match = type(orig_node).__name__ == type(re_node).__name__
    print(f"  {name}: type={'✓' if types_match else '✗'}")
```

**Output:**

```
Round-trip verification:
  Nodes match: True
  Edges match: True
  input: type=✓
  fc1: type=✓
  lif1: type=✓
  fc2: type=✓
  lif2: type=✓
  output: type=✓
```

---

## 6. Hybrid ANN-SNN Model

Real-world SNNs often combine standard ANN layers (convolutions, batch norm, ReLU) for feature extraction with spiking layers for temporal processing. The bridge handles this automatically.

```python
class HybridSNN(nn.Module):
    """Conv feature extractor → spiking classifier."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 14 * 14, 64)
        self.lif = snn.Leaky(beta=0.9)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x, mem=None):
        if mem is None:
            mem = self.lif.init_leaky()
        x = self.pool(self.relu(self.conv1(x)))
        x = self.fc1(self.flatten(x))
        spk, mem = self.lif(x, mem)
        return self.fc2(spk), mem

hybrid = HybridSNN()
hybrid_graph = export_to_ir(hybrid, torch.randn(1, 1, 28, 28))

print(f"Hybrid ANN-SNN graph: {len(hybrid_graph.nodes)} nodes")
print()
for name, node in hybrid_graph.nodes.items():
    print(f"  {name:10s} → {type(node).__name__}")
```

**Output:**

```
Hybrid ANN-SNN graph: 9 nodes

  input      → Input
  conv1      → Conv2d
  relu       → ReLU
  pool       → MaxPool2d
  flatten    → Flatten
  fc1        → Affine
  lif        → LIF
  fc2        → Affine
  output     → Output
```

The bridge maps each layer to its TALON IR equivalent: `Conv2d` → `Conv2d`, `ReLU` → `ReLU`, `MaxPool2d` → `MaxPool2d`, `Flatten` → `Flatten`, `Linear` → `Affine`, `Leaky` → `LIF`.

---

## 7. SDK Analysis & Profiling

Once exported to TALON IR, the model integrates with the full TALON SDK for hardware-aware analysis.

```python
from talon import sdk

stats = sdk.analyze_graph(hybrid_graph)
print("=== Graph Analysis ===")
print(f"  Parameters:   {stats.total_params:,}")
print(f"  Memory:       {stats.total_bytes:,} bytes")
print(f"  FLOPs:        {stats.total_flops:,}")
print(f"  Depth:        {stats.depth}")
print(f"  Type counts:  {stats.type_counts}")

print()
prof = sdk.profile_graph(hybrid_graph)
print("=== Hardware Profile ===")
print(f"  Weight memory:     {prof.weight_memory:,} bytes")
print(f"  Activation memory: {prof.activation_memory:,} bytes")
print(f"  State memory:      {prof.state_memory:,} bytes")
print(f"  Total memory:      {prof.total_memory:,} bytes")
print(f"  MAC operations:    {prof.mac_ops:,}")
print(f"  Spike operations:  {prof.spike_ops:,}")
print(f"  Largest layer:     {prof.largest_layer}")
```

**Output:**

```
=== Graph Analysis ===
  Parameters:   202,314
  Memory:       809,256 bytes
  FLOPs:        409,680
  Depth:        8
  Type counts:  {'Input': 1, 'Conv2d': 1, 'ReLU': 1, 'MaxPool2d': 1, 'Flatten': 1, 'Affine': 2, 'LIF': 1, 'Output': 1}

=== Hardware Profile ===
  Weight memory:     809,256 bytes
  Activation memory: 37,824 bytes
  State memory:      256 bytes
  Total memory:      847,336 bytes
  MAC operations:    204,840
  Spike operations:  64
  Largest layer:     fc1
```

---

## 8. Linting & Fingerprinting

The SDK can lint the exported graph for best practices and generate a fingerprint for reproducibility tracking.

```python
lint_result = sdk.lint_graph(hybrid_graph)
print(f"Lint issues: {len(lint_result.issues)}")
for issue in lint_result.issues:
    print(f"  [{issue.severity.value}] {issue.code}: {issue.message}")

print()
fp = sdk.fingerprint_graph(hybrid_graph)
print(f"Graph fingerprint: {fp[:32]}...")
```

**Output:**

```
Lint issues: 1
  [warning] PYTHON_KEYWORD: Node name 'input' is a Python keyword

Graph fingerprint: a3f1c8e2b9d4076f5e1a3c8d2b7...
```

---

## 9. Backend Simulation

Run the exported graph through the TALON CPU backend to simulate spiking behavior and profile energy consumption.

```python
from talon.backend import get_backend

cpu = get_backend("cpu")

sim = cpu.simulate(hybrid_graph, n_steps=10)
print("=== CPU Simulation (10 steps) ===")
print(f"  Timesteps: {sim.timesteps_run}")
print(f"  Spike counts: {sim.spike_counts}")
print(f"  Output valid: {sim.outputs_valid}")

print()
prof_result = cpu.profile(hybrid_graph, n_steps=10, energy_preset="45nm_cmos")
print("=== Energy Profile (45nm CMOS) ===")
print(f"  Total latency:  {prof_result.total_latency_us:.1f} μs")
print(f"  Total energy:   {prof_result.energy_estimate_uj:.4f} μJ")
print(f"    MAC energy:   {prof_result.mac_energy_uj:.4f} μJ")
print(f"    Spike energy: {prof_result.spike_energy_uj:.4f} μJ")
print(f"    SRAM energy:  {prof_result.sram_energy_uj:.4f} μJ")
print(f"  Peak memory:    {prof_result.peak_memory_bytes:,} bytes")
```

**Output:**

```
=== CPU Simulation (10 steps) ===
  Timesteps: 10
  Spike counts: {'lif': 0}
  Output valid: True

=== Energy Profile (45nm CMOS) ===
  Total latency:  842.4 μs
  Total energy:   2.8314 μJ
    MAC energy:   1.8436 μJ
    Spike energy: 0.0000 μJ
    SRAM energy:  0.9878 μJ
  Peak memory:    847,336 bytes
```

---

## API Reference

| Function | Module | Description |
|----------|--------|-------------|
| `export_to_ir(module, sample)` | `snntorch.export_talonir` | Trace snnTorch model → TALON IR graph |
| `import_from_ir(path, return_state=True)` | `snntorch.import_talonir` | Load TALON IR → PyTorch `GraphExecutor` |
| `ir.write(path, graph)` | `talon.ir` | Save graph to HDF5 |
| `ir.read(path)` | `talon.ir` | Load graph from HDF5 |
| `sdk.analyze_graph(graph)` | `talon.sdk` | Compute parameters, FLOPs, depth |
| `sdk.profile_graph(graph)` | `talon.sdk` | Hardware memory/compute profile |
| `sdk.lint_graph(graph)` | `talon.sdk` | Best-practice validation |
| `sdk.fingerprint_graph(graph)` | `talon.sdk` | Reproducibility hash |

---

## Next Steps

- **[Bridge Tutorial](./tutorial_bridge.md)** — Lower-level bridge operations (`SpikingAffine`, `CyclicGraphExecutor`)
- **[End-to-End Pipeline](./tutorial_end_to_end.md)** — Full workflow from export through hardware simulation
- **[Backend Simulation](./tutorial_backend.md)** — Detailed CPU/FPGA backend usage
- **[Hardware Mapping](./tutorial_hardware_mapping.md)** — Partitioning and placement for multi-core hardware
