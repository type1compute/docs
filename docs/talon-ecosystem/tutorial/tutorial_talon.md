---
sidebar_position: 13
---

# Tutorial: TALON SDK — Analysis, Profiling & Deployment

**TALON** (Tactical AI at Low-power On-device Nodes) provides tools for model conversion, graph analysis, hardware profiling, linting, and fingerprinting.

## What You'll Learn

1. Export PyTorch models to TALON IR and back
2. Analyze graph structure and statistics
3. Profile for hardware deployment (memory, compute, energy)
4. Compare graphs for change detection
5. Lint graphs for issues and best practices
6. Fingerprint graphs for reproducibility

## Prerequisites

```bash
pip install t1c-talon
```

---

## 1. Setup

```python
import os
import numpy as np
import torch
import torch.nn as nn
from talon import ir, bridge, viz, sdk

os.makedirs("models", exist_ok=True)

print("TALON loaded successfully")
print(f"TALON version: {sdk.__version__}")
```

**Output:**

```
TALON loaded successfully
TALON version: 0.0.1
```

---

## 2. Model Conversion: PyTorch to TALON IR

The SDK enables bidirectional conversion:

- `bridge.to_ir()` — Export PyTorch models to TALON IR
- `bridge.ir_to_torch()` — Import TALON IR graphs as PyTorch executors

### Supported PyTorch Modules

| PyTorch | TALON IR Primitive |
|---------|----------------|
| `nn.Linear` | `Affine` |
| `nn.Conv1d` | `Conv1d` |
| `nn.Conv2d` | `Conv2d` |
| `nn.MaxPool2d` | `MaxPool2d` |
| `nn.AvgPool2d` | `AvgPool2d` |
| `nn.Flatten` | `Flatten` |
| `nn.ReLU` | `ReLU` |
| `nn.BatchNorm2d` | `BatchNorm2d` |
| `snntorch.Leaky` | `LIF` |

### Export a Model

```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 14 * 14, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

model = SimpleCNN()
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**Output:**

```
Model parameters: 31,530
```

```python
sample_input = torch.randn(1, 1, 28, 28)

graph = bridge.to_ir(model, sample_input)

print(f"Exported TALON IR graph:")
print(f"  Nodes: {len(graph.nodes)}")
print(f"  Edges: {len(graph.edges)}")
print(f"  Is DAG: {graph.is_dag}")

ir.write("models/simple_cnn.t1c", graph)
print(f"\nSaved to models/simple_cnn.t1c")
```

**Output:**

```
Exported TALON IR graph:
  Nodes: 7
  Edges: 6
  Is DAG: True

Saved to models/simple_cnn.t1c
```

### Round-Trip Verification

```python
executor = bridge.ir_to_torch(graph)

with torch.no_grad():
    orig_out = model(sample_input)
    exec_out = executor(sample_input)
    diff = (orig_out - exec_out).abs().max().item()

print(f"Roundtrip test:")
print(f"  Max difference: {diff:.2e}")
print(f"  Equivalent: {diff < 1e-5}")
```

**Output:**

```
Roundtrip test:
  Max difference: 0.00e+00
  Equivalent: True
```

---

## 3. Graph Analysis

Analyze graph structure, parameter counts, and compute statistics.

```python
stats = sdk.analyze_graph(graph)

print(f"Graph Analysis:")
print(f"  Nodes: {stats.node_count}")
print(f"  Edges: {stats.edge_count}")
print(f"  Total parameters: {stats.total_params:,}")
print(f"  Total memory: {stats.total_bytes:,} bytes")
print(f"  Depth: {stats.depth}")
print(f"  Width: {stats.width}")

print(f"\nLayer breakdown:")
for node_type, count in stats.type_counts.items():
    print(f"  {node_type}: {count}")
```

**Output:**

```
Graph Analysis:
  Nodes: 7
  Edges: 6
  Total parameters: 31,530
  Total memory: 126,120 bytes
  Depth: 6
  Width: 1

Layer breakdown:
  Input: 1
  Conv2d: 1
  ReLU: 1
  MaxPool2d: 1
  Flatten: 1
  Affine: 1
  Output: 1
```

### Trace Paths

```python
paths = sdk.trace_path(graph, "input", "output")

print(f"Paths from input to output:")
for i, path in enumerate(paths):
    print(f"  Path {i+1}: {' -> '.join(path)}")
```

**Output:**

```
Paths from input to output:
  Path 1: input -> conv1 -> relu -> pool -> flatten -> fc -> output
```

---

## 4. Hardware Profiling

Profile graphs for hardware deployment with memory, compute, and energy estimates.

```python
profile = sdk.profile_graph(graph)

print(f"Hardware Profile:")
print(f"  Weight memory: {profile.weight_memory:,} bytes")
print(f"  Activation memory: {profile.activation_memory:,} bytes")
print(f"  Total memory: {profile.total_memory:,} bytes")
print(f"  Estimated 8-bit quantized: {profile.estimated_quantized_memory:,} bytes")
print(f"\n  MAC operations: {profile.mac_ops:,}")
print(f"  Spike operations: {profile.spike_ops:,}")

print(f"\nRecommendations:")
for rec in profile.recommendations:
    print(f"  - {rec}")
```

**Output:**

```
Hardware Profile:
  Weight memory: 126,120 bytes
  Activation memory: 50,176 bytes
  Total memory: 176,296 bytes
  Estimated 8-bit quantized: 81,706 bytes

  MAC operations: 144,256
  Spike operations: 0

Recommendations:
  - Consider using SpikingAffine for FC layers to enable hardware-optimized quantization
```

---

## 5. Graph Comparison

Compare graphs to detect changes (e.g., before/after training).

```python
graph_modified = ir.read("models/simple_cnn.t1c")

conv_node = graph_modified.nodes["conv1"]
conv_node.weight = conv_node.weight + np.random.randn(
    *conv_node.weight.shape
).astype(np.float32) * 0.01

diff = sdk.compare_graphs(graph, graph_modified)

print(f"Graph Comparison:")
print(f"  Identical: {diff.identical}")
print(f"  Structural match: {diff.structural_match}")
print(f"  Numerical match: {diff.numerical_match}")

print(f"\nModified nodes:")
for node_diff in diff.node_diffs:
    if node_diff.status == "modified":
        print(f"  - {node_diff.name}: {node_diff.changes}")
```

**Output:**

```
Graph Comparison:
  Identical: False
  Structural match: True
  Numerical match: False

Modified nodes:
  - conv1: ['input_type differs', 'output_type differs', 'weight: max_diff=3.32e-02']
  - flatten: ['input_type differs', 'output_type differs']
  - pool: ['input_type differs', 'output_type differs']
  - relu: ['input_type differs', 'output_type differs']
```

---

## 6. Linting & Validation

Check graphs for issues and best practices.

```python
result = sdk.lint_graph(graph)

print(f"Lint Results:")
print(f"  Valid: {result.is_valid}")
print(f"  Errors: {len(result.errors)}")
print(f"  Warnings: {len(result.warnings)}")

if result.warnings:
    print(f"\nWarnings:")
    for warn in result.warnings:
        print(f"  [{warn.code}] {warn.message}")
```

**Output:**

```
Lint Results:
  Valid: True
  Errors: 0
  Warnings: 2

Warnings:
  [PYTHON_KEYWORD] Node name 'input' is a Python keyword
  [PYTHON_KEYWORD] Node name 'output' is a Python keyword
```

---

## 7. Fingerprinting & Provenance

Generate deterministic hashes for reproducibility and add metadata stamps.

```python
fp = sdk.fingerprint_graph(graph)
print(f"Graph fingerprint: {fp[:32]}...")

fp2 = sdk.fingerprint_graph(graph)
print(f"Deterministic: {fp == fp2}")
```

**Output:**

```
Graph fingerprint: d8bdfea603097d8140fa86f96ec64467...
Deterministic: True
```

```python
stamped = sdk.stamp_graph(graph, notes="Tutorial CNN model")

ir.write("models/simple_cnn_stamped.t1c", stamped)
print(f"Stamped graph saved with provenance metadata")
```

**Output:**

```
Stamped graph saved with provenance metadata
```

---

## 8. Visualization

```python
os.makedirs("viz", exist_ok=True)

viz.export_html(graph, "viz/simple_cnn.html", title="Simple CNN")
print(f"Visualization exported to viz/simple_cnn.html")
print(f"Open this file in a browser to explore the graph interactively.")
```

**Output:**

```
Visualization exported to viz/simple_cnn.html
Open this file in a browser to explore the graph interactively.
```

---

## Summary

| Function | Purpose |
|----------|---------|
| `bridge.to_ir()` | Export PyTorch models to TALON IR |
| `bridge.ir_to_torch()` | Import as PyTorch executor |
| `sdk.analyze_graph()` | Graph statistics and metrics |
| `sdk.profile_graph()` | Hardware deployment estimates |
| `sdk.compare_graphs()` | Diff two graphs |
| `sdk.lint_graph()` | Check for issues and best practices |
| `sdk.fingerprint_graph()` | Deterministic hashing |
| `sdk.stamp_graph()` | Add provenance metadata |
| `viz.export_html()` | Interactive visualization |

## What's Next

- [Tutorial: Bridge](./tutorial_bridge) — Advanced export/import with stateful LIF
- [Tutorial: Ghost & Detection](./tutorial_ghost_detection) — GhostNet primitives and detection pipeline
- [Tutorial: Hardware Mapping](./tutorial_hardware_mapping) — Partition, route, place on hardware
- [SDK Reference](../talon) — Full SDK API documentation
